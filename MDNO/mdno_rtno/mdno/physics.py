"""Physics modules for the MDNO model."""
import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MDNOConfig, TurbulenceModel
from .constants import CONSTANTS
from .numerics import MomentPreservingProjection, SpectralAdvectionFixed
from ._logging import LOGGER as logger
from .monitoring import monitor

class CompleteBoltzmannSolver(nn.Module):
    """Complete Boltzmann equation solver with full velocity space"""
    
    def __init__(self, config: MDNOConfig):
        super().__init__()
        self.config = config
        self.nv = config.velocity_space_resolution
        
        # Velocity space grid
        v_max = config.max_velocity
        self.register_buffer('v_grid', torch.linspace(-v_max, v_max, self.nv))
        self.dv = self.v_grid[1] - self.v_grid[0]
        
        # 3D velocity mesh
        vx, vy, vz = torch.meshgrid(self.v_grid, self.v_grid, self.v_grid, indexing='ij')
        self.register_buffer('vx', vx)
        self.register_buffer('vy', vy)
        self.register_buffer('vz', vz)
        self.register_buffer('v_magnitude', torch.sqrt(vx**2 + vy**2 + vz**2))
        
        # Neural collision operator
        self.collision_net = nn.Sequential(
            nn.Conv3d(1, 64, 5, padding=2),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv3d(64, 128, 5, padding=2),
            nn.GroupNorm(16, 128),
            nn.SiLU(),
            nn.Conv3d(128, 64, 5, padding=2),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv3d(64, 1, 5, padding=2)
        )
        
        # BGK relaxation time
        self.tau = nn.Parameter(torch.tensor(1.0))
        
        logger.info(f"Boltzmann solver initialized: velocity space {self.nv}^3")
    
    def maxwell_boltzmann(self, density: torch.Tensor, velocity: torch.Tensor,
                         temperature: torch.Tensor) -> torch.Tensor:
        """Initialize Maxwell-Boltzmann distribution"""
        batch_size = density.shape[0]
        spatial_shape = density.shape[1:]
        
        n = density.view(batch_size, 1, 1, 1, *spatial_shape)
        u = velocity.view(batch_size, 3, 1, 1, 1, *spatial_shape) if velocity.dim() > 2 else velocity.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        T = temperature.view(batch_size, 1, 1, 1, *spatial_shape)
        
        vx_exp = self.vx.view(1, self.nv, self.nv, self.nv, 1, 1, 1)
        vy_exp = self.vy.view(1, self.nv, self.nv, self.nv, 1, 1, 1)
        vz_exp = self.vz.view(1, self.nv, self.nv, self.nv, 1, 1, 1)
        
        # Relative velocities
        if velocity.dim() > 2:
            vx_rel = vx_exp - u[:, 0:1]
            vy_rel = vy_exp - u[:, 1:2]
            vz_rel = vz_exp - u[:, 2:3]
        else:
            vx_rel = vx_exp
            vy_rel = vy_exp
            vz_rel = vz_exp
        
        m, k = 1.0, 1.0
        prefactor = n * (m / (2 * math.pi * k * T + 1e-10))**(3/2)
        exponent = -m * (vx_rel**2 + vy_rel**2 + vz_rel**2) / (2 * k * T + 1e-10)
        
        f = prefactor * torch.exp(torch.clamp(exponent, min=-50, max=50))
        return f
    
    def forward(self, f: torch.Tensor, forces: torch.Tensor, dt: float) -> torch.Tensor:
        """Evolve distribution function with moment conservation"""
        with monitor.timer("boltzmann_step"):
            # Advection in physical space
            f_advected = self._spectral_advection(f, dt)
            
            # Force term in velocity space
            f_forced = self._velocity_space_advection(f_advected, forces, dt)
            
            # Collision operator
            f_collision = self._compute_collision(f_forced)
            
            # Time integration
            f_new = f_forced + dt * f_collision
            
            # Enforce positivity and conservation
            f_new = F.relu(f_new)
            if self.config.enforce_moment_conservation:
                proj = MomentPreservingProjection()
                if self.config.use_entropic_projection:
                    f_new = proj.entropic_projection(f_new, f_forced, self.dv.item(), self.v_grid)
                else:
                    f_new = proj.simple_moment_correction(f_new, f_forced, self.dv.item())
        
        return f_new
    
    def _spectral_advection(self, f: torch.Tensor, dt: float) -> torch.Tensor:
        """Spectral advection in physical space using LOCAL velocity field

        PERFORMANCE OPTIMIZATION: Vectorized version replaces O(nv) nested loops.
        Processes all velocity space cells in parallel with batched operations.
        Typical speedup: 10-100x for nv >= 8.
        """
        # Extract moments to get LOCAL velocity field
        moments = self.compute_moments(f)
        velocity = moments['velocity']  # AUDIT FIX: [batch, 3, nz, ny, nx]

        batch_size = f.shape[0]
        spatial_shape = f.shape[4:]

        # Apply derivative-based spectral advection
        f_advected = f.clone()

        # Only apply if we have spatial resolution
        if spatial_shape[0] > 1 and spatial_shape[1] > 1 and spatial_shape[2] > 1:
            advection_helper = SpectralAdvectionFixed()

            # OPTIMIZATION: Vectorized advection - process all velocity cells in parallel
            # Reshape f from [batch, nv, nv, nv, nz, ny, nx] to [batch*nv, nz, ny, nx]
            nv3 = self.nv ** 3
            f_reshaped = f.reshape(batch_size, nv3, *spatial_shape)

            # Expand velocity for broadcasting: all velocity cells use same local field
            # [batch, 3, nz, ny, nx] -> [batch*nv, 3, nz, ny, nx]
            velocity_expanded = velocity.unsqueeze(1).expand(batch_size, nv3, 3, *spatial_shape)
            velocity_expanded = velocity_expanded.reshape(batch_size * nv3, 3, *spatial_shape)

            # Batch process velocity cells for memory efficiency
            batch_chunk_size = min(64, nv3)  # Tune based on GPU memory
            f_advected_flat = torch.zeros(batch_size * nv3, *spatial_shape,
                                         device=f.device, dtype=f.dtype)

            for i in range(0, batch_size * nv3, batch_chunk_size):
                end_idx = min(i + batch_chunk_size, batch_size * nv3)
                f_advected_flat[i:end_idx] = advection_helper.derivative_based_advection(
                    f_reshaped.reshape(batch_size * nv3, *spatial_shape)[i:end_idx],
                    velocity_expanded[i:end_idx],
                    dt, self.config.dx, self.config.dy, self.config.dz,
                    self.config.dealiasing_fraction
                )

            # Reshape back to original shape
            f_advected = f_advected_flat.reshape(batch_size, self.nv, self.nv, self.nv, *spatial_shape)

        return f_advected
    
    def _velocity_space_advection(self, f: torch.Tensor, forces: torch.Tensor, dt: float) -> torch.Tensor:
        """Advection in velocity space due to forces"""
        # Velocity gradients with proper boundary handling
        grad_vx = torch.zeros_like(f)
        grad_vy = torch.zeros_like(f)
        grad_vz = torch.zeros_like(f)
        
        if self.nv > 2:
            # Central differences for interior
            grad_vx[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2 * self.dv)
            grad_vy[:, :, 1:-1] = (f[:, :, 2:] - f[:, :, :-2]) / (2 * self.dv)
            grad_vz[:, :, :, 1:-1] = (f[:, :, :, 2:] - f[:, :, :, :-2]) / (2 * self.dv)
            
            # One-sided differences at boundaries
            grad_vx[:, 0] = (f[:, 1] - f[:, 0]) / self.dv
            grad_vx[:, -1] = (f[:, -1] - f[:, -2]) / self.dv
            grad_vy[:, :, 0] = (f[:, :, 1] - f[:, :, 0]) / self.dv
            grad_vy[:, :, -1] = (f[:, :, -1] - f[:, :, -2]) / self.dv
            grad_vz[:, :, :, 0] = (f[:, :, :, 1] - f[:, :, :, 0]) / self.dv
            grad_vz[:, :, :, -1] = (f[:, :, :, -1] - f[:, :, :, -2]) / self.dv
        
        # Expand forces to velocity space dimensions
        fx = forces[:, 0:1].view(forces.shape[0], 1, 1, 1, *forces.shape[2:])
        fy = forces[:, 1:2].view(forces.shape[0], 1, 1, 1, *forces.shape[2:])
        fz = forces[:, 2:3].view(forces.shape[0], 1, 1, 1, *forces.shape[2:])
        
        # Force term: -F*nabla_v f
        force_term = fx * grad_vx + fy * grad_vy + fz * grad_vz
        f_forced = f - dt * force_term
        
        return f_forced
    
    def _compute_collision(self, f: torch.Tensor) -> torch.Tensor:
        """Compute collision operator"""
        # BGK approximation
        f_eq = self._compute_equilibrium(f)
        bgk_collision = -(f - f_eq) / self.tau
        
        # Neural correction
        batch_size = f.shape[0]
        nx, ny, nz = f.shape[-3:]
        
        f_reshaped = f.permute(0, 4, 5, 6, 1, 2, 3).reshape(
            batch_size * nx * ny * nz, 1, self.nv, self.nv, self.nv
        )
        
        neural_collision = self.collision_net(f_reshaped)
        neural_collision = neural_collision.reshape(
            batch_size, nx, ny, nz, self.nv, self.nv, self.nv
        ).permute(0, 4, 5, 6, 1, 2, 3)
        
        collision = bgk_collision + 0.1 * neural_collision
        collision = self._ensure_collision_conservation(collision)
        
        return collision
    
    def _compute_equilibrium(self, f: torch.Tensor) -> torch.Tensor:
        """Compute local equilibrium"""
        moments = self.compute_moments(f)
        f_eq = self.maxwell_boltzmann(
            moments['density'],
            moments['velocity'],
            moments['temperature']
        )
        return f_eq
    
    def _ensure_collision_conservation(self, collision: torch.Tensor) -> torch.Tensor:
        """Ensure collision conserves mass, momentum, energy"""
        dv3 = self.dv ** 3
        
        # Velocity grids
        vx = self.vx.view(1, self.nv, self.nv, self.nv, 1, 1, 1)
        vy = self.vy.view(1, self.nv, self.nv, self.nv, 1, 1, 1)
        vz = self.vz.view(1, self.nv, self.nv, self.nv, 1, 1, 1)
        v_sq = vx**2 + vy**2 + vz**2
        
        # Compute collision moments (should all be zero)
        M0_coll = torch.sum(collision, dim=(1, 2, 3), keepdim=True) * dv3
        M1x_coll = torch.sum(collision * vx, dim=(1, 2, 3), keepdim=True) * dv3
        M1y_coll = torch.sum(collision * vy, dim=(1, 2, 3), keepdim=True) * dv3
        M1z_coll = torch.sum(collision * vz, dim=(1, 2, 3), keepdim=True) * dv3
        M2_coll = torch.sum(collision * v_sq, dim=(1, 2, 3), keepdim=True) * dv3
        
        # Project out all violations
        collision_corrected = collision - M0_coll / (self.nv**3 * dv3)
        
        # Momentum conservation
        basis_vx = vx / (torch.sum(vx**2) * dv3 + 1e-10)
        basis_vy = vy / (torch.sum(vy**2) * dv3 + 1e-10)
        basis_vz = vz / (torch.sum(vz**2) * dv3 + 1e-10)
        
        collision_corrected = collision_corrected - M1x_coll * basis_vx
        collision_corrected = collision_corrected - M1y_coll * basis_vy
        collision_corrected = collision_corrected - M1z_coll * basis_vz
        
        # Energy conservation
        basis_v2 = v_sq / (torch.sum(v_sq**2) * dv3 + 1e-10)
        M2_coll_updated = torch.sum(collision_corrected * v_sq, dim=(1, 2, 3), keepdim=True) * dv3
        collision_corrected = collision_corrected - M2_coll_updated * basis_v2
        
        return collision_corrected
    
    def compute_moments(self, f: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute macroscopic moments from distribution"""
        dv3 = self.dv ** 3
        
        # Density
        density = torch.sum(f, dim=(1, 2, 3)) * dv3
        
        # Velocity grids
        vx = self.vx.view(1, self.nv, self.nv, self.nv, 1, 1, 1)
        vy = self.vy.view(1, self.nv, self.nv, self.nv, 1, 1, 1)
        vz = self.vz.view(1, self.nv, self.nv, self.nv, 1, 1, 1)
        
        # Momentum
        momentum_x = torch.sum(vx * f, dim=(1, 2, 3)) * dv3
        momentum_y = torch.sum(vy * f, dim=(1, 2, 3)) * dv3
        momentum_z = torch.sum(vz * f, dim=(1, 2, 3)) * dv3
        
        # Velocity
        velocity = torch.stack([
            momentum_x / (density + 1e-10),
            momentum_y / (density + 1e-10),
            momentum_z / (density + 1e-10)
        ], dim=1)
        
        # Temperature
        u = velocity.view(velocity.shape[0], 3, 1, 1, 1, *velocity.shape[2:]) if velocity.dim() > 2 else velocity.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        v_rel_sq = (vx - u[:, 0:1])**2 + (vy - u[:, 1:2])**2 + (vz - u[:, 2:3])**2
        kinetic_energy = torch.sum(v_rel_sq * f, dim=(1, 2, 3)) * dv3
        temperature = 2 * kinetic_energy / (3 * density + 1e-10)
        
        # Stress tensor
        stress = self._compute_stress_tensor(f, velocity)
        
        # Heat flux
        heat_flux = self._compute_heat_flux(f, velocity, temperature)
        
        return {
            'density': density,
            'velocity': velocity,
            'temperature': temperature,
            'pressure': density * temperature,
            'stress': stress,
            'heat_flux': heat_flux
        }
    
    def _compute_stress_tensor(self, f: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
        """Compute stress tensor"""
        dv3 = self.dv ** 3
        
        vx = self.vx.view(1, self.nv, self.nv, self.nv, 1, 1, 1)
        vy = self.vy.view(1, self.nv, self.nv, self.nv, 1, 1, 1)
        vz = self.vz.view(1, self.nv, self.nv, self.nv, 1, 1, 1)
        
        u = velocity.view(velocity.shape[0], 3, 1, 1, 1, *velocity.shape[2:]) if velocity.dim() > 2 else velocity.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        # Peculiar velocities
        cx = vx - u[:, 0:1] if velocity.dim() > 2 else vx
        cy = vy - u[:, 1:2] if velocity.dim() > 2 else vy
        cz = vz - u[:, 2:3] if velocity.dim() > 2 else vz
        
        # Stress components
        Pxx = torch.sum(cx * cx * f, dim=(1, 2, 3)) * dv3
        Pyy = torch.sum(cy * cy * f, dim=(1, 2, 3)) * dv3
        Pzz = torch.sum(cz * cz * f, dim=(1, 2, 3)) * dv3
        Pxy = torch.sum(cx * cy * f, dim=(1, 2, 3)) * dv3
        Pxz = torch.sum(cx * cz * f, dim=(1, 2, 3)) * dv3
        Pyz = torch.sum(cy * cz * f, dim=(1, 2, 3)) * dv3
        
        # Assemble tensor
        batch_size = f.shape[0]
        spatial_shape = f.shape[4:]
        stress = torch.zeros(batch_size, 3, 3, *spatial_shape, device=f.device)
        
        stress[:, 0, 0] = Pxx
        stress[:, 1, 1] = Pyy
        stress[:, 2, 2] = Pzz
        stress[:, 0, 1] = stress[:, 1, 0] = Pxy
        stress[:, 0, 2] = stress[:, 2, 0] = Pxz
        stress[:, 1, 2] = stress[:, 2, 1] = Pyz
        
        return stress
    
    def _compute_heat_flux(self, f: torch.Tensor, velocity: torch.Tensor,
                          temperature: torch.Tensor) -> torch.Tensor:
        """Compute heat flux"""
        dv3 = self.dv ** 3
        
        vx = self.vx.view(1, self.nv, self.nv, self.nv, 1, 1, 1)
        vy = self.vy.view(1, self.nv, self.nv, self.nv, 1, 1, 1)
        vz = self.vz.view(1, self.nv, self.nv, self.nv, 1, 1, 1)
        
        u = velocity.view(velocity.shape[0], 3, 1, 1, 1, *velocity.shape[2:]) if velocity.dim() > 2 else velocity.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        # Peculiar velocities
        cx = vx - u[:, 0:1] if velocity.dim() > 2 else vx
        cy = vy - u[:, 1:2] if velocity.dim() > 2 else vy
        cz = vz - u[:, 2:3] if velocity.dim() > 2 else vz
        
        c_sq = cx**2 + cy**2 + cz**2
        
        # Heat flux components
        qx = torch.sum(cx * c_sq * f, dim=(1, 2, 3)) * dv3 / 2
        qy = torch.sum(cy * c_sq * f, dim=(1, 2, 3)) * dv3 / 2
        qz = torch.sum(cz * c_sq * f, dim=(1, 2, 3)) * dv3 / 2
        
        heat_flux = torch.stack([qx, qy, qz], dim=1)
        
        return heat_flux

# ============================================================================
# COMPLETE PRIMITIVE EQUATIONS
# ============================================================================

class CompletePrimitiveEquations(nn.Module):
    """Complete primitive equations with ALL terms"""
    
    def __init__(self, config: MDNOConfig):
        super().__init__()
        self.config = config
        self.constants = CONSTANTS
        
        # Learnable parameters
        self.nu = nn.Parameter(torch.tensor(10.0))  # Viscosity
        self.kappa = nn.Parameter(torch.tensor(5.0))  # Diffusivity
        
        # Neural subgrid parameterization
        self.subgrid_net = nn.Sequential(
            nn.Conv3d(7, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv3d(128, 256, 3, padding=1),
            nn.GroupNorm(16, 256),
            nn.SiLU(),
            nn.Conv3d(256, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv3d(128, 7, 3, padding=1),
            nn.Tanh()
        )
        
        logger.info("Complete primitive equations initialized")
    
    def compute_tendencies(self, state: Dict[str, torch.Tensor],
                          forcing: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute COMPLETE tendencies with all terms"""
        u, v, w = state['u'], state['v'], state['w']
        T, q, p, rho = state['T'], state['q'], state['p'], state['rho']
        
        dx, dy, dz = self.config.dx, self.config.dy, self.config.dz
        
        # ===== ALL GRADIENTS =====
        du_dx, du_dy, du_dz = [torch.gradient(u, dim=d)[0] / s 
                                for d, s in [(-1, dx), (-2, dy), (-3, dz)]]
        dv_dx, dv_dy, dv_dz = [torch.gradient(v, dim=d)[0] / s 
                                for d, s in [(-1, dx), (-2, dy), (-3, dz)]]
        dw_dx, dw_dy, dw_dz = [torch.gradient(w, dim=d)[0] / s 
                                for d, s in [(-1, dx), (-2, dy), (-3, dz)]]
        
        # Pressure gradients
        dp_dx = torch.gradient(p, dim=-1)[0] / dx
        dp_dy = torch.gradient(p, dim=-2)[0] / dy
        dp_dz = torch.gradient(p, dim=-3)[0] / dz
        
        dT_dx, dT_dy, dT_dz = [torch.gradient(T, dim=d)[0] / s 
                                for d, s in [(-1, dx), (-2, dy), (-3, dz)]]
        drho_dx, drho_dy, drho_dz = [torch.gradient(rho, dim=d)[0] / s 
                                      for d, s in [(-1, dx), (-2, dy), (-3, dz)]]
        
        # ===== CORIOLIS =====
        ny = u.shape[-2]
        lat = torch.linspace(-math.pi/2, math.pi/2, ny, device=u.device)
        f = 2 * self.constants.EARTH_ROTATION_RATE * torch.sin(lat)
        f = f.view(1, 1, -1, 1).expand_as(u)
        
        # ===== ADVECTION =====
        u_adv = u * du_dx + v * du_dy + w * du_dz
        v_adv = u * dv_dx + v * dv_dy + w * dv_dz
        w_adv = u * dw_dx + v * dw_dy + w * dw_dz
        
        # ===== VISCOUS TERMS =====
        d2u_dx2 = torch.gradient(du_dx, dim=-1)[0] / dx
        d2u_dy2 = torch.gradient(du_dy, dim=-2)[0] / dy
        d2u_dz2 = torch.gradient(du_dz, dim=-3)[0] / dz
        visc_u = self.nu * (d2u_dx2 + d2u_dy2 + d2u_dz2)
        
        d2v_dx2 = torch.gradient(dv_dx, dim=-1)[0] / dx
        d2v_dy2 = torch.gradient(dv_dy, dim=-2)[0] / dy
        d2v_dz2 = torch.gradient(dv_dz, dim=-3)[0] / dz
        visc_v = self.nu * (d2v_dx2 + d2v_dy2 + d2v_dz2)
        
        d2w_dx2 = torch.gradient(dw_dx, dim=-1)[0] / dx
        d2w_dy2 = torch.gradient(dw_dy, dim=-2)[0] / dy
        d2w_dz2 = torch.gradient(dw_dz, dim=-3)[0] / dz
        visc_w = self.nu * (d2w_dx2 + d2w_dy2 + d2w_dz2)
        
        # ===== MOMENTUM EQUATIONS (COMPLETE) =====
        du_dt = -u_adv - dp_dx / (rho + 1e-10) + f * v + visc_u + forcing.get('F_u', 0)
        dv_dt = -v_adv - dp_dy / (rho + 1e-10) - f * u + visc_v + forcing.get('F_v', 0)
        dw_dt = -w_adv - dp_dz / (rho + 1e-10) - self.constants.GRAVITY + visc_w + forcing.get('F_w', 0)
        
        # ===== THERMODYNAMIC EQUATION =====
        T_adv = u * dT_dx + v * dT_dy + w * dT_dz
        
        d2T_dx2 = torch.gradient(dT_dx, dim=-1)[0] / dx
        d2T_dy2 = torch.gradient(dT_dy, dim=-2)[0] / dy
        d2T_dz2 = torch.gradient(dT_dz, dim=-3)[0] / dz
        diff_T = self.kappa * (d2T_dx2 + d2T_dy2 + d2T_dz2)
        
        kappa_ratio = self.constants.DRY_AIR_GAS_CONSTANT / self.constants.CP_DRY_AIR
        adiabatic = kappa_ratio * T * w * dp_dz / (p + 1e-10)
        
        dT_dt = -T_adv + diff_T + adiabatic + forcing.get('Q', 0) / self.constants.CP_DRY_AIR
        
        # ===== MOISTURE EQUATION =====
        q_adv = u * torch.gradient(q, dim=-1)[0] / dx + \
                v * torch.gradient(q, dim=-2)[0] / dy + \
                w * torch.gradient(q, dim=-3)[0] / dz
        dq_dt = -q_adv + forcing.get('S', 0)
        
        # ===== CONTINUITY EQUATION =====
        div = du_dx + dv_dy + dw_dz
        drho_dt = -rho * div - u * drho_dx - v * drho_dy - w * drho_dz
        
        # ===== EQUATION OF STATE =====
        dp_dt = self.constants.DRY_AIR_GAS_CONSTANT * (rho * dT_dt + T * drho_dt)
        
        return {
            'u': du_dt, 'v': dv_dt, 'w': dw_dt,
            'T': dT_dt, 'q': dq_dt, 'p': dp_dt, 'rho': drho_dt
        }

# ============================================================================
# HAMILTONIAN DYNAMICS (AUDIT FIX)
# ============================================================================

class HamiltonianDynamics(nn.Module):
    """Hamiltonian dynamics with symplectic integration"""
    
    def __init__(self, config: MDNOConfig):
        super().__init__()
        self.config = config
        
        self.hamiltonian_net = nn.Sequential(
            nn.Linear(2, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        self.casimir_net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 32), nn.Tanh(),
            nn.Linear(32, 4)
        )
        
        logger.info("Hamiltonian dynamics initialized")
    
    def forward(self, state: torch.Tensor, dt: float) -> torch.Tensor:
        """Symplectic integration (Stormer-Verlet)"""
        channels = state.shape[1] // 2
        q = state[:, :channels]
        p = state[:, channels:]
        
        dH_dq = self._compute_gradient_q(q, p)
        p_half = p - 0.5 * dt * dH_dq
        
        dH_dp = self._compute_gradient_p(q, p_half)
        q_new = q + dt * dH_dp
        
        dH_dq_new = self._compute_gradient_q(q_new, p_half)
        p_new = p_half - 0.5 * dt * dH_dq_new
        
        state_new = torch.cat([q_new, p_new], dim=1)
        state_new = self._preserve_casimirs(state_new, state)
        
        return state_new
    
    def _compute_gradient_q(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        q.requires_grad_(True)
        batch_size = q.shape[0]
        q_flat = q.view(batch_size, -1)
        p_flat = p.view(batch_size, -1)
        
        state_flat = torch.cat([q_flat.mean(dim=1, keepdim=True),
                               p_flat.mean(dim=1, keepdim=True)], dim=1)
        H = self.hamiltonian_net(state_flat).sum()
        grad = torch.autograd.grad(H, q, create_graph=True)[0]
        return grad
    
    def _compute_gradient_p(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        p.requires_grad_(True)
        batch_size = p.shape[0]
        q_flat = q.view(batch_size, -1)
        p_flat = p.view(batch_size, -1)
        
        state_flat = torch.cat([q_flat.mean(dim=1, keepdim=True),
                               p_flat.mean(dim=1, keepdim=True)], dim=1)
        H = self.hamiltonian_net(state_flat).sum()
        grad = torch.autograd.grad(H, p, create_graph=True)[0]
        return grad
    
    def _preserve_casimirs(self, state_new: torch.Tensor, state_old: torch.Tensor) -> torch.Tensor:
        casimirs_old = self._compute_casimirs(state_old)
        casimirs_new = self._compute_casimirs(state_new)
        error = casimirs_new - casimirs_old
        correction = 0.1 * error.mean() * state_new / (torch.norm(state_new) + 1e-10)
        return state_new - correction
    
    def _compute_casimirs(self, state: torch.Tensor) -> torch.Tensor:
        batch_size = state.shape[0]
        state_flat = state.view(batch_size, -1)
        state_mean = torch.cat([
            state_flat[:, :state_flat.shape[1]//2].mean(dim=1, keepdim=True),
            state_flat[:, state_flat.shape[1]//2:].mean(dim=1, keepdim=True)
        ], dim=1)
        return self.casimir_net(state_mean)
    
    def compute_energy(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute total energy (kinetic + potential)
        AUDIT FIX: Sum over ALL spatial dimensions
        """
        channels = state.shape[1] // 2
        q = state[:, :channels]
        p = state[:, channels:]
        
        # Sum over every dimension except the batch (dim=0) so the result is
        # a per-sample scalar.  This integrates both the channel contributions
        # and the spatial volume, which preserves the total Hamiltonian energy
        # that the tests check for.  Previously we only reduced over the
        # spatial axes which leaked a residual channel dimension and produced
        # multi-element outputs, breaking the conservation test.
        sum_dims = tuple(range(1, state.ndim))
        T = 0.5 * torch.sum(p**2, dim=sum_dims)
        V = 0.5 * torch.sum(q**2, dim=sum_dims)

        return T + V

# ============================================================================
# TURBULENCE PARAMETERIZATION
# ============================================================================

class TurbulenceParameterization(nn.Module):
    """Advanced turbulence modeling"""
    
    def __init__(self, config: MDNOConfig):
        super().__init__()
        self.config = config
        self.model_type = config.turbulence_model
        
        self.smagorinsky_cs = nn.Parameter(torch.tensor(0.17))
        
        # k-epsilon model
        self.k_epsilon_net = nn.Sequential(
            nn.Conv3d(7, 128, 3, padding=1),
            nn.GroupNorm(8, 128), nn.SiLU(),
            nn.Conv3d(128, 64, 3, padding=1),
            nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv3d(64, 2, 3, padding=1),
            nn.Softplus()
        )
        
        logger.info(f"Turbulence model: {self.model_type.value}")
    
    def compute_turbulent_fluxes(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.model_type == TurbulenceModel.DNS:
            return {}
        elif self.model_type == TurbulenceModel.LES:
            return self._les_closure(state)
        elif self.model_type == TurbulenceModel.RANS:
            return self._rans_closure(state)
        else:
            les = self._les_closure(state)
            rans = self._rans_closure(state)
            alpha = 0.5
            return {k: alpha * les[k] + (1-alpha) * rans.get(k, 0) for k in les}
    
    def _les_closure(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        u, v, w = state['u'], state['v'], state['w']
        
        S11 = torch.gradient(u, dim=-1)[0] / self.config.dx
        S22 = torch.gradient(v, dim=-2)[0] / self.config.dy
        S33 = torch.gradient(w, dim=-3)[0] / self.config.dz
        S12 = 0.5 * (torch.gradient(u, dim=-2)[0] / self.config.dy + 
                     torch.gradient(v, dim=-1)[0] / self.config.dx)
        S13 = 0.5 * (torch.gradient(u, dim=-3)[0] / self.config.dz + 
                     torch.gradient(w, dim=-1)[0] / self.config.dx)
        S23 = 0.5 * (torch.gradient(v, dim=-3)[0] / self.config.dz + 
                     torch.gradient(w, dim=-2)[0] / self.config.dy)
        
        S_mag = torch.sqrt(2 * (S11**2 + S22**2 + S33**2 + 2*(S12**2 + S13**2 + S23**2)))
        delta = (self.config.dx * self.config.dy * self.config.dz)**(1/3)
        nu_t = (self.smagorinsky_cs * delta)**2 * S_mag
        
        tau_11 = -2 * nu_t * S11
        tau_22 = -2 * nu_t * S22
        tau_33 = -2 * nu_t * S33
        tau_12 = -2 * nu_t * S12
        tau_13 = -2 * nu_t * S13
        tau_23 = -2 * nu_t * S23
        
        return {
            'tau_11': tau_11, 'tau_22': tau_22, 'tau_33': tau_33,
            'tau_12': tau_12, 'tau_13': tau_13, 'tau_23': tau_23,
            'nu_t': nu_t
        }
    
    def _rans_closure(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        state_tensor = torch.stack([
            state['u'], state['v'], state['w'],
            state['T'], state['q'], state['p'], state['rho']
        ], dim=1)
        
        k_epsilon = self.k_epsilon_net(state_tensor)
        k = k_epsilon[:, 0]
        epsilon = k_epsilon[:, 1]
        
        C_mu = 0.09
        nu_t = C_mu * k**2 / (epsilon + 1e-8)
        
        return {'nu_t': nu_t, 'k': k, 'epsilon': epsilon}

# ============================================================================
# CLOUD MICROPHYSICS
# ============================================================================

class CloudMicrophysics(nn.Module):
    """Complete cloud microphysics"""
    
    def __init__(self, config: MDNOConfig):
        super().__init__()
        self.config = config
        self.constants = CONSTANTS
        
        self.condensation_net = nn.Sequential(
            nn.Linear(5, 64), nn.SiLU(),
            nn.Linear(64, 32), nn.SiLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )
        
        self.ice_nucleation_net = nn.Sequential(
            nn.Linear(4, 32), nn.SiLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )
        
        self.autoconversion_rate = nn.Parameter(torch.tensor(1e-3))
        self.collection_efficiency = nn.Parameter(torch.tensor(0.8))
        
        logger.info("Cloud microphysics initialized")
    
    def compute_phase_transitions(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        T = state['T']
        p = state['p']
        q_v = state.get('q_vapor', state.get('q', torch.zeros_like(T)))
        q_c = state.get('q_cloud', torch.zeros_like(T))
        q_i = state.get('q_ice', torch.zeros_like(T))
        q_r = state.get('q_rain', torch.zeros_like(T))
        
        transitions = {}
        
        # Saturation vapor pressure
        e_sat = self._saturation_vapor_pressure(T)
        e = q_v * p / (0.622 + 0.378 * q_v)
        RH = e / e_sat
        
        # Condensation
        supersaturation = torch.clamp(RH - 1.0, -1, 1)
        condensation_rate = 0.01 * supersaturation * q_v
        transitions['condensation'] = condensation_rate
        
        # Ice processes
        ice_mask = T < 273.15
        freezing_rate = torch.where(
            ice_mask,
            torch.minimum(q_c, 0.01 * (273.15 - T) / self.config.dt),
            torch.zeros_like(T)
        )
        melting_rate = torch.where(
            ~ice_mask,
            torch.minimum(q_i, 0.01 * (T - 273.15) / self.config.dt),
            torch.zeros_like(T)
        )
        
        transitions['freezing'] = freezing_rate
        transitions['melting'] = melting_rate
        
        # Autoconversion
        q_c_threshold = 1e-3
        autoconversion = torch.where(
            q_c > q_c_threshold,
            self.autoconversion_rate * (q_c - q_c_threshold),
            torch.zeros_like(T)
        )
        transitions['autoconversion'] = autoconversion
        
        # Collection
        collection = self.collection_efficiency * q_r * q_c
        transitions['collection'] = collection
        
        # Rain evaporation
        rain_evap = torch.where(
            RH < 1.0,
            0.001 * q_r * (1 - RH),
            torch.zeros_like(T)
        )
        transitions['rain_evaporation'] = rain_evap
        
        return transitions
    
    def _saturation_vapor_pressure(self, T: torch.Tensor) -> torch.Tensor:
        return 611.2 * torch.exp(17.67 * (T - 273.15) / (T - 29.65))

# ============================================================================
# CHEMICAL TRANSPORT
# ============================================================================

class ChemicalTransport(nn.Module):
    """Atmospheric chemistry"""
    
    def __init__(self, config: MDNOConfig):
        super().__init__()
        self.config = config
        
        self.species = ['O3', 'NO', 'NO2', 'CO', 'CH4', 'SO2']
        self.n_species = len(self.species)
        
        self.reaction_net = nn.Sequential(
            nn.Linear(self.n_species + 4, 128), nn.SiLU(),
            nn.Linear(128, 64), nn.SiLU(),
            nn.Linear(64, self.n_species)
        )
        
        self.photolysis_net = nn.Sequential(
            nn.Linear(3, 32), nn.SiLU(),
            nn.Linear(32, 16), nn.SiLU(),
            nn.Linear(16, 4), nn.Softplus()
        )
        
        logger.info(f"Chemical transport: {self.n_species} species")
    
    def compute_chemistry(self, state: Dict[str, torch.Tensor],
                         concentrations: Dict[str, torch.Tensor],
                         solar_zenith: torch.Tensor) -> Dict[str, torch.Tensor]:
        T = state['T']
        p = state['p']
        
        conc_list = [concentrations.get(s.lower(), torch.zeros_like(T)) for s in self.species]
        conc_tensor = torch.stack(conc_list, dim=-1)
        
        chem_input = torch.cat([
            conc_tensor.view(-1, self.n_species),
            T.flatten().unsqueeze(-1),
            p.flatten().unsqueeze(-1),
            torch.zeros(T.numel(), 2, device=T.device)
        ], dim=-1)
        
        rates = self.reaction_net(chem_input).reshape(*T.shape, self.n_species)
        
        production_loss = {}
        for i, species in enumerate(self.species):
            production_loss[species.lower()] = rates[..., i]
        
        # Simple NOx-O3 chemistry
        if 'o3' in concentrations and 'no' in concentrations:
            k_O3_NO = 1.8e-14 * torch.exp(1370/T)
            reaction_rate = k_O3_NO * concentrations['o3'] * concentrations['no']
            production_loss['o3'] = production_loss.get('o3', 0) - reaction_rate
            production_loss['no'] = production_loss.get('no', 0) - reaction_rate
            production_loss['no2'] = production_loss.get('no2', 0) + reaction_rate
        
        return production_loss

# ============================================================================
# SCALE BRIDGING
# ============================================================================

class ScaleBridging(nn.Module):
    """Moment-based scale bridging"""
    
    def __init__(self, boltzmann_solver: CompleteBoltzmannSolver, config: MDNOConfig):
        super().__init__()
        self.boltzmann_solver = boltzmann_solver
        self.config = config
        
        self.micro_to_meso_net = nn.Sequential(
            nn.Conv3d(10, 64, 3, padding=1),
            nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv3d(64, 64, 3, padding=1),
            nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv3d(64, 5, 3, padding=1)
        )
        
        self.meso_to_macro_net = nn.Sequential(
            nn.Conv3d(5, 32, 3, stride=2, padding=1),
            nn.GroupNorm(8, 32), nn.SiLU(),
            nn.Conv3d(32, 32, 3, stride=2, padding=1),
            nn.GroupNorm(8, 32), nn.SiLU(),
            nn.Conv3d(32, 4, 3, padding=1)
        )
        
        logger.info("Scale bridging initialized")
    
    def micro_to_meso(self, f: torch.Tensor) -> Dict[str, torch.Tensor]:
        moments = self.boltzmann_solver.compute_moments(f)
        
        density = moments['density']
        velocity = moments['velocity']
        temperature = moments['temperature']
        stress = moments['stress']
        
        stress_flat = stress.view(stress.shape[0], -1, *stress.shape[-3:])
        
        combined = torch.cat([
            density.unsqueeze(1),
            velocity,
            temperature.unsqueeze(1),
            stress_flat[:, :5]
        ], dim=1)
        
        corrections = self.micro_to_meso_net(combined)
        
        ns_state = {
            'density': density + corrections[:, 0],
            'velocity': velocity + corrections[:, 1:4],
            'pressure': density * temperature + corrections[:, 4],
            'temperature': temperature
        }
        
        return ns_state
    
    def meso_to_macro(self, ns_state: Dict[str, torch.Tensor]) -> torch.Tensor:
        combined = torch.cat([
            ns_state['density'].unsqueeze(1),
            ns_state['velocity'],
            ns_state['pressure'].unsqueeze(1)
        ], dim=1)
        
        macro_state = self.meso_to_macro_net(combined)
        macro_state = self._ensure_hamiltonian_structure(macro_state)
        
        return macro_state
    
    def _ensure_hamiltonian_structure(self, state: torch.Tensor) -> torch.Tensor:
        if state.shape[1] >= 2:
            u, v = state[:, 0], state[:, 1]
            div = torch.gradient(u, dim=-1)[0] + torch.gradient(v, dim=-2)[0]
            state[:, 0] -= torch.gradient(div, dim=-1)[0]
            state[:, 1] -= torch.gradient(div, dim=-2)[0]
        return state

# ============================================================================
# FNO3D
# ============================================================================


