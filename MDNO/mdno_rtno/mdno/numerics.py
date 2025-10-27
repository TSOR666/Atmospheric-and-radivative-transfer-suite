import math
from typing import Tuple

import torch
import torch.nn.functional as F

from ._logging import LOGGER as logger


class CFLMonitor:
    """Monitor CFL condition and suggest timesteps."""

    @staticmethod
    def _extract_velocity_components(velocity: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Return the available velocity components ordered as (u, v, w).

        The helper is tolerant to both batched tensors with the component
        dimension in ``dim=1`` (shape ``[batch, components, ...]``) and
        unbatched tensors where the first axis enumerates the velocity
        components (shape ``[components, ...]``).  Only the provided
        components are returned, allowing the caller to gracefully handle
        two-dimensional (u, v) or one-dimensional (u) flows.
        """

        if velocity.ndim < 1:
            raise ValueError("Velocity tensor must have at least one dimension")

        if velocity.ndim >= 2 and velocity.size(1) in {1, 2, 3}:
            components = velocity.unbind(dim=1)
        elif velocity.size(0) in {1, 2, 3}:
            components = velocity.unbind(dim=0)
        else:
            raise ValueError(
                "Velocity tensor must expose 1-3 components along the first or second dimension"
            )

        if not components:
            raise ValueError("Velocity tensor does not contain any components")

        return components

    @staticmethod
    def compute_cfl(velocity: torch.Tensor, dx: float, dy: float, dz: float, dt: float) -> float:
        """Compute the CFL number for a velocity field."""

        if dt <= 0:
            raise ValueError("Timestep 'dt' must be positive")
        if dx <= 0 or dy <= 0 or dz <= 0:
            raise ValueError("Grid spacings 'dx', 'dy', and 'dz' must be positive")

        components = CFLMonitor._extract_velocity_components(velocity)
        u = components[0]
        v = components[1] if len(components) > 1 else None
        w = components[2] if len(components) > 2 else None

        cfl_terms = [torch.abs(u) * dt / dx]
        if v is not None:
            cfl_terms.append(torch.abs(v) * dt / dy)
        if w is not None:
            cfl_terms.append(torch.abs(w) * dt / dz)

        total_cfl = torch.stack(cfl_terms).sum(dim=0)
        return float(total_cfl.max().item()) if total_cfl.numel() > 0 else 0.0

    @staticmethod
    def suggest_timestep(
        velocity: torch.Tensor,
        dx: float,
        dy: float,
        dz: float,
        target_cfl: float = 0.5,
    ) -> float:
        """Suggest a stable timestep based on the maximum velocity."""

        if target_cfl <= 0:
            raise ValueError("target_cfl must be positive")
        if dx <= 0 or dy <= 0 or dz <= 0:
            raise ValueError("Grid spacings 'dx', 'dy', and 'dz' must be positive")

        components = CFLMonitor._extract_velocity_components(velocity)
        u = components[0]
        v = components[1] if len(components) > 1 else None
        w = components[2] if len(components) > 2 else None

        eps = 1e-10
        timestep_candidates = [dx / (torch.abs(u).max() + eps)]
        if v is not None:
            timestep_candidates.append(dy / (torch.abs(v).max() + eps))
        if w is not None:
            timestep_candidates.append(dz / (torch.abs(w).max() + eps))

        min_dt = torch.stack(timestep_candidates).min()
        return float(target_cfl * min_dt.item())


class MomentPreservingProjection:
    """Enforce positivity while preserving mass, momentum, and energy."""

    @staticmethod
    def entropic_projection(
        f: torch.Tensor,
        f_target: torch.Tensor,
        dv: float,
        v_grid: torch.Tensor,
        max_iterations: int = 50,
        tol: float = 1e-6,
    ) -> torch.Tensor:
        """Entropic projection to enforce f >= 0 while preserving moments."""
        device = f.device
        dtype = f.dtype
        batch_size = f.shape[0]
        spatial_shape = f.shape[4:]

        # Compute target moments
        M0_target = torch.sum(f_target, dim=(1, 2, 3)) * (dv**3)

        nv = f.shape[1]
        vx, vy, vz = torch.meshgrid(v_grid, v_grid, v_grid, indexing="ij")

        vx = vx.view(1, nv, nv, nv, 1, 1, 1)
        vy = vy.view(1, nv, nv, nv, 1, 1, 1)
        vz = vz.view(1, nv, nv, nv, 1, 1, 1)

        # Momentum and energy targets
        M1x_target = torch.sum(f_target * vx, dim=(1, 2, 3)) * (dv**3)
        M1y_target = torch.sum(f_target * vy, dim=(1, 2, 3)) * (dv**3)
        M1z_target = torch.sum(f_target * vz, dim=(1, 2, 3)) * (dv**3)
        v_sq = vx**2 + vy**2 + vz**2
        M2_target = torch.sum(f_target * v_sq, dim=(1, 2, 3)) * (dv**3)

        # Initialize Lagrange multipliers
        lambda_0 = torch.zeros(batch_size, 1, 1, 1, *spatial_shape, device=device, dtype=dtype)
        lambda_1x = torch.zeros(batch_size, 1, 1, 1, *spatial_shape, device=device, dtype=dtype)
        lambda_1y = torch.zeros(batch_size, 1, 1, 1, *spatial_shape, device=device, dtype=dtype)
        lambda_1z = torch.zeros(batch_size, 1, 1, 1, *spatial_shape, device=device, dtype=dtype)
        lambda_2 = torch.zeros(batch_size, 1, 1, 1, *spatial_shape, device=device, dtype=dtype)

        f_proj = f.clone()

        for iteration in range(max_iterations):
            exponent = lambda_0 + lambda_1x * vx + lambda_1y * vy + lambda_1z * vz + lambda_2 * v_sq
            f_proj = f * torch.exp(torch.clamp(exponent, min=-10, max=10))
            f_proj = torch.clamp(f_proj, min=0)

            # Current moments
            M0_current = torch.sum(f_proj, dim=(1, 2, 3)) * (dv**3)
            M1x_current = torch.sum(f_proj * vx, dim=(1, 2, 3)) * (dv**3)
            M1y_current = torch.sum(f_proj * vy, dim=(1, 2, 3)) * (dv**3)
            M1z_current = torch.sum(f_proj * vz, dim=(1, 2, 3)) * (dv**3)
            M2_current = torch.sum(f_proj * v_sq, dim=(1, 2, 3)) * (dv**3)

            r0 = M0_current - M0_target
            r1x = M1x_current - M1x_target
            r1y = M1y_current - M1y_target
            r1z = M1z_current - M1z_target
            r2 = M2_current - M2_target

            residual = torch.sqrt(r0**2 + r1x**2 + r1y**2 + r1z**2 + r2**2).max()
            if residual < tol:
                logger.debug("Entropic projection converged in %s iterations", iteration + 1)
                break

            lr = 0.01
            residual_norm = residual + 1e-10

            lambda_0 = lambda_0 - lr * r0.reshape(batch_size, 1, 1, 1, *spatial_shape) / residual_norm
            lambda_1x = lambda_1x - lr * r1x.reshape(batch_size, 1, 1, 1, *spatial_shape) / residual_norm
            lambda_1y = lambda_1y - lr * r1y.reshape(batch_size, 1, 1, 1, *spatial_shape) / residual_norm
            lambda_1z = lambda_1z - lr * r1z.reshape(batch_size, 1, 1, 1, *spatial_shape) / residual_norm
            lambda_2 = lambda_2 - lr * r2.reshape(batch_size, 1, 1, 1, *spatial_shape) / residual_norm

        return f_proj

    @staticmethod
    def simple_moment_correction(f: torch.Tensor, f_old: torch.Tensor, dv: float) -> torch.Tensor:
        """Simple moment correction scaling to match mass."""
        M0_old = torch.sum(f_old, dim=(1, 2, 3), keepdim=True) * (dv**3)
        M0_new = torch.sum(f, dim=(1, 2, 3), keepdim=True) * (dv**3)
        return f * (M0_old / (M0_new + 1e-10))


class SpectralAdvectionFixed:
    """Fixed spectral advection with proper k-space cutoffs and FFT protection."""

    @staticmethod
    def compute_wavenumber_cutoffs(
        nx: int,
        ny: int,
        nz: int,
        dx: float,
        dy: float,
        dz: float,
        dealiasing_fraction: float,
    ) -> Tuple[int, int, int]:
        """Compute wavenumber cutoffs in index units."""
        k_cut_x = int(nx * dealiasing_fraction / 2)
        k_cut_y = int(ny * dealiasing_fraction / 2)
        k_cut_z = int(nz * dealiasing_fraction / 2)
        return k_cut_x, k_cut_y, k_cut_z

    @staticmethod
    def derivative_based_advection(
        f: torch.Tensor,
        velocity: torch.Tensor,
        dt: float,
        dx: float,
        dy: float,
        dz: float,
        dealiasing_fraction: float = 2.0 / 3.0,
    ) -> torch.Tensor:
        """Derivative-based spectral advection with FFT protection."""
        with torch.cuda.amp.autocast(enabled=False):
            f32 = f.float()
            device = f32.device

            nz, ny, nx = f32.shape[-3:]
            f_fft = torch.fft.rfftn(f32, dim=(-3, -2, -1))

            kz_idx = torch.fft.fftfreq(nz, d=1.0).to(device) * nz
            ky_idx = torch.fft.fftfreq(ny, d=1.0).to(device) * ny
            kx_idx = torch.fft.rfftfreq(nx, d=1.0).to(device) * nx

            k_cut_x, k_cut_y, k_cut_z = SpectralAdvectionFixed.compute_wavenumber_cutoffs(
                nx, ny, nz, dx, dy, dz, dealiasing_fraction
            )

            mask = (
                (torch.abs(kz_idx).view(-1, 1, 1) <= k_cut_z)
                & (torch.abs(ky_idx).view(1, -1, 1) <= k_cut_y)
                & (torch.abs(kx_idx).view(1, 1, -1) <= k_cut_x)
            )
            f_fft = f_fft * mask

            kz_phys = torch.fft.fftfreq(nz, dz).to(device).view(-1, 1, 1)
            ky_phys = torch.fft.fftfreq(ny, dy).to(device).view(1, -1, 1)
            kx_phys = torch.fft.rfftfreq(nx, dx).to(device).view(1, 1, -1)

            df_dz_fft = 2j * math.pi * kz_phys * f_fft
            df_dy_fft = 2j * math.pi * ky_phys * f_fft
            df_dx_fft = 2j * math.pi * kx_phys * f_fft

            df_dz = torch.fft.irfftn(df_dz_fft, s=(nz, ny, nx), dim=(-3, -2, -1))
            df_dy = torch.fft.irfftn(df_dy_fft, s=(nz, ny, nx), dim=(-3, -2, -1))
            df_dx = torch.fft.irfftn(df_dx_fft, s=(nz, ny, nx), dim=(-3, -2, -1))

        df_dx = df_dx.to(f.dtype)
        df_dy = df_dy.to(f.dtype)
        df_dz = df_dz.to(f.dtype)

        u = velocity[:, 0]
        v = velocity[:, 1]
        w = velocity[:, 2]

        advection = -(u * df_dx + v * df_dy + w * df_dz)
        return f + dt * advection
