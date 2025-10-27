from __future__ import annotations

from typing import Callable, Dict, Tuple

import torch

from .config import RTNOConfig
from .constants import CONSTANTS


class OpticalDepthScaling:
    """Optical depth safeguards (delta-Eddington and safe transmittance)."""

    @staticmethod
    def delta_eddington_scaling(
        tau: torch.Tensor, omega: torch.Tensor, g: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f = g * g
        tau_star = (1 - omega * f) * tau
        omega_star = (1 - f) * omega / (1 - omega * f + 1e-10)
        g_star = (g - f) / (1 - f + 1e-10)
        return tau_star, omega_star, g_star

    @staticmethod
    def safe_transmittance(tau: torch.Tensor, threshold: float = 20.0) -> torch.Tensor:
        small_tau = tau < CONSTANTS.SMALL_TAU_THRESHOLD
        trans_small = 1 - tau + 0.5 * tau**2

        moderate_tau = (tau >= CONSTANTS.SMALL_TAU_THRESHOLD) & (tau < threshold)
        trans_moderate = torch.exp(-tau)

        trans_large = torch.zeros_like(tau)
        return torch.where(small_tau, trans_small, torch.where(moderate_tau, trans_moderate, trans_large))


class MultipleScatteringSolver:
    """Iterative source-iteration solver for multiple scattering."""

    def __init__(self, config: RTNOConfig):
        self.config = config
        self.max_iterations = config.max_scattering_iterations
        self.tolerance = config.scattering_convergence_tol

    def solve_source_iteration(
        self,
        extinction: torch.Tensor,
        scattering: torch.Tensor,
        phase_function: Callable[[torch.Tensor], torch.Tensor],
        thermal_source: torch.Tensor,
        boundary_radiance: torch.Tensor,
        angles: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        batch_size = extinction.shape[0]
        nz, ny, nx = extinction.shape[1:]
        n_angles = len(angles["mu"])
        device = extinction.device

        radiance = torch.zeros(
            batch_size, n_angles, nz, ny, nx, device=device, dtype=self.config.dtype
        )
        radiance[:, :, -1] = boundary_radiance
        omega = scattering / (extinction + 1e-10)

        for _ in range(self.max_iterations):
            radiance_old = radiance.clone()
            scatter_source = self._compute_scatter_source(radiance, phase_function, omega, angles)
            total_source = thermal_source.unsqueeze(1) + scatter_source
            radiance = self._radiative_transfer_step(radiance, extinction, total_source, angles)

            residual = torch.norm(radiance - radiance_old) / (torch.norm(radiance) + 1e-10)
            if residual < self.tolerance:
                break

        return radiance

    def _compute_scatter_source(
        self,
        radiance: torch.Tensor,
        phase_function: Callable[[torch.Tensor], torch.Tensor],
        omega: torch.Tensor,
        angles: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        n_angles = radiance.shape[1]
        scatter_source = torch.zeros_like(radiance)
        mu_grid = angles["mu"]
        weights = angles["weights"]

        for i in range(n_angles):
            mu_out = mu_grid[i]
            cos_theta = mu_out * mu_grid
            P = phase_function(cos_theta)
            scatter_source[:, i] = omega.unsqueeze(1) * torch.sum(
                radiance * P.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) * weights.view(1, -1, 1, 1), dim=1
            )

        return scatter_source

    def _radiative_transfer_step(
        self,
        radiance: torch.Tensor,
        extinction: torch.Tensor,
        source: torch.Tensor,
        angles: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        radiance_new = radiance.clone()
        mu = angles["mu"]
        dz = self.config.dz

        for i, mu_val in enumerate(mu):
            if mu_val == 0:
                continue

            if mu_val > 0:
                for k in range(1, radiance.shape[2]):
                    tau = extinction[:, k] * dz / mu_val
                    trans = OpticalDepthScaling.safe_transmittance(tau)
                    radiance_new[:, i, k] = (
                        radiance_new[:, i, k - 1] * trans
                        + source[:, i, k] * (1 - trans) / (extinction[:, k] + 1e-10)
                    )
            else:
                for k in range(radiance.shape[2] - 2, -1, -1):
                    tau = extinction[:, k] * dz / abs(mu_val)
                    trans = OpticalDepthScaling.safe_transmittance(tau)
                    radiance_new[:, i, k] = (
                        radiance_new[:, i, k + 1] * trans
                        + source[:, i, k] * (1 - trans) / (extinction[:, k] + 1e-10)
                    )

        return radiance_new


class ShortCharacteristics3D:
    """Three-dimensional short characteristics solver for horizontal coupling."""

    def __init__(self, config: RTNOConfig):
        self.config = config
        self.dx = config.dx
        self.dy = config.dy
        self.dz = config.dz

    def solve_3d(
        self,
        extinction: torch.Tensor,
        source: torch.Tensor,
        boundary_conditions: Dict[str, torch.Tensor],
        angles: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        batch_size = extinction.shape[0]
        nz, ny, nx = extinction.shape[1:]
        n_angles = len(angles["mu"])
        device = extinction.device

        radiance = torch.zeros(
            batch_size, n_angles, nz, ny, nx, device=device, dtype=extinction.dtype
        )

        if "top" in boundary_conditions:
            radiance[:, :, -1] = boundary_conditions["top"]

        mu = angles["mu"]
        phi = angles["phi"]

        for i_angle in range(n_angles):
            mu_i = mu[i_angle]
            phi_i = phi[i_angle % len(phi)]
            l_x = torch.sqrt(torch.clamp(1 - mu_i**2, min=0.0)) * torch.cos(phi_i)
            l_y = torch.sqrt(torch.clamp(1 - mu_i**2, min=0.0)) * torch.sin(phi_i)
            l_z = mu_i

            ix_range = range(nx) if l_x >= 0 else range(nx - 1, -1, -1)
            iy_range = range(ny) if l_y >= 0 else range(ny - 1, -1, -1)
            iz_range = range(nz) if l_z >= 0 else range(nz - 1, -1, -1)

            for iz in iz_range:
                for iy in iy_range:
                    for ix in ix_range:
                        I_x = self._get_upstream(radiance, i_angle, iz, iy, ix, l_x, 0, boundary_conditions)
                        I_y = self._get_upstream(radiance, i_angle, iz, iy, ix, l_y, 1, boundary_conditions)
                        I_z = self._get_upstream(radiance, i_angle, iz, iy, ix, l_z, 2, boundary_conditions)

                        ds_x = abs(self.dx / (l_x + 1e-10))
                        ds_y = abs(self.dy / (l_y + 1e-10))
                        ds_z = abs(self.dz / (l_z + 1e-10))
                        ds = min(ds_x, ds_y, ds_z)

                        tau = extinction[:, iz, iy, ix] * ds
                        trans = OpticalDepthScaling.safe_transmittance(tau)
                        I_upstream = (I_x * ds_x + I_y * ds_y + I_z * ds_z) / (ds_x + ds_y + ds_z + 1e-10)

                        S = source[:, i_angle, iz, iy, ix]
                        beta = extinction[:, iz, iy, ix]
                        radiance[:, i_angle, iz, iy, ix] = I_upstream * trans + S * (1 - trans) / (beta + 1e-10)

        return radiance

    def _get_upstream(
        self,
        radiance: torch.Tensor,
        i_angle: int,
        iz: int,
        iy: int,
        ix: int,
        direction: float,
        axis: int,
        boundary_conditions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        nz, ny, nx = radiance.shape[2:]

        if direction >= 0:
            if axis == 0 and ix > 0:
                return radiance[:, i_angle, iz, iy, ix - 1]
            if axis == 1 and iy > 0:
                return radiance[:, i_angle, iz, iy - 1, ix]
            if axis == 2 and iz > 0:
                return radiance[:, i_angle, iz - 1, iy, ix]
        else:
            if axis == 0 and ix < nx - 1:
                return radiance[:, i_angle, iz, iy, ix + 1]
            if axis == 1 and iy < ny - 1:
                return radiance[:, i_angle, iz, iy + 1, ix]
            if axis == 2 and iz < nz - 1:
                return radiance[:, i_angle, iz + 1, iy, ix]

        return boundary_conditions.get("default", torch.zeros_like(radiance[:, i_angle, iz, iy, ix]))


__all__ = ["OpticalDepthScaling", "MultipleScatteringSolver", "ShortCharacteristics3D"]
