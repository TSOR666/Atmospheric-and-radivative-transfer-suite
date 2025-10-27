"""
Ray marching utilities for STRATUS with adaptive optical-depth control.
"""

from __future__ import annotations

import math

import torch

from .config import StratusConfig
from .polarization import MuellerMatrix


def ray_box_exit_distance(
    origins: torch.Tensor,
    directions: torch.Tensor,
    box_min: torch.Tensor,
    box_max: torch.Tensor,
) -> torch.Tensor:
    """Return the distance until each ray exits the axis-aligned bounding box."""

    eps = 1.0e-6
    sign = torch.where(directions >= 0, 1.0, -1.0)
    huge = directions.new_full((), 1.0e10)
    inv_dir = torch.where(
        directions.abs() > eps,
        1.0 / directions,
        sign * huge,
    )
    t0 = (box_min - origins) * inv_dir
    t1 = (box_max - origins) * inv_dir
    torch.minimum(t0, t1)
    t_max = torch.maximum(t0, t1)
    t_exit = torch.minimum(torch.minimum(t_max[..., 0], t_max[..., 1]), t_max[..., 2])
    return torch.clamp_min(t_exit, 0.0)


def trilinear_interpolation(field: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    nx, ny, nz = field.shape[:3]
    x = positions[:, 0].clamp(0, nx - 1.0001)
    y = positions[:, 1].clamp(0, ny - 1.0001)
    z = positions[:, 2].clamp(0, nz - 1.0001)

    x0 = x.floor().long()
    y0 = y.floor().long()
    z0 = z.floor().long()
    x1 = (x0 + 1).clamp_max(nx - 1)
    y1 = (y0 + 1).clamp_max(ny - 1)
    z1 = (z0 + 1).clamp_max(nz - 1)

    fx = (x - x0.float()).unsqueeze(-1)
    fy = (y - y0.float()).unsqueeze(-1)
    fz = (z - z0.float()).unsqueeze(-1)

    c000 = field[x0, y0, z0]
    c001 = field[x0, y0, z1]
    c010 = field[x0, y1, z0]
    c011 = field[x0, y1, z1]
    c100 = field[x1, y0, z0]
    c101 = field[x1, y0, z1]
    c110 = field[x1, y1, z0]
    c111 = field[x1, y1, z1]

    c00 = c000 * (1 - fx) + c100 * fx
    c01 = c001 * (1 - fx) + c101 * fx
    c10 = c010 * (1 - fx) + c110 * fx
    c11 = c011 * (1 - fx) + c111 * fx

    c0 = c00 * (1 - fy) + c10 * fy
    c1 = c01 * (1 - fy) + c11 * fy

    return c0 * (1 - fz) + c1 * fz


class RayMarcher:
    """
    Adaptive ray marcher implementing the corrected STRATUS integration scheme.
    """

    def __init__(self, config: StratusConfig, mueller: MuellerMatrix | None = None) -> None:
        self.config = config
        self.mueller = mueller
        if config.wavelengths is not None:
            lambda_mean = (
                (
                    config.wavelengths.to(config.device)
                    if isinstance(config.wavelengths, torch.Tensor)
                    else torch.tensor(config.wavelengths, device=config.device)
                )
                .float()
                .mean()
            )
        else:
            lambda_mean = torch.tensor(1.0, device=config.device)
        radius = torch.tensor(
            sum(config.voxel_size) / (3 * 2), device=config.device, dtype=torch.float32
        )
        self.mie_size_parameter = 2 * math.pi * radius / lambda_mean

    def march(
        self,
        origins: torch.Tensor,
        directions: torch.Tensor,
        kappa_field: torch.Tensor,
        source_field: torch.Tensor,
        max_distance: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Integrate radiance and transmittance along a batch of rays.

        Parameters
        ----------
        origins:
            Tensor of shape ``[N, 3]`` containing ray origins in voxel coordinates.
        directions:
            Tensor of shape ``[N, 3]`` with **normalized** ray directions.
        kappa_field:
            Tensor of shape ``[Nx, Ny, Nz, n_stokes, n_bands]`` containing extinction
            coefficients.
        source_field:
            Tensor of shape ``[Nx, Ny, Nz, n_stokes, n_bands]`` with in-scattering
            source terms.
        max_distance:
            Optional manual cap on the integration distance.  When omitted the
            bounding-box exit distance is used.

        Returns
        -------
        Tuple[Tensor, Tensor]
            Radiance and transmittance tensors, each shaped ``[N, n_stokes, n_bands]``.
        """

        device = origins.device
        n_rays = origins.shape[0]
        n_stokes = kappa_field.shape[3]
        n_bands = kappa_field.shape[4]
        nx, ny, nz = kappa_field.shape[:3]

        radiance = torch.zeros(n_rays, n_stokes, n_bands, device=device, dtype=torch.float32)
        transmittance = torch.ones_like(radiance)

        box_min = origins.new_zeros(3)
        box_max = torch.tensor(self.config.grid_shape, device=device, dtype=torch.float32)
        if max_distance is None:
            exit_distances = ray_box_exit_distance(origins, directions, box_min, box_max)
            max_distance = float(exit_distances.max().item())
        else:
            exit_distances = torch.full(
                (n_rays,), float(max_distance), device=device, dtype=torch.float32
            )

        base_dt = max_distance / float(self.config.raymarch.ray_marching_steps)
        base_dt = min(base_dt, self.config.raymarch.step_size)
        kappa_flat = kappa_field.reshape(nx, ny, nz, -1)
        source_flat = source_field.reshape(nx, ny, nz, -1)
        travel = torch.zeros(n_rays, device=device, dtype=torch.float32)
        alive = torch.ones(n_rays, device=device, dtype=torch.bool)

        for step in range(self.config.raymarch.ray_marching_steps):
            if not torch.any(alive):
                break

            positions = origins + travel.unsqueeze(-1) * directions

            valid = (
                (positions[:, 0] >= 0)
                & (positions[:, 0] < nx)
                & (positions[:, 1] >= 0)
                & (positions[:, 1] < ny)
                & (positions[:, 2] >= 0)
                & (positions[:, 2] < nz)
            )
            alive &= valid
            valid = alive.clone()

            if not torch.any(valid):
                break

            pos_valid = positions[valid]
            kappa_samples = trilinear_interpolation(kappa_flat, pos_valid).reshape(
                -1, n_stokes, n_bands
            )
            source_samples = trilinear_interpolation(source_flat, pos_valid).reshape(
                -1, n_stokes, n_bands
            )

            if self.config.raymarch.per_ray_adaptive:
                max_kappa = kappa_samples.view(kappa_samples.shape[0], -1).max(dim=1)[0]
                # Ensure dt is never smaller than a minimum threshold (1e-6 * base_dt)
                dt = torch.minimum(
                    torch.full_like(max_kappa, base_dt),
                    self.config.raymarch.max_optical_depth_per_step / (max_kappa + 1.0e-12),
                )
                remaining = (exit_distances[valid] - travel[valid]).clamp_min(0.0)
                dt = torch.minimum(dt, remaining)
                dt = dt.clamp_min(1e-6 * base_dt)
                tau = kappa_samples * dt[:, None, None]
                dt_tensor = dt[:, None, None]
            else:
                max_kappa = kappa_samples.max()
                dt_scalar = min(
                    base_dt,
                    self.config.raymarch.max_optical_depth_per_step / (max_kappa + 1.0e-12),
                )
                dt_scalar = max(dt_scalar, 1e-6 * base_dt)
                dt = torch.full_like(travel[valid], dt_scalar)
                remaining = (exit_distances[valid] - travel[valid]).clamp_min(0.0)
                dt = torch.minimum(dt, remaining)
                tau = kappa_samples * dt[:, None, None]
                dt_tensor = dt[:, None, None]

            tau = tau.clamp_max(self.config.raymarch.max_optical_depth_per_step)
            exp_tau = torch.exp(-tau)

            tv = transmittance[valid]
            rad = radiance[valid]

            tv_new = tv * exp_tau
            mask = tau > 1e-8
            # For tau above the threshold use (source/kappa) * (1 - exp(-tau))
            # For tau close to zero approximate with source * dt to avoid division by zero
            increment = torch.where(
                mask,
                (source_samples / kappa_samples.clamp_min(1e-12)) * (1.0 - exp_tau),
                source_samples * dt_tensor,
            )

            if self.mueller is not None and n_stokes >= 4:
                dir_valid = directions[valid]
                theta = torch.acos(dir_valid[:, 2].clamp(-1.0, 1.0))
                if self.config.scattering_model == "mie":
                    size_parameter = torch.full_like(theta, self.mie_size_parameter)
                    mueller = self.mueller.mie_scattering(
                        size_parameter,
                        theta,
                        self.config.mie_refractive_index,
                        max_order=self.config.mie_max_order,
                    )
                elif self.config.scattering_model == "henyey_greenstein":
                    g = torch.full_like(theta, self.config.henyey_g)
                    mueller = self.mueller.henyey_greenstein(g, theta)
                else:
                    phi = torch.atan2(dir_valid[:, 1], dir_valid[:, 0])
                    mueller = self.mueller.rayleigh_scattering(theta, phi)
                tv_new = torch.einsum("bij,bjk->bik", mueller, tv_new)
                increment = torch.einsum("bij,bjk->bik", mueller, increment)

            rad_new = rad * exp_tau + increment

            transmittance[valid] = tv_new
            radiance[valid] = rad_new
            travel[valid] = travel[valid] + dt

            # Early termination if all rays have negligible transmittance
            if transmittance[valid].max() < self.config.raymarch.min_transmittance:
                break

            finished = travel >= (exit_distances - 1e-6)
            alive &= ~finished

        return radiance, transmittance


__all__ = ["RayMarcher", "ray_box_exit_distance", "trilinear_interpolation"]






