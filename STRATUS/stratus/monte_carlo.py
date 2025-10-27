"""
Monte Carlo radiative-transfer solver with optional polarization support.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from .exceptions import StratusConfigError, StratusPhysicsError
from .polarization import MuellerMatrix

logger = logging.getLogger("stratus.monte_carlo")


@dataclass
class MonteCarloConfig:
    n_rays: int = 10000
    min_weight: float = 1.0e-4
    max_scatter: int = 64
    g: float = 0.0  # Henyey-Greenstein asymmetry
    albedo: float = 0.9
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.n_rays <= 0:
            raise StratusConfigError("MonteCarloConfig.n_rays must be positive.")
        if not torch.isfinite(torch.tensor(self.n_rays, dtype=torch.float32)):
            raise StratusConfigError("MonteCarloConfig.n_rays must be finite.")
        if self.min_weight <= 0.0:
            raise StratusConfigError("MonteCarloConfig.min_weight must be positive.")
        if self.max_scatter <= 0:
            raise StratusConfigError("MonteCarloConfig.max_scatter must be positive.")
        if not -0.999 <= self.g <= 0.999:
            raise StratusConfigError("MonteCarloConfig.g must lie within [-0.999, 0.999].")
        if not 0.0 <= self.albedo <= 1.0:
            raise StratusConfigError("MonteCarloConfig.albedo must lie within [0, 1].")


class MonteCarloRadiativeTransfer:
    """
    Lightweight Monte Carlo engine distilled from the legacy STRATUS implementation.
    """

    def __init__(
        self,
        n_stokes: int,
        grid_shape: tuple[int, int, int],
        config: MonteCarloConfig,
        device: torch.device | None = None,
    ) -> None:
        self.n_stokes = n_stokes
        self.grid_shape = grid_shape
        self.config = config
        if device is None:
            device = torch.device("cpu")
        self.device = device
        if config.seed is not None:
            torch.manual_seed(config.seed)
        self.mueller = MuellerMatrix(device=device) if n_stokes > 1 else None

    # ---------------------------------------------------------------- utilities
    def _initialize_rays(
        self,
        source_positions: torch.Tensor | None,
        source_directions: torch.Tensor | None,
        source_stokes: torch.Tensor | None,
    ) -> dict:
        n = self.config.n_rays

        if source_positions is None:
            x = torch.rand(n, device=self.device) * self.grid_shape[0]
            y = torch.rand(n, device=self.device) * self.grid_shape[1]
            z = torch.zeros(n, device=self.device)
            source_positions = torch.stack([x, y, z], dim=1)

        if source_directions is None:
            theta = torch.zeros(n, device=self.device)
            phi = torch.rand(n, device=self.device) * 2 * torch.pi
            dirs = torch.stack(
                [
                    torch.sin(theta) * torch.cos(phi),
                    torch.sin(theta) * torch.sin(phi),
                    torch.cos(theta),
                ],
                dim=1,
            )
            source_directions = dirs

        if source_stokes is None:
            stokes = torch.zeros(n, self.n_stokes, device=self.device)
            stokes[:, 0] = 1.0
            source_stokes = stokes

        rays = {
            "positions": source_positions,
            "directions": torch.nn.functional.normalize(source_directions, dim=-1),
            "stokes": source_stokes,
            "weights": torch.ones(n, device=self.device),
            "alive": torch.ones(n, dtype=torch.bool, device=self.device),
            "scatter_count": torch.zeros(n, dtype=torch.int32, device=self.device),
        }
        return rays

    def _sample_free_path(self, sigma_t: torch.Tensor, alive_idx: torch.Tensor) -> torch.Tensor:
        if alive_idx.numel() == 0:
            return torch.empty(0, device=self.device)
        u = torch.rand(alive_idx.shape[0], device=self.device).clamp_min(1e-12)
        sigma_mean = sigma_t.mean().clamp_min(1e-8)
        free_path = -torch.log(u) / sigma_mean
        return free_path

    def _update_positions(
        self, rays: dict, alive_idx: torch.Tensor, distance: torch.Tensor
    ) -> None:
        if alive_idx.numel() == 0:
            return
        rays["positions"][alive_idx] += distance.unsqueeze(-1) * rays["directions"][alive_idx]

    def _handle_scattering(self, rays: dict, scatter_idx: torch.Tensor) -> None:
        if scatter_idx.numel() == 0:
            return
        dirs = rays["directions"][scatter_idx]

        u1 = torch.rand_like(dirs[:, 0])
        if abs(self.config.g) < 1e-3:
            cos_theta = 1 - 2 * u1
        else:
            g = self.config.g
            term = (1 - g**2) / (1 - g + 2 * g * u1)
            cos_theta = (1 + g**2 - term**2) / (2 * g)
            cos_theta = cos_theta.clamp(-1.0, 1.0)

        sin_theta = torch.sqrt(torch.clamp(1 - cos_theta**2, min=0.0))
        phi = 2 * torch.pi * torch.rand_like(cos_theta)

        new_dirs = torch.stack(
            [sin_theta * torch.cos(phi), sin_theta * torch.sin(phi), cos_theta],
            dim=1,
        )
        rays["directions"][scatter_idx] = torch.nn.functional.normalize(new_dirs, dim=-1)

        if self.mueller is not None and self.n_stokes >= 4:
            theta = torch.acos(cos_theta)
            mueller = self.mueller.henyey_greenstein(torch.full_like(theta, self.config.g), theta)
            stokes = rays["stokes"][scatter_idx]
            stokes = torch.einsum("bij,bj->bi", mueller, stokes)
            rays["stokes"][scatter_idx] = MuellerMatrix.normalize(stokes)

    def _accumulate(self, rays: dict, detector: torch.Tensor) -> None:
        pos = rays["positions"]
        in_bounds = (
            (pos[:, 0] >= 0)
            & (pos[:, 0] < self.grid_shape[0])
            & (pos[:, 1] >= 0)
            & (pos[:, 1] < self.grid_shape[1])
            & (pos[:, 2] >= 0)
            & (pos[:, 2] < self.grid_shape[2])
        )
        oob = rays["alive"] & ~in_bounds
        if torch.any(oob):
            rays["alive"][oob] = False
            rays["weights"][oob] = 0.0

        idx = torch.where(rays["alive"] & in_bounds)[0]
        if idx.numel() == 0:
            return
        pos = pos[idx].long()
        if self.n_stokes == 1:
            detector[pos[:, 0], pos[:, 1], pos[:, 2]] += rays["weights"][idx]
        else:
            detector[pos[:, 0], pos[:, 1], pos[:, 2], :] += rays["stokes"][idx]

    # ---------------------------------------------------------------- solver
    def solve(
        self,
        sigma_t: torch.Tensor,
        source: torch.Tensor | None = None,
        *,
        source_positions: torch.Tensor | None = None,
        source_directions: torch.Tensor | None = None,
        source_stokes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._last_steps = 0
        rays = self._initialize_rays(source_positions, source_directions, source_stokes)
        detector_shape = (*self.grid_shape, self.n_stokes) if self.n_stokes > 1 else self.grid_shape
        detector = torch.zeros(detector_shape, device=self.device)

        sigma_t = sigma_t.to(self.device)
        if sigma_t.ndim != 3:
            raise StratusPhysicsError("sigma_t must be a 3D tensor matching the grid shape.")
        if sigma_t.shape != self.grid_shape:
            raise StratusPhysicsError(
                f"sigma_t shape {tuple(sigma_t.shape)} does not match grid {self.grid_shape}."
            )
        if not torch.isfinite(sigma_t).all():
            raise StratusPhysicsError("sigma_t contains non-finite values.")
        if (sigma_t < 0).any():
            raise StratusPhysicsError("sigma_t must be non-negative everywhere.")
        sigma_t = sigma_t.clamp_min(1.0e-8)
        # Albedo must be clamped BEFORE use in scattering decisions
        albedo_value = torch.tensor(self.config.albedo, device=self.device).clamp(0.0, 1.0)

        steps = 0
        while steps < self.config.max_scatter and torch.any(rays["alive"]):
            alive_idx = torch.nonzero(rays["alive"], as_tuple=False).squeeze(-1)
            distances = self._sample_free_path(sigma_t, alive_idx)
            if distances.numel() == 0:
                break

            self._update_positions(rays, alive_idx, distances)
            self._accumulate(rays, detector)

            if alive_idx.numel() == 0:
                break
            rays["scatter_count"][alive_idx] += 1
            scatter_prob = torch.rand_like(rays["weights"][alive_idx])
            scatter_mask = scatter_prob < albedo_value
            scatter_idx = alive_idx[scatter_mask]
            absorb_idx = alive_idx[~scatter_mask]

            # For scattered rays, weight unchanged (accounted for in scatter probability)
            # For absorbed rays below, we mark them as dead

            if absorb_idx.numel() > 0:
                rays["alive"][absorb_idx] = False
                rays["weights"][absorb_idx] = 0.0

            low_weight = rays["weights"] < self.config.min_weight
            if torch.any(low_weight):
                rays["alive"][low_weight] = False
                rays["weights"][low_weight] = 0.0

            self._handle_scattering(rays, scatter_idx)
            steps += 1
        self._last_steps = steps

        return detector


__all__ = ["MonteCarloRadiativeTransfer", "MonteCarloConfig"]
