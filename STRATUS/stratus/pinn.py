"""
Physics-informed neural network solver for the radiative transfer equation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .exceptions import StratusConfigError, StratusPhysicsError

logger = logging.getLogger("stratus.pinn")


@dataclass
class PINNConfig:
    hidden_dim: int = 128
    hidden_layers: int = 4
    learning_rate: float = 1.0e-3
    n_epochs: int = 200
    n_samples: int = 2048
    time_final: float = 1.0

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0:
            raise StratusConfigError("PINNConfig.hidden_dim must be positive.")
        if self.hidden_layers <= 0:
            raise StratusConfigError("PINNConfig.hidden_layers must be positive.")
        if self.learning_rate <= 0.0:
            raise StratusConfigError("PINNConfig.learning_rate must be positive.")
        if self.n_epochs <= 0:
            raise StratusConfigError("PINNConfig.n_epochs must be positive.")
        if self.n_samples <= 0:
            raise StratusConfigError("PINNConfig.n_samples must be positive.")
        if self.time_final <= 0.0:
            raise StratusConfigError("PINNConfig.time_final must be positive.")


class PhysicsInformedRTE(nn.Module):
    """
    Simplified physics-informed network approximating the RTE solution.
    """

    def __init__(
        self,
        grid_shape: tuple[int, int, int],
        n_stokes: int,
        config: PINNConfig,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.grid_shape = grid_shape
        self.n_stokes = n_stokes
        self.config = config
        self.device = device

        layers = []
        input_dim = 4  # x, y, z, t
        output_dim = n_stokes

        prev = input_dim
        for _ in range(config.hidden_layers):
            layers.append(nn.Linear(prev, config.hidden_dim))
            layers.append(nn.GELU())
            prev = config.hidden_dim
        layers.append(nn.Linear(prev, output_dim))
        self.network = nn.Sequential(*layers).to(device)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.network(coords)

    def solve(
        self,
        kappa: torch.Tensor,
        source: torch.Tensor,
    ) -> torch.Tensor:
        if len(self.grid_shape) != 3:
            raise StratusConfigError("grid_shape must contain exactly three dimensions.")
        nx, ny, nz = self.grid_shape
        if min(nx, ny, nz) <= 0:
            raise StratusConfigError("grid dimensions must be positive.")
        optimizer = optim.Adam(self.parameters(), lr=self.config.learning_rate)

        if kappa.ndim != 5 or source.ndim != 5:
            raise StratusPhysicsError(
                "kappa and source must be 5D tensors (stokes, x, y, z, bands)."
            )
        kappa = torch.nan_to_num(kappa, nan=0.0, posinf=1e6, neginf=0.0)
        source = torch.nan_to_num(source, nan=0.0, posinf=1e6, neginf=-1e6)

        kappa_grid = kappa.mean(dim=-1).to(self.device)
        source_grid = source.mean(dim=-1).to(self.device)

        if kappa_grid.shape != (self.n_stokes, nx, ny, nz) or source_grid.shape != (
            self.n_stokes,
            nx,
            ny,
            nz,
        ):
            raise StratusPhysicsError(
                "kappa and source grids must match the configured spatial dimensions."
            )

        scale = torch.tensor(
            [max(nx - 1, 1), max(ny - 1, 1), max(nz - 1, 1)],
            device=self.device,
            dtype=torch.float32,
        )

        for epoch in range(self.config.n_epochs):
            optimizer.zero_grad()
            coords = torch.rand(self.config.n_samples, 3, device=self.device)
            coords[:, 0] *= nx - 1
            coords[:, 1] *= ny - 1
            coords[:, 2] *= nz - 1
            time = torch.rand(self.config.n_samples, 1, device=self.device) * self.config.time_final
            input_coords = torch.cat([coords / scale, time], dim=1)
            input_coords.requires_grad_(True)

            prediction = self.forward(input_coords)
            if prediction.shape[-1] != self.n_stokes:
                raise StratusPhysicsError(
                    "PINN network output dimension does not match configured n_stokes."
                )

            gradients = []
            for s in range(self.n_stokes):
                grad_outputs = torch.zeros_like(prediction)
                grad_outputs[:, s] = 1.0
                grads = torch.autograd.grad(
                    outputs=prediction,
                    inputs=input_coords,
                    grad_outputs=grad_outputs,
                    create_graph=True,
                    retain_graph=s < self.n_stokes - 1,
                )[0]
                gradients.append(grads.unsqueeze(0))
            gradients = torch.cat(gradients, dim=0)

            ray_direction = F.normalize(torch.randn_like(coords), dim=-1)
            spatial_gradients = gradients[..., :3] / scale.view(1, 1, 3)
            dI_ds = (spatial_gradients * ray_direction.unsqueeze(0)).sum(dim=-1)

            indices = torch.floor(coords).to(torch.long)
            indices[:, 0] = indices[:, 0].clamp(0, nx - 1)
            indices[:, 1] = indices[:, 1].clamp(0, ny - 1)
            indices[:, 2] = indices[:, 2].clamp(0, nz - 1)

            kappa_samples = kappa_grid[
                :, indices[:, 0], indices[:, 1], indices[:, 2]
            ]
            source_samples = source_grid[
                :, indices[:, 0], indices[:, 1], indices[:, 2]
            ]

            prediction_samples = prediction.T  # [n_stokes, samples]
            residual = dI_ds + kappa_samples * prediction_samples - source_samples
            loss = residual.pow(2).mean()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 50 == 0:
                logger.debug("PINN epoch %d loss %.6f", epoch + 1, loss.item())

        with torch.no_grad():
            grid = torch.stack(
                torch.meshgrid(
                    torch.arange(nx, device=self.device, dtype=torch.float32),
                    torch.arange(ny, device=self.device, dtype=torch.float32),
                    torch.arange(nz, device=self.device, dtype=torch.float32),
                    indexing="ij",
                ),
                dim=-1,
            ).reshape(-1, 3)
            t = torch.full((grid.size(0), 1), self.config.time_final, device=self.device)
            inputs = torch.cat([grid / scale, t], dim=1)
            radiance = self.forward(inputs).reshape(nx, ny, nz, self.n_stokes)
            radiance = radiance.permute(3, 0, 1, 2).unsqueeze(0).unsqueeze(-1)
        return radiance


__all__ = ["PhysicsInformedRTE", "PINNConfig"]
