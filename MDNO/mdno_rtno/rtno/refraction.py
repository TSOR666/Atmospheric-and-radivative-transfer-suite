from __future__ import annotations

import math

import torch
import torch.nn as nn

from .config import RTNOConfig
from .constants import CONSTANTS


class AtmosphericRefraction(nn.Module):
    """Atmospheric refraction with learnable refractivity profile."""

    def __init__(self, config: RTNOConfig):
        super().__init__()
        self.config = config
        self.N0 = nn.Parameter(torch.tensor(300.0))
        self.H = nn.Parameter(torch.tensor(7000.0))
        self.beta = nn.Parameter(torch.tensor(0.1))

    def compute_refractive_index(
        self,
        altitude: torch.Tensor,
        temperature: torch.Tensor,
        pressure: torch.Tensor,
        humidity: torch.Tensor,
    ) -> torch.Tensor:
        N_dry = 77.6890 * pressure / (temperature + 1e-12)
        e_water = humidity * 6.1121 * torch.exp(17.67 * (temperature - 273.15) / (temperature - 29.65 + 1e-12))
        N_wet = 71.2952 * e_water / (temperature + 1e-12) + 375463 * e_water / (temperature**2 + 1e-12)
        N_total = N_dry + N_wet

        N_profile = self.N0 * torch.exp(-altitude / (self.H + 1e-12)) * (1 + self.beta * altitude / (self.H + 1e-12))
        alpha = torch.sigmoid(altitude / 10000)
        N_combined = (1 - alpha) * N_total + alpha * N_profile
        n = 1 + N_combined * 1e-6
        return n

    def ray_bending(
        self,
        zenith_angle: torch.Tensor,
        altitude: torch.Tensor,
        n_profile: torch.Tensor,
    ) -> torch.Tensor:
        R = CONSTANTS.EARTH_RADIUS
        n0 = n_profile[0] if n_profile.numel() > 0 else torch.tensor(1.0, device=n_profile.device)
        r0 = R + altitude[0]
        ray_param = n0 * r0 * torch.sin(zenith_angle)
        r = R + altitude
        n = n_profile
        sin_z_apparent = torch.clamp(ray_param / (n * r + 1e-12), -1.0, 1.0)
        z_apparent = torch.asin(sin_z_apparent)
        bending = zenith_angle.unsqueeze(-1) - z_apparent

        if (zenith_angle > math.pi / 3).any():
            z_deg = zenith_angle * 180 / math.pi
            correction = 1 / torch.tan((z_deg + 7.31 / (z_deg + 4.4)) * math.pi / 180)
            bending = bending + correction * 0.0001

        return bending

    def optical_path_length(self, geometric_path: torch.Tensor, n_profile: torch.Tensor) -> torch.Tensor:
        return torch.sum(n_profile * geometric_path, dim=-1)


__all__ = ["AtmosphericRefraction"]
