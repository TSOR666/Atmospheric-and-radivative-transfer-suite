from __future__ import annotations

import math
from typing import Any, Dict

import torch
import torch.nn as nn

from .config import BoundaryType, RTNOConfig
from .constants import CONSTANTS


class AdvancedBoundaryConditions(nn.Module):
    """Surface and top-of-atmosphere boundary condition models."""

    def __init__(self, config: RTNOConfig):
        super().__init__()
        self.config = config

        self.brdf_net = nn.Sequential(
            nn.Linear(6, 128),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus(),
        )

        self.ocean_brdf = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self.toa_net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, config.n_stokes),
        )

    def surface_brdf(
        self,
        incident: torch.Tensor,
        reflected: torch.Tensor,
        wavelength: torch.Tensor,
        surface_type: torch.Tensor,
        boundary_type: BoundaryType = BoundaryType.MIXED,
    ) -> torch.Tensor:
        theta_i, phi_i = incident[:, 0], incident[:, 1]
        theta_r, phi_r = reflected[:, 0], reflected[:, 1]

        if boundary_type == BoundaryType.LAMBERTIAN:
            brdf = surface_type[:, 0] / math.pi
        elif boundary_type == BoundaryType.SPECULAR:
            delta = 0.01
            spec_condition = torch.abs(theta_i - theta_r) < delta
            brdf = torch.where(
                spec_condition,
                surface_type[:, 0] / (delta * torch.cos(theta_r) + 1e-12),
                torch.zeros_like(theta_i),
            )
        elif boundary_type == BoundaryType.OCEAN:
            wind_speed = surface_type[:, 5] if surface_type.shape[1] > 5 else torch.ones_like(theta_i) * 5.0
            sigma_sq = 0.003 + 0.00512 * wind_speed
            h_theta = (theta_i + theta_r) / 2
            tan_h = torch.tan(h_theta)
            P_slope = torch.exp(-tan_h**2 / (2 * sigma_sq + 1e-12)) / (2 * math.pi * sigma_sq + 1e-12)

            n_water = 1.333
            cos_i = torch.cos(theta_i)
            fresnel = ((n_water - 1) / (n_water + 1)) ** 2 + 0.5 * ((n_water - 1) / (n_water + 1)) ** 2 * (
                1 - cos_i
            ) ** 5

            ocean_features = torch.stack([wind_speed, wavelength, theta_i, theta_r, phi_r - phi_i], dim=-1)
            ocean_correction = self.ocean_brdf(ocean_features).squeeze(-1)
            brdf = fresnel * P_slope / (4 * torch.cos(theta_i) * torch.cos(theta_r) + 1e-8) * ocean_correction
        else:
            lambertian = surface_type[:, 0] / math.pi
            h_theta = (theta_i + theta_r) / 2
            roughness = surface_type[:, 1]
            tan_h = torch.tan(h_theta)
            D = torch.exp(-tan_h**2 / (roughness**2 + 1e-12)) / (
                math.pi * roughness**2 * torch.cos(h_theta) ** 4 + 1e-12
            )
            n = 1.5
            F0 = ((n - 1) / (n + 1)) ** 2
            cos_i = torch.cos(theta_i)
            F = F0 + (1 - F0) * (1 - cos_i) ** 5
            k = roughness * math.sqrt(2 / math.pi)
            G_i = 2 / (1 + torch.sqrt(1 + k**2 * torch.tan(theta_i) ** 2))
            G_r = 2 / (1 + torch.sqrt(1 + k**2 * torch.tan(theta_r) ** 2))
            G = G_i * G_r
            specular = D * F * G / (4 * torch.cos(theta_i) * torch.cos(theta_r) + 1e-12)

            nn_input = torch.stack(
                [theta_i, phi_i, theta_r, phi_r, wavelength, surface_type[:, 2]],
                dim=-1,
            )
            nn_brdf = self.brdf_net(nn_input).squeeze(-1)
            brdf = lambertian + surface_type[:, 3] * specular + surface_type[:, 4] * nn_brdf

        return torch.clamp(brdf, min=0, max=1)

    def top_of_atmosphere(
        self, angles: torch.Tensor, wavelength: torch.Tensor, solar_zenith: torch.Tensor
    ) -> torch.Tensor:
        solar_irradiance = self._solar_spectrum(wavelength)
        toa_input = torch.cat([angles, wavelength.unsqueeze(-1), solar_zenith.unsqueeze(-1)], dim=-1)
        stokes_toa = self.toa_net(toa_input)
        cos_solar = torch.cos(solar_zenith)
        optical_depth_space = 0.1
        stokes_toa[:, 0] = solar_irradiance * torch.exp(-optical_depth_space / (cos_solar + 0.01))
        return stokes_toa

    def _solar_spectrum(self, wavelength: torch.Tensor) -> torch.Tensor:
        T_sun = 5778
        c = CONSTANTS.SPEED_OF_LIGHT
        h = CONSTANTS.PLANCK
        k = CONSTANTS.BOLTZMANN

        lambda_m = wavelength * 1e-9
        B = (2 * h * c**2 / lambda_m**5) / (torch.exp(h * c / (lambda_m * k * T_sun)) - 1)

        AU = 1.496e11
        R_sun = 6.96e8
        solid_angle = math.pi * (R_sun / AU) ** 2
        irradiance = B * solid_angle * 1e-9
        absorption = 1 - 0.1 * torch.exp(-((wavelength - 656) / 10) ** 2)
        irradiance = irradiance * absorption
        return irradiance


__all__ = ["AdvancedBoundaryConditions"]
