from __future__ import annotations

import math
import warnings
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._logging import LOGGER as logger
from .boundary import AdvancedBoundaryConditions
from .config import BoundaryType, RTNOConfig
from .constants import CONSTANTS
from .harmonics import SphericalHarmonics
from .monitoring import monitor
from .optics import MultipleScatteringSolver, OpticalDepthScaling, ShortCharacteristics3D
from .polarization import MuellerMatrixPolarization
from .refraction import AtmosphericRefraction
from .scattering import GasAbsorption, MieScattering


class Complete3DRadiativeTransferSolver(nn.Module):
    """Full 3D radiative-transfer solver including all physical components."""

    def __init__(self, config: RTNOConfig):
        super().__init__()
        self.config = config

        self.spherical_harmonics = SphericalHarmonics(config.spherical_harmonics_order, config.enable_caching)

        self.gas_absorption: Optional[GasAbsorption] = None
        if config.use_gas_absorption:
            self.gas_absorption = GasAbsorption(config.wavelengths, config)
            logger.info("GasAbsorption component bound")
        else:
            logger.info("GasAbsorption disabled")

        self.mie_scattering: Optional[MieScattering] = None
        if config.use_mie_scattering:
            self.mie_scattering = MieScattering(config.wavelengths, config)

        self.refraction: Optional[AtmosphericRefraction] = None
        if config.use_refraction:
            self.refraction = AtmosphericRefraction(config)

        self.boundary_conditions = AdvancedBoundaryConditions(config)

        self.scatter_solver: Optional[MultipleScatteringSolver] = None
        if config.use_multiple_scattering:
            self.scatter_solver = MultipleScatteringSolver(config)

        self.coupling_solver: Optional[ShortCharacteristics3D] = None
        if config.use_horizontal_coupling:
            self.coupling_solver = ShortCharacteristics3D(config)

        self.mueller_polarization = MuellerMatrixPolarization()

        in_channels = config.n_stokes * len(config.wavelengths)
        self.neural_correction = nn.Sequential(
            nn.Conv3d(in_channels, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv3d(128, 256, 3, padding=1),
            nn.GroupNorm(16, 256),
            nn.SiLU(),
            nn.Conv3d(256, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv3d(128, in_channels, 3, padding=1),
            nn.Tanh(),
        )

        logger.info("Complete 3D RT Solver initialized with functional polarization")

    def solve(
        self,
        atmospheric_state: Dict[str, torch.Tensor],
        boundary_conditions_dict: Dict[str, Any],
        max_iterations: Optional[int] = None,
    ) -> torch.Tensor:
        max_iterations = max_iterations or self.config.max_iterations

        with monitor.timer("rt_solve"):
            if self.config.validate_inputs:
                self._validate_inputs(atmospheric_state)

            T = atmospheric_state["temperature"]
            P = atmospheric_state["pressure"]
            H = atmospheric_state.get("humidity", torch.zeros_like(T))
            density = atmospheric_state.get("density", P / (CONSTANTS.DRY_AIR_GAS_CONSTANT * T))

            batch_size, nz, ny, nx = T.shape
            n_wavelengths = len(self.config.wavelengths)

            radiance = torch.zeros(
                batch_size,
                self.config.n_stokes,
                n_wavelengths,
                nz,
                ny,
                nx,
                device=T.device,
                dtype=self.config.dtype,
            )

            optical_props = self._compute_optical_properties(T, P, H, density, atmospheric_state)

            if self.config.use_delta_eddington:
                optical_props = self._apply_delta_eddington(optical_props)

            angles = self._setup_angular_discretization()

            for w_idx in range(n_wavelengths):
                wavelength = self.config.wavelengths[w_idx]
                props_wl = {
                    "extinction": optical_props["extinction"][:, w_idx],
                    "scattering": optical_props["scattering"][:, w_idx],
                    "absorption": optical_props["absorption"][:, w_idx],
                    "phase_function": optical_props.get("phase_function"),
                }

                g = optical_props.get("asymmetry_parameter", torch.tensor(0.7, device=T.device))

                def phase_function(cos_theta: torch.Tensor, g_val=g) -> torch.Tensor:
                    return (1 - g_val**2) / (4 * math.pi * (1 + g_val**2 - 2 * g_val * cos_theta) ** (3 / 2))

                thermal_source = self._compute_thermal_source(T, props_wl["extinction"], wavelength)
                n_angles = len(angles["mu"])
                boundary_radiance = torch.zeros(
                    batch_size, n_angles, ny, nx, device=T.device, dtype=self.config.dtype
                )

                if "surface" in boundary_conditions_dict:
                    surface_state = boundary_conditions_dict["surface"]
                    incident = self._compute_incident_angles(n_angles, device=T.device)
                    reflected = self._compute_reflected_angles(n_angles, device=T.device)
                    surface_type = surface_state.get(
                        "surface_type",
                        torch.ones(batch_size, 6, device=T.device, dtype=self.config.dtype),
                    )
                    boundary_type = surface_state.get("boundary_type", BoundaryType.MIXED)
                    if isinstance(boundary_type, str):
                        boundary_type = BoundaryType(boundary_type)
                    brdf = self.boundary_conditions.surface_brdf(
                        incident,
                        reflected,
                        torch.ones(n_angles, device=T.device) * wavelength,
                        surface_type,
                        boundary_type,
                    )
                    solar_zenith = surface_state.get("solar_zenith", torch.zeros(batch_size, device=T.device))
                    solar_flux = torch.cos(solar_zenith).clamp(min=0)
                    boundary_radiance[:, :, :, :] = (
                        brdf.view(1, -1, 1, 1) * solar_flux.view(-1, 1, 1, 1)
                    )

                if "top_of_atmosphere" in boundary_conditions_dict:
                    toa = boundary_conditions_dict["top_of_atmosphere"]
                    angles_toa = self._compute_incident_angles(n_angles, device=T.device)
                    solar_zenith = toa.get("solar_zenith", torch.zeros(batch_size, device=T.device))
                    toa_radiance = self.boundary_conditions.top_of_atmosphere(
                        angles_toa, torch.ones(n_angles, device=T.device) * wavelength, solar_zenith
                    )
                    boundary_radiance[:, :, -1] = toa_radiance[:, :n_angles]

                scatter_solver = self.scatter_solver
                if scatter_solver is not None:
                    rt_radiance = scatter_solver.solve_source_iteration(
                        props_wl["extinction"],
                        props_wl["scattering"],
                        phase_function,
                        thermal_source,
                        boundary_radiance,
                        angles,
                    )
                else:
                    rt_radiance = self._scalar_radiative_transfer(
                        props_wl["extinction"], thermal_source, boundary_radiance, angles
                    )

                if self.config.n_stokes == 4:
                    stokes = self._apply_mueller_coupling_functional(
                        rt_radiance.unsqueeze(1).repeat(1, 4, 1, 1, 1),
                        angles,
                        {
                            "single_scattering_albedo": optical_props["single_scattering_albedo"][:, w_idx],
                        },
                    )
                    radiance[:, :, w_idx] = stokes
                else:
                    radiance[:, 0, w_idx] = rt_radiance.mean(dim=1)

            radiance = self._apply_neural_correction(radiance)

            return radiance

    def _compute_optical_properties(
        self,
        temperature: torch.Tensor,
        pressure: torch.Tensor,
        humidity: torch.Tensor,
        density: torch.Tensor,
        state: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        n_wavelengths = len(self.config.wavelengths)
        extinction = torch.zeros_like(temperature).unsqueeze(1).repeat(1, n_wavelengths, 1, 1, 1)
        scattering = torch.zeros_like(extinction)
        absorption = torch.zeros_like(extinction)
        asymmetry = torch.zeros_like(extinction)

        if self.config.use_mie_scattering and self.mie_scattering is not None:
            radius = state.get("particle_radius", torch.ones_like(temperature) * 1e-6)
            refractive_index = complex(1.45, 1e-3)
            number_density = density * 1e6
            for w_idx, wavelength in enumerate(self.config.wavelengths):
                mie_props = self.mie_scattering.compute_optical_properties(
                    wavelength, radius, refractive_index, number_density
                )
                extinction[:, w_idx] = mie_props["extinction"]
                scattering[:, w_idx] = mie_props["scattering"]
                absorption[:, w_idx] = mie_props["absorption"]
                asymmetry[:, w_idx] = mie_props["asymmetry_parameter"]
        else:
            extinction += 1e-4

        if self.config.use_gas_absorption and self.gas_absorption is not None:
            concentrations = {
                key: state[key]
                for key in ("o3", "h2o", "co2")
                if key in state
            }
            for w_idx in range(n_wavelengths):
                cross_section = self.gas_absorption.compute_cross_section(
                    temperature, pressure, concentrations
                )
                absorption[:, w_idx] += cross_section.view(1, -1, 1, 1)

        single_scattering_albedo = scattering / (extinction + 1e-10)
        optical_props: Dict[str, torch.Tensor] = {
            "extinction": extinction,
            "scattering": scattering,
            "absorption": absorption,
            "single_scattering_albedo": single_scattering_albedo,
            "asymmetry_parameter": asymmetry.mean(dim=1),
        }

        return optical_props

    def _apply_delta_eddington(self, optical_props: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        tau = optical_props["extinction"]
        omega = optical_props["single_scattering_albedo"]
        g = optical_props["asymmetry_parameter"]

        tau_star, omega_star, g_star = OpticalDepthScaling.delta_eddington_scaling(tau, omega, g)
        optical_props["extinction"] = tau_star
        optical_props["single_scattering_albedo"] = omega_star
        optical_props["asymmetry_parameter"] = g_star
        return optical_props

    def _setup_angular_discretization(self) -> Dict[str, torch.Tensor]:
        n_streams = self.config.discrete_ordinates_streams
        mu = torch.linspace(-1 + 1 / n_streams, 1 - 1 / n_streams, n_streams, device=self.config.get_device())
        weights = torch.ones_like(mu) * (2 / n_streams)
        phi = torch.linspace(0, 2 * math.pi, n_streams, device=self.config.get_device(), dtype=self.config.dtype)
        return {"mu": mu, "weights": weights, "phi": phi}

    def _compute_thermal_source(
        self, temperature: torch.Tensor, extinction: torch.Tensor, wavelength: torch.Tensor
    ) -> torch.Tensor:
        planck = (2 * CONSTANTS.PLANCK * CONSTANTS.SPEED_OF_LIGHT**2) / (
            (wavelength * 1e-9) ** 5 * (torch.exp(
                CONSTANTS.PLANCK * CONSTANTS.SPEED_OF_LIGHT / (wavelength * 1e-9 * CONSTANTS.BOLTZMANN * temperature)
            ) - 1)
        )
        return planck * (1 - torch.exp(-extinction))

    def _compute_incident_angles(self, n_angles: int, device: torch.device) -> torch.Tensor:
        theta = torch.linspace(0, math.pi / 2, n_angles, device=device)
        phi = torch.linspace(0, 2 * math.pi, n_angles, device=device)
        return torch.stack([theta, phi], dim=-1)

    def _compute_reflected_angles(self, n_angles: int, device: torch.device) -> torch.Tensor:
        theta = torch.linspace(0, math.pi / 2, n_angles, device=device)
        phi = torch.linspace(0, 2 * math.pi, n_angles, device=device)
        return torch.stack([theta, phi], dim=-1)

    def _scalar_radiative_transfer(
        self,
        extinction: torch.Tensor,
        source: torch.Tensor,
        boundary_radiance: torch.Tensor,
        angles: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        batch_size, nz, ny, nx = extinction.shape
        n_angles = len(angles["mu"])
        radiance = torch.zeros(batch_size, n_angles, nz, ny, nx, device=extinction.device, dtype=extinction.dtype)

        for i, mu in enumerate(angles["mu"]):
            if mu >= 0:
                radiance[:, i, 0] = boundary_radiance[:, i]
                for k in range(1, nz):
                    tau = extinction[:, k] * self.config.dz / (mu + 1e-10)
                    trans = OpticalDepthScaling.safe_transmittance(tau)
                    radiance[:, i, k] = radiance[:, i, k - 1] * trans + source[:, k] * (1 - trans) / (
                        extinction[:, k] + 1e-10
                    )
            else:
                radiance[:, i, -1] = boundary_radiance[:, i, -1]
                for k in range(nz - 2, -1, -1):
                    tau = extinction[:, k] * self.config.dz / abs(mu)
                    trans = OpticalDepthScaling.safe_transmittance(tau)
                    radiance[:, i, k] = radiance[:, i, k + 1] * trans + source[:, k] * (1 - trans) / (
                        extinction[:, k] + 1e-10
                    )
        return radiance

    def _apply_mueller_coupling_functional(
        self,
        radiance: torch.Tensor,
        angles: Dict[str, torch.Tensor],
        optical_props: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if self.config.n_stokes == 1:
            return radiance

        batch_size = radiance.shape[0]
        n_wavelengths = radiance.shape[2]
        spatial_shape = radiance.shape[3:]
        n_angles = len(angles["mu"])

        stokes_full = (
            torch.zeros(batch_size, 4, n_wavelengths, *spatial_shape, device=radiance.device, dtype=radiance.dtype)
            if radiance.shape[1] < 4
            else radiance.clone()
        )
        if radiance.shape[1] < 4:
            stokes_full[:, 0] = radiance[:, 0]

        for w_idx in range(n_wavelengths):
            omega = optical_props["single_scattering_albedo"]
            mueller_matrices = []
            for mu in angles["mu"]:
                mueller_matrices.append(self.mueller_polarization.rayleigh_mueller_matrix(mu))
            mueller_stack = torch.stack(mueller_matrices, dim=0)
            stokes_wl = stokes_full[:, :, w_idx].unsqueeze(2).expand(-1, -1, n_angles, *spatial_shape)
            stokes_scattered = self.mueller_polarization.apply_mueller_scattering(
                stokes_wl, mueller_stack, omega
            )
            stokes_full[:, :, w_idx] = stokes_scattered.mean(dim=2)

        return stokes_full

    def _apply_neural_correction(self, radiance: torch.Tensor) -> torch.Tensor:
        batch_size = radiance.shape[0]
        radiance_flat = radiance.view(batch_size, -1, *radiance.shape[-3:])
        correction = self.neural_correction(radiance_flat)
        radiance_corrected = radiance_flat + 0.05 * correction
        return F.relu(radiance_corrected).view_as(radiance)

    def _validate_inputs(self, state: Dict[str, torch.Tensor]) -> None:
        for key in ("temperature", "pressure"):
            if key not in state:
                raise ValueError(f"Missing required key: {key}")
            tensor = state[key]
            if torch.isnan(tensor).any():
                raise ValueError(f"NaN detected in {key}")
            if torch.isinf(tensor).any():
                raise ValueError(f"Inf detected in {key}")
            if key == "temperature":
                if (tensor < 0).any():
                    raise ValueError("Temperature cannot be negative")
                if (tensor < 100).any() or (tensor > 400).any():
                    warnings.warn("Temperature outside typical range [100, 400]K")
            elif key == "pressure":
                if (tensor <= 0).any():
                    raise ValueError("Pressure must be positive")


__all__ = ["Complete3DRadiativeTransferSolver"]
