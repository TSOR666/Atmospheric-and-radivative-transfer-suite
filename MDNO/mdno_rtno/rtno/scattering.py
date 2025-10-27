from __future__ import annotations

import math
from typing import Dict, Tuple, Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import RTNOConfig
from .constants import CONSTANTS


class MieScattering(nn.Module):
    """Mie scattering implementation with Riccati-Bessel functions."""

    def __init__(self, wavelengths: torch.Tensor, config: RTNOConfig):
        super().__init__()
        self.register_buffer("wavelengths", wavelengths)
        self.config = config
        self._mie_cache: Dict = {}
        self.size_distribution = nn.Parameter(torch.tensor([1e-6, 0.5]))

    def size_parameter(self, wavelength: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
        return 2 * math.pi * radius / (wavelength * 1e-9)

    def _riccati_bessel(
        self, n: int, x: torch.Tensor, derivative: bool = False
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        psi = torch.zeros(n + 2, *x.shape, dtype=torch.complex64, device=x.device)
        chi = torch.zeros(n + 2, *x.shape, dtype=torch.complex64, device=x.device)

        psi[0] = torch.sin(x)
        psi[1] = psi[0] / (x + 1e-12) - torch.cos(x)
        chi[0] = -torch.cos(x)
        chi[1] = -chi[0] / (x + 1e-12) - torch.sin(x)

        for i in range(2, n + 2):
            psi[i] = (2 * i - 1) / (x + 1e-12) * psi[i - 1] - psi[i - 2]
            chi[i] = (2 * i - 1) / (x + 1e-12) * chi[i - 1] - chi[i - 2]

        xi = psi - 1j * chi

        if derivative:
            psi_prime = torch.zeros_like(psi)
            xi_prime = torch.zeros_like(xi)

            psi_prime[0] = torch.cos(x)
            xi_prime[0] = torch.cos(x) + 1j * torch.sin(x)

            for i in range(1, n + 1):
                psi_prime[i] = psi[i - 1] - i / (x + 1e-12) * psi[i]
                xi_prime[i] = xi[i - 1] - i / (x + 1e-12) * xi[i]

            return psi, xi, psi_prime, xi_prime

        return psi, xi

    def compute_mie_coefficients(self, x: torch.Tensor, m: complex) -> Tuple[torch.Tensor, torch.Tensor]:
        x_tensor = torch.as_tensor(x)
        x_flat = x_tensor.detach().reshape(-1)

        if x_flat.numel() == 1:
            key_values = (float(x_flat.item()),)
        else:
            key_values = tuple(float(val) for val in x_flat[:10].cpu().tolist())

        shape_key = tuple(x_tensor.shape)
        cache_key = (key_values, shape_key, complex(m))
        if cache_key in self._mie_cache:
            return self._mie_cache[cache_key]

        x_max = float(x_flat.abs().max().item()) if x_flat.numel() > 0 else 0.0
        nmax = max(int(x_max + 4.0 * x_max ** (1 / 3) + 2.0), 1)
        nmax = min(nmax, 200)

        mx = m * x
        psi, xi, psi_prime, xi_prime = cast(
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            self._riccati_bessel(nmax, x, derivative=True),
        )
        psi_mx, xi_mx, psi_mx_prime, xi_mx_prime = cast(
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            self._riccati_bessel(nmax, mx, derivative=True),
        )

        an = torch.zeros(nmax, dtype=torch.complex64, device=x.device)
        bn = torch.zeros(nmax, dtype=torch.complex64, device=x.device)

        for n in range(1, nmax + 1):
            numerator_a = m * psi_mx[n] * psi_prime[n] - psi[n] * psi_mx_prime[n]
            denominator_a = m * psi_mx[n] * xi_prime[n] - xi[n] * psi_mx_prime[n]
            an[n - 1] = numerator_a / (denominator_a + 1e-30)

            numerator_b = psi_mx[n] * psi_prime[n] - m * psi[n] * psi_mx_prime[n]
            denominator_b = psi_mx[n] * xi_prime[n] - m * xi[n] * psi_mx_prime[n]
            bn[n - 1] = numerator_b / (denominator_b + 1e-30)

        if len(self._mie_cache) < 1000:
            self._mie_cache[cache_key] = (an, bn)

        return an, bn

    def compute_phase_function(
        self,
        cos_theta: torch.Tensor,
        wavelength: torch.Tensor,
        radius: torch.Tensor,
        refractive_index: complex,
    ) -> torch.Tensor:
        x = self.size_parameter(wavelength, radius)
        an, bn = self.compute_mie_coefficients(x, refractive_index)
        nmax = len(an)

        S1 = torch.zeros_like(cos_theta, dtype=torch.complex64)
        S2 = torch.zeros_like(cos_theta, dtype=torch.complex64)

        pi_n = torch.zeros(nmax + 1, *cos_theta.shape, device=cos_theta.device)
        tau_n = torch.zeros(nmax + 1, *cos_theta.shape, device=cos_theta.device)

        pi_n[0] = 1.0
        pi_n[1] = 3.0 * cos_theta
        tau_n[1] = cos_theta

        for n in range(1, nmax + 1):
            if n > 1:
                pi_n[n] = ((2 * n + 1) * cos_theta * pi_n[n - 1] - (n + 1) * pi_n[n - 2]) / n
                tau_n[n] = n * cos_theta * pi_n[n] - (n + 1) * pi_n[n - 1]

            factor = (2 * n + 1) / (n * (n + 1))
            S1 += factor * (an[n - 1] * pi_n[n] + bn[n - 1] * tau_n[n])
            S2 += factor * (an[n - 1] * tau_n[n] + bn[n - 1] * pi_n[n])

        phase = (torch.abs(S1) ** 2 + torch.abs(S2) ** 2) / (2 * x**2)
        return phase.real

    def compute_optical_properties(
        self,
        wavelength: torch.Tensor,
        radius: torch.Tensor,
        refractive_index: complex,
        number_density: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        x = self.size_parameter(wavelength, radius)
        an, bn = self.compute_mie_coefficients(x, refractive_index)
        nmax = len(an)

        Q_ext = torch.zeros((), dtype=torch.float32, device=x.device)
        Q_sca = torch.zeros((), dtype=torch.float32, device=x.device)
        g_sum = torch.zeros((), dtype=torch.float32, device=x.device)

        for n in range(1, nmax + 1):
            Q_ext = Q_ext + (2 * n + 1) * (an[n - 1].real + bn[n - 1].real)
            Q_sca = Q_sca + (2 * n + 1) * (torch.abs(an[n - 1]) ** 2 + torch.abs(bn[n - 1]) ** 2)

            if n < nmax:
                g_sum = g_sum + (n * (n + 2) / (n + 1)) * (
                    an[n - 1] * an[n].conj() + bn[n - 1] * bn[n].conj()
                ).real
                g_sum = g_sum + ((2 * n + 1) / (n * (n + 1))) * (an[n - 1] * bn[n - 1].conj()).real

        Q_ext = (2 / x**2) * Q_ext
        Q_sca = (2 / x**2) * Q_sca
        Q_abs = Q_ext - Q_sca
        g = (4 / (x**2 * Q_sca + 1e-30)) * g_sum

        area = math.pi * radius**2
        sigma_ext = Q_ext * area
        sigma_sca = Q_sca * area
        sigma_abs = Q_abs * area

        k_ext = number_density * sigma_ext
        k_sca = number_density * sigma_sca
        k_abs = number_density * sigma_abs

        return {
            "extinction": k_ext,
            "scattering": k_sca,
            "absorption": k_abs,
            "single_scattering_albedo": Q_sca / (Q_ext + 1e-30),
            "asymmetry_parameter": g,
        }


class GasAbsorption(nn.Module):
    """Gas absorption with Voigt profiles, continuum, and line mixing."""

    def __init__(self, wavelengths: torch.Tensor, config: RTNOConfig):
        super().__init__()
        self.register_buffer("wavelengths", wavelengths)
        self.config = config
        self._init_line_database()

        self.line_mixing_net = nn.Sequential(
            nn.Linear(10, 64),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, len(wavelengths)),
            nn.Sigmoid(),
        )

        self.continuum_net = nn.Sequential(
            nn.Linear(5, 32),
            nn.SiLU(),
            nn.Linear(32, len(wavelengths)),
            nn.Softplus(),
        )

    def _init_line_database(self) -> None:
        self.o3_lines = torch.tensor(
            [
                [254.0, 1.15e-17, 2.0, 0.05],
                [280.0, 5.0e-18, 3.0, 0.04],
                [310.0, 1.9e-19, 2.5, 0.03],
            ]
        )
        self.h2o_lines = torch.tensor(
            [
                [940.0, 2.5e-23, 3.0, 0.4],
                [1130.0, 8.2e-24, 2.5, 0.35],
                [1380.0, 1.8e-22, 4.0, 0.45],
            ]
        )
        self.co2_lines = torch.tensor(
            [
                [1400.0, 3.5e-25, 3.0, 0.5],
                [1600.0, 7.8e-26, 2.5, 0.48],
                [2000.0, 4.2e-24, 4.0, 0.52],
                [2700.0, 1.2e-24, 3.5, 0.49],
            ]
        )

    def voigt_profile(self, nu: torch.Tensor, nu0: float, gamma_L: torch.Tensor, gamma_D: torch.Tensor) -> torch.Tensor:
        sigma = gamma_D / (math.sqrt(2 * math.log(2)) + 1e-12)
        x = (nu - nu0) / (sigma + 1e-12)
        y = gamma_L / (sigma + 1e-12)

        t = y - 1j * x
        s = torch.abs(x) + y

        mask1 = s >= 15
        mask2 = (s < 15) & (torch.abs(x) >= 5.5)
        mask3 = ~(mask1 | mask2)

        w1 = t * 0.5641896 / (0.5 + t**2)
        u = t**2
        w2 = t * (1.410474 + u * 0.5641896) / (0.75 + u * (3.0 + u))
        w3 = torch.exp(-x**2) * torch.cos(2 * x * y) / math.sqrt(math.pi) + 2 * y / math.pi * torch.sin(x**2) / (
            x**2 + y**2 + 1e-10
        )

        result = torch.where(mask1, w1.real, torch.where(mask2, w2.real, w3))
        return result / (sigma * math.sqrt(math.pi) + 1e-12)

    def compute_cross_section(
        self,
        temperature: torch.Tensor,
        pressure: torch.Tensor,
        concentrations: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        device = temperature.device
        n_wavelengths = len(self.wavelengths)
        cross_sections = torch.zeros(n_wavelengths, device=device, dtype=self.config.dtype)

        T_ref = CONSTANTS.STANDARD_TEMPERATURE
        P_ref = CONSTANTS.STANDARD_PRESSURE

        for species, lines in (("O3", self.o3_lines), ("H2O", self.h2o_lines), ("CO2", self.co2_lines)):
            if species.lower() in concentrations:
                conc = concentrations[species.lower()]
                for line in lines:
                    center_wl = line[0]
                    strength = line[1]
                    width = line[2]
                    temp_exp = line[3]

                    strength_T = strength * (T_ref / (temperature + 1e-12)) ** temp_exp
                    gamma_L = width * (pressure / (P_ref + 1e-12)) * torch.sqrt(T_ref / (temperature + 1e-12))
                    mass = {"O3": 48.0, "H2O": 18.0, "CO2": 44.0}[species]
                    gamma_D = center_wl / CONSTANTS.SPEED_OF_LIGHT * torch.sqrt(
                        2 * CONSTANTS.BOLTZMANN * temperature * CONSTANTS.AVOGADRO / (mass + 1e-12)
                    )

                    for i, wl in enumerate(self.wavelengths):
                        profile = self.voigt_profile(wl, center_wl, gamma_L, gamma_D)
                        cross_sections[i] += conc * strength_T * profile

        continuum_features = torch.stack(
            [
                temperature / (T_ref + 1e-12),
                pressure / (P_ref + 1e-12),
                concentrations.get("h2o", torch.tensor(0.0, device=device, dtype=self.config.dtype)),
                torch.tensor(1.0, device=device, dtype=self.config.dtype),
                torch.tensor(0.0, device=device, dtype=self.config.dtype),
            ]
        )

        continuum = self.continuum_net(continuum_features)
        cross_sections += continuum

        mixing_features = torch.cat(
            [
                temperature.unsqueeze(0) / (T_ref + 1e-12),
                pressure.unsqueeze(0) / (P_ref + 1e-12),
                cross_sections[:8],
            ]
        )

        if mixing_features.shape[0] < 10:
            mixing_features = F.pad(mixing_features, (0, 10 - mixing_features.shape[0]))

        mixing_correction = self.line_mixing_net(mixing_features[:10])
        cross_sections = cross_sections * (1 + 0.1 * (mixing_correction - 0.5))

        return cross_sections


__all__ = ["MieScattering", "GasAbsorption"]
