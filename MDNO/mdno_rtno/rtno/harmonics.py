from __future__ import annotations

import math
from functools import lru_cache
from typing import Dict, Tuple

import numpy as np
import torch

from .utils import SCIPY_AVAILABLE, sp


class SphericalHarmonics:
    """Complete spherical harmonics helper with caching and SciPy fallback."""

    def __init__(self, max_order: int = 16, enable_cache: bool = True):
        self.max_order = max_order
        self.enable_cache = enable_cache
        self._cache: Dict[Tuple[int, int, Tuple[int, ...], str], torch.Tensor] = {}
        self._norm_cache = self._precompute_normalization()
        self._legendre_cache: Dict[Tuple[int, int, Tuple[int, ...], str], torch.Tensor] = {}

    def _precompute_normalization(self) -> Dict[Tuple[int, int], float]:
        norms: Dict[Tuple[int, int], float] = {}
        for l in range(self.max_order + 1):
            for m in range(-l, l + 1):
                if m == 0:
                    norms[(l, m)] = math.sqrt((2 * l + 1) / (4 * math.pi))
                else:
                    factor = 1.0
                    for k in range(l - abs(m) + 1, l + abs(m) + 1):
                        factor *= k
                    factor = 1.0 / factor
                    norms[(l, m)] = math.sqrt((2 * l + 1) * factor / (2 * math.pi))
        return norms

    @lru_cache(maxsize=1024)
    def _associated_legendre(self, l: int, m: int, x: torch.Tensor) -> torch.Tensor:
        m_abs = abs(m)
        cache_key = (l, m_abs, tuple(x.shape), x.device.type)
        if cache_key in self._legendre_cache:
            return self._legendre_cache[cache_key]

        if l == m_abs:
            if m_abs == 0:
                result = torch.ones_like(x)
            else:
                sqrt_term = torch.sqrt(torch.clamp(1 - x * x, min=0.0) + 1e-12)
                double_factorial = 1.0
                for i in range(1, m_abs + 1):
                    double_factorial *= (2 * i - 1)
                result = ((-1) ** m_abs) * double_factorial * (sqrt_term**m_abs)
        elif l == m_abs + 1:
            P_mm = self._associated_legendre(m_abs, m_abs, x)
            result = x * (2 * m_abs + 1) * P_mm
        else:
            P_lm_2 = self._associated_legendre(m_abs, m_abs, x)
            P_lm_1 = self._associated_legendre(m_abs + 1, m_abs, x)
            for ll in range(m_abs + 2, l + 1):
                denom = ll - m_abs
                P_lm = ((2 * ll - 1) * x * P_lm_1 - (ll + m_abs - 1) * P_lm_2) / denom
                P_lm_2, P_lm_1 = P_lm_1, P_lm
            result = P_lm_1

        if self.enable_cache:
            self._legendre_cache[cache_key] = result
        return result

    def compute_ylm(self, l: int, m: int, theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        cache_key = (l, m, tuple(theta.shape), theta.device.type)
        if self.enable_cache and cache_key in self._cache:
            return self._cache[cache_key]

        cos_theta = torch.cos(theta)

        if SCIPY_AVAILABLE and theta.numel() < 10000:
            theta_np = theta.detach().cpu().numpy()
            phi_np = phi.detach().cpu().numpy()
            if m >= 0:
                ylm = sp.sph_harm(m, l, phi_np, theta_np).real
            else:
                ylm = sp.sph_harm(-m, l, phi_np, theta_np).imag * ((-1) ** m)
            result = torch.from_numpy(ylm).to(theta.device, dtype=theta.dtype)
        else:
            P_lm = self._associated_legendre(l, abs(m), cos_theta)
            norm = self._norm_cache.get((l, m), 1.0)
            if m == 0:
                result = norm * P_lm
            elif m > 0:
                result = math.sqrt(2) * norm * P_lm * torch.cos(m * phi)
            else:
                result = math.sqrt(2) * norm * P_lm * torch.sin(abs(m) * phi)

        if self.enable_cache and result.numel() < 100000:
            self._cache[cache_key] = result

        return result

    def expand_field(
        self, field: torch.Tensor, theta_grid: torch.Tensor, phi_grid: torch.Tensor
    ) -> Dict[Tuple[int, int], torch.Tensor]:
        coeffs: Dict[Tuple[int, int], torch.Tensor] = {}
        for l in range(self.max_order + 1):
            for m in range(-l, l + 1):
                ylm = self.compute_ylm(l, m, theta_grid, phi_grid)
                coeffs[(l, m)] = (
                    torch.sum(field * ylm * torch.sin(theta_grid))
                    * (2 * np.pi / theta_grid.shape[0])
                    * (np.pi / phi_grid.shape[1])
                )
        return coeffs

    def reconstruct_field(
        self, coeffs: Dict[Tuple[int, int], torch.Tensor], theta: torch.Tensor, phi: torch.Tensor
    ) -> torch.Tensor:
        field = torch.zeros_like(theta)
        for (l, m), coeff in coeffs.items():
            if l <= self.max_order:
                ylm = self.compute_ylm(l, m, theta, phi)
                field += coeff * ylm
        return field

    def clear_cache(self) -> None:
        self._cache.clear()
        self._legendre_cache.clear()
        self._associated_legendre.cache_clear()


__all__ = ["SphericalHarmonics"]
