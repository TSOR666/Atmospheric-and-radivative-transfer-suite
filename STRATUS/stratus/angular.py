"""
Angular sampling utilities including spherical harmonics support.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .exceptions import StratusConfigError


@dataclass
class AngularSample:
    directions: torch.Tensor
    weights: torch.Tensor


class SphericalHarmonicsSampler:
    """
    Generate deterministic direction sets compatible with low-order SH expansions.
    """

    def __init__(self, order: int, device: torch.device, dtype: torch.dtype) -> None:
        if order < 0 or order > 4:
            raise StratusConfigError("Spherical harmonics order must be between 0 and 4.")
        self.order = order
        self.device = device
        self.dtype = dtype

    def sample(self) -> AngularSample:
        if self.order == 0:
            dirs = torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, -1.0],
                ],
                device=self.device,
                dtype=self.dtype,
            )
            weights = torch.full((6,), 1.0 / 6.0, device=self.device, dtype=self.dtype)
            return AngularSample(directions=dirs, weights=weights)

        if self.order == 1:
            dirs = torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, -1.0],
                    [1.0, 1.0, 0.0],
                    [1.0, -1.0, 0.0],
                    [-1.0, 1.0, 0.0],
                    [-1.0, -1.0, 0.0],
                    [0.0, 1.0, 1.0],
                    [0.0, 1.0, -1.0],
                    [0.0, -1.0, 1.0],
                    [0.0, -1.0, -1.0],
                    [1.0, 0.0, 1.0],
                    [1.0, 0.0, -1.0],
                    [-1.0, 0.0, 1.0],
                    [-1.0, 0.0, -1.0],
                ],
                device=self.device,
                dtype=self.dtype,
            )
            dirs = torch.nn.functional.normalize(dirs, dim=-1)
            weights = torch.full(
                (dirs.shape[0],), 1.0 / dirs.shape[0], device=self.device, dtype=self.dtype
            )
            return AngularSample(directions=dirs, weights=weights)

        # Higher order: use Fibonacci lattice
        n = max(20, (self.order + 1) ** 2 * 2)
        indices = torch.arange(0, n, device=self.device, dtype=self.dtype) + 0.5
        phi = torch.sqrt(torch.tensor(5.0, device=self.device, dtype=self.dtype) + 1.0) / 2.0
        theta = torch.acos(1 - 2 * indices / n)
        azimuth = 2 * math.pi * indices / phi

        dirs = torch.stack(
            [
                torch.sin(theta) * torch.cos(azimuth),
                torch.sin(theta) * torch.sin(azimuth),
                torch.cos(theta),
            ],
            dim=-1,
        )
        weights = torch.full((n,), 1.0 / n, device=self.device, dtype=self.dtype)
        return AngularSample(directions=dirs, weights=weights)


def real_spherical_harmonics(order: int, directions: torch.Tensor) -> torch.Tensor:
    """
    Evaluate real spherical harmonics up to the requested order.
    Returns tensor with shape [N, (order+1)^2].
    """

    if order > 2:
        raise NotImplementedError("Real spherical harmonics implemented up to order 2.")

    x, y, z = directions.unbind(-1)
    theta = torch.acos(z.clamp(-1.0, 1.0))
    phi = torch.atan2(y, x)

    num_coeffs = (order + 1) ** 2
    result = directions.new_empty((*directions.shape[:-1], num_coeffs))
    idx = 0
    result[..., idx] = 0.5 / math.sqrt(math.pi)
    idx += 1

    if order >= 1:
        coeff = math.sqrt(3.0 / (4 * math.pi))
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        result[..., idx] = coeff * sin_theta * torch.cos(phi)
        idx += 1
        result[..., idx] = coeff * cos_theta
        idx += 1
        result[..., idx] = coeff * sin_theta * torch.sin(phi)
        idx += 1

    if order >= 2:
        coeff_15_16 = math.sqrt(15.0 / (16 * math.pi))
        coeff_15_4 = math.sqrt(15.0 / (4 * math.pi))
        coeff_5_16 = math.sqrt(5.0 / (16 * math.pi))
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        sin_theta_sq = sin_theta**2
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)
        result[..., idx] = coeff_15_16 * sin_theta_sq * torch.cos(2 * phi)
        idx += 1
        result[..., idx] = coeff_15_4 * sin_theta * cos_theta * cos_phi
        idx += 1
        result[..., idx] = coeff_5_16 * (3 * cos_theta**2 - 1)
        idx += 1
        result[..., idx] = coeff_15_4 * sin_theta * cos_theta * sin_phi
        idx += 1
        result[..., idx] = coeff_15_16 * sin_theta_sq * torch.sin(2 * phi)

    return result


__all__ = ["SphericalHarmonicsSampler", "AngularSample", "real_spherical_harmonics"]
