"""
Validation helpers for the STRATUS framework.

The routines in this module provide lightweight physics and numerical
consistency checks that can be executed as part of integration tests or manual
smoke tests.  They supersede the dispersed validation snippets buried in the
legacy scripts.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import torch

from .config import StratusConfig
from .model import StratusRadianceModel

logger = logging.getLogger("stratus.validation")


@dataclass
class ValidationResult:
    """Container for validation metrics."""

    name: str
    passed: bool
    metrics: dict[str, float]


class RadiativeValidator:
    """
    Execute basic physics and numerical sanity checks on a STRATUS model.
    """

    def __init__(self, model: StratusRadianceModel, config: StratusConfig) -> None:
        self.model = model
        self.config = config

    # ---------------------------------------------------------------- analysis
    def uniform_slab_test(
        self,
        *,
        kappa_value: float = 0.05,
        source_value: float = 0.01,
        point: torch.Tensor | None = None,
    ) -> ValidationResult:
        """
        Compare the numerical solution against the analytic solution for a
        homogeneous slab with constant absorption/emission.
        """

        device = self.config.device
        dtype = self.config.dtype
        nx, ny, nz = self.config.grid_shape

        kappa = torch.full(
            (1, self.config.n_stokes, nx, ny, nz, self.config.n_bands),
            fill_value=kappa_value,
            device=device,
            dtype=dtype,
        )
        source = torch.full_like(kappa, fill_value=source_value)

        if point is None:
            point = torch.tensor([[nx / 2.0, ny / 2.0, nz / 4.0]], device=device, dtype=dtype)
        else:
            point = point.to(device=device, dtype=dtype).unsqueeze(0)

        ray_dir = torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=dtype)

        with torch.no_grad():
            result = self.model(
                kappa,
                source,
                point,
                ray_directions=ray_dir,
                ray_weights=torch.ones(1, device=device, dtype=dtype),
            )

        radiance = result["radiance"][0, 0, 0, 0].item()

        z_exit = nz * self.config.voxel_size[2]
        z_origin = point[0, 2].item() * self.config.voxel_size[2]
        path_length = max(z_exit - z_origin, 0.0)
        analytical = (source_value / kappa_value) * (
            1.0 - float(math.exp(-kappa_value * path_length))
        )

        rel_error = abs(radiance - analytical) / max(abs(analytical), 1e-8)
        passed = rel_error < 0.05

        metrics = {
            "numerical_radiance": float(radiance),
            "analytic_radiance": float(analytical),
            "relative_error": float(rel_error),
        }

        return ValidationResult("uniform_slab", passed, metrics)

    def cpu_gpu_parity_test(
        self,
        *,
        n_points: int = 64,
        tolerance: float = 1.0e-2,
    ) -> ValidationResult:
        """
        Confirm that CPU and GPU executions produce comparable radiance values.
        """

        if not torch.cuda.is_available():
            logger.info("Skipping CPU/GPU parity test; CUDA not available.")
            return ValidationResult("cpu_gpu_parity", True, {"skipped": 1.0})

        config_cpu = StratusConfig(
            grid_shape=self.config.grid_shape,
            voxel_size=self.config.voxel_size,
            spectral_method=self.config.spectral_method,
            n_stokes=self.config.n_stokes,
            n_bands=self.config.n_bands,
            boundary_condition=self.config.boundary_condition,
            raymarch=self.config.raymarch,
            device=torch.device("cpu"),
            dtype=self.config.dtype,
        )
        cpu_model = StratusRadianceModel(config_cpu)

        torch.manual_seed(0)
        nx, ny, nz = self.config.grid_shape
        kappa = (
            torch.rand(
                1,
                self.config.n_stokes,
                nx,
                ny,
                nz,
                self.config.n_bands,
                device=self.config.device,
                dtype=self.config.dtype,
            )
            * 0.1
        )
        source = torch.rand_like(kappa) * 0.05

        points = torch.rand(n_points, 3, device=self.config.device, dtype=self.config.dtype)
        points[:, 0] *= nx - 1
        points[:, 1] *= ny - 1
        points[:, 2] *= nz - 1

        with torch.no_grad():
            gpu_out = self.model(kappa, source, points)["radiance"].cpu()
            cpu_out = cpu_model(
                kappa.to(device="cpu"), source.to(device="cpu"), points.to(device="cpu")
            )["radiance"]

        relative_diff = (gpu_out - cpu_out).abs() / (cpu_out.abs() + 1.0e-8)
        max_diff = float(relative_diff.max().item())
        passed = max_diff < tolerance

        metrics = {
            "max_relative_difference": max_diff,
            "tolerance": tolerance,
        }
        return ValidationResult("cpu_gpu_parity", passed, metrics)

    # ---------------------------------------------------------------- orchestrator
    def run_all(self) -> dict[str, ValidationResult]:
        """Execute the full validation suite and return results."""
        results = {
            "uniform_slab": self.uniform_slab_test(),
            "cpu_gpu_parity": self.cpu_gpu_parity_test(),
        }

        for name, result in results.items():
            status = "PASS" if result.passed else "FAIL"
            logger.info("Validation %s: %s metrics=%s", name, status, result.metrics)
        return results


__all__ = ["RadiativeValidator", "ValidationResult"]
