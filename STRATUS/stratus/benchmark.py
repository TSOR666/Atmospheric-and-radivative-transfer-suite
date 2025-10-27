"""
Utility functions to benchmark STRATUS models for throughput and latency.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from .config import StratusConfig
from .model import StratusRadianceModel


@dataclass
class BenchmarkResult:
    latency_ms: float
    throughput_samples_s: float
    memory_mb: float


def profile_model(
    model: StratusRadianceModel,
    config: StratusConfig,
    *,
    batch_size: int = 1,
    evaluation_points: int | None = None,
    warmup: int = 5,
    iters: int = 20,
) -> BenchmarkResult:
    device = config.device
    dtype = config.dtype
    model = model.to(device)
    model.eval()

    nx, ny, nz = config.grid_shape
    shape = (batch_size, config.n_stokes, nx, ny, nz, config.n_bands)
    kappa = torch.rand(shape, device=device, dtype=dtype)
    source = torch.rand(shape, device=device, dtype=dtype)

    eval_points_tensor = None
    if evaluation_points is not None and evaluation_points > 0:
        points = torch.rand(batch_size, evaluation_points, 3, device=device, dtype=dtype)
        points[..., 0] *= nx - 1
        points[..., 1] *= ny - 1
        points[..., 2] *= nz - 1
        eval_points_tensor = points

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(warmup):
            model(kappa, source, eval_points_tensor)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(iters):
            model(kappa, source, eval_points_tensor)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / iters

    latency_ms = elapsed * 1000.0
    throughput = batch_size / elapsed
    if torch.cuda.is_available():
        memory = torch.cuda.max_memory_allocated(device) / 1e6
        torch.cuda.reset_peak_memory_stats(device)
    else:
        memory = 0.0

    return BenchmarkResult(latency_ms=latency_ms, throughput_samples_s=throughput, memory_mb=memory)


__all__ = ["profile_model", "BenchmarkResult"]
