"""
Performance monitoring utilities for STRATUS training and inference loops.

This module provides lightweight instrumentation inspired by the legacy
`PerformanceMonitor` class but with safer lifecycle management and improved
logging controls.
"""

from __future__ import annotations

import logging
import statistics
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager

import torch

logger = logging.getLogger("stratus.monitoring")

# Optional dependencies -------------------------------------------------------
try:  # pragma: no cover - optional dependency
    import psutil  # type: ignore
except (ImportError, ModuleNotFoundError):  # pragma: no cover - optional dependency
    psutil = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import GPUtil  # type: ignore
except (ImportError, ModuleNotFoundError):  # pragma: no cover - optional dependency
    GPUtil = None  # type: ignore


class PerformanceMonitor:
    """
    Aggregate timing, CPU, and GPU statistics across iterative workloads.

    Parameters
    ----------
    log_interval:
        Emit a log entry every `log_interval` steps.  Set to zero to disable.
    window:
        Number of samples used when computing moving averages.
    enable_gpu_stats:
        Gather GPU utilisation and memory metrics when CUDA is available.
    """

    def __init__(
        self,
        *,
        log_interval: int = 10,
        window: int = 100,
        enable_gpu_stats: bool = True,
    ) -> None:
        self.log_interval = max(0, log_interval)
        self.window = max(1, window)
        self.enable_gpu_stats = enable_gpu_stats and torch.cuda.is_available()

        self.step_times: deque[float] = deque(maxlen=self.window)
        self.cpu_times: deque[float] = deque(maxlen=self.window)
        self.gpu_memory: deque[float] = deque(maxlen=self.window)
        self._step = 0

        self._process = psutil.Process() if psutil is not None else None

    # ------------------------------------------------------------------ helpers
    def __enter__(self) -> PerformanceMonitor:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @contextmanager
    def monitor_step(self) -> Iterator[None]:
        """
        Context-manager that measures wall-clock duration of one iteration.

        Example
        -------
        >>> monitor = PerformanceMonitor(log_interval=20)
        >>> for step in range(num_steps):
        ...     with monitor.monitor_step():
        ...         train_step()
        ...     monitor.maybe_log(step)
        """

        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.step_times.append(elapsed)

            if self._process is not None:
                cpu_percent = self._process.cpu_percent(interval=None)
                self.cpu_times.append(cpu_percent)

            if self.enable_gpu_stats:
                current_memory = torch.cuda.memory_allocated() / 1e6
                self.gpu_memory.append(current_memory)

            self._step += 1

    def maybe_log(self, step: int | None = None) -> None:
        """
        Emit a log line if the log interval condition is satisfied.
        """

        if self.log_interval == 0:
            return

        current_step = self._step if step is None else step
        if current_step == 0 or current_step % self.log_interval != 0:
            return

        metrics = self.current_metrics()
        logger.info("Step %s metrics: %s", current_step, metrics)

    # ---------------------------------------------------------------- metrics
    def current_metrics(self) -> dict[str, float]:
        """
        Return a dictionary with the latest moving-average statistics.
        """

        metrics: dict[str, float] = {}

        if self.step_times:
            metrics["step_time_ms"] = 1e3 * statistics.mean(self.step_times)
            metrics["throughput_steps_s"] = 1.0 / max(statistics.mean(self.step_times), 1e-9)

        if self.cpu_times:
            metrics["cpu_percent"] = statistics.mean(self.cpu_times)

        if self.gpu_memory:
            metrics["gpu_memory_mb"] = statistics.mean(self.gpu_memory)
            if GPUtil is not None:  # pragma: no cover - optional dependency
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        metrics["gpu_util_percent"] = statistics.mean(
                            gpu.load * 100.0 for gpu in gpus
                        )
                except Exception as exc:
                    logger.debug("GPUtil query failed: %s", exc)

        return metrics

    def reset(self) -> None:
        """Clear accumulated statistics."""
        self.step_times.clear()
        self.cpu_times.clear()
        self.gpu_memory.clear()
        self._step = 0

    def close(self) -> None:
        """Release any cached state explicitly."""
        self.reset()


__all__ = ["PerformanceMonitor"]
