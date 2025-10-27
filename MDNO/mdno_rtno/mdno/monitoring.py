import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import numpy as np
import torch

from ._logging import LOGGER as logger


class PerformanceMonitor:
    """Comprehensive performance monitoring helper."""

    def __init__(self, name: str = "MDNO"):
        self.name = name
        self.metrics: DefaultDict[str, List[float]] = defaultdict(list)
        self.timings: DefaultDict[str, List[float]] = defaultdict(list)
        self.memory_usage: List[Tuple[str, float]] = []

    @contextmanager
    def timer(self, operation: str):
        """Record timing and GPU memory usage for an operation."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            duration = time.perf_counter() - start
            end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

            self.timings[operation].append(duration)
            self.memory_usage.append((operation, (end_memory - start_memory) / 1e9))

    def log_metrics(self, iteration: Optional[int] = None):
        """Log rolling performance metrics."""
        for name, times in self.timings.items():
            if times:
                avg_time = np.mean(times[-100:])
                logger.info("%s/%s: %.4fs", self.name, name, avg_time)

        if self.memory_usage:
            total_memory = sum(m[1] for m in self.memory_usage[-100:])
            logger.info("%s/memory: %.2f GB", self.name, total_memory)

    def get_summary(self) -> Dict[str, Any]:
        """Return a summary dictionary with mean/std for each metric."""
        summary: Dict[str, Any] = {}
        for name, times in self.timings.items():
            if times:
                summary[f"{name}_mean"] = np.mean(times)
                summary[f"{name}_std"] = np.std(times)
        return summary


monitor = PerformanceMonitor("MDNO_v5_3")
