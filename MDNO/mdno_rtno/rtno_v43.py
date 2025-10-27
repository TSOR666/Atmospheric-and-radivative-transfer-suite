"""Compatibility wrapper exposing the refactored RTNO v4.3 interfaces."""

from rtno import (
    BoundaryType,
    CONSTANTS,
    Complete3DRadiativeTransferSolver,
    EnhancedRTNO_v43,
    PerformanceMonitor,
    RTNOConfig,
    ScatteringType,
    monitor,
    test_complete_rtno_v43,
)

__all__ = [
    "BoundaryType",
    "CONSTANTS",
    "Complete3DRadiativeTransferSolver",
    "EnhancedRTNO_v43",
    "PerformanceMonitor",
    "RTNOConfig",
    "ScatteringType",
    "monitor",
    "test_complete_rtno_v43",
]


if __name__ == "__main__":
    test_complete_rtno_v43()
