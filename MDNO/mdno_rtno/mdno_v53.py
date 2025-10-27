"""Compatibility wrapper exposing the refactored MDNO v5.3 interfaces."""

from mdno import (
    AtmosphericConstants,
    CONSTANTS,
    MDNOConfig,
    PhysicsConstraintType,
    TurbulenceModel,
    PerformanceMonitor,
    monitor,
    EnhancedMDNO_v53_Complete,
    MDNOTrainer,
    test_complete_mdno_v53,
)

__all__ = [
    "AtmosphericConstants",
    "CONSTANTS",
    "MDNOConfig",
    "PhysicsConstraintType",
    "TurbulenceModel",
    "PerformanceMonitor",
    "monitor",
    "EnhancedMDNO_v53_Complete",
    "MDNOTrainer",
    "test_complete_mdno_v53",
]

if __name__ == "__main__":
    test_complete_mdno_v53()
