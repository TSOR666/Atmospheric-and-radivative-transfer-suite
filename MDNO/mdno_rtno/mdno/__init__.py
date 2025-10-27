"""Package entry point for the refactored MDNO framework.

Version: 5.3.1 (Optimized)
- Boltzmann solver vectorized (10-100x speedup)
- Physics validation suite added
- Production ready with comprehensive audit
"""

from .constants import AtmosphericConstants, CONSTANTS
from .config import MDNOConfig, PhysicsConstraintType, TurbulenceModel
from .demo import test_complete_mdno_v53
from .model import EnhancedMDNO_v53_Complete
from .monitoring import PerformanceMonitor, monitor
from .trainer import MDNOTrainer
from .validation import PhysicsValidator, run_comprehensive_validation

__version__ = "5.3.1"

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
    "PhysicsValidator",
    "run_comprehensive_validation",
]
