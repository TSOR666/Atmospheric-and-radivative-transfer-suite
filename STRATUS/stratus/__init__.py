"""
STRATUS modular radiative-transfer toolkit.

This package exposes the cleaned and production-ready interfaces for building
STRATUS radiative models after refactoring the legacy monolithic scripts.
"""

from .angular import AngularSample, SphericalHarmonicsSampler, real_spherical_harmonics
from .benchmark import BenchmarkResult, profile_model
from .config import (
    AngularBasis,
    BoundaryCondition,
    RayMarchConfig,
    SpectralMethod,
    StratusConfig,
)
from .data import StratusDataset, create_dataloader
from .exceptions import StratusConfigError, StratusError, StratusPhysicsError
from .export import ExportInputs, StratusExporter
from .model import StratusRadianceModel
from .monitoring import PerformanceMonitor
from .monte_carlo import MonteCarloConfig, MonteCarloRadiativeTransfer
from .multiscale import MultiscaleFeatureEncoder
from .pinn import PhysicsInformedRTE, PINNConfig
from .polarization import MuellerMatrix
from .raymarch import RayMarcher
from .validation import RadiativeValidator, ValidationResult

__all__ = [
    "StratusConfig",
    "RayMarchConfig",
    "SpectralMethod",
    "BoundaryCondition",
    "AngularBasis",
    "StratusRadianceModel",
    "StratusDataset",
    "PerformanceMonitor",
    "RayMarcher",
    "create_dataloader",
    "ExportInputs",
    "StratusExporter",
    "RadiativeValidator",
    "ValidationResult",
    "MonteCarloRadiativeTransfer",
    "MonteCarloConfig",
    "MuellerMatrix",
    "PhysicsInformedRTE",
    "PINNConfig",
    "SphericalHarmonicsSampler",
    "AngularSample",
    "real_spherical_harmonics",
    "MultiscaleFeatureEncoder",
    "profile_model",
    "BenchmarkResult",
    "StratusError",
    "StratusConfigError",
    "StratusPhysicsError",
]
