from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch

from ._logging import LOGGER as logger


class TurbulenceModel(Enum):
    """Turbulence modeling approaches."""

    DNS = "direct_numerical_simulation"
    LES = "large_eddy_simulation"
    RANS = "reynolds_averaged"
    HYBRID = "hybrid_rans_les"


class PhysicsConstraintType(Enum):
    """Types of physics constraints."""

    NAVIER_STOKES = "navier_stokes"
    BOLTZMANN = "boltzmann"
    HAMILTONIAN = "hamiltonian"
    CONSERVATION_LAWS = "conservation_laws"
    THERMODYNAMICS = "thermodynamics"


@dataclass
class MDNOConfig:
    """Complete configuration for the MDNO v5.3 model."""

    # Multi-scale grid configuration
    grid_shapes: Dict[str, Tuple[int, ...]] = field(
        default_factory=lambda: {
            "micro": (16, 16, 8),
            "meso": (32, 32, 16),
            "macro": (64, 64, 32),
        }
    )

    # Spatial resolution
    dx: float = 1000.0
    dy: float = 1000.0
    dz: float = 500.0
    dt: float = 60.0

    # Velocity space (Boltzmann)
    velocity_space_resolution: int = 8
    max_velocity: float = 3.0

    # CFL monitoring
    cfl_number: float = 0.5
    adaptive_timestep: bool = True

    # Spectral anti-aliasing
    use_antialiasing: bool = True
    dealiasing_fraction: float = 2.0 / 3.0
    use_derivative_advection: bool = True

    # Conservation enforcement
    enforce_moment_conservation: bool = True
    use_entropic_projection: bool = True
    conservation_tolerance: float = 1e-6

    # Physics constraints
    physics_constraints: List[PhysicsConstraintType] = field(
        default_factory=lambda: [
            PhysicsConstraintType.NAVIER_STOKES,
            PhysicsConstraintType.BOLTZMANN,
            PhysicsConstraintType.CONSERVATION_LAWS,
            PhysicsConstraintType.THERMODYNAMICS,
        ]
    )

    # Model options
    turbulence_model: TurbulenceModel = TurbulenceModel.LES
    use_cloud_microphysics: bool = True
    use_chemistry: bool = True
    use_boundary_layer: bool = True
    use_hamiltonian: bool = True
    use_radiative_transfer: bool = False
    rtno_config: Optional[Dict[str, Any]] = None

    # Neural network
    hidden_dim: int = 256
    operator_width: int = 128
    operator_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.0
    activation: str = "silu"

    # Spectral configuration
    spectral_truncation: int = 85

    # Numerical settings
    max_iterations: int = 100
    convergence_tolerance: float = 1e-6
    numerical_stability_eps: float = 1e-8

    # Performance
    use_mixed_precision: bool = True
    gradient_checkpointing: bool = False
    compile_model: bool = False
    distributed: bool = False

    # Validation
    validate_inputs: bool = True
    validate_physics: bool = True
    validate_conservation: bool = True

    # Loss weights
    constraint_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "navier_stokes": 1.0,
            "boltzmann": 1.0,
            "conservation": 0.5,
            "hamiltonian": 0.8,
            "thermodynamics": 0.3,
        }
    )

    # Training
    batch_size: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    warmup_steps: int = 1000

    # Monitoring
    enable_monitoring: bool = True
    log_frequency: int = 10
    checkpoint_frequency: int = 100

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32

    def validate(self) -> None:
        """Validate configuration assumptions."""
        for scale, shape in self.grid_shapes.items():
            if not all(s > 0 for s in shape):
                raise ValueError(f"Invalid grid shape for {scale}")
        if self.velocity_space_resolution <= 0:
            raise ValueError("Velocity space resolution must be positive")
        if self.dt <= 0:
            raise ValueError("Timestep must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("Dropout must be in [0, 1)")
        if not 0 < self.cfl_number < 1:
            raise ValueError("CFL number must be in (0, 1)")
        if not 0 < self.dealiasing_fraction <= 1:
            raise ValueError("Dealiasing fraction must be in (0, 1]")
        if self.use_radiative_transfer and self.rtno_config is not None and not isinstance(self.rtno_config, dict):
            raise ValueError("rtno_config must be a dict of keyword arguments when provided")
        logger.info("[OK] Configuration validated")

    def get_device(self) -> torch.device:
        """Return the configured device."""
        return torch.device(self.device)
