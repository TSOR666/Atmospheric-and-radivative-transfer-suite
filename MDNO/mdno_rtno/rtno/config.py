from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import torch

from ._logging import LOGGER as logger


class BoundaryType(Enum):
    """Supported lower boundary condition types."""

    LAMBERTIAN = "lambertian"
    SPECULAR = "specular"
    MIXED = "mixed"
    OCEAN = "ocean"
    VACUUM = "vacuum"


class ScatteringType(Enum):
    """Supported scattering process types."""

    RAYLEIGH = "rayleigh"
    MIE = "mie"
    RAMAN = "raman"
    MULTIPLE = "multiple"


@dataclass
class RTNOConfig:
    """Complete configuration for Enhanced RTNO v4.3."""

    # Model architecture
    hidden_dim: int = 256
    num_layers: int = 8
    num_heads: int = 16
    dim_head: int = 64
    dropout: float = 0.0
    activation: str = "silu"

    # Physics switches
    use_mie_scattering: bool = True
    use_gas_absorption: bool = True
    use_spherical_harmonics: bool = True
    use_polarization: bool = True
    use_refraction: bool = True
    use_aerosol_cloud: bool = True
    use_raman_scattering: bool = False
    use_non_lte: bool = False

    # Multiple scattering
    use_multiple_scattering: bool = True
    max_scattering_iterations: int = 10
    scattering_convergence_tol: float = 1e-6

    # 3D coupling
    use_horizontal_coupling: bool = False
    coupling_method: str = "short_characteristics"

    # Optical depth safeguards
    use_delta_eddington: bool = True
    use_safe_transmittance: bool = True

    # Correlated-k
    use_correlated_k: bool = False
    k_distribution_path: Optional[str] = None

    # Optical parameters
    wavelengths: torch.Tensor = field(default_factory=lambda: torch.linspace(250, 2500, 64))
    refractive_indices: torch.Tensor = field(
        default_factory=lambda: torch.complex(torch.ones(64) * 1.5, torch.ones(64) * 0.01)
    )

    # Numerical settings
    spherical_harmonics_order: int = 16
    gauss_quadrature_points: int = 16
    discrete_ordinates_streams: int = 16
    max_scattering_orders: int = 3
    max_iterations: int = 20
    legendre_polynomial_order: int = 32

    # Grid settings
    nx: int = 64
    ny: int = 64
    nz: int = 32
    dx: float = 1000.0
    dy: float = 1000.0
    dz: float = 500.0

    # Stokes parameters
    n_stokes: int = 4  # I, Q, U, V
    use_mueller_coupling: bool = True

    # Performance
    use_mixed_precision: bool = True
    gradient_checkpointing: bool = False
    enable_caching: bool = True
    compile_model: bool = False
    batch_size: int = 1
    cache_size: int = 1024

    # Validation
    validate_inputs: bool = True
    validate_physics: bool = True
    numerical_stability_eps: float = 1e-8
    enable_monitoring: bool = True

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32

    def validate(self) -> None:
        """Validate configuration before constructing the model."""
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if len(self.wavelengths) == 0:
            raise ValueError("wavelengths tensor must not be empty")
        if self.nx <= 0 or self.ny <= 0 or self.nz <= 0:
            raise ValueError("grid dimensions must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.n_stokes not in (1, 4):
            raise ValueError("n_stokes must be 1 or 4")

        if self.use_non_lte:
            raise NotImplementedError("Non-LTE mode not yet implemented; set use_non_lte=False")

        if self.use_correlated_k and not self.k_distribution_path:
            logger.warning(
                "Correlated-k enabled but no k-distribution path provided; falling back to simplified bands."
            )

        logger.info("[OK] RTNO configuration validated")

    def get_device(self) -> torch.device:
        """Return the configured torch.device."""
        return torch.device(self.device)


__all__ = ["BoundaryType", "ScatteringType", "RTNOConfig"]
