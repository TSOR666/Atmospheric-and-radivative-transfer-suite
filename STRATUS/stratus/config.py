"""
Configuration objects and enumerations for the STRATUS framework.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import torch

from .exceptions import StratusConfigError
from .monte_carlo import MonteCarloConfig
from .pinn import PINNConfig

logger = logging.getLogger("stratus.config")

MIN_MIE_SAMPLES = 16


CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    DEVICE_CAPABILITY = torch.cuda.get_device_capability()
    TENSOR_CORES_AVAILABLE = DEVICE_CAPABILITY[0] >= 7
else:
    DEVICE_CAPABILITY = (0, 0)
    TENSOR_CORES_AVAILABLE = False


class SpectralMethod(Enum):
    FULL_4D = "full_4d"
    BAND_AVERAGED = "band_averaged"
    CORRELATED_K = "correlated_k"
    SPHERICAL_HARMONICS = "spherical_harmonics"
    HYBRID_MC_NN = "hybrid_mc_nn"


class BoundaryCondition(Enum):
    VACUUM = "vacuum"
    REFLECTIVE = "reflective"
    PERIODIC = "periodic"
    BLACKBODY = "blackbody"
    ISOTROPIC = "isotropic"


class AngularBasis(Enum):
    QUADRATURE = "quadrature"
    SPHERICAL_HARMONICS = "spherical_harmonics"


def _default_device() -> torch.device:
    return torch.device("cuda" if CUDA_AVAILABLE else "cpu")


@dataclass
class RayMarchConfig:
    step_size: float = 0.5
    max_optical_depth_per_step: float = 0.1
    min_transmittance: float = 1.0e-2
    ray_marching_steps: int = 256
    per_ray_adaptive: bool = True

    def validate(self) -> None:
        if self.step_size <= 0.0:
            raise StratusConfigError("step_size must be positive.")
        if self.max_optical_depth_per_step <= 0.0:
            raise StratusConfigError("max_optical_depth_per_step must be positive.")
        if not (0.0 < self.min_transmittance <= 1.0):
            raise StratusConfigError("min_transmittance must be in (0, 1].")
        if self.ray_marching_steps <= 0:
            raise StratusConfigError("ray_marching_steps must be positive.")


@dataclass
class StratusConfig:
    """
    High-level configuration for STRATUS.
    """

    # Spatial
    grid_shape: tuple[int, int, int] = (128, 128, 64)
    spatial_method: str = "fno"
    voxel_size: tuple[float, float, float] = (1.0, 1.0, 1.0)

    # Spectral
    spectral_method: SpectralMethod = SpectralMethod.CORRELATED_K
    wavelengths: torch.Tensor | None = None
    n_bands: int = 20
    n_stokes: int = 1

    # Ray marching
    raymarch: RayMarchConfig = field(default_factory=RayMarchConfig)
    n_angular_samples: int = 26
    use_gpu_ray_marching: bool = CUDA_AVAILABLE
    angular_basis: AngularBasis = AngularBasis.QUADRATURE
    sh_max_degree: int = 2

    # Neural operator
    fno_modes: tuple[int, int, int] = (16, 16, 8)
    fno_width: int = 64
    fno_layers: int = 4
    hidden_dim: int = 128
    use_multiscale: bool = False
    multiscale_levels: int = 2
    multiscale_wavelet: str = "haar"

    scattering_model: str = "rayleigh"  # "rayleigh", "mie", "henyey_greenstein"

    # Monte Carlo / stochastic solver
    monte_carlo: MonteCarloConfig = field(default_factory=MonteCarloConfig)

    # Physics / loss
    use_path_integration: bool = True
    adaptive_sampling: bool = True
    physics_informed: bool = True
    physics_loss_weight: float = 0.1
    energy_conservation: bool = True
    reciprocity_constraint: bool = True
    pinn: PINNConfig = field(default_factory=PINNConfig)

    # Boundary
    boundary_condition: BoundaryCondition = BoundaryCondition.VACUUM
    boundary_temperature: float = 300.0
    mie_max_order: int = 8
    mie_refractive_index: float = 1.33
    mie_samples: int = 128
    henyey_g: float = 0.0

    # Tensor cores / precision
    use_tensor_cores: bool = TENSOR_CORES_AVAILABLE
    tensor_core_precision: str = "fp16"  # "fp16", "bf16", "tf32"
    channels_last_3d: bool = True
    mixed_precision: bool = True

    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    checkpoint_interval: int = 100
    keep_last_n_checkpoints: int = 3

    # Device
    device: torch.device = field(default_factory=_default_device)
    dtype: torch.dtype = torch.float32
    compute_dtype: torch.dtype = torch.float16 if TENSOR_CORES_AVAILABLE else torch.float32

    def __post_init__(self) -> None:
        if isinstance(self.device, str):
            self.device = torch.device(self.device)

        if self.n_bands < 1 or self.n_stokes < 1:
            raise StratusConfigError("n_bands and n_stokes must be >= 1.")

        if len(self.grid_shape) != 3:
            raise StratusConfigError("grid_shape must have length 3.")
        if any(dim <= 1 for dim in self.grid_shape):
            raise StratusConfigError("grid_shape entries must be > 1.")
        if len(self.voxel_size) != 3:
            raise StratusConfigError("voxel_size must have length 3.")
        if any(v <= 0.0 for v in self.voxel_size):
            raise StratusConfigError("voxel_size entries must be positive.")

        self.raymarch.validate()

        if not isinstance(self.monte_carlo, MonteCarloConfig):
            raise TypeError("monte_carlo must be a MonteCarloConfig instance.")
        if not isinstance(self.pinn, PINNConfig):
            raise TypeError("pinn must be a PINNConfig instance.")

        if not isinstance(self.angular_basis, AngularBasis):
            raise StratusConfigError("angular_basis must be an AngularBasis enum value.")
        if self.angular_basis is AngularBasis.SPHERICAL_HARMONICS:
            if self.sh_max_degree < 0 or self.sh_max_degree > 4:
                raise StratusConfigError("sh_max_degree must be between 0 and 4 for stability.")
            if self.n_angular_samples < (self.sh_max_degree + 1) ** 2:
                self.n_angular_samples = (self.sh_max_degree + 1) ** 2
                logger.info(
                    "Adjusted n_angular_samples to %s to satisfy spherical harmonics coverage.",
                    self.n_angular_samples,
                )

        if self.use_multiscale and self.multiscale_levels < 1:
            raise StratusConfigError("multiscale_levels must be >= 1 when use_multiscale is True.")

        # Validate that multiscale levels don't exceed grid dimensions
        if self.use_multiscale:
            min_grid_dim = min(self.grid_shape)
            max_levels = int(torch.log2(torch.tensor(min_grid_dim, dtype=torch.float32)).item())
            if self.multiscale_levels > max_levels:
                logger.warning(
                    "multiscale_levels (%d) exceeds maximum for grid shape (%d). Clamping to %d.",
                    self.multiscale_levels,
                    max_levels,
                    max_levels,
                )
                self.multiscale_levels = max_levels

        if self.scattering_model not in {"rayleigh", "mie", "henyey_greenstein"}:
            raise StratusConfigError("Invalid scattering_model specified.")

        if self.mie_max_order < 1:
            raise StratusConfigError("mie_max_order must be >= 1.")
        if self.mie_samples < MIN_MIE_SAMPLES:
            raise StratusConfigError(
                f"mie_samples must be >= {MIN_MIE_SAMPLES} to maintain quadrature stability."
            )
        if abs(self.henyey_g) >= 0.999:
            raise StratusConfigError("henyey_g must be in (-1, 1).")

        if self.use_tensor_cores and not TENSOR_CORES_AVAILABLE:
            logger.warning("Tensor cores requested but not available; disabling.")
            self.use_tensor_cores = False
            self.compute_dtype = self.dtype

        if self.use_tensor_cores:
            if self.tensor_core_precision == "bf16" and hasattr(torch, "bfloat16"):
                self.compute_dtype = torch.bfloat16
            elif self.tensor_core_precision == "fp16":
                self.compute_dtype = torch.float16
            else:
                self.compute_dtype = torch.float32
        else:
            self.compute_dtype = self.dtype

        if self.mixed_precision is False:
            self.compute_dtype = self.dtype

        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        if self.use_gpu_ray_marching and not CUDA_AVAILABLE:
            logger.warning("GPU ray marching requested but CUDA not available; using CPU.")
            self.use_gpu_ray_marching = False

    @property
    def channels(self) -> int:
        return self.n_stokes * self.n_bands

    def get_device(self) -> torch.device:
        return self.device

    def grid_tensor(self) -> torch.Tensor:
        return torch.tensor(self.grid_shape, dtype=self.dtype, device=self.device)


__all__ = [
    "BoundaryCondition",
    "SpectralMethod",
    "RayMarchConfig",
    "StratusConfig",
]
