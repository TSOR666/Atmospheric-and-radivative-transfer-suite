import math

import pytest

torch = pytest.importorskip("torch")

from stratus.config import RayMarchConfig, SpectralMethod, StratusConfig
from stratus.raymarch import RayMarcher


def test_ray_marcher_adaptive_matches_exponential() -> None:
    grid_shape = (4, 4, 4)
    config = StratusConfig(
        grid_shape=grid_shape,
        n_bands=1,
        n_stokes=1,
        spectral_method=SpectralMethod.CORRELATED_K,
        raymarch=RayMarchConfig(
            ray_marching_steps=64,
            step_size=0.5,
            per_ray_adaptive=True,
            max_optical_depth_per_step=0.2,
            min_transmittance=1.0e-6,
        ),
    )

    marcher = RayMarcher(config)
    kappa_value = 0.2
    kappa_field = torch.full((*grid_shape, 1, 1), kappa_value)
    source_field = torch.zeros_like(kappa_field)

    origin = torch.tensor([[0.5, 0.5, 0.0]], dtype=torch.float32)
    direction = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)

    _, trans = marcher.march(origin, direction, kappa_field, source_field)

    expected = math.exp(-kappa_value * (grid_shape[2] - origin[0, 2]))
    assert torch.isfinite(trans).all()
    assert torch.allclose(trans.squeeze(), torch.tensor(expected), atol=5e-3, rtol=1e-3)
