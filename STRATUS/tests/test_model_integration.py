import pytest

torch = pytest.importorskip("torch")

from stratus.config import RayMarchConfig, SpectralMethod, StratusConfig
from stratus.model import StratusRadianceModel
from stratus.monte_carlo import MonteCarloConfig
from stratus.pinn import PINNConfig


@pytest.fixture()
def base_grid() -> tuple[int, int, int]:
    return (2, 2, 2)


def test_model_monte_carlo_forward(base_grid: tuple[int, int, int]) -> None:
    config = StratusConfig(
        grid_shape=base_grid,
        n_bands=1,
        n_stokes=1,
        spectral_method=SpectralMethod.CORRELATED_K,
        spatial_method="monte_carlo",
        monte_carlo=MonteCarloConfig(n_rays=16, max_scatter=3, seed=0),
        raymarch=RayMarchConfig(ray_marching_steps=8, step_size=0.25),
    )

    model = StratusRadianceModel(config)
    kappa = torch.ones(1, config.n_stokes, *base_grid, config.n_bands)
    source = torch.zeros_like(kappa)

    result = model(kappa, source)

    assert result["radiance"].shape == (1, 1, *base_grid, 1)
    assert torch.isfinite(result["radiance"]).all()
    assert result["transmittance"].shape == (1, 1, *base_grid, 1)
    assert torch.isfinite(result["transmittance"]).all()


def test_model_physics_informed_forward(base_grid: tuple[int, int, int]) -> None:
    pinn_config = PINNConfig(
        hidden_dim=8,
        hidden_layers=1,
        learning_rate=1e-3,
        n_epochs=1,
        n_samples=16,
        time_final=0.5,
    )
    config = StratusConfig(
        grid_shape=base_grid,
        n_bands=1,
        n_stokes=1,
        spectral_method=SpectralMethod.CORRELATED_K,
        spatial_method="physics_informed",
        pinn=pinn_config,
        raymarch=RayMarchConfig(ray_marching_steps=8, step_size=0.25),
    )

    model = StratusRadianceModel(config)
    kappa = torch.ones(1, config.n_stokes, *base_grid, config.n_bands)
    source = torch.zeros_like(kappa)

    result = model(kappa, source)

    assert result["radiance"].shape == (1, 1, *base_grid, 1)
    assert torch.isfinite(result["radiance"]).all()
    assert result["transmittance"].shape == (1, 1, *base_grid, 1)
    assert torch.isfinite(result["transmittance"]).all()
