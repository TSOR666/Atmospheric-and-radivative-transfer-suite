import pytest

torch = pytest.importorskip("torch")

from stratus.config import SpectralMethod, StratusConfig
from stratus.exceptions import StratusConfigError, StratusPhysicsError
from stratus.model import StratusRadianceModel
from stratus.monte_carlo import MonteCarloConfig, MonteCarloRadiativeTransfer
from stratus.pinn import PhysicsInformedRTE, PINNConfig


def test_monte_carlo_config_validation() -> None:
    with pytest.raises(StratusConfigError):
        MonteCarloConfig(n_rays=0)
    with pytest.raises(StratusConfigError):
        MonteCarloConfig(albedo=1.5)


def test_monte_carlo_negative_sigma_rejected() -> None:
    config = MonteCarloConfig(n_rays=4, max_scatter=2)
    solver = MonteCarloRadiativeTransfer(1, (2, 2, 2), config)
    sigma_t = torch.full((2, 2, 2), -1.0)
    with pytest.raises(StratusPhysicsError):
        solver.solve(sigma_t)


def test_monte_carlo_zero_albedo_extinguishes_rays() -> None:
    config = MonteCarloConfig(n_rays=16, max_scatter=5, albedo=0.0, min_weight=1.0e-6)
    solver = MonteCarloRadiativeTransfer(1, (4, 4, 4), config)
    sigma_t = torch.ones(4, 4, 4)

    detector = solver.solve(sigma_t)

    assert torch.isfinite(detector).all()
    assert detector.sum() <= config.n_rays + 1e-4


def test_pinn_solver_runs_with_minimal_configuration() -> None:
    config = PINNConfig(hidden_dim=8, hidden_layers=1, n_epochs=1, n_samples=8, learning_rate=1e-3)
    solver = PhysicsInformedRTE((2, 2, 2), 1, config, torch.device("cpu"))

    kappa = torch.ones(1, 2, 2, 2, 1)
    source = torch.zeros(1, 2, 2, 2, 1)

    result = solver.solve(kappa, source)
    assert result.shape == (1, 1, 2, 2, 2, 1)
    assert torch.isfinite(result).all()


def test_pinn_solver_supports_multiple_stokes() -> None:
    config = PINNConfig(hidden_dim=8, hidden_layers=1, n_epochs=1, n_samples=8, learning_rate=1e-3)
    solver = PhysicsInformedRTE((2, 2, 2), 3, config, torch.device("cpu"))

    kappa = torch.ones(3, 2, 2, 2, 1)
    source = torch.zeros(3, 2, 2, 2, 1)

    result = solver.solve(kappa, source)
    assert result.shape == (1, 3, 2, 2, 2, 1)
    assert torch.isfinite(result).all()


def test_energy_constraint_zero_for_balanced_field() -> None:
    config = StratusConfig(
        grid_shape=(2, 2, 2),
        n_bands=1,
        n_stokes=1,
        spectral_method=SpectralMethod.CORRELATED_K,
        energy_conservation=True,
        reciprocity_constraint=False,
    )
    model = StratusRadianceModel(config)
    field = torch.ones(1, 1, 2, 2, 2, 1)
    losses = model.compute_physics_constraints(field, field, field)
    assert torch.isclose(losses["energy_conservation"], torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(losses["total_physics_loss"], torch.tensor(0.0), atol=1e-6)


def test_monte_carlo_terminates_when_rays_exit_domain() -> None:
    config = MonteCarloConfig(n_rays=32, max_scatter=50, albedo=1.0, seed=123)
    solver = MonteCarloRadiativeTransfer(1, (2, 2, 2), config)

    sigma_t = torch.ones(2, 2, 2)
    start_positions = torch.zeros(config.n_rays, 3)
    directions = torch.zeros(config.n_rays, 3)
    directions[:, 2] = 1.0

    detector = solver.solve(
        sigma_t,
        source_positions=start_positions,
        source_directions=directions,
    )

    assert solver._last_steps < config.max_scatter
    assert torch.isfinite(detector).all()
    assert detector.sum() <= config.n_rays + 1e-4
