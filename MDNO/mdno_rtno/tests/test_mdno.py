import pytest

torch = pytest.importorskip("torch", reason="PyTorch is required for MDNO tests")

from mdno.config import MDNOConfig
from mdno.model import EnhancedMDNO_v53_Complete


@pytest.fixture(scope="module")
def mdno_model():
    config = MDNOConfig(
        grid_shapes={
            "micro": (6, 6, 4),
            "meso": (10, 10, 6),
            "macro": (14, 14, 8),
        },
        velocity_space_resolution=3,
        enforce_moment_conservation=True,
        use_entropic_projection=False,
        use_antialiasing=True,
        use_derivative_advection=True,
        adaptive_timestep=True,
    )
    model = EnhancedMDNO_v53_Complete(config)
    model.eval()
    return model, config


def test_micro_mass_conservation(mdno_model):
    model, config = mdno_model
    torch.manual_seed(0)
    nv = config.velocity_space_resolution
    micro_input = {
        "micro": torch.randn(1, nv, nv, nv, *config.grid_shapes["micro"]).abs()
    }
    dv = model.boltzmann_solver.dv.item()
    mass_before = torch.sum(micro_input["micro"]) * (dv ** 3)

    with torch.no_grad():
        outputs = model(micro_input)

    mass_after = torch.sum(outputs["micro"]) * (dv ** 3)
    rel_error = torch.abs(mass_after - mass_before) / torch.clamp_min(mass_before, 1e-8)
    assert rel_error.item() < 1e-3, f"Mass conservation degraded: {rel_error.item():.3e}"


def test_macro_energy_conservation(mdno_model):
    model, config = mdno_model
    torch.manual_seed(1)
    macro_input = {"macro": torch.randn(1, 4, *config.grid_shapes["macro"]) }
    energy_before = model.hamiltonian.compute_energy(macro_input["macro"])

    with torch.no_grad():
        outputs = model({"macro": macro_input["macro"]})

    energy_after = model.hamiltonian.compute_energy(outputs["macro"])
    rel_error = torch.abs(energy_after - energy_before) / torch.clamp_min(energy_before, 1e-8)
    assert rel_error.item() < 1e-3, f"Energy drift detected: {rel_error.item():.3e}"
