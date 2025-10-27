import pytest

torch = pytest.importorskip("torch", reason="PyTorch is required for RTNO tests")

from rtno.config import RTNOConfig
from rtno.model import EnhancedRTNO_v43
from rtno.optics import OpticalDepthScaling
from rtno.polarization import MuellerMatrixPolarization


def test_radiance_non_negative_and_transmittance():
    config = RTNOConfig(
        nx=8,
        ny=8,
        nz=4,
        wavelengths=torch.linspace(400.0, 700.0, 4),
        use_multiple_scattering=True,
        use_horizontal_coupling=False,
        use_delta_eddington=True,
        use_mie_scattering=True,
        use_gas_absorption=True,
        use_polarization=True,
        use_refraction=True,
    )
    model = EnhancedRTNO_v43(config)
    model.eval()

    state = {
        "temperature": torch.full((1, config.nz, config.ny, config.nx), 288.15),
        "pressure": torch.full((1, config.nz, config.ny, config.nx), 101325.0),
        "humidity": torch.full((1, config.nz, config.ny, config.nx), 0.5),
    }

    with torch.no_grad():
        outputs = model(state, return_diagnostics=True)

    radiance = outputs["radiance_corrected"]
    assert torch.all(radiance >= -1e-6), "Radiance should remain non-negative"

    tau_small = torch.tensor([1e-5, 1e-4, 1e-3])
    safe_small = OpticalDepthScaling.safe_transmittance(tau_small)
    taylor = 1 - tau_small
    assert torch.allclose(safe_small, taylor, atol=1e-6, rtol=1e-4)

    tau_large = torch.tensor([25.0, 50.0, 100.0])
    safe_large = OpticalDepthScaling.safe_transmittance(tau_large)
    assert torch.all(safe_large < 1e-6)

    tau = torch.tensor([30.0])
    omega_val = torch.tensor([0.99])
    g = torch.tensor([0.85])
    tau_star, omega_star, g_star = OpticalDepthScaling.delta_eddington_scaling(tau, omega_val, g)
    assert torch.isfinite(tau_star).all()
    assert torch.isfinite(omega_star).all()
    assert torch.isfinite(g_star).all()


def test_vectorized_mueller_matches_reference():
    cos_theta = torch.tensor([0.0, 0.5], dtype=torch.float32)
    mueller = MuellerMatrixPolarization.rayleigh_mueller_matrix(cos_theta)
    torch.manual_seed(42)
    stokes_in = torch.randn(2, 4, 3, 1, 1, 1)
    omega = torch.full((2, 1, 1, 1), 0.9)

    # Reference implementation using explicit loops (mirrors pre-vectorized code)
    stokes_flat = stokes_in.reshape(2, 4, 3, -1)
    expected = torch.zeros_like(stokes_flat)
    for batch_idx in range(2):
        for angle_idx in range(3):
            S = stokes_flat[batch_idx, :, angle_idx]
            scattered = torch.matmul(mueller[angle_idx], S)
            expected[batch_idx, :, angle_idx] = scattered
    expected = expected.reshape_as(stokes_in) * omega.reshape(2, 1, 1, 1)

    actual = MuellerMatrixPolarization.apply_mueller_scattering(stokes_in, mueller, omega)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-4)
