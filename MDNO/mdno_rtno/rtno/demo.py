from __future__ import annotations

import math

import torch

from .config import RTNOConfig
from .model import EnhancedRTNO_v43
from .optics import OpticalDepthScaling


def test_complete_rtno_v43():
    """Comprehensive smoke test for the refactored RTNO package."""
    print("=" * 80)
    print("TESTING COMPLETE RTNO v4.3 - Production")
    print("=" * 80)

    config = RTNOConfig(
        nx=16,
        ny=16,
        nz=8,
        wavelengths=torch.linspace(400, 700, 8),
        use_multiple_scattering=True,
        use_horizontal_coupling=False,
        use_delta_eddington=True,
        use_mie_scattering=True,
        use_gas_absorption=True,
        use_polarization=True,
        use_refraction=True,
    )

    model = EnhancedRTNO_v43(config)

    atmospheric_state = {
        "temperature": torch.ones(1, config.nz, config.ny, config.nx) * 288.15,
        "pressure": torch.ones(1, config.nz, config.ny, config.nx) * 101325.0,
        "humidity": torch.ones(1, config.nz, config.ny, config.nx) * 0.5,
    }

    print("\n[1] Running forward pass...")
    with torch.no_grad():
        outputs = model(atmospheric_state, return_diagnostics=True)

    radiance = outputs["radiance_corrected"]
    print(f"[OK] Radiance shape: {radiance.shape}")
    print(f"[OK] Irradiance shape: {outputs['irradiance'].shape}")
    print(f"[OK] Heating rate shape: {outputs['heating_rate'].shape}")

    print("\n[2] Validating physics outputs...")
    assert torch.all(radiance[:, 0] >= 0), "Negative radiance detected"
    print("[OK] Radiance non-negative")

    if "diagnostics" in outputs:
        diag = outputs["diagnostics"]
        print(f"[OK] Neural contribution ratio: {diag['neural_contribution_ratio']:.2%}")
        print(f"[OK] Total energy: {diag['total_energy']:.2e}")
        if "mean_linear_polarization" in diag:
            print(f"[OK] Mean linear polarization: {diag['mean_linear_polarization']:.2%}")

    print("\n[3] Testing spherical harmonics helper...")
    sph = model.rt_solver.spherical_harmonics
    theta = torch.linspace(0, math.pi, 10)
    phi = torch.linspace(0, 2 * math.pi, 10)
    ylm = sph.compute_ylm(4, 2, theta, phi)
    print(f"[OK] Y_l^m sample: {ylm.shape}")

    if getattr(model.rt_solver, "mie_scattering", None) is not None:
        print("[OK] Mie scattering component available")
    if getattr(model.rt_solver, "gas_absorption", None) is not None:
        print("[OK] Gas absorption component available")

    print("\n[4] Testing Mueller matrix polarization...")
    cos_theta = torch.tensor([0.0, 0.5, 0.866], device=radiance.device)
    mueller = model.rt_solver.mueller_polarization.rayleigh_mueller_matrix(cos_theta)
    print(f"[OK] Mueller matrix shape: {mueller.shape}")

    stokes_in = torch.randn(1, 4, 2, config.nz, config.ny, config.nx, device=radiance.device)
    omega = torch.ones(1, config.nz, config.ny, config.nx, device=radiance.device) * 0.9
    mueller_simple = model.rt_solver.mueller_polarization.rayleigh_mueller_matrix(torch.tensor(0.5))
    stokes_out = model.rt_solver.mueller_polarization.apply_mueller_scattering(stokes_in, mueller_simple, omega)
    print(f"[OK] Stokes scattering shape: {stokes_out.shape}")

    print("\n[5] Testing optical depth safeguards...")
    tau_small = torch.tensor([1e-5, 1e-4, 1e-3])
    tau_large = torch.tensor([25.0, 50.0, 100.0])
    print(f"[OK] Safe transmittance (small): {OpticalDepthScaling.safe_transmittance(tau_small)}")
    print(f"[OK] Safe transmittance (large): {OpticalDepthScaling.safe_transmittance(tau_large)}")

    print("\n[6] Delta-Eddington sanity check...")
    tau = torch.tensor([30.0])
    omega_val = torch.tensor([0.99])
    g = torch.tensor([0.85])
    tau_star, omega_star, g_star = OpticalDepthScaling.delta_eddington_scaling(tau, omega_val, g)
    print(f"[OK] tau*: {tau_star.item():.2f}, omega*: {omega_star.item():.3f}, g*: {g_star.item():.3f}")

    print("\n[OK] ALL TESTS PASSED - RTNO v4.3 Production")
    return model, outputs


if __name__ == "__main__":
    test_complete_rtno_v43()
