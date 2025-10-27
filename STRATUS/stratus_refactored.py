"""
Convenience entry point for the refactored STRATUS framework.
"""

from __future__ import annotations

import torch

from stratus import (
    AngularBasis,
    ExportInputs,
    MonteCarloConfig,
    PerformanceMonitor,
    PINNConfig,
    RadiativeValidator,
    RayMarchConfig,
    SpectralMethod,
    StratusConfig,
    StratusExporter,
    StratusRadianceModel,
)


def create_stratus_model(**overrides) -> tuple[StratusRadianceModel, StratusConfig]:
    """
    Create a :class:`StratusRadianceModel` with optional configuration overrides.

    Example
    -------
    >>> model, config = create_stratus_model(
    ...     grid_shape=(64, 64, 32),
    ...     n_bands=8,
    ...     raymarch=RayMarchConfig(step_size=0.25),
    ... )
    """

    config_kwargs: dict[str, object] = {}
    config_kwargs.update(overrides)

    if "raymarch" in config_kwargs and isinstance(config_kwargs["raymarch"], dict):
        config_kwargs["raymarch"] = RayMarchConfig(**config_kwargs["raymarch"])
    if "monte_carlo" in config_kwargs and isinstance(config_kwargs["monte_carlo"], dict):
        config_kwargs["monte_carlo"] = MonteCarloConfig(**config_kwargs["monte_carlo"])
    if "pinn" in config_kwargs and isinstance(config_kwargs["pinn"], dict):
        config_kwargs["pinn"] = PINNConfig(**config_kwargs["pinn"])

    config = StratusConfig(**config_kwargs)
    model = StratusRadianceModel(config)
    return model, config


def _smoke_test() -> None:
    """Run a minimal forward pass to verify the setup."""

    model, config = create_stratus_model(
        grid_shape=(16, 16, 8),
        n_bands=2,
        spectral_method=SpectralMethod.CORRELATED_K,
        raymarch=RayMarchConfig(step_size=0.25, max_optical_depth_per_step=0.1),
        angular_basis=AngularBasis.SPHERICAL_HARMONICS,
        sh_max_degree=1,
        use_multiscale=True,
        scattering_model="mie",
        mie_refractive_index=1.4,
    )
    device = config.device
    dtype = config.dtype

    batch = 2
    nx, ny, nz = config.grid_shape
    channels = (batch, config.n_stokes, nx, ny, nz, config.n_bands)

    kappa = torch.rand(*channels, device=device, dtype=dtype) * 0.05
    source = torch.rand(*channels, device=device, dtype=dtype) * 0.01

    # Evaluate at eight random voxels
    coords = [
        torch.randint(0, limit, (batch, 8), device=device, dtype=torch.long)
        for limit in config.grid_shape
    ]
    evaluation_points = torch.stack(coords, dim=-1).to(dtype=dtype)

    with torch.no_grad():
        output = model(kappa, source, evaluation_points, compute_physics_loss=True)

    print("Radiance shape:", tuple(output["radiance"].shape))
    print("Transmittance range:", output["transmittance"].amin(), output["transmittance"].amax())
    if "total_physics_loss" in output:
        print("Physics loss:", output["total_physics_loss"].item())

    validator = RadiativeValidator(model, config)
    result = validator.uniform_slab_test()
    print("Uniform slab test:", "PASS" if result.passed else "FAIL", result.metrics)

    monitor = PerformanceMonitor(log_interval=1)
    with monitor.monitor_step():
        with torch.no_grad():
            _ = model(kappa, source, evaluation_points)
    print("Step metrics:", monitor.current_metrics())

    mc_model, mc_config = create_stratus_model(
        grid_shape=(8, 8, 4),
        n_bands=1,
        n_stokes=4,
        spatial_method="monte_carlo",
        monte_carlo={"n_rays": 2048, "max_scatter": 6, "g": 0.2},
    )
    mc_device = mc_config.device
    mc_dtype = mc_config.dtype
    mc_channels = (1, mc_config.n_stokes, *mc_config.grid_shape, mc_config.n_bands)
    mc_kappa = torch.rand(*mc_channels, device=mc_device, dtype=mc_dtype) * 0.02
    mc_source = torch.rand(*mc_channels, device=mc_device, dtype=mc_dtype) * 0.01
    with torch.no_grad():
        mc_output = mc_model(mc_kappa, mc_source)
    print("Monte Carlo radiance shape:", tuple(mc_output["radiance"].shape))

    pinn_model, pinn_config = create_stratus_model(
        grid_shape=(8, 8, 4),
        n_bands=1,
        n_stokes=1,
        spatial_method="physics_informed",
        pinn={"n_epochs": 10, "n_samples": 512, "hidden_layers": 2, "hidden_dim": 32},
    )
    pinn_device = pinn_config.device
    pinn_dtype = pinn_config.dtype
    pinn_channels = (1, pinn_config.n_stokes, *pinn_config.grid_shape, pinn_config.n_bands)
    pinn_kappa = torch.rand(*pinn_channels, device=pinn_device, dtype=pinn_dtype) * 0.03
    pinn_source = torch.rand(*pinn_channels, device=pinn_device, dtype=pinn_dtype) * 0.01
    pinn_output = pinn_model(pinn_kappa, pinn_source)
    print("PINN radiance shape:", tuple(pinn_output["radiance"].shape))

    StratusExporter(model, config)
    example_inputs = ExportInputs(kappa=kappa, source=source, evaluation_points=evaluation_points)
    # Export paths are not written during smoke test to avoid filesystem side-effects.
    _ = example_inputs  # Keep for illustration without touching disk.


if __name__ == "__main__":
    _smoke_test()
