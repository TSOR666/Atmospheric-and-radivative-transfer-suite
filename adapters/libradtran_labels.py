"""
libRadtran/DISORT parity label generator.

Because libRadtran is a heavy optional dependency, we adopt a soft-fail
approach: if bindings are unavailable we return a clear status message; when
present, we perform a small DISORT call to produce polarized radiance labels.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


try:  # pragma: no cover - optional dependency
    import pyradlib  # type: ignore
except Exception:  # pragma: no cover
    pyradlib = None  # type: ignore


@dataclass
class DISORTConfig:
    """Minimal configuration necessary to run DISORT via libRadtran."""

    atmosphere: str
    wavelength_nm: float
    solar_zenith_deg: float
    sensor_zenith_deg: float
    relative_azimuth_deg: float
    n_stokes: int = 1
    n_streams: int = 8
    albedo: float = 0.1
    altitude_km: float = 0.0

    @classmethod
    def from_sample(cls, sample: Dict[str, Any]) -> "DISORTConfig":
        cfg = sample.get("config", {})
        if not isinstance(cfg, dict):
            raise TypeError("sample['config'] must be a mapping.")
        required = ["atmosphere", "wavelength_nm", "solar_zenith_deg", "sensor_zenith_deg", "relative_azimuth_deg"]
        missing = [field for field in required if field not in cfg]
        if missing:
            raise KeyError(f"Missing libRadtran configuration keys: {', '.join(missing)}")
        return cls(**cfg)


def build_libradtran_labels(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Produce polarized radiance labels using libRadtran/DISORT when available.

    Parameters
    ----------
    sample:
        Mapping with at minimum ``config`` describing the DISORT scene. Optional
        keys ``aerosol_optical_depth`` and ``surface_albedo`` override defaults.
    """
    if pyradlib is None:
        return {
            "labels": None,
            "meta": {
                "tool": "libradtran",
                "status": "unavailable",
                "reason": "pyradlib (libRadtran bindings) not installed.",
            },
        }

    try:
        config = DISORTConfig.from_sample(sample)
    except Exception as exc:
        return {
            "labels": None,
            "meta": {
                "tool": "libradtran",
                "status": "error",
                "reason": str(exc),
            },
        }

    surface_albedo = sample.get("surface_albedo", config.albedo)
    aerosol_optical_depth = sample.get("aerosol_optical_depth", 0.05)

    solver = pyradlib.Disort()
    solver.set_atmosphere(config.atmosphere)
    solver.set_geometry(
        solzen=config.solar_zenith_deg,
        szen=config.sensor_zenith_deg,
        relazi=config.relative_azimuth_deg,
    )
    solver.set_wavelength(config.wavelength_nm / 1000.0)  # convert to microns
    solver.set_surface_albedo(surface_albedo)
    solver.set_number_of_streams(config.n_streams)
    solver.set_vmr_profile("H2O", aerosol_optical_depth)
    solver.set_number_of_stokes(config.n_stokes)
    solver.set_observation_altitude(config.altitude_km)

    try:
        result = solver.run()
    except Exception as exc:  # pragma: no cover - external dependency
        return {
            "labels": None,
            "meta": {
                "tool": "libradtran",
                "status": "error",
                "reason": f"DISORT execution failed: {exc}",
            },
        }

    radiance = np.asarray(result["radiance"])
    stokes = radiance.reshape(config.n_stokes, -1)
    return {
        "labels": {
            "radiance": stokes.tolist(),
            "wavelength_nm": config.wavelength_nm,
            "geometry": {
                "solar_zenith_deg": config.solar_zenith_deg,
                "sensor_zenith_deg": config.sensor_zenith_deg,
                "relative_azimuth_deg": config.relative_azimuth_deg,
            },
        },
        "meta": {
            "tool": "libradtran",
            "status": "ok",
            "n_stokes": config.n_stokes,
            "n_streams": config.n_streams,
        },
    }
