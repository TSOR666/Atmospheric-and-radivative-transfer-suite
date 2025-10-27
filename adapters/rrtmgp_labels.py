"""
RRTMGP parity label generator.

The adapter attempts to call the ``rrtmgp`` Python bindings when installed. If
they are missing, we fall back to a Beer-Lambert two-stream approximation so
test fixtures still receive deterministic labels.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

import numpy as np


try:  # pragma: no cover - optional heavy dependency
    import rrtmgp  # type: ignore
except Exception:  # pragma: no cover
    rrtmgp = None  # type: ignore


@dataclass
class ColumnState:
    pressure: np.ndarray  # Pa
    temperature: np.ndarray  # K
    gas_mix: Mapping[str, np.ndarray]  # volume mixing ratios
    surface_temperature: float
    solar_zenith_deg: float

    @classmethod
    def from_sample(cls, sample: Mapping[str, Any]) -> "ColumnState":
        state = sample.get("state", {})
        if not state:
            raise KeyError("sample['state'] must contain pressure/temperature profiles.")

        pressure = np.asarray(state.get("pressure"))
        temperature = np.asarray(state.get("temperature"))
        if pressure.size == 0 or temperature.size == 0:
            raise ValueError("pressure and temperature profiles cannot be empty.")

        gases = sample.get("gases", {})
        if not isinstance(gases, Mapping):
            raise TypeError("sample['gases'] must be a mapping of gas -> profile.")

        surface_temperature = float(sample.get("surface_temperature", temperature[-1]))
        solar_zenith_deg = float(sample.get("solar_zenith_deg", 45.0))
        return cls(pressure, temperature, gases, surface_temperature, solar_zenith_deg)


def _fallback_two_stream(column: ColumnState, k_ext: Sequence[float]) -> Dict[str, Any]:
    """Beer-Lambert fallback when RRTMGP bindings are unavailable."""
    pressure = column.pressure
    temperature = column.temperature
    k_ext = np.asarray(k_ext, dtype=float)
    if k_ext.size != pressure.size:
        k_ext = np.resize(k_ext, pressure.size)

    dz = np.abs(np.gradient(pressure)) / np.maximum(pressure, 1.0)
    optical_depth = np.cumsum(k_ext * dz)

    surface_flux = 5.670374419e-8 * column.surface_temperature**4
    up_flux = surface_flux * np.exp(-optical_depth)

    return {
        "labels": {
            "optical_depth": optical_depth.tolist(),
            "upwelling_flux": up_flux.tolist(),
        },
        "meta": {
            "tool": "rrtmgp-fallback",
            "status": "approximate",
            "reason": "rrtmgp bindings not installed; used Beer-Lambert approximation.",
        },
    }


def build_rrtmgp_labels(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Produce scalar radiative-transfer labels using RRTMGP if available.

    Parameters
    ----------
    sample:
        Mapping describing a single column atmosphere. Required keys:
        ``state`` (with ``pressure`` and ``temperature`` arrays) and ``gases``.
    """
    column = ColumnState.from_sample(sample)

    if rrtmgp is None:
        k_ext = sample.get("extinction_profile", np.full_like(column.pressure, 1e-4, dtype=float))
        return _fallback_two_stream(column, k_ext)

    try:  # pragma: no cover - depends on heavy external library
        solver = rrtmgp.LWFluxCalculator()
        solver.load_coefficients(sample.get("coefficients_path"))
        result = solver.compute_fluxes(
            pressure=column.pressure,
            temperature=column.temperature,
            gas_mixings=column.gas_mix,
            surface_temperature=column.surface_temperature,
            solar_zenith=column.solar_zenith_deg,
        )
        return {
            "labels": {
                "upwelling_flux": result["up_flux"].tolist(),
                "downwelling_flux": result["down_flux"].tolist(),
                "heating_rate": result["heating_rate"].tolist(),
            },
            "meta": {
                "tool": "rrtmgp",
                "status": "ok",
            },
        }
    except Exception as exc:
        return {
            "labels": None,
            "meta": {
                "tool": "rrtmgp",
                "status": "error",
                "reason": str(exc),
            },
        }

