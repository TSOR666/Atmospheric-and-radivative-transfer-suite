"""
Physics invariants validator.

Provides lightweight diagnostics for conservation, positivity, and stability
that can run on CPU-only environments.  It accepts simple dictionary inputs so
that tests and partner QA tools can reuse the same logic without depending on
training infrastructure.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, MutableMapping

import numpy as np


def _to_numpy(value: Any) -> np.ndarray:
    """Convert torch/xarray like objects into a numpy array."""
    if value is None:
        raise TypeError("Field is None; cannot evaluate invariants.")

    if hasattr(value, "detach"):  # torch.Tensor
        return value.detach().cpu().numpy()

    if hasattr(value, "values"):  # xarray.DataArray
        return np.asarray(value.values)

    if hasattr(value, "to_numpy"):
        return np.asarray(value.to_numpy())

    return np.asarray(value)


def _register_positive(field: str, array: np.ndarray, errors: MutableMapping[str, str]) -> None:
    min_val = float(np.nanmin(array))
    if min_val < 0.0:
        errors[field] = f"{field}: contains negative values (min={min_val:.3e})."


def check_invariants(sample: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Run a suite of invariants on a fluid state.

    Parameters
    ----------
    sample:
        Mapping with keys:
          * ``state`` - mapping of field name -> array-like (required)
          * ``dt``, ``dx``, ``dy``, ``dz`` - grid spacing metadata (optional but
            required for CFL check)

    Returns
    -------
    dict
        ``errors`` (list[str]): Hard failures.
        ``warnings`` (list[str]): Potential issues / skipped checks.
        ``metrics`` (dict[str, float]): Summary statistics for logging.
    """
    if "state" not in sample:
        raise KeyError("sample must contain a 'state' mapping.")

    state = sample["state"]
    if not isinstance(state, Mapping):
        raise TypeError("sample['state'] must be a mapping of field names to arrays.")

    errors_map: Dict[str, str] = {}
    warnings: list[str] = []
    metrics: Dict[str, float] = {}

    # Positivity and summary statistics
    for field in ("rho", "density", "T", "temperature", "q", "specific_humidity"):
        if field not in state:
            continue
        arr = _to_numpy(state[field])
        _register_positive(field, arr, errors_map)
        metrics[f"{field}_mean"] = float(np.nanmean(arr))
        metrics[f"{field}_std"] = float(np.nanstd(arr))

    # CFL condition
    dt = sample.get("dt")
    dx = sample.get("dx")
    dy = sample.get("dy")
    dz = sample.get("dz")
    if None not in (dt, dx, dy, dz) and all(axis in state for axis in ("u", "v", "w")):
        try:
            u = _to_numpy(state["u"])
            v = _to_numpy(state["v"])
            w = _to_numpy(state["w"])
            umax = float(np.nanmax(np.abs(u)))
            vmax = float(np.nanmax(np.abs(v)))
            wmax = float(np.nanmax(np.abs(w)))
            courant = umax * dt / dx + vmax * dt / dy + wmax * dt / dz
            metrics["courant_number"] = courant
            if courant > 1.0:
                warnings.append(
                    f"CFL condition violated (C={courant:.3f} > 1); reduce dt or enlarge grid spacing."
                )
        except Exception as exc:  # pragma: no cover - defensive
            warnings.append(f"CFL computation failed: {exc}")
    else:
        warnings.append("CFL check skipped (missing u/v/w or grid spacing metadata).")

    # Conservation integrals
    if all(name in state for name in ("rho", "u", "v", "w", "T")):
        rho = _to_numpy(state["rho"])
        u = _to_numpy(state["u"])
        v = _to_numpy(state["v"])
        w = _to_numpy(state["w"])
        temperature = _to_numpy(state["T"])

        metrics["total_mass"] = float(np.nansum(rho))

        kinetic = 0.5 * rho * (u ** 2 + v ** 2 + w ** 2)
        metrics["total_kinetic_energy"] = float(np.nansum(kinetic))

        cp = float(sample.get("cp", 1004.0))
        internal = cp * rho * (temperature - np.nanmean(temperature))
        metrics["total_internal_energy"] = float(np.nansum(internal))

        if not np.all(np.isfinite(kinetic)):
            errors_map["energy"] = "Non-finite kinetic energy detected."

    # Moisture boundedness
    if "q" in state:
        q = _to_numpy(state["q"])
        q_max = float(np.nanmax(q))
        if q_max > 1.0:
            warnings.append(f"Specific humidity exceeds 1 kg/kg (max={q_max:.2f}); verify units.")

    return {
        "errors": [msg for _, msg in sorted(errors_map.items())],
        "warnings": warnings,
        "metrics": metrics,
    }

