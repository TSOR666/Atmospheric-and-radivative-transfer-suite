"""
CF/NetCDF schema validator used by the `psuite-validate` CLI.

The goal is to offer immediate, actionable feedback when a partner supplies a
small sample file.  Checks favour signal over strictness: we report malformed
or suspicious metadata as warnings, and reserve errors for structural issues
that would break downstream tooling.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Tuple

import math

import numpy as np


CoordLike = Mapping[str, object]


@dataclass
class ValidationReport:
    errors: List[str]
    warnings: List[str]
    metrics: Dict[str, float]

    def as_dict(self) -> Dict[str, object]:
        return {
            "errors": self.errors,
            "warnings": self.warnings,
            "metrics": self.metrics,
        }


def _get_coord(ds, names: Iterable[str]):
    for name in names:
        if name in ds.coords:
            return name, ds.coords[name]
        if name in ds.variables:
            return name, ds.variables[name]
    return None, None


def _ensure_units(coord, expected_keywords: Tuple[str, ...], warnings: MutableMapping[str, List[str]]) -> None:
    units = getattr(coord, "attrs", {}).get("units")
    if not units:
        warnings.setdefault("units", []).append(f"{coord.name}: missing units attribute")
        return
    if not any(keyword in units.lower() for keyword in expected_keywords):
        hints = ", ".join(expected_keywords)
        warnings.setdefault("units", []).append(f"{coord.name}: unexpected units '{units}' (expected to contain {hints})")


def _data_min_max(array) -> Tuple[float, float]:
    if hasattr(array, "values"):
        array = array.values
    if hasattr(array, "to_numpy"):
        array = array.to_numpy()
    array = np.asarray(array)
    return float(np.nanmin(array)), float(np.nanmax(array))


def validate_cf(ds) -> Dict[str, object]:
    """
    Validate a dataset against a thin slice of CF conventions and project
    expectations.

    Parameters
    ----------
    ds:
        An :class:`xarray.Dataset` (the function only assumes ``dims``,
        ``coords`` and ``variables`` attributes).

    Returns
    -------
    dict
        Mapping with ``errors`` (list[str]), ``warnings`` (list[str]), and
        ``metrics`` (dict[str, float]).
    """
    report = ValidationReport(errors=[], warnings=[], metrics={})
    warning_groups: Dict[str, List[str]] = {}

    expected_dims = {"time", "lat", "lon"}
    present_dims = set(ds.dims)
    missing_dims = sorted(expected_dims - present_dims)
    if missing_dims:
        report.errors.append(f"Missing required dimensions: {', '.join(missing_dims)}")

    # Latitude / longitude sanity checks
    lat_name, lat_coord = _get_coord(ds, ("lat", "latitude"))
    lon_name, lon_coord = _get_coord(ds, ("lon", "longitude"))

    if lat_coord is not None:
        lat_min, lat_max = _data_min_max(lat_coord)
        if lat_min < -90.0 or lat_max > 90.0 or math.isnan(lat_min) or math.isnan(lat_max):
            report.errors.append(
                f"{lat_name}: values outside [-90, 90] (min={lat_min:.2f}, max={lat_max:.2f})"
            )
        _ensure_units(lat_coord, ("degree",), warning_groups)
    else:
        report.errors.append("Latitude coordinate (lat/latitude) is missing")

    if lon_coord is not None:
        lon_min, lon_max = _data_min_max(lon_coord)
        within_globe = (-180.0 <= lon_min <= 360.0) and (-180.0 <= lon_max <= 360.0)
        if not within_globe or math.isnan(lon_min) or math.isnan(lon_max):
            report.errors.append(
                f"{lon_name}: values outside typical ranges [-180, 180] or [0, 360] "
                f"(min={lon_min:.2f}, max={lon_max:.2f})"
            )
        _ensure_units(lon_coord, ("degree",), warning_groups)
    else:
        report.errors.append("Longitude coordinate (lon/longitude) is missing")

    # Time coordinate
    time_name, time_coord = _get_coord(ds, ("time",))
    if time_coord is None:
        report.errors.append("Time coordinate is missing")
    else:
        if getattr(time_coord.dtype, "kind", None) not in {"M", "O"}:
            warning_groups.setdefault("time", []).append(
                f"{time_name}: dtype {time_coord.dtype} is not datetime64 or object datetime"
            )
        calendar = getattr(time_coord, "attrs", {}).get("calendar")
        if calendar and calendar.lower() not in {"standard", "gregorian", "proleptic_gregorian"}:
            warning_groups.setdefault("time", []).append(
                f"{time_name}: uncommon calendar '{calendar}'"
            )

    # Level / pressure coordinate monotonicity (optional)
    for level_name in ("level", "plev", "pressure"):
        if level_name in ds.dims or level_name in ds.coords:
            _, level_coord = _get_coord(ds, (level_name,))
            if level_coord is not None:
                level_vals = np.asarray(level_coord)
                if level_vals.ndim == 1:
                    diffs = np.diff(level_vals)
                    if not np.all(diffs < 0) and not np.all(diffs > 0):
                        warning_groups.setdefault("level", []).append(
                            f"{level_name}: not strictly monotonic ({len(diffs)} differences)"
                        )
                _ensure_units(level_coord, ("pascal", "pa", "hpa"), warning_groups)
            break

    # Core variable coverage
    expected_vars = {
        "temperature": ("K",),
        "specific_humidity": ("kg", "kg kg-1"),
        "surface_pressure": ("Pa", "hPa"),
        "cloud_fraction": ("1", "kg kg-1"),
    }
    present_vars = set(ds.data_vars)
    missing_vars = sorted(set(expected_vars) - present_vars)
    if missing_vars:
        warning_groups.setdefault("variables", []).append(
            f"Dataset missing recommended variables: {', '.join(missing_vars)}"
        )
    for name, unit_hints in expected_vars.items():
        if name in ds.data_vars:
            _ensure_units(ds.data_vars[name], unit_hints, warning_groups)

    report.metrics.update(
        {
            "n_dimensions": float(len(ds.dims)),
            "n_variables": float(len(ds.data_vars)),
            "size_bytes": float(
                getattr(ds, "nbytes", np.sum(v.nbytes for v in ds.data_vars.values()))
            ),
        }
    )

    # Flatten grouped warnings
    for group in warning_groups.values():
        report.warnings.extend(sorted(set(group)))

    return report.as_dict()


def cli() -> None:
    import json
    import sys

    try:
        import xarray as xr
    except Exception:  # pragma: no cover - import guard
        print("xarray is required for psuite-validate. Try: pip install xarray")
        sys.exit(2)

    if len(sys.argv) != 2:
        print("Usage: psuite-validate <dataset.nc>")
        sys.exit(2)

    ds = xr.open_dataset(sys.argv[1])
    print(json.dumps(validate_cf(ds), indent=2))


if __name__ == "__main__":
    cli()
