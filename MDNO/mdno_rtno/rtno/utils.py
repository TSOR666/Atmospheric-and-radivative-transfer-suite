"""
Shared utilities for the RTNO package.
"""
from __future__ import annotations

import warnings

try:
    import scipy.special as sp  # type: ignore
    from scipy import integrate  # type: ignore
    from scipy.interpolate import interp1d  # type: ignore
    SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    sp = None  # type: ignore
    integrate = None  # type: ignore
    interp1d = None  # type: ignore
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available, using fallback implementations")

__all__ = ["sp", "integrate", "interp1d", "SCIPY_AVAILABLE"]
