"""Common exception hierarchy for the STRATUS framework."""

from __future__ import annotations


class StratusError(Exception):
    """Base class for all STRATUS specific exceptions."""


class StratusConfigError(StratusError, ValueError):
    """Raised when configuration validation fails."""


class StratusPhysicsError(StratusError, RuntimeError):
    """Raised when physical constraints are violated during simulation."""


__all__ = ["StratusError", "StratusConfigError", "StratusPhysicsError"]
