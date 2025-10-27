"""
Unified physics framework interface bundling MDNO and RTNO components.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch.nn as nn

from mdno import (
    MDNOConfig,
    MDNOTrainer,
    EnhancedMDNO_v53_Complete,
    PhysicsConstraintType,
    TurbulenceModel,
    monitor as mdno_monitor,
)
from rtno import (
    BoundaryType,
    RTNOConfig,
    EnhancedRTNO_v43,
    PerformanceMonitor as RTNOPerformanceMonitor,
    monitor as rtno_monitor,
)

FRAMEWORKS = {"mdno", "rtno"}


def create_model(
    model: str,
    config: Optional[Any] = None,
    *,
    config_kwargs: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    """
    Factory helper returning a configured physics model.

    Parameters
    ----------
    model:
        Name of the model to create (``"mdno"`` or ``"rtno"``).
    config:
        Optional pre-built configuration instance. If omitted, a new configuration
        will be created from ``config_kwargs``.
    config_kwargs:
        Optional keyword arguments used to build a configuration when ``config`` is not
        supplied.
    """
    model_key = model.lower()
    cfg_kwargs = config_kwargs or {}

    if model_key == "mdno":
        if config is None:
            mdno_config = MDNOConfig(**cfg_kwargs)
        elif isinstance(config, MDNOConfig):
            mdno_config = config
        else:
            raise TypeError("MDNO configuration must be an instance of MDNOConfig")
        return EnhancedMDNO_v53_Complete(mdno_config)

    if model_key == "rtno":
        if config is None:
            rtno_config = RTNOConfig(**cfg_kwargs)
        elif isinstance(config, RTNOConfig):
            rtno_config = config
        else:
            raise TypeError("RTNO configuration must be an instance of RTNOConfig")
        return EnhancedRTNO_v43(rtno_config)

    raise ValueError(f"Unknown model '{model}'. Available options: {sorted(FRAMEWORKS)}")


__all__ = [
    "FRAMEWORKS",
    "BoundaryType",
    "EnhancedMDNO_v53_Complete",
    "EnhancedRTNO_v43",
    "MDNOConfig",
    "MDNOTrainer",
    "PhysicsConstraintType",
    "RTNOConfig",
    "RTNOPerformanceMonitor",
    "TurbulenceModel",
    "create_model",
    "mdno_monitor",
    "rtno_monitor",
]
