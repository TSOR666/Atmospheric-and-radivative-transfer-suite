"""
Multiscale feature encoders inspired by the legacy wavelet decomposition.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiscaleFeatureEncoder(nn.Module):
    """
    Construct a Laplacian pyramid and blend the detail coefficients back into the input.
    """

    def __init__(self, levels: int) -> None:
        super().__init__()
        self.levels = levels

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        # tensor shape: [B, C, X, Y, Z]
        current = tensor
        combined = tensor.clone()
        base_shape = tensor.shape[-3:]
        weight = 1.0

        for _ in range(self.levels):
            pooled = F.avg_pool3d(current, kernel_size=2, stride=2, ceil_mode=True)
            upsampled = F.interpolate(
                pooled, size=current.shape[-3:], mode="trilinear", align_corners=False
            )
            detail = current - upsampled

            if detail.shape[-3:] != base_shape:
                detail = F.interpolate(
                    detail,
                    size=base_shape,
                    mode="trilinear",
                    align_corners=False,
                )

            weight *= 0.5
            combined = combined + weight * detail
            current = pooled

            if min(current.shape[-3:]) <= 1:
                break

        return combined


__all__ = ["MultiscaleFeatureEncoder"]
