"""Neural operator layers for MDNO."""
from typing import Tuple

import torch
import torch.nn as nn

from .config import MDNOConfig

class FourierLayer3D(nn.Module):
    """3D Fourier layer"""
    
    def __init__(self, width: int, modes: Tuple[int, int, int]):
        super().__init__()
        self.width = width
        self.modes = modes
        
        self.weights = nn.Parameter(
            torch.randn(width, width, modes[0], modes[1], modes[2], 2) / width
        )
        self.local_conv = nn.Conv3d(width, width, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        
        out_ft = torch.zeros_like(x_ft)
        weight_complex = torch.complex(self.weights[..., 0], self.weights[..., 1])
        
        out_ft[:, :, :self.modes[0], :self.modes[1], :self.modes[2]] = torch.einsum(
            "bixyz,ioxyz->boxyz",
            x_ft[:, :, :self.modes[0], :self.modes[1], :self.modes[2]],
            weight_complex
        )
        
        x_spectral = torch.fft.irfftn(out_ft, s=x.shape[-3:])
        x_local = self.local_conv(x)
        
        return x_spectral + x_local

class FNO3D(nn.Module):
    """3D Fourier Neural Operator"""
    
    def __init__(self, config: MDNOConfig, in_channels: int, out_channels: int):
        super().__init__()
        width = config.operator_width
        
        self.input_proj = nn.Sequential(
            nn.Conv3d(in_channels, width, 1),
            nn.GroupNorm(min(8, width), width),
            nn.GELU()
        )
        
        self.fourier_layers = nn.ModuleList([
            FourierLayer3D(width, (16, 16, 8))
            for _ in range(config.operator_layers)
        ])
        
        self.output_proj = nn.Sequential(
            nn.Conv3d(width, width // 2, 1),
            nn.GroupNorm(min(4, width // 2), width // 2),
            nn.GELU(),
            nn.Conv3d(width // 2, out_channels, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for layer in self.fourier_layers:
            x = layer(x) + x
        return self.output_proj(x)

# ============================================================================
# MAIN ENHANCED MDNO v5.3 (Audit ready)
# ============================================================================

