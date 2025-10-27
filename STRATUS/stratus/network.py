"""
Neural operator building blocks for STRATUS.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import StratusConfig


class TensorCoreLinear(nn.Module):
    """
    Linear layer with optional padding to align shapes for tensor-core execution.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        config: StratusConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.in_features = in_features
        self.out_features = out_features
        self.padded_in = ((in_features + 7) // 8) * 8
        self.padded_out = ((out_features + 7) // 8) * 8

        self.weight = nn.Parameter(torch.empty(self.padded_out, self.padded_in))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.padded_out))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.shape[-1] != self.padded_in:
            pad_width = self.padded_in - input.shape[-1]
            input = F.pad(input, (0, pad_width))

        output = F.linear(input, self.weight, self.bias)

        if output.shape[-1] != self.out_features:
            output = output[..., : self.out_features]
        return output


class TensorCoreConv3d(nn.Module):
    """
    3D convolution that pads channels to align with tensor-core friendly widths.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        config: StratusConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padded_in = ((in_channels + 7) // 8) * 8
        self.padded_out = ((out_channels + 7) // 8) * 8

        self.conv = nn.Conv3d(
            self.padded_in,
            self.padded_out,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.shape[1] != self.padded_in:
            pad_channels = self.padded_in - input.shape[1]
            padding = torch.zeros(
                input.shape[0],
                pad_channels,
                *input.shape[2:],
                device=input.device,
                dtype=input.dtype,
            )
            input = torch.cat([input, padding], dim=1)

        if self.config.channels_last_3d and input.dim() == 5:
            input = input.contiguous(memory_format=torch.channels_last_3d)

        output = self.conv(input)

        if output.shape[1] != self.out_channels:
            output = output[:, : self.out_channels]
        return output


class SpectralConv3d(nn.Module):
    """
    Spectral convolution used by the Fourier Neural Operator.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: tuple[int, int, int],
        *,
        config: StratusConfig,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.config = config
        scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale
            * torch.randn(
                in_channels,
                out_channels,
                modes[0],
                modes[1],
                modes[2] // 2 + 1,
                2,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Save input shape for inverse transform
        original_shape = x.shape[-3:]

        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        out_ft = torch.zeros_like(
            x_ft[:, : self.out_channels, ...], device=x.device, dtype=x_ft.dtype
        )

        weight = torch.view_as_complex(self.weights)

        m0 = min(self.modes[0], x_ft.size(-3))
        m1 = min(self.modes[1], x_ft.size(-2))
        m2 = min(self.modes[2], x_ft.size(-1))

        out_ft[..., :m0, :m1, :m2] = torch.einsum(
            "bixyz,ioxyz->boxyz",
            x_ft[:, : self.in_channels, :m0, :m1, :m2],
            weight[:, :, :m0, :m1, :m2],
        )

        x = torch.fft.irfftn(out_ft, s=original_shape, dim=[-3, -2, -1])
        return x


class TrueFNO3D(nn.Module):
    """
    Three-dimensional Fourier Neural Operator used as spatial backbone.
    """

    def __init__(self, config: StratusConfig) -> None:
        super().__init__()
        self.config = config
        in_channels = config.n_stokes * config.n_bands

        self.fc0 = TensorCoreConv3d(in_channels, config.fno_width, 1, config=config)
        self.spectral_layers = nn.ModuleList(
            [
                SpectralConv3d(config.fno_width, config.fno_width, config.fno_modes, config=config)
                for _ in range(config.fno_layers)
            ]
        )
        self.conv_layers = nn.ModuleList(
            [
                TensorCoreConv3d(config.fno_width, config.fno_width, 1, config=config)
                for _ in range(config.fno_layers)
            ]
        )
        self.norms = nn.ModuleList(
            [nn.InstanceNorm3d(config.fno_width) for _ in range(config.fno_layers)]
        )
        self.fc1 = TensorCoreConv3d(config.fno_width, in_channels, 1, config=config)
        nn.init.zeros_(self.fc1.conv.weight)
        if self.fc1.conv.bias is not None:
            nn.init.zeros_(self.fc1.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = x.permute(0, 1, 5, 2, 3, 4)
        spatial_dims = x.shape[-3:]
        x = x.reshape(batch_size, -1, *spatial_dims)

        if self.config.channels_last_3d and x.dim() == 5:
            x = x.contiguous(memory_format=torch.channels_last_3d)

        x = self.fc0(x)

        for spectral, conv, norm in zip(self.spectral_layers, self.conv_layers, self.norms):
            x1 = spectral(x)
            x2 = conv(x)
            x = norm(x1 + x2)
            x = F.gelu(x)
            # Clamp to prevent numerical overflow in subsequent layers
            x = x.clamp(-1e6, 1e6)

        x = self.fc1(x)
        x = x.reshape(
            batch_size,
            self.config.n_stokes,
            self.config.n_bands,
            *x.shape[-3:],
        ).permute(0, 1, 3, 4, 5, 2)
        return x


__all__ = ["TensorCoreLinear", "TensorCoreConv3d", "SpectralConv3d", "TrueFNO3D"]
