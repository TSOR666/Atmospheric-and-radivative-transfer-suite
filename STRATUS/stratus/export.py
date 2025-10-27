"""
Model export utilities for STRATUS.

The functions provided here produce ONNX and TorchScript artefacts while handling
optional evaluation points gracefully.  This replaces the improvised exporters
from the legacy utilities and ensures consistent behaviour across the framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from .config import StratusConfig

# Optional dependency ---------------------------------------------------------
try:  # pragma: no cover - optional dependency
    import onnx  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    onnx = None  # type: ignore


@dataclass
class ExportInputs:
    """
    Convenience container describing the tensors required for export.

    Attributes
    ----------
    kappa:
        Absorption coefficient tensor.
    source:
        Emission coefficient tensor.
    evaluation_points:
        Optional evaluation points tensor.
    """

    kappa: torch.Tensor
    source: torch.Tensor
    evaluation_points: torch.Tensor | None = None


class StratusExporter:
    """Export STRATUS models to deployment-friendly formats."""

    def __init__(self, model: nn.Module, config: StratusConfig) -> None:
        self.model = model
        self.config = config
        self.model.eval()

    # ------------------------------------------------------------------ ONNX
    def export_onnx(
        self,
        save_path: str | Path,
        inputs: ExportInputs,
        *,
        opset_version: int = 17,
        with_evaluation_points: bool = True,
    ) -> Path:
        """
        Export the model to ONNX.

        Parameters
        ----------
        save_path:
            Destination file path.
        inputs:
            Example inputs used to trace the model.
        opset_version:
            ONNX opset to target.
        with_evaluation_points:
            Include the optional evaluation point argument when true.
        """

        if onnx is None:  # pragma: no cover - optional dependency
            raise RuntimeError("onnx is not installed. Install `onnx` to enable ONNX export.")

        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        class _Wrapper(nn.Module):
            def __init__(self, model: nn.Module, include_points: bool) -> None:
                super().__init__()
                self.model = model
                self.include_points = include_points

            def forward(
                self,
                kappa: torch.Tensor,
                source: torch.Tensor,
                evaluation_points: torch.Tensor | None = None,
            ) -> torch.Tensor:
                if self.include_points:
                    return self.model(kappa, source, evaluation_points)["radiance"]
                return self.model(kappa, source)["radiance"]

        wrapper = _Wrapper(self.model, with_evaluation_points)

        export_args_list = [inputs.kappa, inputs.source]
        input_names = ["kappa", "source"]
        dynamic_axes = {"kappa": {0: "batch"}, "source": {0: "batch"}}

        if with_evaluation_points:
            eval_points = inputs.evaluation_points
            if eval_points is None:
                raise ValueError(
                    "evaluation_points must be provided when with_evaluation_points=True"
                )
            export_args_list.append(eval_points)
            input_names.append("evaluation_points")
            dynamic_axes["evaluation_points"] = {0: "batch", 1: "n_points"}

        export_args = tuple(export_args_list)

        torch.onnx.export(
            wrapper,
            export_args,
            path.as_posix(),
            opset_version=opset_version,
            input_names=input_names,
            output_names=["radiance"],
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
        )

        return path

    # ---------------------------------------------------------------- torchscript
    def export_torchscript(
        self,
        save_path: str | Path,
        inputs: ExportInputs,
        *,
        with_evaluation_points: bool = True,
        optimize: bool = True,
    ) -> Path:
        """
        Export the model to TorchScript using `torch.jit.trace`.
        """

        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        def _call_model(
            kappa: torch.Tensor,
            source: torch.Tensor,
            evaluation_points: torch.Tensor | None = None,
        ) -> torch.Tensor:
            output = (
                self.model(kappa, source, evaluation_points)
                if with_evaluation_points
                else self.model(kappa, source)
            )
            return output["radiance"]

        if with_evaluation_points and inputs.evaluation_points is None:
            raise ValueError("evaluation_points must be provided when with_evaluation_points=True")

        example_args = (
            inputs.kappa,
            inputs.source,
            inputs.evaluation_points if with_evaluation_points else None,
        )

        traced = torch.jit.trace(_call_model, example_args, strict=False)

        if optimize:
            traced = torch.jit.optimize_for_inference(traced)

        traced.save(path.as_posix())
        return path


__all__ = ["ExportInputs", "StratusExporter"]
