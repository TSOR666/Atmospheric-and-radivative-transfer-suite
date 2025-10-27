"""
High-level STRATUS radiance model combining the FNO backbone and ray marcher.
"""

from __future__ import annotations

import math
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

from .angular import SphericalHarmonicsSampler
from .config import AngularBasis, BoundaryCondition, SpectralMethod, StratusConfig
from .exceptions import StratusConfigError, StratusPhysicsError
from .monte_carlo import MonteCarloRadiativeTransfer
from .multiscale import MultiscaleFeatureEncoder
from .network import TensorCoreLinear, TrueFNO3D
from .pinn import PhysicsInformedRTE
from .polarization import MuellerMatrix
from .raymarch import RayMarcher, ray_box_exit_distance


class StratusRadianceModel(nn.Module):
    """
    End-to-end STRATUS implementation with deterministic physics integration.
    """

    def __init__(self, config: StratusConfig) -> None:
        super().__init__()
        self.config = config
        self.spatial_model = TrueFNO3D(config)
        self.mueller = MuellerMatrix(config.device) if config.n_stokes > 1 else None
        self.ray_marcher = RayMarcher(config, mueller=self.mueller)
        self.monte_carlo: MonteCarloRadiativeTransfer | None = None
        self.pinn_solver: PhysicsInformedRTE | None = None
        self.angular_sampler = (
            SphericalHarmonicsSampler(config.sh_max_degree, config.device, config.dtype)
            if config.angular_basis is AngularBasis.SPHERICAL_HARMONICS
            else None
        )
        self.multiscale_encoder = (
            MultiscaleFeatureEncoder(config.multiscale_levels) if config.use_multiscale else None
        )
        self.boundary_handler = self._build_boundary_handler()
        self.spectral_processor = self._build_spectral_processor()

    # ------------------------------------------------------------------ builders
    def _build_boundary_handler(self) -> nn.ModuleDict:
        return nn.ModuleDict(
            {
                "encoder": nn.Sequential(
                    TensorCoreLinear(6, self.config.hidden_dim, config=self.config),
                    nn.SiLU(),
                    nn.LayerNorm(self.config.hidden_dim),
                ),
                "decoder": nn.Sequential(
                    TensorCoreLinear(
                        self.config.hidden_dim, self.config.hidden_dim // 2, config=self.config
                    ),
                    nn.SiLU(),
                    TensorCoreLinear(
                        self.config.hidden_dim // 2,
                        self.config.n_stokes * self.config.n_bands,
                        config=self.config,
                    ),
                ),
            }
        )

    def _build_spectral_processor(self) -> nn.Module:
        if self.config.spectral_method == SpectralMethod.CORRELATED_K:

            class _SpectralResidual(nn.Module):
                def __init__(self, config: StratusConfig) -> None:
                    super().__init__()
                    self.layers = nn.Sequential(
                        TensorCoreLinear(config.n_bands, config.hidden_dim, config=config),
                        nn.SiLU(),
                        TensorCoreLinear(config.hidden_dim, config.n_bands, config=config),
                    )
                    final_linear = self.layers[-1]
                    nn.init.zeros_(final_linear.weight)
                    if final_linear.bias is not None:
                        nn.init.zeros_(final_linear.bias)
                    self.activation = nn.Softplus()

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    correction = self.activation(self.layers(x)) - math.log(2.0)
                    return x + correction

            return _SpectralResidual(self.config)
        return nn.Identity()

    # ---------------------------------------------------------------- utilities
    def apply_boundary_conditions(
        self,
        radiance: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.boundary_condition == BoundaryCondition.VACUUM:
            return radiance

        batch_size, n_points = positions.shape[:2]
        pos_flat = positions.reshape(-1, 3)
        norm = pos_flat / torch.tensor(
            self.config.grid_shape,
            device=pos_flat.device,
            dtype=pos_flat.dtype,
        )
        features = torch.cat([pos_flat, norm], dim=-1)
        encoded = self.boundary_handler["encoder"](features)
        boundary = self.boundary_handler["decoder"](encoded)
        boundary = boundary.view(batch_size, n_points, self.config.n_stokes, self.config.n_bands)
        return radiance + 0.1 * boundary

    def _apply_multiscale(self, field: torch.Tensor) -> torch.Tensor:
        if self.multiscale_encoder is None:
            return field
        batch, stokes, nx, ny, nz, bands = field.shape
        tensor = field.permute(0, 1, 5, 2, 3, 4).reshape(batch, stokes * bands, nx, ny, nz)
        encoded = self.multiscale_encoder(tensor)
        encoded = encoded.reshape(batch, stokes, bands, nx, ny, nz).permute(0, 1, 3, 4, 5, 2)
        return encoded

    def compute_physics_constraints(
        self,
        radiance: torch.Tensor,
        kappa: torch.Tensor,
        source: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}

        rad = radiance
        if rad.dim() == 4 and rad.shape[1] != self.config.n_stokes:
            # Radiance from sampled evaluation points: [B, P, S, B] -> [B, S, P, B]
            rad = rad.permute(0, 2, 1, 3)

        src = source
        if src.dim() == 4 and src.shape[1] != self.config.n_stokes:
            src = src.permute(0, 2, 1, 3)

        spatial_dims = tuple(range(2, rad.dim() - 1)) if rad.dim() > 3 else ()

        if self.config.energy_conservation:
            if spatial_dims:
                integrated = rad.mean(dim=spatial_dims)
            else:
                integrated = rad

            if source.dim() >= 6:
                source_term = source.mean(dim=(2, 3, 4))
            elif src.dim() > 3:
                reduce_dims = tuple(range(2, src.dim() - 1))
                source_term = src.mean(dim=reduce_dims)
            else:
                source_term = src

            losses["energy_conservation"] = F.mse_loss(integrated, source_term)

        if self.config.reciprocity_constraint:
            if spatial_dims:
                flipped = torch.flip(rad, dims=spatial_dims)
                losses["reciprocity"] = F.mse_loss(rad, flipped)

        if self.config.n_stokes >= 4:
            I = rad[:, 0, ...]
            Q = rad[:, 1, ...]
            U = rad[:, 2, ...]
            V = rad[:, 3, ...]
            losses["stokes_non_negative"] = F.relu(-I).mean()
            losses["stokes_norm"] = F.relu(Q**2 + U**2 + V**2 - I**2).mean()

        total = sum(losses.values()) if losses else torch.tensor(0.0, device=radiance.device)
        losses["total_physics_loss"] = total
        return losses

    def _monte_carlo_solver(self) -> MonteCarloRadiativeTransfer:
        if self.monte_carlo is None:
            self.monte_carlo = MonteCarloRadiativeTransfer(
                self.config.n_stokes,
                self.config.grid_shape,
                self.config.monte_carlo,
                device=self.config.device,
            )
        return self.monte_carlo

    def _physics_informed_solver(self) -> PhysicsInformedRTE:
        if self.pinn_solver is None:
            self.pinn_solver = PhysicsInformedRTE(
                self.config.grid_shape,
                self.config.n_stokes,
                self.config.pinn,
                self.config.device,
            )
        return self.pinn_solver

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        kappa: torch.Tensor,
        source: torch.Tensor,
        evaluation_points: torch.Tensor | None = None,
        *,
        compute_physics_loss: bool = False,
        ray_directions: torch.Tensor | None = None,
        ray_weights: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        device = kappa.device
        batch_size = kappa.shape[0]
        dtype = kappa.dtype

        if self.config.spatial_method == "monte_carlo":
            solver = self._monte_carlo_solver()
            radiance_batches = []
            for b in range(batch_size):
                sigma_t = kappa[b, 0].mean(dim=-1)
                detector = solver.solve(sigma_t)
                if self.config.n_stokes == 1:
                    det = detector.unsqueeze(0).unsqueeze(-1)
                else:
                    det = detector.permute(3, 0, 1, 2).unsqueeze(-1)
                radiance_batches.append(det)
            radiance = torch.stack(radiance_batches, dim=0)
            trans = torch.ones_like(radiance)
            return {"radiance": radiance, "transmittance": trans}
        if self.config.spatial_method == "physics_informed":
            solver = self._physics_informed_solver()
            fields = []
            for b in range(batch_size):
                field = solver.solve(kappa[b].to(device), source[b].to(device))
                if field.dim() >= 1 and field.shape[0] == 1:
                    field = field.squeeze(0)
                fields.append(field)
            radiance = torch.stack(fields, dim=0)
            trans = torch.ones_like(radiance)
            return {"radiance": radiance, "transmittance": trans}

        kappa_input = self._apply_multiscale(kappa)
        source_input = self._apply_multiscale(source)

        autocast_device = "cuda" if device.type == "cuda" else "cpu"
        autocast_enabled = self.config.mixed_precision
        autocast_dtype = self.config.compute_dtype if autocast_device == "cuda" else torch.bfloat16
        if autocast_device == "cpu" and self.config.compute_dtype != torch.bfloat16:
            autocast_enabled = False

        if hasattr(torch, "autocast"):
            autocast_cm = torch.autocast(
                device_type=autocast_device,
                dtype=autocast_dtype,
                enabled=autocast_enabled,
            )
        else:  # pragma: no cover - legacy fallback
            try:
                from torch.cuda.amp import autocast as cuda_autocast
            except ImportError:  # pragma: no cover - defensive
                cuda_autocast = None

            if autocast_device == "cuda" and cuda_autocast is not None:
                autocast_cm = cuda_autocast(enabled=self.config.mixed_precision)
            else:
                autocast_cm = nullcontext()

        with autocast_cm:
            spatial_features = self.spatial_model(kappa_input)
            enhanced_source = source_input + 0.1 * spatial_features

            if evaluation_points is None:
                radiance_field = enhanced_source
                if self.config.spectral_method != SpectralMethod.BAND_AVERAGED:
                    b, s, nx, ny, nz, nb = radiance_field.shape
                    flat = radiance_field.permute(0, 2, 3, 4, 1, 5).reshape(-1, nb)
                    processed = self.spectral_processor(flat)
                    radiance_field = processed.reshape(b, nx, ny, nz, s, nb).permute(
                        0, 4, 1, 2, 3, 5
                    )
                result = {"radiance": radiance_field}
            else:
                if evaluation_points.dim() == 2:
                    evaluation_points = evaluation_points.unsqueeze(0).expand(batch_size, -1, -1)
                radiance_batches = []
                for b in range(batch_size):
                    kappa_field = kappa_input[b].permute(1, 2, 3, 0, 4).contiguous()
                    source_field = enhanced_source[b].permute(1, 2, 3, 0, 4).contiguous()
                    eval_pts = evaluation_points[b]
                    n_points = eval_pts.shape[0]
                    directions: torch.Tensor
                    weight_matrix: torch.Tensor

                    if ray_directions is not None:
                        dirs_input = ray_directions
                        if dirs_input.dim() == 4:
                            dirs_input = dirs_input[b]
                        if dirs_input.dim() == 2:
                            dirs_input = dirs_input.unsqueeze(0)
                        elif dirs_input.dim() == 3:
                            pass
                        else:
                            raise StratusConfigError(
                                "ray_directions must have shape [N,3], [P,N,3], or [B,P,N,3]."
                            )

                        if dirs_input.shape[0] == 1 and n_points > 1:
                            dirs_input = dirs_input.expand(n_points, -1, -1)
                        elif dirs_input.shape[0] != n_points:
                            raise StratusConfigError(
                                "ray_directions must match number of evaluation points."
                            )

                        directions = F.normalize(dirs_input.to(device=device, dtype=dtype), dim=-1)
                        n_dirs = directions.shape[1]

                        if ray_weights is not None:
                            weights_input = ray_weights
                            if weights_input.dim() == 3:
                                weights_input = weights_input[b]
                            if weights_input.dim() == 1:
                                if weights_input.numel() != n_dirs:
                                    raise StratusConfigError(
                                        "ray_weights must align with ray_directions count."
                                    )
                                weights_input = weights_input.unsqueeze(0)
                            elif weights_input.dim() == 2:
                                pass
                            else:
                                raise StratusConfigError(
                                    "ray_weights must have shape [N], [P,N], or [B,P,N]."
                                )

                            if weights_input.shape[0] == 1 and n_points > 1:
                                weights_input = weights_input.expand(n_points, -1)
                            elif weights_input.shape[0] != n_points:
                                raise StratusConfigError(
                                    "ray_weights must match number of evaluation points."
                                )
                            if weights_input.shape[1] != n_dirs:
                                raise StratusConfigError(
                                    "ray_weights must align with ray_directions count."
                                )

                            weight_matrix = weights_input.to(device=device, dtype=dtype)
                        else:
                            weight_matrix = torch.full(
                                (n_points, n_dirs),
                                1.0 / max(n_dirs, 1),
                                device=device,
                                dtype=dtype,
                            )
                    elif self.angular_sampler is not None:
                        sample = self.angular_sampler.sample()
                        dir_set = sample.directions.to(device=device, dtype=dtype)
                        weights = sample.weights.to(device=device, dtype=dtype)
                        n_dirs = dir_set.shape[0]
                        directions = dir_set.unsqueeze(0).expand(n_points, -1, 3)
                        weight_matrix = weights.unsqueeze(0).expand(n_points, -1)
                    else:
                        n_dirs = self.config.n_angular_samples
                        phi = torch.rand(n_points, n_dirs, device=device, dtype=dtype) * 2 * math.pi
                        cos_theta = 1 - 2 * torch.rand(n_points, n_dirs, device=device, dtype=dtype)
                        sin_theta = torch.sqrt(torch.clamp(1 - cos_theta**2, min=0.0))
                        directions = torch.stack(
                            [
                                sin_theta * torch.cos(phi),
                                sin_theta * torch.sin(phi),
                                cos_theta,
                            ],
                            dim=-1,
                        )
                        weight_matrix = torch.full(
                            (n_points, n_dirs),
                            1.0 / n_dirs,
                            device=device,
                            dtype=dtype,
                        )

                    # Normalize weights to sum to 1, avoiding division by zero
                    weight_sums = weight_matrix.sum(dim=1, keepdim=True).clamp_min(1e-8)
                    weight_matrix = weight_matrix / weight_sums

                    origins = eval_pts.unsqueeze(1).expand(-1, n_dirs, -1).reshape(-1, 3)
                    dirs = directions.reshape(-1, 3)

                    if __debug__:
                        assert torch.allclose(
                            dirs.norm(dim=-1),
                            torch.ones_like(dirs[:, 0]),
                            atol=1e-5,
                        ), "Ray directions must remain normalized."

                    box_min = torch.zeros(3, device=device)
                    box_max = torch.tensor(
                        self.config.grid_shape, device=device, dtype=torch.float32
                    )
                    exit_dist = ray_box_exit_distance(origins, dirs, box_min, box_max)
                    max_distance = exit_dist.max().item()

                    radiance, trans = self.ray_marcher.march(
                        origins,
                        dirs,
                        kappa_field,
                        source_field,
                        max_distance=max_distance,
                    )
                    radiance = radiance.reshape(
                        n_points, n_dirs, self.config.n_stokes, self.config.n_bands
                    )
                    trans = trans.reshape(
                        n_points, n_dirs, self.config.n_stokes, self.config.n_bands
                    )
                    weights_expanded = weight_matrix.unsqueeze(-1).unsqueeze(-1)
                    radiance = torch.sum(radiance * weights_expanded, dim=1)
                    trans = torch.sum(trans * weights_expanded, dim=1)
                    radiance = self.apply_boundary_conditions(
                        radiance.unsqueeze(0), eval_pts.unsqueeze(0)
                    ).squeeze(0)
                    radiance_batches.append((radiance, trans))

                if not radiance_batches:
                    raise StratusPhysicsError("No radiance samples were produced for the batch.")

                radiance_list: list[torch.Tensor] = [r for r, _ in radiance_batches]
                trans_list: list[torch.Tensor] = [t for _, t in radiance_batches]
                reference_shape = radiance_list[0].shape
                for tensor in radiance_list[1:]:
                    if tensor.shape != reference_shape:
                        raise StratusPhysicsError(
                            "Inconsistent radiance tensor shapes across the batch."
                        )

                radiance_stack = torch.stack(radiance_list, dim=0).contiguous()
                trans_stack = torch.stack(trans_list, dim=0).contiguous()
                result = {"radiance": radiance_stack, "transmittance": trans_stack}
                result["evaluation_points"] = evaluation_points

        if compute_physics_loss:
            result.update(self.compute_physics_constraints(result["radiance"], kappa, source))
        return result


__all__ = ["StratusRadianceModel"]
