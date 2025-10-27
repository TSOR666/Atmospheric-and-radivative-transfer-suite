from __future__ import annotations

import numpy as np
import torch


class MuellerMatrixPolarization:
    """Mueller matrix based polarization handling."""

    @staticmethod
    def rayleigh_mueller_matrix(cos_theta: torch.Tensor) -> torch.Tensor:
        device = cos_theta.device
        M = torch.zeros(*cos_theta.shape, 4, 4, device=device)
        cos2 = cos_theta**2
        sin2 = 1 - cos2
        M[..., 0, 0] = 0.75 * (1 + cos2)
        M[..., 0, 1] = M[..., 1, 0] = -0.75 * sin2
        M[..., 1, 1] = 0.75 * (cos2 + 1)
        M[..., 2, 2] = M[..., 3, 3] = 1.5 * cos_theta
        return M

    @staticmethod
    def mie_mueller_matrix(cos_theta: torch.Tensor, S1: torch.Tensor, S2: torch.Tensor) -> torch.Tensor:
        device = cos_theta.device
        M = torch.zeros(*cos_theta.shape, 4, 4, device=device)
        S11 = 0.5 * (torch.abs(S2) ** 2 + torch.abs(S1) ** 2)
        S12 = 0.5 * (torch.abs(S2) ** 2 - torch.abs(S1) ** 2)
        S33 = (S2 * S1.conj()).real
        S34 = (S2 * S1.conj()).imag
        M[..., 0, 0] = S11
        M[..., 0, 1] = M[..., 1, 0] = S12
        M[..., 1, 1] = S11
        M[..., 2, 2] = S33
        M[..., 2, 3] = -S34
        M[..., 3, 2] = S34
        M[..., 3, 3] = S33
        return M

    @staticmethod
    def apply_mueller_scattering(
        stokes_in: torch.Tensor,
        mueller_matrix: torch.Tensor,
        omega: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = stokes_in.shape[0]
        n_stokes = stokes_in.shape[1]
        if n_stokes == 1:
            return stokes_in

        original_shape = stokes_in.shape
        spatial_dims = original_shape[3:]
        n_angles = original_shape[2]
        n_positions = int(np.prod(spatial_dims))

        stokes_flat = stokes_in.reshape(batch_size, 4, n_angles, n_positions)
        stokes_perm = stokes_flat.permute(0, 2, 3, 1)  # (B, A, P, 4)

        if mueller_matrix.dim() == 3:
            mueller_expanded = (
                mueller_matrix.unsqueeze(0)
                .unsqueeze(2)
                .expand(batch_size, n_angles, n_positions, 4, 4)
            )
        else:
            mueller_expanded = mueller_matrix.reshape(batch_size, n_angles, -1, 4, 4)
            if mueller_expanded.shape[2] == 1 and n_positions > 1:
                mueller_expanded = mueller_expanded.expand(-1, -1, n_positions, -1, -1)

        scattered = torch.matmul(
            mueller_expanded,
            stokes_perm.unsqueeze(-1)
        ).squeeze(-1)

        omega_flat = omega.reshape(batch_size, 1, n_positions, 1)
        scattered = scattered * omega_flat

        scattered = scattered.permute(0, 3, 1, 2).reshape(original_shape)
        return scattered

    @staticmethod
    def rotate_stokes(stokes: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        if stokes.shape[1] == 1:
            return stokes
        cos2 = torch.cos(2 * angle)
        sin2 = torch.sin(2 * angle)
        I = stokes[:, 0]
        Q = stokes[:, 1]
        U = stokes[:, 2]
        V = stokes[:, 3]
        Q_rot = Q * cos2 + U * sin2
        U_rot = -Q * sin2 + U * cos2
        return torch.stack([I, Q_rot, U_rot, V], dim=1)


__all__ = ["MuellerMatrixPolarization"]
