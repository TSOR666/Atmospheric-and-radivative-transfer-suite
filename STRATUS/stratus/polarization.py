"""
Polarization utilities and Mueller-matrix helpers for STRATUS.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor

try:  # SciPy provides high-fidelity special functions when available.
    from scipy import special as sp_special
except Exception:  # pragma: no cover - SciPy is optional.
    sp_special = None

logger = logging.getLogger("stratus.polarization")


def _spherical_bessel_sequences(order: int, x: Tensor) -> tuple[Tensor, Tensor]:
    """
    Return psi_n(x) and psi'_n(x) for n = 0..order using recurrence relations.
    """
    device = x.device
    dtype = x.dtype

    if sp_special is not None:
        x_cpu = x.detach().cpu()
        x_np = x_cpu.numpy()
        psi = []
        psi_prime = []
        for n in range(0, order + 1):
            jn = sp_special.spherical_jn(n, x_np)
            jn_prime = sp_special.spherical_jn(n, x_np, derivative=True)
            psi_n = torch.from_numpy(x_np * jn).to(device=device, dtype=dtype)
            psi_prime_n = torch.from_numpy(jn + x_np * jn_prime).to(device=device, dtype=dtype)
            psi.append(psi_n)
            psi_prime.append(psi_prime_n)
        return torch.stack(psi, dim=0), torch.stack(psi_prime, dim=0)

    psi = []
    psi_prime = []
    psi_0 = torch.sin(x)
    psi.append(psi_0)
    psi_prime.append(torch.cos(x))

    prev = psi_0
    for n in range(1, order + 1):
        jn = torch.special.spherical_jn(n, x)
        psi_n = x * jn
        psi_nm1 = psi[-1] if n > 1 else prev
        psi_prime_n = psi_nm1 - (n + 1) / x.clamp_min(1e-12) * psi_n
        psi.append(psi_n)
        psi_prime.append(psi_prime_n)

    return torch.stack(psi, dim=0), torch.stack(psi_prime, dim=0)


def _spherical_hankel_sequences(order: int, x: Tensor) -> tuple[Tensor, Tensor]:
    """
    Return xi_n(x) and xi'_n(x) for n = 0..order.
    """
    device = x.device
    dtype = x.dtype
    complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128

    if sp_special is not None:
        x_cpu = x.detach().cpu()
        x_np = x_cpu.numpy()
        xi = []
        xi_prime = []
        for n in range(0, order + 1):
            jn = sp_special.spherical_jn(n, x_np)
            jn_prime = sp_special.spherical_jn(n, x_np, derivative=True)
            yn = sp_special.spherical_yn(n, x_np)
            yn_prime = sp_special.spherical_yn(n, x_np, derivative=True)

            psi_n = torch.from_numpy(x_np * jn).to(device=device, dtype=dtype)
            psi_prime_n = torch.from_numpy(jn + x_np * jn_prime).to(device=device, dtype=dtype)
            chi_n = torch.from_numpy(-x_np * yn).to(device=device, dtype=dtype)
            chi_prime_n = torch.from_numpy(-(yn + x_np * yn_prime)).to(device=device, dtype=dtype)

            xi_n = psi_n.to(complex_dtype) - 1j * chi_n.to(complex_dtype)
            xi_prime_n = psi_prime_n.to(complex_dtype) - 1j * chi_prime_n.to(complex_dtype)

            xi.append(xi_n)
            xi_prime.append(xi_prime_n)

        return torch.stack(xi, dim=0), torch.stack(xi_prime, dim=0)

    j_seq, _ = _spherical_bessel_sequences(order, x)
    y_seq = []
    for n in range(0, order + 1):
        y_seq.append(torch.special.spherical_yn(n, x))
    y_seq = torch.stack(y_seq, dim=0)
    chi = -x * y_seq
    xi = j_seq.to(complex_dtype) - 1j * chi.to(complex_dtype)

    xi_prime = []
    for n in range(0, order + 1):
        if n == 0:
            xi_prime.append((torch.cos(x) - 1j * torch.sin(x)).to(complex_dtype))
        else:
            term = xi[n - 1] - (n + 1) / x.clamp_min(1e-12) * xi[n]
            xi_prime.append(term)
    xi_prime = torch.stack(xi_prime, dim=0)
    return xi, xi_prime


class MuellerMatrix:
    """
    Collection of Mueller-matrix builders for common radiative processes.
    """

    def __init__(self, device: torch.device | None = None) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.dtype = torch.float32

    # ---------------------------------------------------------------- identity
    def identity(self, *shape: int) -> torch.Tensor:
        base = torch.eye(4, device=self.device)
        if not shape:
            return base
        expanded = base.expand(*shape, 4, 4)
        return expanded.clone()

    # ----------------------------------------------------------- absorption
    def isotropic_absorption(self, kappa: torch.Tensor) -> torch.Tensor:
        """
        Return a diagonal Mueller matrix modelling isotropic extinction.
        """
        kappa = torch.nan_to_num(kappa.to(self.device), nan=0.0, posinf=1e6, neginf=-1e6)
        diag = -kappa.unsqueeze(-1).unsqueeze(-1)
        matrix = self.identity(*kappa.shape)
        for i in range(4):
            matrix[..., i, i] = diag[..., 0, 0]
        return matrix

    # -------------------------------------------------------- rayleigh
    def rayleigh_scattering(
        self,
        theta: torch.Tensor,
        phi: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Angular Mueller matrix for Rayleigh scattering.
        """
        theta = torch.nan_to_num(theta.to(self.device))
        if phi is not None:
            phi = torch.nan_to_num(phi.to(self.device))

        shape = theta.shape
        matrix = torch.zeros((*shape, 4, 4), device=self.device)

        cos_theta = torch.cos(theta)
        cos2_theta = cos_theta**2

        matrix[..., 0, 0] = 0.75 * (1.0 + cos2_theta)
        matrix[..., 1, 1] = 0.75 * (1.0 + cos2_theta)
        off = 0.75 * (cos2_theta - 1.0)
        matrix[..., 0, 1] = off
        matrix[..., 1, 0] = off

        if phi is not None:
            sin_2phi = torch.sin(2.0 * phi)
            cos_2phi = torch.cos(2.0 * phi)
            matrix[..., 2, 2] = 1.5 * cos_theta * cos_2phi
            matrix[..., 3, 3] = 1.5 * cos_theta * sin_2phi
            matrix[..., 2, 3] = 0.75 * cos_theta * sin_2phi * cos_2phi
            matrix[..., 3, 2] = matrix[..., 2, 3]
        else:
            matrix[..., 2, 2] = 0.75 * cos_theta
            matrix[..., 3, 3] = 0.75 * cos_theta
        return matrix

    # --------------------------------------------------------- heney greenstein
    def henyey_greenstein(
        self,
        g: torch.Tensor,
        theta: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mueller matrix approximation using Henyey-Greenstein phase functions.
        """
        g = g.to(self.device)
        theta = theta.to(self.device)
        cos_theta = torch.cos(theta)
        denom = (1 + g**2 - 2 * g * cos_theta).clamp_min(1e-6)
        phase = (1 - g**2) / (denom**1.5)
        matrix = self.identity(*theta.shape)
        matrix[..., 0, 0] = phase
        matrix[..., 1, 1] = phase * (1 - g**2)
        matrix[..., 2, 2] = phase * (1 - g**2)
        matrix[..., 3, 3] = phase * (1 - g**2)
        return matrix

    # ------------------------------------------------------------ fresnel
    def fresnel_reflection(
        self,
        n1: float,
        n2: float,
        incidence: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mueller matrix for Fresnel reflection at dielectric interface.
        """
        incidence = incidence.to(self.device)
        cos_i = torch.cos(incidence).clamp(-1.0, 1.0)
        sin_t2 = (n1 / n2) ** 2 * (1 - cos_i**2)
        total = sin_t2 > 1.0
        cos_t = torch.sqrt(torch.clamp(1 - sin_t2, min=0.0))

        rs = ((n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)) ** 2
        rp = ((n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t)) ** 2

        rs[total] = 1.0
        rp[total] = 1.0

        matrix = torch.zeros((*incidence.shape, 4, 4), device=self.device)
        matrix[..., 0, 0] = 0.5 * (rs + rp)
        matrix[..., 1, 1] = 0.5 * (rs - rp)
        matrix[..., 2, 2] = torch.sqrt(rs * rp)
        matrix[..., 3, 3] = matrix[..., 2, 2]
        return matrix

    # ---------------------------------------------------------- mie scattering
    def mie_scattering(
        self,
        size_parameter: torch.Tensor,
        theta: torch.Tensor,
        refractive_index: float,
        max_order: int = 8,
    ) -> torch.Tensor:
        """
        Compute Mueller matrix using Lorenz-Mie theory for non-absorbing spheres.
        """
        x = size_parameter.to(self.device).clamp_min(1e-6)
        m = torch.tensor(refractive_index, device=self.device, dtype=x.dtype)

        theta = theta.to(self.device)
        cos_theta = torch.cos(theta)

        try:
            order = int(max_order)
            psi_x, psi_prime_x = _spherical_bessel_sequences(order, x)
            psi_mx, psi_prime_mx = _spherical_bessel_sequences(order, m * x)
            xi_x, xi_prime_x = _spherical_hankel_sequences(order, x)

            def _stabilize(value: Tensor, epsilon: float) -> Tensor:
                magnitude = value.abs()
                mask = magnitude < epsilon
                if not torch.any(mask):
                    return value

                direction = torch.sgn(value)
                direction = torch.where(
                    mask & (direction == 0), torch.ones_like(direction), direction
                )
                eps_tensor = value.new_tensor(epsilon)
                replacement = direction * eps_tensor
                return torch.where(mask, replacement, value)

            a_coeffs = []
            b_coeffs = []
            for n in range(1, order + 1):
                numerator_a = m * psi_mx[n] * psi_prime_x[n] - psi_x[n] * psi_prime_mx[n]
                denominator_a = m * psi_mx[n] * xi_prime_x[n] - xi_x[n] * psi_prime_mx[n]
                eps = 1e-10
                denominator_a = _stabilize(denominator_a, eps)
                numerator_b = psi_mx[n] * psi_prime_x[n] - m * psi_x[n] * psi_prime_mx[n]
                denominator_b = psi_mx[n] * xi_prime_x[n] - m * xi_x[n] * psi_prime_mx[n]
                denominator_b = _stabilize(denominator_b, eps)
                a_coeffs.append(numerator_a / denominator_a)
                b_coeffs.append(numerator_b / denominator_b)

            a = torch.stack(a_coeffs, dim=0)
            b = torch.stack(b_coeffs, dim=0)

            pi_prev = torch.zeros_like(cos_theta)
            pi_curr = torch.ones_like(cos_theta)
            S1 = torch.zeros_like(cos_theta, dtype=torch.complex64)
            S2 = torch.zeros_like(cos_theta, dtype=torch.complex64)

            for n in range(1, order + 1):
                factor = (2 * n + 1) / (n * (n + 1))
                tau_n = n * cos_theta * pi_curr - (n + 1) * pi_prev
                S1 = S1 + factor * (a[n - 1] * pi_curr + b[n - 1] * tau_n)
                S2 = S2 + factor * (a[n - 1] * tau_n + b[n - 1] * pi_curr)

                if n < order:
                    pi_next = (2 * n + 1) / (n + 1) * cos_theta * pi_curr - n / (n + 1) * pi_prev
                    pi_prev, pi_curr = pi_curr, pi_next

            S1_mag = torch.abs(S1) ** 2
            S2_mag = torch.abs(S2) ** 2
            I = 0.5 * (S1_mag + S2_mag)
            Q = 0.5 * (S2_mag - S1_mag)
            U = S1.real * S2.real + S1.imag * S2.imag
            V = S1.real * S2.imag - S1.imag * S2.real

            matrix = torch.zeros((*theta.shape, 4, 4), device=self.device, dtype=theta.dtype)
            matrix[..., 0, 0] = I
            matrix[..., 0, 1] = Q
            matrix[..., 1, 0] = Q
            matrix[..., 1, 1] = I
            matrix[..., 2, 2] = U
            matrix[..., 2, 3] = -V
            matrix[..., 3, 2] = V
            matrix[..., 3, 3] = U
            return matrix
        except Exception as exc:
            logger.warning("Mie scattering failed (%s). Falling back to Henyey-Greenstein.", exc)
            g = torch.zeros_like(theta)
            return self.henyey_greenstein(g, theta)

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def apply(matrix: torch.Tensor, stokes: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...ij,...j->...i", matrix, stokes)

    @staticmethod
    def normalize(stokes: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        I = stokes[..., 0:1]
        pol = torch.norm(stokes[..., 1:], dim=-1, keepdim=True)
        mask = pol > (I + eps)
        stokes = stokes.clone()
        scale = torch.ones_like(stokes[..., 1:])
        scale = scale * ((I + eps) / (pol + eps))
        scale = torch.where(mask.expand_as(scale), scale, torch.ones_like(scale))
        stokes[..., 1:] = stokes[..., 1:] * scale
        stokes[..., 0] = stokes[..., 0].clamp_min(0.0)
        return stokes


__all__ = ["MuellerMatrix"]
