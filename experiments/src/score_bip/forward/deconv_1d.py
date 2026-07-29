"""1D Gaussian-blur forward operator for deconvolution experiments."""
from __future__ import annotations

import math

import torch
from torch import nn


class Identity1D(nn.Module):
    """Identity forward on R^d (full observation), ``G u = u``.

    Useful for sanity-checking the Langevin sampler against Gaussian
    conjugate posteriors.
    """

    def forward(self, u: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return u

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return y


class GaussianBlur1D(nn.Module):
    """Circular Gaussian convolution.

    y(x) = (k * u)(x), with k a discretized Gaussian of bandwidth `sigma_kernel`.
    Implemented in the Fourier domain to keep gradients exact and to allow
    spectral-truncation experiments cleanly.
    """

    def __init__(self, dim: int = 64, sigma_kernel: float = 0.05) -> None:
        super().__init__()
        self.dim = dim
        # frequency-domain kernel: exp(-2 pi^2 sigma^2 xi^2)
        freqs = torch.fft.fftfreq(dim) * dim
        decay = torch.exp(-2.0 * math.pi**2 * (sigma_kernel**2) * freqs**2)
        self.register_buffer("filter_fft", decay)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        U = torch.fft.fft(u, dim=-1)
        Y = U * self.filter_fft
        return torch.fft.ifft(Y, dim=-1).real

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        # Self-adjoint (real, symmetric filter), but keep an explicit method
        # so theory.discretization scripts can call .adjoint(...) explicitly.
        return self.forward(y)
