"""One-dimensional Matern-2.5 Gaussian-process samples for diagnostics."""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor


def matern25_kernel(x: np.ndarray, y: np.ndarray, length_scale: float = 0.1, sigma: float = 1.0) -> np.ndarray:
    """Evaluate the Matern-2.5 covariance matrix ``K(x, y)``."""
    d = np.abs(x[:, None] - y[None, :])
    r = d / length_scale
    factor = math.sqrt(5.0) * r
    return (sigma ** 2) * (1.0 + factor + factor ** 2 / 3.0) * np.exp(-factor)


class MaternGP1D:
    """Cached sampler from a 1D Matern-2.5 Gaussian process on [0, 1].

    Args:
        dim: number of equally-spaced grid points in [0, 1].
        length_scale: kernel bandwidth.
        sigma: kernel variance (set the prior covariance scale).
        seed: optional RNG seed for reproducible Cholesky.
    """

    def __init__(self, dim: int = 64, length_scale: float = 0.1, sigma: float = 1.0, seed: int | None = None) -> None:
        self.dim = dim
        grid = np.linspace(0.0, 1.0, dim)
        K = matern25_kernel(grid, grid, length_scale=length_scale, sigma=sigma)
        # Jitter for numerical stability of the Cholesky factor.
        K += 1e-6 * np.eye(dim)
        L = np.linalg.cholesky(K).astype(np.float32)
        self._L = torch.from_numpy(L)
        self.length_scale = length_scale
        self.kernel_sigma = sigma
        self._gen: torch.Generator | None
        if seed is not None:
            self._gen = torch.Generator().manual_seed(seed)
        else:
            self._gen = None

    @property
    def trace(self) -> float:
        """Return the trace of the discretized covariance matrix."""
        return float((self._L @ self._L.T).diagonal().sum().item())

    def sample(self, batch: int, device: str | torch.device = "cpu") -> Tensor:
        """Draw a batch of GP samples of shape (batch, dim)."""
        if self._gen is not None:
            z = torch.randn(batch, self.dim, generator=self._gen)
        else:
            z = torch.randn(batch, self.dim)
        L = self._L.to(device)
        return (z.to(device) @ L.T)


class GPDataset(torch.utils.data.Dataset):
    """Lazy iterable wrapper around `MaternGP1D` for use with DataLoader."""

    def __init__(self, gp: MaternGP1D, size: int = 50_000) -> None:
        self.gp = gp
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Tensor:  # noqa: ARG002 - iid
        return self.gp.sample(1)[0]
