"""Finite-dimensional time-conditioned MLP used by the pilot scripts.

Spectral normalization can be enabled for its main linear layers. This module
does not include the covariance factorization in the manuscript and does not,
by itself, certify any infinite-dimensional drift assumption.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm


class ScoreMLP1D(nn.Module):
    """Time-conditioned MLP score network with controllable Lipschitz constant.

    Args:
        dim: Spatial dimension of the discretized 1D function (#pixels).
        hidden: Width of hidden layers.
        depth: Number of hidden layers.
        lipschitz: If True, wrap the main projection and residual-block
            linear layers with ``spectral_norm``.
    """

    def __init__(
        self,
        dim: int = 64,
        hidden: int = 256,
        depth: int = 4,
        lipschitz: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.time_embed = nn.Linear(1, hidden)

        def wrap(layer: nn.Linear) -> nn.Module:
            return spectral_norm(layer) if lipschitz else layer

        self.in_proj = wrap(nn.Linear(dim + hidden, hidden))
        self.blocks = nn.ModuleList(
            [wrap(nn.Linear(hidden, hidden)) for _ in range(depth)]
        )
        self.out_proj = wrap(nn.Linear(hidden, dim))
        self.act = nn.SiLU()

    def forward(self, t: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Score evaluation s_theta(t, u).

        Args:
            t: shape (B,) time-step in [0, T].
            u: shape (B, dim).

        Returns:
            shape (B, dim).
        """
        te = self.act(self.time_embed(t.unsqueeze(-1)))
        h = torch.cat([u, te], dim=-1)
        h = self.act(self.in_proj(h))
        for blk in self.blocks:
            h = self.act(blk(h)) + h
        return self.out_proj(h)
