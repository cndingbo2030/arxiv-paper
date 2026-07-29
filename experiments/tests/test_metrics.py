"""Smoke tests for utilities and shape contracts."""
from __future__ import annotations

import torch

from score_bip.forward import GaussianBlur1D
from score_bip.priors import ScoreMLP1D
from score_bip.utils.metrics import empirical_contraction_radius, hellinger_mc


def test_hellinger_self_zero() -> None:
    log_p = torch.zeros(1000)
    assert torch.allclose(hellinger_mc(log_p, log_p), torch.tensor(0.0), atol=1e-6)


def test_contraction_radius_shape() -> None:
    samples = torch.randn(2000, 64)
    truth = torch.zeros(64)
    r = empirical_contraction_radius(samples, truth, quantile=0.95)
    assert r.ndim == 0
    assert r > 0


def test_score_mlp_shape() -> None:
    net = ScoreMLP1D(dim=64, hidden=128, depth=2)
    t = torch.rand(8)
    u = torch.randn(8, 64)
    out = net(t, u)
    assert out.shape == u.shape


def test_blur_self_adjoint() -> None:
    op = GaussianBlur1D(dim=64, sigma_kernel=0.05)
    u = torch.randn(4, 64)
    v = torch.randn(4, 64)
    lhs = (op(u) * v).sum()
    rhs = (u * op.adjoint(v)).sum()
    assert torch.allclose(lhs, rhs, atol=1e-5)
