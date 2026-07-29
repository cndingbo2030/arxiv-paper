"""Smoke tests for the score training pipeline."""
from __future__ import annotations

import torch

from score_bip.data import MaternGP1D
from score_bip.data.gp_1d import GPDataset
from score_bip.priors import ScoreMLP1D
from score_bip.training import ScoreLightningModule, dsm_loss_vp, marginal_alpha_sigma


def test_marginal_shapes() -> None:
    t = torch.rand(16)
    alpha, sigma = marginal_alpha_sigma(t)
    assert alpha.shape == sigma.shape == (16,)
    assert (alpha > 0).all()
    assert (sigma > 0).all()


def test_gp_sample_finite_trace() -> None:
    gp = MaternGP1D(dim=16, length_scale=0.1, sigma=1.0, seed=0)
    u = gp.sample(8)
    assert u.shape == (8, 16)
    assert torch.isfinite(u).all()
    assert gp.trace > 0


def test_gp_dataset_iter() -> None:
    gp = MaternGP1D(dim=8, length_scale=0.1, sigma=1.0, seed=0)
    ds = GPDataset(gp, size=4)
    sample = ds[0]
    assert sample.shape == (8,)


def test_dsm_loss_finite_and_decreasing() -> None:
    """Verify DSM loss is well-defined and short SGD reduces it."""
    torch.manual_seed(0)
    net = ScoreMLP1D(dim=16, hidden=32, depth=2, lipschitz=False)
    gp = MaternGP1D(dim=16, length_scale=0.1, sigma=1.0, seed=0)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    initial_loss = None
    for step in range(20):
        u0 = gp.sample(64)
        loss = dsm_loss_vp(net, u0)
        if step == 0:
            initial_loss = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert torch.isfinite(loss)
    assert loss.item() < initial_loss, "DSM loss did not decrease after 20 SGD steps"


def test_lightning_module_forward() -> None:
    net = ScoreMLP1D(dim=8, hidden=16, depth=2)
    module = ScoreLightningModule(net, ema_decay=0.9)
    t = torch.rand(4)
    u = torch.randn(4, 8)
    out = module(t, u)
    assert out.shape == u.shape
    ema_out = module.ema_forward(t, u)
    assert ema_out.shape == u.shape
