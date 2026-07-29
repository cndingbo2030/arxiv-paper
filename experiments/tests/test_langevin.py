"""Sanity tests for the preconditioned Langevin posterior sampler."""
from __future__ import annotations

import pytest
import torch

from score_bip.forward import Identity1D
from score_bip.posterior import PreconditionedLangevin


def gaussian_prior_score(u: torch.Tensor) -> torch.Tensor:
    """Score of N(0, I): grad log p(u) = -u."""
    return -u


def test_langevin_gaussian_posterior_mean() -> None:
    """With G=I and p_0=N(0,I), posterior mean is y/(1+sigma^2)."""
    torch.manual_seed(0)
    dim = 8
    sigma_obs = 0.5
    y = torch.full((dim,), 1.2)
    forward = Identity1D()

    smpl = PreconditionedLangevin(
        score_fn=gaussian_prior_score,
        forward=forward,
        sigma_obs=sigma_obs,
        step_size=8.0e-3,
        n_steps=6000,
    )

    n_chains = 128
    endpoints: list[torch.Tensor] = []
    for _ in range(n_chains):
        u0 = torch.randn(dim)
        endpoints.append(smpl.sample(y, u0))
    mean_u = torch.stack(endpoints).mean(dim=0)

    expected = y / (1.0 + sigma_obs**2)
    assert torch.allclose(mean_u, expected, atol=0.12, rtol=0.05)


def test_preconditioner_requires_matching_noise_transform() -> None:
    """A covariance preconditioner cannot also stand in for its square root."""
    forward = Identity1D()
    with pytest.raises(ValueError, match="noise_transform"):
        PreconditionedLangevin(
            score_fn=gaussian_prior_score,
            forward=forward,
            precond=lambda u: 2.0 * u,
        )
