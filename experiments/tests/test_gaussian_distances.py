"""Tests for the closed-form Gaussian-distance utilities."""
from __future__ import annotations

import numpy as np
import pytest

from score_bip.utils.gaussian_distances import (
    gaussian_posterior_linear_gaussian,
    hellinger_squared_mvn,
)


def test_hellinger_identical_zero() -> None:
    d = 5
    m = np.zeros(d)
    s = np.eye(d)
    h2 = hellinger_squared_mvn(m, s, m, s)
    assert h2 == pytest.approx(0.0, abs=1e-9)


def test_full_truncation_matches_reference_posterior() -> None:
    """Spectral truncation with all modes equals prior → same posterior."""
    rng = np.random.default_rng(2)
    dim = 12
    k = np.eye(dim) + 0.5 * np.ones((dim, dim))
    k = 0.5 * (k + k.T)
    g = np.eye(dim)
    z = rng.standard_normal(dim)
    u = np.linalg.cholesky(k) @ z
    y = g @ u + 0.2 * rng.standard_normal(dim)
    sigma = 0.2

    m0, s0 = gaussian_posterior_linear_gaussian(y, g, k, sigma)
    m1, s1 = gaussian_posterior_linear_gaussian(y, g, k, sigma)

    h2 = hellinger_squared_mvn(m0, s0, m1, s1)
    assert h2 == pytest.approx(0.0, abs=1e-8)


def test_ill_conditioned_prior_produces_positive_definite_posterior() -> None:
    dim = 10
    prior_eigenvalues = np.geomspace(1.0, 1.0e-10, dim)
    prior_cov = np.diag(prior_eigenvalues)
    g = np.eye(dim)
    y = np.linspace(-1.0, 1.0, dim)

    mean, covariance = gaussian_posterior_linear_gaussian(
        y,
        g,
        prior_cov,
        sigma_obs=0.1,
    )

    np.linalg.cholesky(covariance)
    assert np.all(np.isfinite(mean))
    assert hellinger_squared_mvn(mean, covariance, mean, covariance) == pytest.approx(
        0.0,
        abs=1e-8,
    )
