"""Closed-form Hellinger distance between nondegenerate multivariate Gaussians."""
from __future__ import annotations

import numpy as np


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


def _cholesky_logdet(matrix: np.ndarray) -> float:
    factor = np.linalg.cholesky(_symmetrize(matrix))
    return 2.0 * float(np.log(np.diag(factor)).sum())


def hellinger_squared_mvn(
    mean0: np.ndarray,
    cov0: np.ndarray,
    mean1: np.ndarray,
    cov1: np.ndarray,
) -> float:
    """Squared Hellinger distance :math:`H^2(N_0,N_1)` for full-rank Gaussian laws.

    Uses the standard Bhattacharyya parametrisation
    :math:`H^2 = 1 - BC`, see e.g. the explicit MVN formula in standard references.
    """
    m0 = np.asarray(mean0, dtype=np.float64).reshape(-1)
    m1 = np.asarray(mean1, dtype=np.float64).reshape(-1)
    s0 = np.asarray(cov0, dtype=np.float64)
    s1 = np.asarray(cov1, dtype=np.float64)
    if m0.shape != m1.shape or s0.shape != s1.shape or s0.shape[0] != m0.shape[0]:
        raise ValueError("incompatible shapes")

    s0 = _symmetrize(s0)
    s1 = _symmetrize(s1)
    s_bar = 0.5 * (s0 + s1)
    diff = m0 - m1
    quad = 0.125 * float(diff @ np.linalg.solve(s_bar, diff))

    try:
        ld0 = _cholesky_logdet(s0)
        ld1 = _cholesky_logdet(s1)
        ldb = _cholesky_logdet(s_bar)
    except np.linalg.LinAlgError as exc:
        raise ValueError("covariance must be positive definite") from exc

    db = quad + 0.5 * ldb - 0.25 * ld0 - 0.25 * ld1
    bc = float(np.exp(min(0.0, -db)))
    return float(np.clip(1.0 - bc, 0.0, 1.0))


def gaussian_posterior_linear_gaussian(
    y: np.ndarray,
    g: np.ndarray,
    prior_cov: np.ndarray,
    sigma_obs: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Posterior mean and covariance for scalar Gaussian noise and prior :math:`N(0,K)`.

    Model: :math:`y = Gu + \\sigma \\varepsilon`, :math:`u \\sim N(0,K)`, zero mean prior.
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    g = np.asarray(g, dtype=np.float64)
    k = _symmetrize(np.asarray(prior_cov, dtype=np.float64))
    d = y.shape[0]
    if g.shape != (d, d) or k.shape != (d, d):
        raise ValueError("y, G, K must be compatible d×d")
    if sigma_obs <= 0:
        raise ValueError("sigma_obs must be positive")

    # Whiten u = Lz.  Solving in z-coordinates avoids explicitly inverting
    # an ill-conditioned prior covariance.
    try:
        prior_factor = np.linalg.cholesky(k)
    except np.linalg.LinAlgError as exc:
        raise ValueError("prior_cov must be positive definite") from exc

    sigma2 = float(sigma_obs) ** 2
    whitened_forward = g @ prior_factor
    precision_z = np.eye(d) + whitened_forward.T @ whitened_forward / sigma2
    rhs_z = whitened_forward.T @ y / sigma2
    mean_z = np.linalg.solve(precision_z, rhs_z)
    covariance_z = np.linalg.solve(precision_z, np.eye(d))

    mean_post = prior_factor @ mean_z
    cov_post = _symmetrize(prior_factor @ covariance_z @ prior_factor.T)
    return mean_post, cov_post
