"""Numerical metrics used in experiments."""
from __future__ import annotations

import torch


def hellinger_mc(
    log_density_p: torch.Tensor,
    log_density_q: torch.Tensor,
) -> torch.Tensor:
    """Monte-Carlo estimate of squared Hellinger distance.

    Given log-densities log p(x_i), log q(x_i) at samples drawn from some
    common dominating measure r (e.g. one of p or q), the squared Hellinger
    distance is

        H^2(p,q) = 1 - E_r[ sqrt(p/r) sqrt(q/r) ].

    Here we assume samples are from p (i.e. r = p) and return
    1 - mean(sqrt(q / p)).
    """
    if log_density_p.shape != log_density_q.shape:
        raise ValueError("log_density_p and log_density_q must have the same shape")
    log_ratio = 0.5 * (log_density_q - log_density_p)
    return 1.0 - log_ratio.exp().mean()


def empirical_contraction_radius(
    samples: torch.Tensor,
    truth: torch.Tensor,
    quantile: float = 0.95,
) -> torch.Tensor:
    """Return the radius of a ball around `truth` containing `quantile` of samples."""
    if samples.ndim != 2 or truth.ndim != 1:
        raise ValueError("Expected samples of shape (n, d) and truth of shape (d,)")
    dists = (samples - truth).norm(dim=-1)
    return torch.quantile(dists, quantile)
