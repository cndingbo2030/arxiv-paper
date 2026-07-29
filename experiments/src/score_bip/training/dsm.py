"""Denoising score-matching loss for a finite-dimensional VP/OU process.

The forward noising process is
    dU_s = -1/2 U_s ds + dW_s,    U_0 ~ pi_data,
whose marginal at time t is
    U_t = alpha(t) U_0 + sigma(t) eps,    eps ~ N(0, I),
with
    alpha(t) = exp(-t/2),
    sigma(t) = sqrt(1 - exp(-t)).

The conditional score is
    nabla_u log p_{t|0}(u_t | u_0) = -eps / sigma(t).

The implementation minimizes

    L(theta) = E_t E_u_0 E_eps [ lambda(t) * || s_theta(t, u_t) * sigma(t) + eps ||^2 ],

with lambda(t) = 1, weighting the prediction in epsilon space rather than
score space. This finite-dimensional training loss is not the pathwise
Cameron--Martin drift error in the manuscript.
"""
from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor


def marginal_alpha_sigma(t: Tensor) -> tuple[Tensor, Tensor]:
    """Marginal alpha(t), sigma(t) of the OU SDE."""
    alpha = torch.exp(-0.5 * t)
    sigma = torch.sqrt(1.0 - torch.exp(-t))
    return alpha, sigma


def dsm_loss_vp(
    score_net: Callable[[Tensor, Tensor], Tensor],
    u0: Tensor,
    t_eps: float = 1.0e-3,
    t_max: float = 1.0,
) -> Tensor:
    """Denoising score matching loss for the VP/OU SDE.

    Args:
        score_net: function (t: (B,), u: (B, D)) -> (B, D) returning the
            current estimate of nabla_u log p_t(u).
        u0: clean prior samples, shape (B, D).
        t_eps: lower bound on t to avoid the singular sigma(0) = 0.
        t_max: upper bound on t.

    Returns:
        Scalar loss tensor.
    """
    b, _ = u0.shape
    t = torch.rand(b, device=u0.device) * (t_max - t_eps) + t_eps
    alpha, sigma = marginal_alpha_sigma(t)
    eps = torch.randn_like(u0)
    u_t = alpha.unsqueeze(-1) * u0 + sigma.unsqueeze(-1) * eps
    score_pred = score_net(t, u_t)
    target_eps = -score_pred * sigma.unsqueeze(-1)
    return (target_eps - eps).pow(2).mean()
