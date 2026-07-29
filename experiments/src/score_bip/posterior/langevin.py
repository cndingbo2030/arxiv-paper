"""Preconditioned Langevin dynamics with score-based prior.

Implements preconditioned overdamped Langevin targeting the posterior
    pi(u) \\propto p_0(u) \\, p(y \\mid u).

If `score_fn(u)` approximates \\nabla \\log p_0(u) and the Gaussian
observational model is `y = Gu + \\sigma_{\\mathrm{obs}} \\eta`, then
    \\nabla \\log \\pi(u) = \\nabla \\log p_0(u) + \\nabla \\log p(y\\mid u),
with
    \\nabla \\log p(y\\mid u) = -\\sigma_{\\mathrm{obs}}^{-2} G^*(Gu - y).

The Euler update is
    u <- u + \\Delta\\, C(\\nabla\\log\\pi(u))
         + \\sqrt{2\\Delta}\\, C^{1/2}[\\xi],
with separate callables for `C` and `C^{1/2}` (both identity by default).
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import torch


class ForwardWithAdjoint(Protocol):
    """Structural type for a differentiable linear forward pair."""

    def __call__(self, u: torch.Tensor) -> torch.Tensor: ...

    def adjoint(self, y: torch.Tensor) -> torch.Tensor: ...


def _identity(x: torch.Tensor) -> torch.Tensor:
    return x


class PreconditionedLangevin:
    """Run preconditioned Langevin sampling from the posterior.

    Args:
        score_fn: Callable taking ``u`` and returning the prior score
                  ``grad log p_0(u)`` (e.g. trained network at small diffusion time).
        forward: Forward operator G with callable evaluation and an
            ``adjoint`` method.
        precond: Callable implementing multiplication by C.
        noise_transform: Callable implementing multiplication by C^(1/2).
            It is required whenever ``precond`` is supplied.
        sigma_obs: Observational noise level (called sigma in the paper).
    """

    def __init__(
        self,
        score_fn: Callable[[torch.Tensor], torch.Tensor],
        forward: ForwardWithAdjoint,
        precond: Callable[[torch.Tensor], torch.Tensor] | None = None,
        noise_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        sigma_obs: float = 0.05,
        step_size: float = 1e-3,
        n_steps: int = 5000,
    ) -> None:
        if sigma_obs <= 0:
            raise ValueError("sigma_obs must be positive")
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        if n_steps < 1:
            raise ValueError("n_steps must be at least one")
        if precond is not None and noise_transform is None:
            raise ValueError(
                "noise_transform implementing C^(1/2) is required with precond"
            )

        self.score_fn = score_fn
        self.forward = forward
        self.precond = precond or _identity
        self.noise_transform = noise_transform or _identity
        self.sigma_obs = sigma_obs
        self.step_size = step_size
        self.n_steps = n_steps

    @torch.no_grad()
    def sample(self, y: torch.Tensor, u0: torch.Tensor) -> torch.Tensor:
        """Run Langevin from initial state u0, return the chain endpoint."""
        u = u0.clone()
        sqrt2dt = (2.0 * self.step_size) ** 0.5
        sigma2 = self.sigma_obs**2
        for _ in range(self.n_steps):
            residual = self.forward(u) - y
            grad_log_lik = -self.forward.adjoint(residual) / sigma2
            drift = self.precond(self.score_fn(u) + grad_log_lik)
            noise = torch.randn_like(u)
            u = u + self.step_size * drift + sqrt2dt * self.noise_transform(noise)
        return u
