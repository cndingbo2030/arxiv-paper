"""Exploratory code for learned-prior Bayesian inverse problems.

Companion code for the paper:
    Posterior Stability and Lifted Discretization for Bayesian Inverse
    Problems with Learned Ornstein--Uhlenbeck Drift Priors.

Module map:
    priors/      -- score-network parameterizations (1D MLP in code today)
    forward/     -- forward operators (1D Gaussian blur, identity sanity-check)
    posterior/   -- posterior sampling (preconditioned Langevin)
    theory/      -- finite-dimensional diagnostics and posterior-radius pilots
    utils/       -- shared utilities (metrics)
"""

__version__ = "0.1.0"
