"""Finite-dimensional Gaussian spectral-truncation diagnostic.

This script implements a Gaussian Bayesian inverse problem on
:math:`\\mathbb{R}^d`
with a Matérn-2.5 covariance and a 1D Gaussian-blur forward map, compares the
exact posteriors obtained when the prior covariance is spectrally truncated
at rank :math:`N` against a full-rank reference prior, and records the
closed-form squared Hellinger distance :math:`H^2` between these Gaussian
posteriors.

The truncated covariance is regularized by a positive ridge, so every
finite-dimensional Gaussian is nondegenerate. This diagnostic is not the
full-space lifted approximation analysed in the manuscript and does not
verify a theorem about learned drift priors.

Run:
    PYTHONPATH=src python -m score_bip.theory.gaussian_discretization_diagnostic

    PYTHONPATH=src python -m score_bip.theory.gaussian_discretization_diagnostic --fast
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from score_bip.data.gp_1d import matern25_kernel
from score_bip.forward import GaussianBlur1D
from score_bip.utils.gaussian_distances import (
    gaussian_posterior_linear_gaussian,
    hellinger_squared_mvn,
)


def build_k_core(dim: int, length_scale: float, jitter: float) -> np.ndarray:
    grid = np.linspace(0.0, 1.0, dim, dtype=np.float64)
    k = matern25_kernel(grid, grid, length_scale=length_scale, sigma=1.0)
    k += jitter * np.eye(dim)
    return k


def spectral_truncate(k: np.ndarray, n_modes: int, rank_deficit_ridge: float) -> np.ndarray:
    """Keep top ``n_modes`` eigenpairs; add ``rank_deficit_ridge``·I only if rank-deficient."""
    dim = k.shape[0]
    k_sym = 0.5 * (k + k.T)
    w, v = np.linalg.eigh(k_sym)
    order = np.argsort(w)[::-1]
    w, v = w[order], v[:, order]
    n_keep = min(int(n_modes), dim)
    w_used = w[:n_keep]
    v_used = v[:, :n_keep]
    k_n = (v_used * w_used) @ v_used.T
    if n_keep < dim:
        k_n += rank_deficit_ridge * np.eye(dim)
    return k_n


def blur_matrix(dim: int, sigma_kernel: float) -> np.ndarray:
    op = GaussianBlur1D(dim=dim, sigma_kernel=sigma_kernel)
    g = np.zeros((dim, dim), dtype=np.float64)
    with torch.no_grad():
        for j in range(dim):
            e = torch.zeros(dim, dtype=torch.float32)
            e[j] = 1.0
            g[:, j] = op(e.unsqueeze(0))[0].double().numpy()
    return g


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument(
        "--n-values",
        type=int,
        nargs="+",
        default=[8, 16, 24, 32, 48, 64],
        help="Spectral ranks N for the truncated prior covariance.",
    )
    parser.add_argument("--length-scale", type=float, default=0.1)
    parser.add_argument("--kernel-sigma", type=float, default=0.05, dest="kernel_sigma")
    parser.add_argument("--sigma-obs", type=float, default=0.1, dest="sigma_obs")
    parser.add_argument("--jitter", type=float, default=1e-6, help="Diagonal jitter on Matérn K.")
    parser.add_argument(
        "--trunc-ridge",
        type=float,
        default=1e-6,
        dest="trunc_ridge",
        help="Extra ridge when truncated prior is rank-deficient.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/gaussian_discretization_diagnostic.npz"),
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.fast:
        args.dim = 32
        args.n_values = [4, 8, 16, 24, 32]

    dim = int(args.dim)
    rng = np.random.default_rng(args.seed)
    k_core = build_k_core(dim, args.length_scale, args.jitter)
    g = blur_matrix(dim, args.kernel_sigma)

    z = rng.standard_normal(dim)
    u_true = np.linalg.cholesky(k_core) @ z
    y = g @ u_true + args.sigma_obs * rng.standard_normal(dim)

    m_ref, s_ref = gaussian_posterior_linear_gaussian(y, g, k_core, args.sigma_obs)

    n_list: list[int] = []
    h2_list: list[float] = []
    for n_modes in args.n_values:
        if n_modes < 1 or n_modes > dim:
            logging.warning("skip N=%s (outside [1,%s])", n_modes, dim)
            continue
        k_n = spectral_truncate(k_core, n_modes, args.trunc_ridge)
        m_n, s_n = gaussian_posterior_linear_gaussian(y, g, k_n, args.sigma_obs)
        h2 = hellinger_squared_mvn(m_n, s_n, m_ref, s_ref)
        n_list.append(int(n_modes))
        h2_list.append(float(h2))
        logging.info("N=%3d  H^2(post_N, post_ref)=%.6e", n_modes, h2)

    disclaimer = (
        "finite_dimensional_gaussian_diagnostic_only;"
        " not_full_space_lift;"
        " not_learned_drift_prior;"
        " see module docstring."
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        n_modes=np.array(n_list, dtype=np.int32),
        hellinger_squared=np.array(h2_list, dtype=np.float64),
        dim=dim,
        sigma_obs=args.sigma_obs,
        kernel_sigma=args.kernel_sigma,
        disclaimer=np.array(disclaimer),
        fast=args.fast,
    )
    logging.info("wrote %s", args.out.resolve())


if __name__ == "__main__":
    main()
