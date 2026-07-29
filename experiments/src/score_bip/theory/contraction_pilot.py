"""Exploratory posterior-radius sweep for a 1D deconvolution model.

The script fits a descriptive log--log slope to finite-grid sampler output.
It does not test the prior-mass, sieve, testing, or identifiability conditions
of a posterior-contraction theorem.

Run:
    PYTHONPATH=src python -m score_bip.theory.contraction_pilot \\
        --config configs/1d_deconv.yaml

``--medium`` selects a moderate exploratory preset (order-10 CPU minutes).

Quick CI-sized smoke:
    PYTHONPATH=src python -m score_bip.theory.contraction_pilot --fast
"""
from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from score_bip.data.gp_1d import MaternGP1D
from score_bip.forward import GaussianBlur1D
from score_bip.posterior import ForwardWithAdjoint, PreconditionedLangevin
from score_bip.priors import ScoreMLP1D
from score_bip.training import dsm_loss_vp
from score_bip.utils.metrics import empirical_contraction_radius


def fit_loglog_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Least-squares slope and intercept in log-log space (base e)."""
    lx, ly = np.log(x), np.log(y)
    slope, intercept = np.polyfit(lx, ly, 1)
    return float(slope), float(intercept)


def _cfg_get(cfg: dict[str, Any], *path: str, default: Any = None) -> Any:
    cur: Any = cfg
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def train_score_mlp(
    gp: MaternGP1D,
    dim: int,
    hidden: int,
    depth: int,
    lipschitz: bool,
    steps: int,
    batch: int,
    lr: float,
    device: torch.device,
) -> tuple[ScoreMLP1D, float]:
    """Short score-matching training on OU marginals.

    Returns the trained network and the final-batch DSM loss (last optimisation step).
    """
    torch.manual_seed(0)
    net = ScoreMLP1D(dim=dim, hidden=hidden, depth=depth, lipschitz=lipschitz).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    net.train()
    loss_val = float("nan")
    for s in range(steps):
        u0 = gp.sample(batch, device=str(device))
        loss = dsm_loss_vp(net, u0)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        loss_val = float(loss.item())
        if s % max(1, steps // 5) == 0:
            logging.info("train step %s/%s loss=%.4f", s, steps, loss_val)
    net.eval()
    return net, loss_val


def make_marginal_score(net: ScoreMLP1D, t_score: float) -> Callable[[torch.Tensor], torch.Tensor]:
    """Wrap ``(t,u) -> score`` MLP as ``u -> nabla log p_0(u)`` at small diffusion time."""

    def score(u: torch.Tensor) -> torch.Tensor:
        if u.ndim == 1:
            t = torch.tensor([t_score], device=u.device, dtype=u.dtype)
            return net(t, u.unsqueeze(0))[0]
        t = torch.full((u.shape[0],), t_score, device=u.device, dtype=u.dtype)
        return net(t, u)

    return score


@torch.no_grad()
def radius_at_sigma(
    sigma_obs: float,
    *,
    gp: MaternGP1D,
    forward: ForwardWithAdjoint,
    score_fn: Callable[[torch.Tensor], torch.Tensor],
    dim: int,
    langevin_steps: int,
    step_size: float,
    n_chains: int,
    n_truths: int,
    device: torch.device,
    quantile: float = 0.9,
) -> tuple[float, float]:
    """Mean and std of empirical contraction radii over independent truths."""
    sigma2 = float(sigma_obs) ** 2
    # Stabilise Euler: gradient of log-lik scales like 1/sigma^2.
    stable_step = min(step_size, 0.5 * sigma2)
    n_steps_eff = int(min(12_000, langevin_steps * (step_size / max(stable_step, 1e-12))))
    radii: list[float] = []
    for _ in range(n_truths):
        u_dag = gp.sample(1, device=str(device))[0]
        y = forward(u_dag.unsqueeze(0))[0] + sigma_obs * torch.randn(dim, device=device)

        smpl = PreconditionedLangevin(
            score_fn=score_fn,
            forward=forward,
            sigma_obs=sigma_obs,
            step_size=stable_step,
            n_steps=n_steps_eff,
        )

        endpoints: list[torch.Tensor] = []
        for _c in range(n_chains):
            u0 = gp.sample(1, device=str(device))[0]
            endpoints.append(smpl.sample(y, u0))
        samples = torch.stack(endpoints)
        r = empirical_contraction_radius(samples, u_dag, quantile=quantile)
        radii.append(float(r.item()))
    arr = np.array(radii)
    return float(arr.mean()), float(arr.std(ddof=1) if len(arr) > 1 else 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/1d_deconv.yaml"))
    parser.add_argument("--fast", action="store_true", help="Small grid for smoke / CI.")
    parser.add_argument(
        "--medium",
        action="store_true",
        help="Moderate exploratory settings (tens of CPU minutes).",
    )
    parser.add_argument("--out", type=Path, default=Path("results/contraction_pilot.npz"))
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    device = torch.device(args.device)

    if args.fast and args.medium:
        raise SystemExit("choose at most one of --fast / --medium")

    if args.config.exists() and not args.fast and not args.medium:
        with args.config.open() as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}

    if args.fast:
        dim, hidden, depth = 24, 96, 2
        kernel_sigma = 0.05
        train_steps, batch, lr = 600, 32, 2.0e-4
        lipschitz = False
        langevin_steps, langevin_step = 350, 5.0e-4
        n_chains, n_truths = 10, 3
        sigma_list = [0.25, 0.125, 0.0625, 0.03125, 0.015625]
        quantile = 0.9
    elif args.medium:
        dim, hidden, depth = 32, 128, 3
        kernel_sigma = 0.05
        train_steps, batch, lr = 4000, 128, 2.0e-4
        lipschitz = False
        langevin_steps, langevin_step = 800, 1.0e-3
        n_chains, n_truths = 16, 6
        sigma_list = [2.0**-i for i in range(2, 9)]
        quantile = 0.9
    else:
        dim = int(_cfg_get(cfg, "problem", "dim", default=64))
        hidden = int(_cfg_get(cfg, "prior", "hidden", default=256))
        depth = int(_cfg_get(cfg, "prior", "depth", default=4))
        kernel_sigma = float(_cfg_get(cfg, "problem", "kernel_sigma", default=0.05))
        train_steps = int(_cfg_get(cfg, "train", "steps", default=8_000))
        batch = int(_cfg_get(cfg, "train", "batch_size", default=128))
        lr = float(_cfg_get(cfg, "train", "lr", default=2.0e-4))
        lipschitz = bool(_cfg_get(cfg, "prior", "lipschitz", default=True))
        langevin_steps = int(_cfg_get(cfg, "posterior", "n_steps", default=4_000))
        langevin_step = float(_cfg_get(cfg, "posterior", "step_size", default=1.0e-3))
        n_chains = int(_cfg_get(cfg, "sweep", "n_replicates_per_sigma", default=24))
        n_truths = min(8, n_chains // 2) if n_chains >= 4 else 4
        sigma_list = list(
            _cfg_get(
                cfg,
                "sweep",
                "sigma_obs_values",
                default=[2.0**-i for i in range(1, 9)],
            )
        )
        quantile = 0.9

    logging.info("dim=%s; fitted slopes are descriptive diagnostics only", dim)

    gp = MaternGP1D(dim=dim, length_scale=0.1, sigma=1.0, seed=42)
    forward = GaussianBlur1D(dim=dim, sigma_kernel=kernel_sigma).to(device)

    net, _ = train_score_mlp(
        gp, dim, hidden, depth, lipschitz, train_steps, batch, lr, device
    )
    score_fn = make_marginal_score(net, t_score=1.0e-3)

    sigmas: list[float] = []
    means: list[float] = []
    stds: list[float] = []
    for sig in sigma_list:
        m, s = radius_at_sigma(
            float(sig),
            gp=gp,
            forward=forward,
            score_fn=score_fn,
            dim=dim,
            langevin_steps=langevin_steps,
            step_size=langevin_step,
            n_chains=n_chains,
            n_truths=n_truths,
            device=device,
            quantile=quantile,
        )
        sigmas.append(float(sig))
        means.append(m)
        stds.append(s)
        logging.info("sigma_obs=%.5f  mean_radius=%.4f +/- %.4f", sig, m, s)

    sig_arr = np.array(sigmas)
    mean_arr = np.array(means)
    slope, intercept = fit_loglog_slope(sig_arr, mean_arr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        sigma_obs=sig_arr,
        radius_mean=mean_arr,
        radius_std=np.array(stds),
        loglog_slope=slope,
        loglog_intercept=intercept,
        dim=dim,
        quantile=quantile,
        fast=args.fast,
        medium=args.medium,
    )
    logging.info("descriptive log-log slope=%.4f; saved %s", slope, args.out.resolve())


if __name__ == "__main__":
    main()
