"""Exploratory posterior-radius sweep with sparse linear forwards.

Reuses training, marginal score extraction, and Langevin machinery from
``score_bip.theory.contraction_pilot``; only the forward operator
``G`` is swapped for a full-rank random projection or a Fourier subsample.

CLI example:

    PYTHONPATH=src python sweep_sparse.py --config configs/sparse_default.yaml

CI micro-run:

    PYTHONPATH=src python sweep_sparse.py --fast-smoke --out /tmp/sparse.npz
"""
from __future__ import annotations

import argparse
import logging
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn

from score_bip.data.gp_1d import MaternGP1D
from score_bip.posterior import PreconditionedLangevin
from score_bip.theory.contraction_pilot import (
    fit_loglog_slope,
    make_marginal_score,
    train_score_mlp,
)
from score_bip.utils.metrics import empirical_contraction_radius

# ---------------------------------------------------------------------------#
# Forward operators (linear): y = G u in R^{obs_dim}


class GaussianSparseForward(nn.Module):
    """Full-rank random projection: ``y = M u`` with ``M \\in R^{d\\times d}``.

    Entries are i.i.d. ``N(0, 1/d)`` so columns have expected unit Euclidean
    norm (scaling ``1/sqrt(d)`` matches the ``1/k`` variance prescription for
    ``k = d``).
    """

    def __init__(self, dim: int, *, seed: int) -> None:
        super().__init__()
        self.dim = dim
        self.obs_dim = dim
        g = torch.Generator()
        g.manual_seed(int(seed))
        # k = d => Var[M_ij] = 1/k = 1/d
        M = torch.randn(dim, dim, generator=g) / math.sqrt(float(dim))
        self.register_buffer("M", M)

    def forward(self, u: torch.Tensor) -> torch.Tensor:  # noqa: D102
        u = u.to(dtype=self.M.dtype)
        if u.ndim == 1:
            return u @ self.M.T
        return u @ self.M.T

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:  # noqa: D102
        y = y.to(dtype=self.M.dtype)
        if y.ndim == 1:
            return y @ self.M
        return y @ self.M


class FourierSubsampleForward(nn.Module):
    """Stack real and imaginary parts of ``k = d//2`` random Fourier rows."""

    def __init__(self, dim: int, *, seed: int) -> None:
        super().__init__()
        self.dim = dim
        k_freq = max(1, dim // 2)
        rng = np.random.default_rng(int(seed))
        idx = np.sort(rng.choice(dim, size=k_freq, replace=False))
        j = torch.arange(dim, dtype=torch.float64)
        n = torch.arange(dim, dtype=torch.float64)
        phase = -2.0 * math.pi * (j.unsqueeze(1) * n.unsqueeze(0)) / float(dim)
        F = torch.polar(torch.ones_like(phase), phase) / math.sqrt(float(dim))
        rows_c = F[torch.as_tensor(idx, dtype=torch.long)]  # k_freq x d
        M = torch.cat([rows_c.real, rows_c.imag], dim=0).to(dtype=torch.float32)
        self.register_buffer("M", M)
        self.obs_dim = int(M.shape[0])

    def forward(self, u: torch.Tensor) -> torch.Tensor:  # noqa: D102
        u = u.to(dtype=self.M.dtype)
        if u.ndim == 1:
            return u @ self.M.T
        return u @ self.M.T

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:  # noqa: D102
        y = y.to(dtype=self.M.dtype)
        if y.ndim == 1:
            return y @ self.M
        return y @ self.M


def build_forward(
    measurement: str,
    dim: int,
    seed: int,
    device: torch.device,
) -> nn.Module:
    if measurement == "gaussian-sparse":
        f = GaussianSparseForward(dim, seed=seed)
    elif measurement == "fourier-subsample":
        f = FourierSubsampleForward(dim, seed=seed)
    else:
        raise ValueError(f"unknown measurement {measurement!r}")
    return f.to(device)


def _cfg_get(cfg: dict[str, Any], *path: str, default: Any = None) -> Any:
    cur: Any = cfg
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def _parse_sigma_grid(s: str) -> list[float]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        msg = "sigma-grid must list at least one float, comma-separated"
        raise argparse.ArgumentTypeError(msg)
    return [float(p) for p in parts]


@torch.no_grad()
def radius_at_sigma(
    sigma_obs: float,
    *,
    gp: MaternGP1D,
    forward: nn.Module,
    score_fn: Callable[[torch.Tensor], torch.Tensor],
    obs_dim: int,
    langevin_steps: int,
    step_size: float,
    n_chains: int,
    n_truths: int,
    device: torch.device,
    quantile: float = 0.9,
) -> tuple[float, float]:
    """Mean and standard deviation of finite-sampler radius diagnostics."""
    sigma2 = float(sigma_obs) ** 2
    stable_step = min(step_size, 0.5 * sigma2)
    n_steps_eff = int(min(12_000, langevin_steps * (step_size / max(stable_step, 1e-12))))
    radii: list[float] = []
    for _ in range(n_truths):
        u_dag = gp.sample(1, device=str(device))[0]
        y = forward(u_dag.unsqueeze(0))[0] + sigma_obs * torch.randn(obs_dim, device=device)

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
    parser.add_argument(
        "--measurement",
        choices=["gaussian-sparse", "fourier-subsample"],
        default=None,
        help="Linear forward: i.i.d. Gaussian matrix (k=d) vs random Fourier rows.",
    )
    parser.add_argument("--matrix-seed", type=int, default=None, help="RNG for M / frequency draw.")
    parser.add_argument("--config", type=Path, default=Path("configs/sparse_default.yaml"))
    parser.add_argument(
        "--fast-smoke",
        action="store_true",
        help="Tiny run for CI (seconds); ignores most YAML timings.",
    )
    out_group = parser.add_mutually_exclusive_group()
    out_group.add_argument(
        "--out",
        type=Path,
        dest="result_out",
        default=None,
        help="Output .npz path (default: pilot_data/sparse_contraction.npz).",
    )
    out_group.add_argument(
        "--output",
        type=Path,
        dest="result_out",
        default=None,
        help="Alias of --out.",
    )
    parser.add_argument(
        "--d",
        type=int,
        default=None,
        dest="cli_dim",
        metavar="INT",
        help="Override config problem.dim (ignored with --fast-smoke).",
    )
    parser.add_argument(
        "--n-truths",
        type=int,
        default=None,
        dest="cli_n_truths",
        metavar="INT",
        help="Override config sweep.n_truths (ignored with --fast-smoke).",
    )
    parser.add_argument(
        "--sigma-grid",
        type=str,
        default=None,
        dest="cli_sigma_grid",
        metavar="STR",
        help="Comma-separated sigmas overriding sweep.sigma_obs_values (ignored with --fast-smoke).",
    )
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    out_path: Path = (
        args.result_out if args.result_out is not None else Path("pilot_data/sparse_contraction.npz")
    )
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    device = torch.device(args.device)
    cfg: dict[str, Any] = {}
    if args.config.exists() and not args.fast_smoke:
        with args.config.open() as f:
            cfg = yaml.safe_load(f) or {}

    matrix_seed = int(
        args.matrix_seed
        if args.matrix_seed is not None
        else _cfg_get(cfg, "problem", "matrix_seed", default=12_345)
    )
    yaml_meas = _cfg_get(cfg, "problem", "measurement", default=None)
    measurement = str(args.measurement or yaml_meas or "gaussian-sparse")

    if args.fast_smoke:
        dim = 8
        hidden, depth = 32, 2
        train_steps, batch, lr = 100, 16, 2.0e-4
        lipschitz = False
        langevin_steps, langevin_step = 80, 5.0e-4
        n_chains, n_truths = 4, 2
        sigma_list = [0.4, 0.12]
        quantile = 0.9
        torch.manual_seed(0)
    else:
        dim = int(_cfg_get(cfg, "problem", "dim", default=32))
        hidden = int(_cfg_get(cfg, "prior", "hidden", default=128))
        depth = int(_cfg_get(cfg, "prior", "depth", default=3))
        train_steps = int(_cfg_get(cfg, "train", "steps", default=4_000))
        batch = int(_cfg_get(cfg, "train", "batch_size", default=128))
        lr = float(_cfg_get(cfg, "train", "lr", default=2.0e-4))
        lipschitz = bool(_cfg_get(cfg, "prior", "lipschitz", default=False))
        langevin_steps = int(_cfg_get(cfg, "posterior", "n_steps", default=800))
        langevin_step = float(_cfg_get(cfg, "posterior", "step_size", default=1.0e-3))
        n_chains = int(_cfg_get(cfg, "sweep", "n_replicates_per_sigma", default=16))
        n_truths = int(_cfg_get(cfg, "sweep", "n_truths", default=max(1, n_chains // 4)))
        sigma_list = list(
            _cfg_get(
                cfg,
                "sweep",
                "sigma_obs_values",
                default=[2.0**-i for i in range(2, 9)],
            )
        )
        quantile = 0.9
        if args.cli_dim is not None:
            dim = int(args.cli_dim)
        if args.cli_n_truths is not None:
            n_truths = int(args.cli_n_truths)
        if args.cli_sigma_grid is not None:
            sigma_list = _parse_sigma_grid(args.cli_sigma_grid)

    gp = MaternGP1D(dim=dim, length_scale=0.1, sigma=1.0, seed=42)
    forward = build_forward(measurement, dim, matrix_seed, device)
    obs_dim = int(forward.obs_dim)  # type: ignore[attr-defined]

    net, dsm_loss_final = train_score_mlp(
        gp,
        dim,
        hidden,
        depth,
        lipschitz,
        train_steps,
        batch,
        lr,
        device,
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
            obs_dim=obs_dim,
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

    sig_arr = np.asarray(sigmas, dtype=np.float64)
    mean_arr = np.asarray(means, dtype=np.float64)
    std_arr = np.asarray(stds, dtype=np.float64)
    slope, intercept = fit_loglog_slope(sig_arr, mean_arr)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        sigma_grid=sig_arr,
        sigma_obs=sig_arr,
        radii_mean=mean_arr,
        radius_mean=mean_arr,
        radii_std=std_arr,
        radius_std=std_arr,
        radii_quantile=np.float64(quantile),
        loglog_slope=np.float64(slope),
        loglog_intercept=np.float64(intercept),
        dim=np.int32(dim),
        obs_dim=np.int32(obs_dim),
        n_truths=np.int32(n_truths),
        n_chains=np.int32(n_chains),
        n_replicates_per_sigma=np.int32(n_chains),
        measurement=np.array(measurement, dtype=object),
        matrix_seed=np.int32(matrix_seed),
        fast_smoke=bool(args.fast_smoke),
        train_steps=np.int32(train_steps),
        dsm_loss_final=np.float64(dsm_loss_final),
    )
    logging.info("saved %s (log-log OLS slope=%.4f)", out_path.resolve(), slope)


if __name__ == "__main__":
    main()
