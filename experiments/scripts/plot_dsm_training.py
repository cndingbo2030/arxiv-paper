"""Plot DSM training loss for ScoreMLP1D on a Matern-2.5 GP.

This script trains a small ScoreMLP1D, logs the loss curve, and writes
the figure to `paper/figures/dsm_training_curve.pdf`. The loss curve is
a training diagnostic and is not an estimate of the pathwise drift error in
the manuscript.

Run from the repo root:
    cd experiments && PYTHONPATH=src python3 scripts/plot_dsm_training.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from score_bip.data import MaternGP1D  # noqa: E402
from score_bip.priors import ScoreMLP1D  # noqa: E402
from score_bip.training import dsm_loss_vp  # noqa: E402


def train_and_log(
    dim: int = 64,
    hidden: int = 128,
    depth: int = 3,
    steps: int = 2000,
    batch: int = 64,
    lr: float = 2.0e-4,
    seed: int = 0,
) -> tuple[list[float], list[float]]:
    """Train one model and return (steps, losses, smoothed_losses)."""
    torch.manual_seed(seed)
    gp = MaternGP1D(dim=dim, length_scale=0.1, sigma=1.0, seed=seed)
    net = ScoreMLP1D(dim=dim, hidden=hidden, depth=depth, lipschitz=False)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)

    losses: list[float] = []
    for _ in range(steps):
        u0 = gp.sample(batch)
        loss = dsm_loss_vp(net, u0)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())

    window = 50
    smoothed = np.convolve(losses, np.ones(window) / window, mode="valid").tolist()
    return losses, smoothed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("../paper/figures/dsm_training_curve.pdf"),
    )
    args = parser.parse_args()

    losses, smoothed = train_and_log(steps=args.steps)

    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.plot(losses, color="C0", alpha=0.25, lw=0.6, label="raw")
    offset = (len(losses) - len(smoothed)) // 2
    ax.plot(range(offset, offset + len(smoothed)), smoothed, color="C0", lw=1.4, label="50-step EMA")
    ax.set_xlabel("training step")
    ax.set_ylabel("DSM loss")
    ax.set_title(r"ScoreMLP1D on Mat\'ern-2.5 GP, $d=64$, OU SDE")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")
    print(f"initial={sum(losses[:20])/20:.4f}  final={sum(losses[-20:])/20:.4f}  ratio={sum(losses[-20:])/sum(losses[:20]):.3f}")


if __name__ == "__main__":
    main()
