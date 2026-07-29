"""Short run that exercises the DSM training path.

Used locally and in CI as a deterministic software check. It is not a
convergence test for the statistical estimator.
"""
from __future__ import annotations

import argparse

import torch

from score_bip.data import MaternGP1D
from score_bip.priors import ScoreMLP1D
from score_bip.training import dsm_loss_vp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--lipschitz", action="store_true", help="Use spectral-norm layers (slower).")
    args = parser.parse_args()

    torch.manual_seed(0)
    gp = MaternGP1D(dim=args.dim, length_scale=0.1, sigma=1.0, seed=0)
    net = ScoreMLP1D(dim=args.dim, hidden=args.hidden, depth=args.depth, lipschitz=args.lipschitz)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)

    losses: list[float] = []
    for step in range(args.steps):
        u0 = gp.sample(args.batch)
        loss = dsm_loss_vp(net, u0)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        if step % 50 == 0 or step == args.steps - 1:
            window = losses[-50:] if step >= 50 else losses
            print(f"step={step:4d}  loss={loss.item():.4f}  ema50={sum(window)/len(window):.4f}")

    initial = sum(losses[:20]) / 20
    final = sum(losses[-20:]) / 20
    print(f"\ninitial(loss[:20])={initial:.4f}")
    print(f"final  (loss[-20:])={final:.4f}")
    print(f"ratio final/initial={final/initial:.3f}")
    # Pipeline is healthy if 500 steps cuts the loss by ~40% or more on
    # this 64-d MLP; tight enough to catch regressions, loose enough
    # not to flake on CI.
    assert final < 0.6 * initial, f"DSM training stalled: final/initial={final/initial:.3f}"


if __name__ == "__main__":
    main()
