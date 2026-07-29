"""Plot a finite-dimensional Gaussian spectral-truncation diagnostic."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--npz",
        type=Path,
        default=_ROOT / "pilot_data" / "gaussian_discretization_fast.npz",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_ROOT.parent / "paper" / "figures" / "fig_gaussian_discretization.pdf",
    )
    args = parser.parse_args()

    data = np.load(args.npz)
    n = data["n_modes"].astype(int)
    h2 = data["hellinger_squared"]

    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.semilogy(n, h2, "o-", ms=7, lw=1.2)
    ax.set_xlabel(r"spectral rank $N$ of truncated Mat\'ern prior")
    ax.set_ylabel(r"squared Hellinger $H^2(\Pi^y_N,\Pi^y_{\mathrm{ref}})$")
    ax.set_title("Finite-dimensional Gaussian diagnostic")
    ax.grid(True, which="both", ls="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
