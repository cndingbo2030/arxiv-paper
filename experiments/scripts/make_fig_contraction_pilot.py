"""Plot descriptive posterior-radius output against observation noise."""
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
        default=_ROOT / "pilot_data" / "contraction_pilot.npz",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_ROOT.parent / "paper" / "figures" / "fig_contraction_pilot.pdf",
    )
    args = parser.parse_args()

    data = np.load(args.npz)
    s = data["sigma_obs"]
    r = data["radius_mean"]
    slope = float(data["loglog_slope"])

    fit_x = np.linspace(s.min(), s.max(), 50)
    log_fit = np.polyval(np.polyfit(np.log(s), np.log(r), 1), np.log(fit_x))

    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.loglog(s, r, "o", ms=6, label="empirical radius")
    ax.loglog(fit_x, np.exp(log_fit), "-", lw=1.2, label=f"OLS fit (slope={slope:.2f})")
    ax.set_xlabel(r"observation noise $\sigma$")
    ax.set_ylabel(r"finite-sampler radius (90% quantile)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, which="both", ls="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
