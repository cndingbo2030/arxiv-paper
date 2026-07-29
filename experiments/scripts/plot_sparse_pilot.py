"""Plot a sparse-forward posterior-radius pilot on log--log axes."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_REPO = _ROOT.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--npz",
        type=Path,
        default=_REPO / "pilot_data" / "sparse_contraction.npz",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_REPO / "paper" / "figures" / "fig_sparse_pilot.pdf",
    )
    args = parser.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    s = data["sigma_grid"].astype(float)
    r = data["radii_mean"].astype(float)
    rsd = data["radii_std"].astype(float)

    n_mono = 5
    s5, r5 = s[:n_mono], r[:n_mono]
    coef = np.polyfit(np.log(s5), np.log(r5), 1)
    slope5 = float(coef[0])
    log_pred = np.polyval(coef, np.log(s5))
    ss_res = float(np.sum((np.log(r5) - log_pred) ** 2))
    ss_tot = float(np.sum((np.log(r5) - np.mean(np.log(r5))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    slope_full = float(data["loglog_slope"])

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.errorbar(s, r, yerr=rsd, fmt="o", ms=5, capsize=2, lw=1, label="mean $\\pm$ std")

    fit_x = np.linspace(float(s5.min()), float(s5.max()), 80)
    ax.loglog(fit_x, np.exp(np.polyval(coef, np.log(fit_x))), "-", lw=1.3, label="OLS, first $5$ points")

    ax.set_xlabel(r"observation noise $\sigma$")
    ax.set_ylabel(r"mean $90\%$ finite-sampler radius")
    ax.legend(frameon=False, fontsize=7, loc="best")
    ax.grid(True, which="both", ls="--", alpha=0.35)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(
        f"wrote {args.out.resolve()} "
        f"(resolved OLS slope={slope5:.6f}, R^2={r2:.6f}; full OLS={slope_full:.6f})"
    )


if __name__ == "__main__":
    main()
