#!/usr/bin/env bash
# Drive experiments in one of three modes.
# Usage: scripts/run_all_experiments.sh [smoke|figures-only|full]
set -euo pipefail

MODE="${1:-smoke}"
cd "$(dirname "$0")/../experiments"

case "$MODE" in
  smoke)
    # Just runs the test suite; safe to call from CI.
    pytest -q
    ;;
  figures-only)
    # Generate fresh fast-preset data and exploratory plots.
    export PYTHONPATH="${PWD}/src"
    python3 -m score_bip.theory.gaussian_discretization_diagnostic \
      --fast --out pilot_data/gaussian_discretization_fast.npz
    python3 -m score_bip.theory.contraction_pilot \
      --fast --out pilot_data/contraction_pilot.npz
    python3 sweep_sparse.py \
      --fast-smoke --out ../pilot_data/sparse_contraction.npz
    python3 scripts/plot_dsm_training.py --steps 200
    python3 scripts/make_fig_contraction_pilot.py
    python3 scripts/plot_sparse_pilot.py
    python3 scripts/make_fig_gaussian_discretization.py
    ;;
  full)
    export PYTHONPATH="${PWD}/src"
    python3 -m score_bip.theory.gaussian_discretization_diagnostic \
      --out results/gaussian_discretization_diagnostic.npz
    python3 -m score_bip.theory.contraction_pilot \
      --medium --out results/contraction_pilot.npz
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    exit 1
    ;;
esac
