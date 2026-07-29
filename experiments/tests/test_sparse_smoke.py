"""Fast regression for ``sweep_sparse.py`` (random linear forward, not deconv)."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def test_sparse_sweep_creates_npz_and_monotone_radii(tmp_path: Path) -> None:
    exp_root = Path(__file__).resolve().parents[1]
    out = tmp_path / "sparse_contraction.npz"
    env = {**os.environ, "PYTHONPATH": str(exp_root / "src")}
    t0 = time.perf_counter()
    r = subprocess.run(
        [
            sys.executable,
            str(exp_root / "sweep_sparse.py"),
            "--fast-smoke",
            "--measurement",
            "gaussian-sparse",
            "--matrix-seed",
            "4242",
            "--out",
            str(out),
        ],
        cwd=str(exp_root),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - t0
    assert r.returncode == 0, r.stderr
    # Typical runtime is a few seconds on warm runners; allow margin for CI cold-start.
    assert elapsed < 30.0, f"smoke took {elapsed:.1f}s"
    assert out.is_file()
    data = np.load(out, allow_pickle=True)
    mean = data["radii_mean"]
    sig = data["sigma_grid"]
    assert mean.shape == sig.shape
    assert "dsm_loss_final" in data.files
    assert float(data["dsm_loss_final"]) > 0
    assert int(data["train_steps"]) == 100
    # Descending observation noise -> strictly smaller posterior quantile radii.
    assert np.all(np.diff(mean) < 0), (mean, sig)
