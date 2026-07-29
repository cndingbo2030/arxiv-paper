"""Smoke-test the Gaussian discretization diagnostic CLI."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_gaussian_discretization_fast_exits_zero(tmp_path: Path) -> None:
    exp_root = Path(__file__).resolve().parents[1]
    out = tmp_path / "gaussian_diagnostic.npz"
    env = {**os.environ, "PYTHONPATH": str(exp_root / "src")}
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "score_bip.theory.gaussian_discretization_diagnostic",
            "--fast",
            "--out",
            str(out),
        ],
        cwd=str(exp_root),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    assert out.is_file()
