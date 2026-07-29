"""CLI to train the 1D ScoreMLP1D on Matern-2.5 GP samples.

Usage:
    PYTHONPATH=src python scripts/train_1d.py --config configs/1d_deconv.yaml

The default config runs short (smoke) training; pass `--steps N` to
override.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import lightning as L
import torch
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from score_bip.data import MaternGP1D
from score_bip.data.gp_1d import GPDataset
from score_bip.priors import ScoreMLP1D
from score_bip.training import ScoreLightningModule


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/1d_deconv.yaml"))
    parser.add_argument("--steps", type=int, default=None, help="Override training steps.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out", type=Path, default=Path("results/1d_deconv"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.config.exists():
        with args.config.open() as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}
    p = cfg.get("problem", {})
    pr = cfg.get("prior", {})
    tr = cfg.get("train", {})

    torch.manual_seed(cfg.get("experiment", {}).get("seed", 0))

    gp = MaternGP1D(dim=p.get("dim", 64), length_scale=0.1, sigma=1.0, seed=0)
    dataset = GPDataset(gp, size=tr.get("batch_size", 128) * 200)
    loader = DataLoader(dataset, batch_size=tr.get("batch_size", 128), shuffle=True, num_workers=0)

    net = ScoreMLP1D(
        dim=p.get("dim", 64),
        hidden=pr.get("hidden", 256),
        depth=pr.get("depth", 4),
        lipschitz=pr.get("lipschitz", True),
    )
    module = ScoreLightningModule(net, lr=tr.get("lr", 2.0e-4), ema_decay=pr.get("ema_decay", 0.999))

    max_steps = args.steps if args.steps is not None else tr.get("steps", 50_000)
    args.out.mkdir(parents=True, exist_ok=True)
    ckpt = ModelCheckpoint(dirpath=args.out, filename="score-mlp-{step}", every_n_train_steps=tr.get("ckpt_every", 5000))
    trainer = L.Trainer(
        max_steps=max_steps,
        accelerator="cpu" if args.device == "cpu" else "auto",
        log_every_n_steps=10,
        callbacks=[ckpt],
        default_root_dir=args.out,
        enable_progress_bar=False,
    )
    trainer.fit(module, loader)
    logging.info("Training done. Final EMA available via module.ema_forward.")


if __name__ == "__main__":
    main()
