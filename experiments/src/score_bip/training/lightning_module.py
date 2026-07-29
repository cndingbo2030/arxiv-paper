"""Lightning module wrapping ScoreMLP1D + DSM loss + EMA + optimizer."""
from __future__ import annotations

import copy
from collections.abc import Mapping

import lightning as L
import torch
from torch import Tensor, nn

from .dsm import dsm_loss_vp


class _EMA:
    """Exponential-moving-average shadow of a module, applied for evaluation."""

    def __init__(self, module: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = copy.deepcopy(module).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, module: nn.Module) -> None:
        for p_ema, p in zip(self.shadow.parameters(), module.parameters(), strict=False):
            p_ema.mul_(self.decay).add_(p, alpha=1.0 - self.decay)


class ScoreLightningModule(L.LightningModule):
    """Generic Lightning wrapper for score-network training.

    Designed to be architecture-agnostic: pass in any nn.Module taking
    `(t: (B,), u: (B, D)) -> (B, D)` as its `score_net`.
    """

    def __init__(
        self,
        score_net: nn.Module,
        lr: float = 2.0e-4,
        ema_decay: float = 0.999,
        weight_decay: float = 0.0,
        t_eps: float = 1.0e-3,
        t_max: float = 1.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["score_net"])
        self.score_net = score_net
        self.ema = _EMA(score_net, decay=ema_decay)
        self.t_eps = t_eps
        self.t_max = t_max
        self._lr = lr
        self._weight_decay = weight_decay

    def forward(self, t: Tensor, u: Tensor) -> Tensor:
        return self.score_net(t, u)

    def ema_forward(self, t: Tensor, u: Tensor) -> Tensor:
        """Score evaluation with EMA weights (use this for sampling)."""
        return self.ema.shadow(t, u)

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:  # noqa: ARG002
        if isinstance(batch, Mapping):
            batch = batch["u0"]
        loss = dsm_loss_vp(self.score_net, batch, t_eps=self.t_eps, t_max=self.t_max)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def on_after_backward(self) -> None:
        self.ema.update(self.score_net)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.score_net.parameters(),
            lr=self._lr,
            weight_decay=self._weight_decay,
        )
