"""Losses of the variants. They see only the input frames and the actions.

    E0  reconstruction D(z_j) -> x_j                                                     capacity check
    E1  ||z^_{j+1} - z_{j+1}||^2 (gradient through the target) + SIGReg                  pruning reference (A3)
    E2  E1 with the target detached                                                       A4 analogue

Token losses are means over the 82 tokens: each tile, item, vital, facing, light and sleeping counts once.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SIGReg(nn.Module):
    """Sketched isotropic Gaussian regulariser (LeJEPA): random 1-D projections of the batch, their empirical
    characteristic function against the standard normal's, weighted over t in [0, 3]."""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots)
        dt = 3 / (knots - 1)
        w = torch.full((knots,), 2 * dt)
        w[[0, -1]] = dt
        window = torch.exp(-t.square() / 2)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", w * window)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z (T, B, D): one statistic per T over the B samples, averaged."""
        a = torch.randn(z.shape[-1], self.num_proj, device=z.device)
        a = a / a.norm(dim=0)
        x = (z @ a).unsqueeze(-1) * self.t
        err = (x.cos().mean(-3) - self.phi).square() + x.sin().mean(-3).square()
        return ((err @ self.weights) * z.shape[-2]).mean()


def token_losses(out: dict, f: dict, w: dict | None = None) -> torch.Tensor:
    """Per-token loss (..., 82) of decoder outputs against frames. `w` (optional, the readout's class balance):
    per-class weights `block` (17), `item` (10), `vital` (10), `facing` (4) and positive weights `mob` (4), `sleep`."""
    w = w or {}

    def ce(logits, target, weight=None):
        return F.cross_entropy(logits.flatten(0, -2), target.flatten(), weight=weight, reduction="none").view(
            target.shape
        )

    tile = ce(out["block"], f["block"], w.get("block"))
    tile = tile + F.binary_cross_entropy_with_logits(out["mob"], f["mob"], pos_weight=w.get("mob"), reduction="none").sum(-1)
    item = ce(out["item"], f["item"], w.get("item"))
    vital = ce(out["vital"], f["vital"], w.get("vital"))
    facing = ce(out["facing"], f["facing"], w.get("facing"))
    light = (out["light"] - f["light"]).square()
    sleep = F.binary_cross_entropy_with_logits(out["sleep"], f["sleep"].float(), pos_weight=w.get("sleep"), reduction="none")
    return torch.cat([tile, item, vital, facing[..., None], light[..., None], sleep[..., None]], -1)


def variant_loss(model, f: dict, m: torch.Tensor, cfg, sigreg: SIGReg | None = None) -> tuple[torch.Tensor, dict]:
    """The loss of one batch: frames f (B, C + 1, ...), macro-actions m (B, C, k). Returns the loss and the logged
    values (floats; the diagnostic ones without gradient)."""
    v = cfg.variant
    z = model.encoder(f)  # (B, C + 1, D)
    logs = {"z_std": z.detach().std(0).mean().item()}
    if v == "E0":
        rec = token_losses(model.decoder(z), f).mean()
        return rec, {**logs, "recon": rec.item()}
    if v not in ("E1", "E2"):
        raise ValueError(f"unknown variant {v}")
    target = z[:, 1:] if v == "E1" else z[:, 1:].detach()
    pred = model.predictor(z[:, :-1], m)
    pred_loss = (pred - target).square().mean()
    with torch.no_grad():
        copy = (z[:, :-1] - z[:, 1:]).square().mean()
        logs.update(pred=pred_loss.item(), copy=copy.item(), pred_over_copy=(pred_loss / copy.clamp(min=1e-12)).item())
    sig = sigreg(z.transpose(0, 1))
    logs["sigreg"] = sig.item()
    return pred_loss + cfg.sigreg_weight * sig, logs
