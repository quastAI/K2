"""The loss of E0: reconstruction D(z) -> x of every frame, per token, as the mean over the 82 tokens (each tile, item,
vital, facing, light and sleeping counts once). It sees only the input frames.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


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


def variant_loss(model, f: dict, m: torch.Tensor, cfg, step: int = 0, diag: bool = True):
    """The loss of one batch: frames f (B, C + 1, ...), macro-actions m (B, C, k). Returns the loss and the logged values."""
    if cfg.variant != "E0":
        raise ValueError(f"unknown variant {cfg.variant}")
    z = model.encoder(f)  # (B, C + 1, D)
    rec = token_losses(model.decoder(z), f).mean()
    return rec, {"z_std": z.detach().std(0).mean().item(), "recon": rec.item()}
