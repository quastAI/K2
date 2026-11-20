"""The losses of the variants. They see only the input frames and the actions.

    E0  reconstruction D(z) -> x of every frame, the mean over the 82 tokens                 capacity reference
    E1  reconstruction of every frame (error-weighted, the sum over the tokens: nats per frame)
        + lam * whitened prediction error of z_{t+k}, summed over the latent's directions     the coarse-graining test

E1's prediction error is measured in the coordinates that whiten the target batch: shrinking the latent or one of its
directions buys nothing, and each direction costs its own unpredictable share (about 1 for a direction the predictor
cannot foresee at all), whatever the latent size. The sum over the tokens makes lam a price in nats per frame: a fully
unpredictable direction stays only if it carries more than about lam nats per frame about the frame. `target_from`:
the step from which the prediction's gradient reaches the target latent z_{t+k} (0: from the start; -1: never, a
stop-gradient). The error weights are computed from the input's own reconstruction error (no labels).
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


def error_weights(loss: torch.Tensor, power: float, cap: float) -> torch.Tensor:
    """Weights from the losses' own size (stop-gradient): (loss / mean)^power, normalised to mean 1, capped."""
    with torch.no_grad():
        w = (loss / loss.mean().clamp(min=1e-12)).clamp(min=1e-3) ** power
        return (w / w.mean()).clamp(max=cap)


def whitened_error(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """E[e^T C^-1 e] with e = pred - target and C the target batch's covariance (+ eps x its mean variance, relative):
    the prediction error summed over the target's whitened directions. pred, target (N, D)."""
    t = target.double()
    tc = t - t.mean(0)
    cov = tc.T @ tc / max(len(t) - 1, 1)
    cov = cov + eps * cov.diagonal().mean() * torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)
    chol = torch.linalg.cholesky(cov)
    e = torch.linalg.solve_triangular(chol, (pred.double() - t).T, upper=False)  # (D, N): whitened errors
    return e.square().sum(0).mean().to(pred.dtype)


def e1_loss(model, f: dict, m: torch.Tensor, cfg, step: int, diag: bool):
    """E1. The target z_{t+k} gets the prediction's gradient from step `cfg.target_from` on (never if it is -1)."""
    z = model.encoder(f)  # (B, C + 1, D)
    to_target = cfg.target_from >= 0 and step >= cfg.target_from
    target = z[:, 1:] if to_target else z[:, 1:].detach()
    pred = model.predictor(z[:, :-1], m)
    pe = whitened_error(pred.flatten(0, 1), target.flatten(0, 1), cfg.white_eps)
    tok = token_losses(model.decoder(z), f)  # (B, C + 1, 82)
    loss = cfg.lam * pe
    if cfg.rec_weight > 0:
        loss = loss + cfg.rec_weight * (error_weights(tok, cfg.error_power, cfg.error_cap) * tok).sum(-1).mean()
    logs = {"z_std": z.detach().std(0).mean().item(), "pred_white": pe.item(), "to_target": float(to_target)}
    if diag:
        with torch.no_grad():
            copy = whitened_error(z[:, :-1].flatten(0, 1), z[:, 1:].flatten(0, 1), cfg.white_eps)
        logs.update(recon=tok.detach().sum(-1).mean().item(), copy_white=copy.item())
    return loss, logs


def variant_loss(model, f: dict, m: torch.Tensor, cfg, step: int = 0, diag: bool = True):
    """The loss of one batch: frames f (B, C + 1, ...), macro-actions m (B, C, k). Returns the loss and the logged values."""
    if cfg.variant == "E1":
        return e1_loss(model, f, m, cfg, step, diag)
    if cfg.variant != "E0":
        raise ValueError(f"unknown variant {cfg.variant}")
    z = model.encoder(f)  # (B, C + 1, D)
    rec = token_losses(model.decoder(z), f).mean()
    return rec, {"z_std": z.detach().std(0).mean().item(), "recon": rec.item()}
