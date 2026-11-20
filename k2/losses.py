"""Losses of the variants. They see only the input frames and the actions.

    E0  reconstruction D(z_j) -> x_j                                                     capacity check
    E1  ||z^_{j+1} - z_{j+1}||^2 (gradient through the target) + SIGReg                  pruning reference (A3)
    E2  E1 with the target detached                                                       A4 analogue
    E3  the Grace model (Grace/docs/aac_recipe.md), two stages in one run, no SIGReg, target always detached:
        stage 1 (step < switch)   error-weighted reconstruction D(z) -> x of every frame + ||z^_{j+1} - sg(z_{j+1})||^2
        stage 2 (step >= switch)  decoded prediction D(z^_{j+1}) -> x_{j+1}
                                  + prediction-gated reconstruction exp(-sg(e)/tau) * [D(z_{j+1}) -> x_{j+1}] per token,
                                    e = the token's loss of the decoded prediction
                                  + ||z^_{j+1} - sg(z_{j+1})||^2

The target carries the gradient in E1 and is detached (stop-gradient) in E2 and E3.

E3's error weight is computed from the input's own reconstruction error (no labels): each token's loss over the batch
mean, to the power `error_power`, with a stop-gradient, normalised to mean 1 and capped at `error_cap`. Badly
reconstructed tokens count more, so rare content a plain mean would neglect is pulled in. The gate keeps in the latent
what the predictor foresees (a token whose decoded prediction is right) and pays nothing for what it cannot foresee.

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


def error_weights(loss: torch.Tensor, power: float, cap: float) -> torch.Tensor:
    """Weights from the losses' own size (stop-gradient): (loss / mean)^power, normalised to mean 1, capped."""
    with torch.no_grad():
        w = (loss / loss.mean().clamp(min=1e-12)).clamp(min=1e-3) ** power
        return (w / w.mean()).clamp(max=cap)


def grace_loss(model, f: dict, m: torch.Tensor, cfg, z: torch.Tensor, step: int, diag: bool, logs: dict):
    """E3. `diag`: also compute the other stage's terms without gradient, so that every logged row has the same keys."""
    stage2 = step >= cfg.switch
    pred = model.predictor(z[:, :-1], m)
    pred_loss = (pred - z[:, 1:].detach()).square().mean()
    f_next = {key: x[:, 1:] for key, x in f.items()}
    if stage2 or diag:  # in stage 1 only a diagnostic: what the gate would do
        with torch.set_grad_enabled(stage2):
            dp = token_losses(model.decoder(pred), f_next)  # (B, C, tokens): the decoded prediction against the next frame
            gate = torch.exp(-dp.detach() / cfg.gate_tau)
            gated = (gate * token_losses(model.decoder(z[:, 1:]), f_next)).mean()
    if stage2:
        loss = pred_loss + dp.mean() + gated
        if diag:
            with torch.no_grad():
                rec = token_losses(model.decoder(z), f).mean()
    else:
        rec_all = token_losses(model.decoder(z), f)  # (B, C + 1, tokens): every frame of the step
        rec = rec_all.mean()
        loss = pred_loss + cfg.rec_weight * (error_weights(rec_all, cfg.error_power, cfg.error_cap) * rec_all).mean()
    if diag:
        with torch.no_grad():
            copy = (z[:, :-1] - z[:, 1:]).square().mean()
            logs.update(stage=2 if stage2 else 1, recon=rec.item(), pred=pred_loss.item(), copy=copy.item(),
                        pred_over_copy=(pred_loss / copy.clamp(min=1e-12)).item(), decoded_pred=dp.mean().item(),
                        gated_recon=gated.item(), gate_open=gate.mean().item(), gate_shut=(gate < 0.1).float().mean().item())  # fmt: skip
    return loss, logs


def variant_loss(model, f: dict, m: torch.Tensor, cfg, sigreg: SIGReg | None = None, step: int = 0, diag: bool = True):
    """The loss of one batch: frames f (B, C + 1, ...), macro-actions m (B, C, k); `step` selects E3's stage.
    Returns the loss and the logged values (floats; the diagnostic ones without gradient, filled when `diag`)."""
    v = cfg.variant
    z = model.encoder(f)  # (B, C + 1, D)
    logs = {"z_std": z.detach().std(0).mean().item()}
    if v == "E0":
        rec = token_losses(model.decoder(z), f).mean()
        return rec, {**logs, "recon": rec.item()}
    if v == "E3":
        return grace_loss(model, f, m, cfg, z, step, diag, logs)
    if v not in ("E1", "E2"):
        raise ValueError(f"unknown variant {v}")
    target = z[:, 1:] if v == "E1" else z[:, 1:].detach()
    pred = model.predictor(z[:, :-1], m)
    pred_loss = (pred - target).square().mean()
    with torch.no_grad():
        copy = (z[:, :-1] - z[:, 1:]).square().mean()
        logs.update(pred=pred_loss.item(), copy=copy.item(), pred_over_copy=(pred_loss / copy.clamp(min=1e-12)).item())
    sig = sigreg(z.transpose(0, 1))
    logs.update(sigreg=sig.item())
    return pred_loss + cfg.sigreg_weight * sig, logs
