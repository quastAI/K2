"""Losses of the variants. They see only the input frames and the actions.

    E0  reconstruction D(z_j) -> x_j                                                     capacity check
    E1  ||z^_{j+1} - z_{j+1}||^2 (gradient through the target) + SIGReg                  pruning reference (A3)
    E2  E1 with the target detached                                                       A4 analogue
    E3  detached prediction + decoded prediction D(z^_{j+1}) -> x_{j+1}
        + reconstruction D(z_{j+1}) -> x_{j+1}                                            keep everything (Sieve stage 1)
    E4  E3 with the reconstruction gated per token                                        V5's relative gate

E3 and E4 reconstruct the same frames (the predicted ones, x_1 .. x_C), so the gate is their only difference.
The gate: G = floor + (1 - floor) exp(-g / (beta mean g)), g the per-token divergence between D(z^_{j+1}) and
D(sg(z_{j+1})), both detached; the mean runs over the batch's tokens (kept on purpose, as in V5).
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


def token_divergence(pred: dict, target: dict) -> torch.Tensor:
    """Per-token divergence (..., 82) of two decoder outputs, detached: KL(target || pred) of every categorical and
    binary head (a tile sums its block and mob terms), squared difference of the light level."""
    pred = {k: v.detach() for k, v in pred.items()}
    target = {k: v.detach() for k, v in target.items()}

    def kl(p_logits, q_logits):
        return (q_logits.softmax(-1) * (q_logits.log_softmax(-1) - p_logits.log_softmax(-1))).sum(-1)

    def bkl(p_logits, q_logits):
        q = torch.sigmoid(q_logits)
        return q * (F.logsigmoid(q_logits) - F.logsigmoid(p_logits)) + (1 - q) * (
            F.logsigmoid(-q_logits) - F.logsigmoid(-p_logits)
        )

    tile = kl(pred["block"], target["block"]) + bkl(pred["mob"], target["mob"]).sum(-1)
    return torch.cat(
        [
            tile,
            kl(pred["item"], target["item"]),
            kl(pred["vital"], target["vital"]),
            kl(pred["facing"], target["facing"])[..., None],
            (pred["light"] - target["light"]).square()[..., None],
            bkl(pred["sleep"], target["sleep"])[..., None],
        ],
        -1,
    )


def relative_gate(g: torch.Tensor, floor=0.1, beta=1.0) -> torch.Tensor:
    """floor + (1 - floor) exp(-g / (beta mean g)): scale-free, about uniform when the errors are alike, never below the
    floor. Call it on detached errors."""
    return floor + (1 - floor) * torch.exp(-g / (beta * g.mean().clamp(min=1e-12)))


def variant_loss(model, f: dict, m: torch.Tensor, cfg, sigreg: SIGReg | None = None) -> tuple[torch.Tensor, dict]:
    """The loss of one batch: frames f (B, C + 1, ...), macro-actions m (B, C, k). Returns the loss and the logged
    values (floats; the diagnostic ones without gradient)."""
    v = cfg.variant
    z = model.encoder(f)  # (B, C + 1, D)
    nxt = {key: x[:, 1:] for key, x in f.items()}
    logs = {"z_std": z.detach().std(0).mean().item()}
    if v == "E0":
        rec = token_losses(model.decoder(z), f).mean()
        return rec, {**logs, "recon": rec.item()}
    target = z[:, 1:] if v == "E1" else z[:, 1:].detach()
    pred = model.predictor(z[:, :-1], m)
    pred_loss = (pred - target).square().mean()
    with torch.no_grad():
        copy = (z[:, :-1] - z[:, 1:]).square().mean()
        logs.update(pred=pred_loss.item(), copy=copy.item(), pred_over_copy=(pred_loss / copy.clamp(min=1e-12)).item())
    if v in ("E1", "E2"):
        sig = sigreg(z.transpose(0, 1))
        logs["sigreg"] = sig.item()
        return pred_loss + cfg.sigreg_weight * sig, logs
    decoded_pred = model.decoder(pred)
    decoded_real = model.decoder(z[:, 1:])
    dp = token_losses(decoded_pred, nxt)
    rec = token_losses(decoded_real, nxt)
    if v == "E4":
        gate = relative_gate(token_divergence(decoded_pred, decoded_real), cfg.gate_floor, cfg.gate_beta)
        with torch.no_grad():
            logs.update(gate_mean=gate.mean().item(), gate_std=gate.std().item(),
                        gate_shut=(gate < cfg.gate_floor + 0.05 * (1 - cfg.gate_floor)).float().mean().item())  # fmt: skip
    elif v == "E3":
        gate = torch.ones_like(rec)
    else:
        raise ValueError(f"unknown variant {v}")
    gated = (gate * rec).mean()
    logs.update(decoded_pred=dp.mean().item(), recon=rec.mean().item(), gated_recon=gated.item())
    return pred_loss + dp.mean() + gated, logs
