"""Losses of the variants. They see only the input frames and the actions.

    E0  reconstruction D(z_j) -> x_j                                                     capacity check
    E1  ||z^_{j+1} - z_{j+1}||^2 (gradient through the target) + SIGReg                  pruning reference (A3)
    E2  E1 with the target detached                                                       A4 analogue
    E3  E2 until the predictor plateaus on every latent dimension, then the target's gradient ramps up to E1's

The target's gradient share g: target = g z + (1 - g) sg(z), the value z, the gradient scaled by g (E1: 1, E2: 0,
E3: from `TargetRamp`).

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


class TargetRamp:
    """E3's target-gradient share g. It is 0 (E2) until the predictor's error has plateaued on every latent dimension,
    then rises linearly to 1 (E1) over `ramp_steps` and stays there; the trigger fires once.

    Plateau: an EMA of the per-dimension squared prediction error (time constant window / 5 steps) is compared every
    `window` steps with its value one window earlier; the predictor has plateaued when every dimension improved by less
    than `tol` (relative; a dimension whose error grew counts as plateaued). Checks start after `start` steps (the
    learning-rate warmup). It reads only the training loss."""

    def __init__(self, window=250, tol=0.01, ramp_steps=1000, start=0):
        if window < 1 or tol <= 0 or ramp_steps < 1:
            raise ValueError(f"need window >= 1, tol > 0, ramp_steps >= 1; got {window}, {tol}, {ramp_steps}")
        self.window, self.tol, self.ramp_steps, self.start = window, tol, ramp_steps, start
        self.decay = max(0.0, 1 - 5 / window)  # no smoothing for a window under 5 steps
        self.ema = self.snapshot = None
        self.trigger = None  # the step the ramp started at
        self.plateau_share = 0.0  # share of the dimensions plateaued at the last check

    def value(self, step: int) -> float:
        if self.trigger is None:
            return 0.0
        return min(1.0, max(0.0, (step - self.trigger) / max(self.ramp_steps, 1)))

    def marks(self) -> set[int]:
        """The steps worth a checkpoint: the trigger and the end of the ramp."""
        return set() if self.trigger is None else {self.trigger, self.trigger + self.ramp_steps}

    def update(self, step: int, err: torch.Tensor) -> bool:
        """Feed the per-dimension squared error (D,) of the step just taken; returns True when the ramp starts now."""
        err = err.detach().float().cpu()
        self.ema = err.clone() if self.ema is None else self.decay * self.ema + (1 - self.decay) * err
        if self.trigger is not None or step < self.start or step % self.window:
            return False
        if self.snapshot is not None:
            gain = (self.snapshot - self.ema) / self.snapshot.clamp(min=1e-12)
            plateaued = gain < self.tol
            self.plateau_share = plateaued.float().mean().item()
            if plateaued.all():
                self.trigger = step
        self.snapshot = self.ema.clone()
        return self.trigger == step

    def state_dict(self) -> dict:
        return {k: getattr(self, k) for k in ("ema", "snapshot", "trigger", "plateau_share")}

    def load_state_dict(self, state: dict):
        for k, v in state.items():
            setattr(self, k, v)


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


def variant_loss(model, f: dict, m: torch.Tensor, cfg, sigreg: SIGReg | None = None, target_grad: float | None = None):
    """The loss of one batch: frames f (B, C + 1, ...), macro-actions m (B, C, k); `target_grad` is E3's share g.
    Returns the loss, the logged values (floats; the diagnostic ones without gradient) and the per-dimension squared
    prediction error (D,), detached (None for E0)."""
    v = cfg.variant
    z = model.encoder(f)  # (B, C + 1, D)
    logs = {"z_std": z.detach().std(0).mean().item()}
    if v == "E0":
        rec = token_losses(model.decoder(z), f).mean()
        return rec, {**logs, "recon": rec.item()}, None
    g = {"E1": 1.0, "E2": 0.0}.get(v, target_grad)
    if v not in ("E1", "E2", "E3") or g is None:
        raise ValueError(f"unknown variant {v}, or E3 without target_grad")
    target = z[:, 1:] if g == 1 else z[:, 1:].detach() if g == 0 else g * z[:, 1:] + (1 - g) * z[:, 1:].detach()
    pred = model.predictor(z[:, :-1], m)
    sq = (pred - target).square()
    pred_loss = sq.mean()
    with torch.no_grad():
        copy = (z[:, :-1] - z[:, 1:]).square().mean()
        logs.update(pred=pred_loss.item(), copy=copy.item(), pred_over_copy=(pred_loss / copy.clamp(min=1e-12)).item())
    sig = sigreg(z.transpose(0, 1))
    logs.update(sigreg=sig.item(), target_grad=g)
    return pred_loss + cfg.sigreg_weight * sig, logs, sq.detach().mean((0, 1))
