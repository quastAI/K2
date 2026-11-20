"""Readouts of a run's checkpoints (evaluation only: nothing here reaches a training loss).

1. What the latent keeps: a fresh decoder D' (and a linear probe) trained on frozen z of train frames with a
   class-balanced loss, scored on held-out frames (val) per class: recall per block type and mob, balanced accuracy
   over the values of each item, vital, facing and sleeping, light within a tolerance.
2. Kept against predictable: on the oracle's frames (`oracle_k<k>.npz`), per factor of the outcome frame, kept = D'
   decodes it correctly from z, predictable = the oracle's p at a line (all replays agreed; 0.9 and 0.75 as checks).
   A 2 x 2 table per kind and per class for the factors that change; the ones that do not change are reported apart
   (kept count only). The headlines are the share of predictable factors kept and the share of unpredictable ones.
3. Prediction: per class, the error of D'(z^_{j+h}) after an H-step rollout, over the error of copying the last
   context frame; also the error of D'(z_{j+h}) (the decoder's own floor) and the latent error over copying.
4. Geometry: along expert paths of H level-2 steps, the rank correlation of ||z_j - z_goal|| with the steps to go,
   and the share of steps that bring z closer to the goal.

    python -m k2.readouts runs/E1_k1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import layout as L
from .cache import load
from .data import Frames
from .losses import token_losses
from .model import Decoder, LinearDecoder
from .oracle import LINES, changing, load_oracle, predictable
from .train import limit_threads, load_config, load_model


@torch.no_grad()
def encode(encoder, frames: Frames, rows, batch=4096) -> torch.Tensor:
    rows = torch.as_tensor(rows, device=frames.device)
    return torch.cat([encoder(frames.frames(rows[a : a + batch])) for a in range(0, len(rows), batch)])


@torch.no_grad()
def decode(decoder, z: torch.Tensor, batch=4096) -> dict:
    outs = [decoder(z[a : a + batch]) for a in range(0, len(z), batch)]
    return {k: torch.cat([o[k] for o in outs]) for k in outs[0]}


def balanced_weights(f: dict, cap=100.0) -> dict:
    """Class weights for the readout's loss, from the frames it is trained on: N / (classes * count) per class (classes
    present only), capped; for the binary heads the positive weight negatives / positives, capped."""

    def balance(values, n):
        counts = torch.bincount(values.flatten(), minlength=n).float()
        present = counts > 0
        w = torch.zeros(n, device=values.device)
        w[present] = counts.sum() / (present.sum() * counts[present])
        return w.clamp(max=cap)

    pos = f["mob"].flatten(0, -2).mean(0)
    sleep = f["sleep"].float().mean()
    return {
        "block": balance(f["block"], L.NUM_BLOCKS),
        "item": balance(f["item"], L.LEVELS),
        "vital": balance(f["vital"], L.LEVELS),
        "facing": balance(f["facing"], 4),
        "mob": ((1 - pos) / pos.clamp(min=1e-6)).clamp(max=cap),
        "sleep": ((1 - sleep) / sleep.clamp(min=1e-6)).clamp(max=cap),
    }


def fit_decoder(z: torch.Tensor, f: dict, kind="fresh", steps=2000, batch=1024, lr=1e-3, seed=0, latent_cfg=None):
    """A decoder (`fresh`: the model's decoder architecture; `linear`: the probe) trained on frozen z (n, D)."""
    torch.manual_seed(seed)
    d = z.shape[-1]
    dec = (Decoder(d, *latent_cfg) if latent_cfg else Decoder(d)) if kind == "fresh" else LinearDecoder(d)
    dec = dec.to(z.device)
    w = balanced_weights(f)
    opt = torch.optim.Adam(dec.parameters(), lr=lr)
    gen = torch.Generator(device=z.device).manual_seed(seed)
    for _ in range(steps):
        i = torch.randint(len(z), (batch,), device=z.device, generator=gen)
        loss = token_losses(dec(z[i]), {k: v[i] for k, v in f.items()}, w).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return dec.eval()


def predicted(out: dict) -> dict:
    """The decoded frame of decoder outputs."""
    return {
        "block": out["block"].argmax(-1),
        "mob": (out["mob"] > 0).float(),
        "item": out["item"].argmax(-1),
        "vital": out["vital"].argmax(-1),
        "facing": out["facing"].argmax(-1),
        "light": out["light"],
        "sleep": (out["sleep"] > 0).long(),
    }


def correct(pred: dict, f: dict) -> np.ndarray:
    """Per factor (n, 82): the decoded frame matches the true one (a tile: its block and every mob bit)."""
    tile = (pred["block"] == f["block"]) & (pred["mob"] == f["mob"]).all(-1)
    out = torch.cat(
        [
            tile,
            pred["item"] == f["item"],
            pred["vital"] == f["vital"],
            (pred["facing"] == f["facing"])[..., None],
            ((pred["light"] - f["light"]).abs() <= L.LIGHT_TOLERANCE)[..., None],
            (pred["sleep"] == f["sleep"])[..., None],
        ],
        -1,
    )
    return out.cpu().numpy()


def class_scores(pred: dict, f: dict) -> dict:
    """Per class: block recall, mob balanced accuracy, balanced accuracy over the values of every item, vital, facing,
    sleeping; light within the tolerance. Also the head means and the mean over the rare classes."""
    pred = {k: v.cpu().numpy() for k, v in pred.items()}
    f = {k: v.cpu().numpy() for k, v in f.items()}

    def bal(p, t):
        vals = np.unique(t)
        return float(np.mean([np.mean(p[t == v] == v) for v in vals]))

    s = {}
    for b, name in enumerate(L.BLOCKS):
        m = f["block"] == b
        if m.any():
            s[name] = float(np.mean(pred["block"][m] == b))
    for j, name in enumerate(L.MOBS):
        t, p = f["mob"][..., j] > 0.5, pred["mob"][..., j] > 0.5
        if t.any():
            s[name] = float((np.mean(p[t]) + np.mean(~p[~t])) / 2)
    for i, name in enumerate(L.ITEMS):
        s[f"item_{name}"] = bal(pred["item"][:, i], f["item"][:, i])
    for i, name in enumerate(L.VITALS):
        s[name] = bal(pred["vital"][:, i], f["vital"][:, i])
    s["facing"] = bal(pred["facing"], f["facing"])
    s["light"] = float(np.mean(np.abs(pred["light"] - f["light"]) <= L.LIGHT_TOLERANCE))
    s["sleeping"] = bal(pred["sleep"], f["sleep"])
    blocks = [s[n] for n in L.BLOCKS if n in s]
    rare = [s[n] for n in L.RARE_BLOCKS + L.MOBS if n in s]
    summary = {"blocks": float(np.mean(blocks)), "mobs": float(np.mean([s[n] for n in L.MOBS if n in s])),
               "rare": float(np.mean(rare)), "items": float(np.mean([s[f"item_{n}"] for n in L.ITEMS])),
               "vitals": float(np.mean([s[n] for n in L.VITALS]))}  # fmt: skip
    return {"classes": s, "summary": summary}


def count_2x2(kept: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> dict:
    """Kept against predictable over the factors in `mask`."""
    return {
        "pred_kept": int((mask & pred & kept).sum()), "pred_dropped": int((mask & pred & ~kept).sum()),
        "unpred_kept": int((mask & ~pred & kept).sum()), "unpred_dropped": int((mask & ~pred & ~kept).sum()),
    }  # fmt: skip


def kept_vs_predictable(encoder, dec, val: Frames, oracle: dict) -> dict:
    """Kept against predictable on the oracle's outcome frames, per kind and per class, for the factors that change and
    (apart) the ones that do not. Kept: D'(z) decodes the factor of the outcome frame correctly. Per line of the
    oracle (`LINES`, keys "1", "0.9", "0.75") the 2 x 2 counts of the changing factors; the unchanged ones are
    deterministic by definition, so only their kept count is given."""
    k = int(oracle["k"])
    rows = torch.as_tensor(oracle["rows"] + k)
    kept = correct(predicted(decode(dec, encode(encoder, val, rows))), val.frames(rows))
    chg = changing(oracle, val.table)
    pred = {f"{line:g}": predictable(oracle["p"], line) for line in LINES}
    c = L.take(val.table, oracle["rows"] + k)

    def stats(mask):
        still, moving = mask & ~chg, mask & chg
        return {"unchanged": {"n": int(still.sum()), "kept": int((still & kept).sum())},
                "changing": {"n": int(moving.sum()), "kept": int((moving & kept).sum()),
                             "by_line": {line: count_2x2(kept, p, moving) for line, p in pred.items()}}}  # fmt: skip

    return {
        "lines": list(pred), "n_keys": int(oracle["n_keys"]), "all": stats(np.ones_like(kept)),
        "kinds": {n: stats(m) for n, m in L.kind_masks(c).items()},
        "classes": {n: stats(m) for n, m in L.factor_classes(c).items()},
    }  # fmt: skip


@torch.no_grad()
def prediction(model, dec, val: Frames, cfg, horizon=8, n=2048, seed=0) -> dict:
    """Per class and horizon h = 1 .. H: error of the decoded rollout, of the decoded true latent, and of copying."""
    pool = val.starts(cfg.k, cfg.context, cfg.sources, extra=horizon)
    g = np.random.default_rng(seed)
    t0 = pool[torch.as_tensor(g.choice(len(pool), min(n, len(pool)), replace=False), device=val.device)]
    f, m = val.steps(t0, cfg.k, cfg.context - 1 + horizon)
    z = model.encoder(f)
    c = cfg.context
    zhat = model.predictor.rollout(z[:, :c], m)
    last = {k: v[:, c - 1] for k, v in f.items()}
    out = {}
    for h in (1, horizon) if horizon > 1 else (1,):
        true = {k: v[:, c - 1 + h] for k, v in f.items()}
        ok_pred = correct(predicted(dec(zhat[:, h - 1])), true)
        ok_real = correct(predicted(dec(z[:, c - 1 + h])), true)
        tv = lambda x: L.token_values({**{k: v.cpu().numpy() for k, v in x.items()}, "mob": _mob_byte(x["mob"])})  # noqa: E731
        ok_copy = tv(last) == tv(true)
        classes = L.factor_classes({k: (v.cpu().numpy() if k != "mob" else _mob_byte(v)) for k, v in true.items()})
        per = {}
        for name, msk in classes.items():
            e_pred, e_real, e_copy = 1 - ok_pred[msk].mean(), 1 - ok_real[msk].mean(), 1 - ok_copy[msk].mean()
            per[name] = {"err_pred": float(e_pred), "err_decoded_true": float(e_real), "err_copy": float(e_copy),
                         "pred_over_copy": float(e_pred / e_copy) if e_copy > 0 else None}  # fmt: skip
        target = z[:, c - 1 + h]
        lat = ((zhat[:, h - 1] - target).square().sum(-1) / (z[:, c - 1] - target).square().sum(-1).clamp(min=1e-12))
        out[f"h{h}"] = {"classes": per, "latent_pred_over_copy": float(lat.median())}
    return out


def _mob_byte(mob: torch.Tensor) -> np.ndarray:
    """Mob bits (..., 63, 4) as the cache's bit byte (..., 63)."""
    return (mob.cpu().numpy() > 0.5).astype(np.uint8) @ (1 << np.arange(L.NUM_MOBS)).astype(np.uint8)


@torch.no_grad()
def geometry(encoder, val: Frames, k: int, horizon=8, n=512, seed=0) -> dict:
    """Along expert paths of `horizon` level-2 steps: Spearman of distance to the goal with the steps to go, and the
    share of steps that bring the latent closer to the goal."""
    pool = val.starts(k, horizon, sources=(0,))
    g = np.random.default_rng(seed)
    t0 = pool[torch.as_tensor(g.choice(len(pool), min(n, len(pool)), replace=False), device=val.device)]
    f, _ = val.steps(t0, k, horizon)
    z = encoder(f)
    d = (z - z[:, -1:]).norm(dim=-1)[:, :-1].cpu().numpy()  # (n, H): distance at steps to go H .. 1
    togo = np.arange(horizon, 0, -1)
    rank = lambda x: np.argsort(np.argsort(x, -1), -1)  # noqa: E731
    rd, rt = rank(d), rank(np.broadcast_to(togo, d.shape))
    rho = [np.corrcoef(a, b)[0, 1] if a.std() > 0 else np.nan for a, b in zip(rd, rt)]
    closer = np.diff(np.concatenate([d, np.zeros((len(d), 1))], 1), axis=1) < 0
    return {"spearman": float(np.nanmean(rho)), "closer_share": float(closer.mean())}


def readout_checkpoint(path, cache, device, decoder_steps=2000, n_train=65536, n_eval=16384, horizon=8,
                       frames=None) -> dict:  # fmt: skip
    cfg, model = load_model(path, device)
    train_frames, val = frames
    g = np.random.default_rng(0)
    tr = torch.as_tensor(g.choice(len(train_frames), min(n_train, len(train_frames)), replace=False))
    ev = torch.as_tensor(g.choice(len(val), min(n_eval, len(val)), replace=False))
    z_tr, z_ev = encode(model.encoder, train_frames, tr), encode(model.encoder, val, ev)
    f_tr, f_ev = train_frames.frames(tr), val.frames(ev)
    res = {}
    decs = {}
    for kind in ("fresh", "linear"):
        dec = fit_decoder(z_tr, f_tr, kind, decoder_steps, latent_cfg=(cfg.width, cfg.dec_layers, cfg.heads))
        decs[kind] = dec
        res[kind] = class_scores(predicted(decode(dec, z_ev)), f_ev)
    try:
        res["kept"] = kept_vs_predictable(model.encoder, decs["fresh"], val, load_oracle(cfg.k, cache))
    except FileNotFoundError:
        res["kept"] = None
    if model.predictor is not None:
        res["prediction"] = prediction(model, decs["fresh"], val, cfg, horizon)
    res["geometry"] = geometry(model.encoder, val, cfg.k, horizon)
    return res


def run_readouts(run_dir, device=None, decoder_steps=2000, n_train=65536, n_eval=16384, horizon=8) -> dict:
    """Readouts of every checkpoint of a run into `<run>/readouts.json` (steps already there are skipped)."""
    run_dir = Path(run_dir)
    cfg = load_config(run_dir)
    device = device or cfg.device
    path = run_dir / "readouts.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    frames = (Frames(load("train", cfg.cache), device), Frames(load("val", cfg.cache), device))
    for ckpt in sorted(run_dir.glob("ckpt_*.pt")):
        step = int(ckpt.stem.split("_")[1])
        if str(step) in results:
            continue
        results[str(step)] = readout_checkpoint(ckpt, cfg.cache, device, decoder_steps, n_train, n_eval, horizon,
                                                frames)  # fmt: skip
        path.write_text(json.dumps(results, indent=1))
        print(f"readouts of step {step}: rare {results[str(step)]['fresh']['summary']['rare']:.3f}")
    return results


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run")
    p.add_argument("--device", default=None)
    p.add_argument("--decoder-steps", type=int, default=2000)
    p.add_argument("--n-train", type=int, default=65536)
    p.add_argument("--n-eval", type=int, default=16384)
    p.add_argument("--horizon", type=int, default=8)
    a = p.parse_args(argv)
    limit_threads()
    run_readouts(a.run, a.device, a.decoder_steps, a.n_train, a.n_eval, a.horizon)


if __name__ == "__main__":
    main()
