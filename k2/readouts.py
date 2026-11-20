"""Readouts of a run's checkpoints (evaluation only: nothing here reaches a training loss).

1. What the latent holds: a fresh decoder D' (and a linear probe) trained on frozen z of train frames with a
   class-balanced loss, scored on held-out frames (val) per class: recall per block type and mob, balanced accuracy
   over the values of each item, vital, facing and sleeping, light within a tolerance. The readouts read z whitened on
   their train frames (invertible, so the information is the same; a raw latent stretched along a few directions
   starves a short readout). The summaries leave out the classes the readout cannot read even from the raw state
   (`layout.LOW_CEILING`).
2. Prediction: per class, the error of D'(z^_{j+h}) after an H-step rollout, over the error of copying the last
   context frame; also the error of D'(z_{j+h}) (the decoder's own floor) and the latent error over copying.
3. The raw latent's shape: participation ratio and rank of z, and the median slowness of its whitened directions over
   one level-2 step (0: constant, 2: unrelated).
4. Capacity: the model's own decoder on val frames, the share of frames and of factors reconstructed exactly.
5. What the planner's distance sees: how far the raw latent moves when one factor of the state is edited, in units of
   the median distance between frames one level-2 step apart.

    python -m k2.readouts runs/E0_k1
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
from .train import limit_threads, load_config, load_model

VERSION = 3  # readouts of another version are computed again


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


class Whitened(torch.nn.Module):
    """A readout that whitens its input first: (z - mean) W with W from the eigenvectors of z's covariance on the
    readout's train frames (directions below 1e-6 of the largest variance are floored there)."""

    def __init__(self, dec: torch.nn.Module, z: torch.Tensor):
        super().__init__()
        self.dec = dec
        zd = z.detach().double().cpu()
        mean = zd.mean(0)
        ev, U = torch.linalg.eigh(torch.cov((zd - mean).T))
        ev = ev.clamp(min=0) + 1e-6 * ev.max()
        self.register_buffer("mean", mean.float().to(z.device))
        self.register_buffer("W", (U / ev.sqrt()).float().to(z.device))

    def white(self, z: torch.Tensor) -> torch.Tensor:
        return (z - self.mean) @ self.W

    def forward(self, z: torch.Tensor) -> dict:
        return self.dec(self.white(z))


def fit_decoder(z: torch.Tensor, f: dict, kind="fresh", steps=2000, batch=1024, lr=1e-3, seed=0, latent_cfg=None):
    """A readout (`fresh`: the model's decoder architecture; `linear`: the probe) trained on frozen, whitened z (n, D)."""
    torch.manual_seed(seed)
    d = z.shape[-1]
    dec = (Decoder(d, *latent_cfg) if latent_cfg else Decoder(d)) if kind == "fresh" else LinearDecoder(d)
    dec = Whitened(dec.to(z.device), z)
    zw = dec.white(z)
    w = balanced_weights(f)
    opt = torch.optim.Adam(dec.dec.parameters(), lr=lr)
    gen = torch.Generator(device=z.device).manual_seed(seed)
    for _ in range(steps):
        i = torch.randint(len(z), (batch,), device=z.device, generator=gen)
        loss = token_losses(dec.dec(zw[i]), {k: v[i] for k, v in f.items()}, w).mean()
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
    """Per factor (n, 145): the decoded frame matches the true one (a tile's block; a tile's mobs: every mob bit)."""
    out = torch.cat(
        [
            pred["block"] == f["block"],
            (pred["mob"] == f["mob"]).all(-1),
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
    sleeping; light within the tolerance. Also the head means and the mean over the rare classes, without the classes
    the readout cannot read even from the raw state (`LOW_CEILING`)."""
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
    readable = lambda names: [s[n] for n in names if n in s and n not in L.LOW_CEILING]  # noqa: E731
    summary = {"blocks": float(np.mean(readable(L.BLOCKS))), "mobs": float(np.mean(readable(L.MOBS))),
               "rare": float(np.mean(readable(L.RARE_BLOCKS + L.MOBS))),
               "items": float(np.mean(readable([f"item_{n}" for n in L.ITEMS]))),
               "vitals": float(np.mean(readable(L.VITALS)))}  # fmt: skip
    return {"classes": s, "summary": summary}


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
        tv = lambda x: L.factor_values({**{k: v.cpu().numpy() for k, v in x.items()}, "mob": _mob_byte(x["mob"])})  # noqa: E731
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
def latent_stats(encoder, val: Frames, k: int, n=8192, seed=0) -> dict:
    """Participation ratio and rank (eigenvalues above 1e-6 of the largest) of z, and the median slowness of z's
    whitened directions over one level-2 step: per direction E[dy^2] with unit variance (0 constant, 2 unrelated)."""
    pool = val.starts(k, 1)
    g = np.random.default_rng(seed)
    t0 = pool[torch.as_tensor(g.choice(len(pool), min(n, len(pool)), replace=False), device=val.device)]
    f, _ = val.steps(t0, k, 1)
    z = torch.cat([encoder({key: x[a : a + 2048] for key, x in f.items()}) for a in range(0, len(t0), 2048)])
    z = z.double().cpu()
    x = z.flatten(0, 1)
    ev, U = torch.linalg.eigh(torch.cov((x - x.mean(0)).T))
    keep = ev > ev.max() * 1e-6
    d = (z[:, 1] - z[:, 0]) @ (U[:, keep] / ev[keep].sqrt())
    slow = torch.linalg.eigvalsh(d.T @ d / len(d))
    return {"participation_ratio": float(ev.sum() ** 2 / (ev**2).sum()), "rank": int(keep.sum()),
            "slowness_median": float(slow.median())}  # fmt: skip


@torch.no_grad()
def own_reconstruction(model, val: Frames, rows) -> dict:
    """The model's own decoder on val frames: the share of frames reconstructed exactly (all 145 factors), the share of
    factors, and the class scores. Does the latent hold the whole state?"""
    f = val.frames(torch.as_tensor(rows))
    pred = predicted(decode(model.decoder, encode(model.encoder, val, rows)))
    ok = correct(pred, f)
    return {"frames_exact": float(ok.all(1).mean()), "factors_exact": float(ok.mean()), **class_scores(pred, f)}


EDIT_BLOCKS = ["grass", "water", "stone", "tree", "sand", "path", "coal", "iron", "diamond", "crafting_table", "furnace",
               "lava", "plant"]  # fmt: skip
RINGS = {1: [(3, 3), (3, 5), (2, 4), (4, 4)], 3: [(3, 1), (3, 7), (0, 4), (6, 4)]}  # tiles 1 and 3 away from the player


def _frames_of(c: dict, device) -> dict:
    """Torch frames from a compact batch."""
    tmp = Frames({**c, "action": np.zeros(len(c["block"]), np.int8)}, device)
    return tmp.frames(torch.arange(len(c["block"]), device=tmp.device))


@torch.no_grad()
def sensitivity(encoder, val: Frames, k: int, n=1024, seed=0) -> dict:
    """What the planner's distance ||z - z_goal|| sees of each factor: edit one factor of a val state and measure how far
    the raw latent moves, in units of the median distance between frames one level-2 step (k frames) apart. Per edit,
    the median over the frames where the edit changes something. Edited states need not be reachable in the game.
    `unrelated frame`: the distance to another, random val frame, for scale."""
    g = np.random.default_rng(seed)
    pool = val.starts(k, 1).cpu().numpy()
    t0 = g.choice(pool, min(n, len(pool)), replace=False)
    c = L.take(val.table, t0)

    def enc(cc):
        f = _frames_of(cc, val.device)
        return torch.cat([encoder({key: x[a : a + 2048] for key, x in f.items()}) for a in range(0, len(t0), 2048)])

    z0 = enc(c)
    unit = (encode(encoder, val, t0 + k) - z0).norm(dim=-1).median().item()
    out = {"unrelated frame": float((encode(encoder, val, g.permutation(t0)) - z0).norm(dim=-1).median() / unit)}

    def record(name, cc, changed):
        if changed.any():
            idx = torch.as_tensor(np.nonzero(changed)[0], device=z0.device)
            out[name] = float(((enc(cc) - z0)[idx]).norm(dim=-1).median() / unit)

    rows = np.arange(len(t0))
    for r, tiles in RINGS.items():
        pick = np.array([t[0] * L.COLS + t[1] for t in tiles])[g.integers(0, len(tiles), len(t0))]
        for b in EDIT_BLOCKS:
            cc = {key: v.copy() for key, v in c.items()}
            bi = L.BLOCKS.index(b)
            changed = cc["block"][rows, pick] != bi
            cc["block"][rows, pick] = bi
            record(f"block {b}, {r} away", cc, changed)
        for j, m in enumerate(L.MOBS):
            cc = {key: v.copy() for key, v in c.items()}
            bit = np.uint8(1 << j)
            changed = (cc["mob"][rows, pick] & bit) == 0
            cc["mob"][rows, pick] |= bit
            record(f"mob {m}, {r} away", cc, changed)
    for field, names in (("item", L.ITEMS), ("vital", L.VITALS)):
        for i, name in enumerate(names):
            cc = {key: v.copy() for key, v in c.items()}
            v = cc[field][:, i]
            cc[field][:, i] = np.where(v < L.LEVELS - 1, v + 1, v - 1)
            record(f"{field} {name} ±1", cc, np.ones(len(t0), bool))
    cc = {key: v.copy() for key, v in c.items()}
    cc["facing"] = (cc["facing"] + 1) % 4
    record("facing", cc, np.ones(len(t0), bool))
    cc = {key: v.copy() for key, v in c.items()}
    cc["sleep"] = 1 - cc["sleep"]
    record("sleeping", cc, np.ones(len(t0), bool))
    return {"k": k, "unit": unit, "edits": out}


def readout_checkpoint(path, device, decoder_steps=2000, n_train=65536, n_eval=16384, horizon=8,
                       frames=None, stride=None) -> dict:  # fmt: skip
    """`stride`: the level-2 stride the latent's slowness and the edit distances are measured in (default: the run's k)."""
    cfg, model = load_model(path, device)
    stride = stride or cfg.k
    train_frames, val = frames
    g = np.random.default_rng(0)
    tr = torch.as_tensor(g.choice(len(train_frames), min(n_train, len(train_frames)), replace=False))
    ev = torch.as_tensor(g.choice(len(val), min(n_eval, len(val)), replace=False))
    z_tr, z_ev = encode(model.encoder, train_frames, tr), encode(model.encoder, val, ev)
    f_tr, f_ev = train_frames.frames(tr), val.frames(ev)
    res = {"version": VERSION}
    decs = {}
    for kind in ("fresh", "linear"):
        dec = fit_decoder(z_tr, f_tr, kind, decoder_steps, latent_cfg=(cfg.width, cfg.dec_layers, cfg.heads))
        decs[kind] = dec
        res[kind] = class_scores(predicted(decode(dec, z_ev)), f_ev)
    if model.predictor is not None:
        res["prediction"] = prediction(model, decs["fresh"], val, cfg, horizon)
    res["latent"] = latent_stats(model.encoder, val, stride)
    if model.decoder is not None:
        res["own"] = own_reconstruction(model, val, ev)
    res["sensitivity"] = sensitivity(model.encoder, val, stride)
    return res


def run_readouts(run_dir, device=None, decoder_steps=2000, n_train=65536, n_eval=16384, horizon=8, cache=None,
                 stride=None) -> dict:  # fmt: skip
    """Readouts of every checkpoint of a run into `<run>/readouts.json` (steps already read out by this version are
    skipped). `cache` overrides the run's cache folder (a run trained on another machine)."""
    run_dir = Path(run_dir)
    cfg = load_config(run_dir)
    device = device or cfg.device
    cache = cache or cfg.cache
    path = run_dir / "readouts.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    stale = {s: r for s, r in results.items() if r.get("version") != VERSION}
    if stale:  # readouts of an older version are kept once, under their version's name
        backup = run_dir / f"readouts_v{next(iter(stale.values())).get('version', 1)}.json"
        if not backup.exists():
            backup.write_text(json.dumps(stale, indent=1))
    results = {s: r for s, r in results.items() if r.get("version") == VERSION}
    frames = (Frames(load("train", cache), device), Frames(load("val", cache), device))
    for ckpt in sorted(run_dir.glob("ckpt_*.pt")):
        step = int(ckpt.stem.split("_")[1])
        if str(step) in results:
            continue
        results[str(step)] = readout_checkpoint(ckpt, device, decoder_steps, n_train, n_eval, horizon, frames, stride)
        path.write_text(json.dumps(results, indent=1))
        print(f"readouts of step {step}: rare {results[str(step)]['fresh']['summary']['rare']:.3f}")
    return results


# --- the content at three resolutions: what z holds, what the predictor foresees, what copying gives -----------------

RES_CLASSES = ["grass", "water", "stone", "tree", "sand", "path", "coal", "iron", "diamond", "crafting_table", "furnace",
               "lava", "plant"] + L.MOBS  # fmt: skip
GROUPS = {"terrain": ["grass", "water", "stone", "tree", "sand", "path"],
          "ores and stations": ["coal", "iron", "diamond", "crafting_table", "furnace"],
          "mobs": ["zombie", "cow", "skeleton"]}  # fmt: skip
_r, _c = np.divmod(np.arange(L.TILES), L.COLS)
QUADRANTS = np.stack([(_r < L.PLAYER[0]) & (_c < L.PLAYER[1]), (_r < L.PLAYER[0]) & (_c > L.PLAYER[1]),
                      (_r > L.PLAYER[0]) & (_c < L.PLAYER[1]), (_r > L.PLAYER[0]) & (_c > L.PLAYER[1])]).astype(np.float32)  # fmt: skip


def content(x: np.ndarray) -> dict:
    """The content of frames given as `state` vectors (n, 1345): per class its tiles (63 indicators), its count per
    quadrant (4, without the player's row and column) and its count in view (1); the items, vitals, facing, sleeping."""
    tiles = x[:, : L.VIEW_DIM].reshape(-1, L.TILES, L.CHANNELS)
    out = {}
    for name in RES_CLASSES:
        t = tiles[:, :, L.BLOCKS.index(name) if name in L.BLOCKS else L.NUM_BLOCKS + L.MOBS.index(name)]
        out[f"{name}|tile"], out[f"{name}|quadrant"], out[f"{name}|view"] = t, t @ QUADRANTS.T, t.sum(1, keepdims=True)
    v = L.VIEW_DIM
    out["items|-"], out["vitals|-"] = x[:, v : v + 12], x[:, v + 12 : v + 16]
    out["facing|-"], out["sleeping|-"] = x[:, v + 16 : v + 20], x[:, v + 21 : v + 22]
    return out


def _r2(pred: np.ndarray, y: np.ndarray) -> float:
    sst = ((y - y.mean(0)) ** 2).sum(0)
    ok = sst > 0
    return float(1 - ((pred - y) ** 2).sum(0)[ok].sum() / sst[ok].sum()) if ok.any() else float("nan")


@torch.no_grad()
def resolution_readout(model, cfg, frames, n_fit=30000, n_eval=20000, stride=None, seed=0) -> dict:
    """R^2 on val, per class and resolution, of a ridge regression from the whitened latent (fit on train frames):
    `holds` reads z_t; `foresees` reads the predicted z^_{t+k} against the content at t+k; `copy` reads z_t against the
    content at t+k. Also the group means (terrain, ores and stations, mobs) and the raw latent's shape."""
    from .data import flat_state

    train_frames, val = frames
    k = stride or cfg.k
    g = np.random.default_rng(seed)
    fit = torch.as_tensor(g.choice(len(train_frames), min(n_fit, len(train_frames)), replace=False))
    z_fit = encode(model.encoder, train_frames, fit).double().cpu()
    y_fit = content(flat_state(train_frames.frames(fit)).cpu().numpy())
    mean = z_fit.mean(0)
    ev, u = torch.linalg.eigh(torch.cov((z_fit - mean).T))
    w = u / (ev.clamp(min=0) + 1e-6 * ev.max()).sqrt()
    zf = np.c_[((z_fit - mean) @ w).numpy(), np.ones(len(z_fit))]
    gram = zf.T @ zf + 1e-3 * np.eye(zf.shape[1])
    beta = {key: np.linalg.solve(gram, zf.T @ y) for key, y in y_fit.items()}

    def read(z):
        zz = np.c_[((z.double().cpu() - mean) @ w).numpy(), np.ones(len(z))]
        return {key: zz @ b for key, b in beta.items()}

    pool = val.starts(k, 1).cpu().numpy()
    t0 = torch.as_tensor(g.choice(pool, min(n_eval, len(pool)), replace=False))
    f, m = val.steps(t0, k, 1)
    z = torch.cat([model.encoder({key: x[a : a + 4096] for key, x in f.items()}) for a in range(0, len(t0), 4096)])
    now, nxt = content(flat_state({key: x[:, 0] for key, x in f.items()}).cpu().numpy()), content(
        flat_state({key: x[:, 1] for key, x in f.items()}).cpu().numpy())  # fmt: skip
    res = {"holds": {key: _r2(p, now[key]) for key, p in read(z[:, 0]).items()},
           "copy": {key: _r2(p, nxt[key]) for key, p in read(z[:, 0]).items()}}  # fmt: skip
    if model.predictor is not None:
        zhat = torch.cat([model.predictor(z[a : a + 4096, :1], m[a : a + 4096]) for a in range(0, len(t0), 4096)])[:, 0]
        res["foresees"] = {key: _r2(p, nxt[key]) for key, p in read(zhat).items()}
    for part in list(res.values()):
        for gname, members in GROUPS.items():
            for r in ("tile", "quadrant", "view"):
                part[f"{gname}|{r}"] = float(np.mean([part[f"{c}|{r}"] for c in members]))
    res["latent"] = latent_stats(model.encoder, val, k)
    return res


def run_resolution(run_dir, device=None, stride=None, cache=None) -> dict:
    """The resolution readout of every checkpoint of a run into `<run>/resolution.json` (steps already there skipped)."""
    run_dir = Path(run_dir)
    cfg = load_config(run_dir)
    device = device or cfg.device
    path = run_dir / "resolution.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    frames = (Frames(load("train", cache or cfg.cache), device), Frames(load("val", cache or cfg.cache), device))
    for ckpt in sorted(run_dir.glob("ckpt_*.pt")):
        step = ckpt.stem.split("_")[1].lstrip("0") or "0"
        if step not in results:
            _, model = load_model(ckpt, device)
            results[step] = resolution_readout(model, cfg, frames, stride=stride)
            path.write_text(json.dumps(results, indent=1))
    return results


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run")
    p.add_argument("--device", default=None)
    p.add_argument("--decoder-steps", type=int, default=2000)
    p.add_argument("--n-train", type=int, default=65536)
    p.add_argument("--n-eval", type=int, default=16384)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--cache", default=None, help="the cache folder, if not the one in the run's config")
    p.add_argument("--stride", type=int, default=None, help="the level-2 stride of the latent readouts (default: the run's k)")
    a = p.parse_args(argv)
    limit_threads()
    run_readouts(a.run, a.device, a.decoder_steps, a.n_train, a.n_eval, a.horizon, a.cache, a.stride)


if __name__ == "__main__":
    main()
