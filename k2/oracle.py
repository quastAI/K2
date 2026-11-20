"""The oracle table: what is predictable at stride k given the macro-action, before any model exists.

For every sampled held-out frame t (`<split>_full.npz`, `sample`), the true state is restored from `full_state` and
the k executed actions a_t .. a_{t+k-1} are replayed with N fresh step keys (no auto-reset). Per factor of the outcome
frame (`layout.factor_values`: each tile's block and its mobs apart), `p` is the share of the N replays that match the
most common outcome (`mode`).

Predictable at a line c: p >= c. The main line is 1: all N replays agreed, no tuned number. The lines 0.9 and 0.75
check that a result does not depend on it (the histogram of p has no gap to put a cut in). A factor *changes* when
some replay ends elsewhere than the start frame: its most common outcome differs from the start value, or the replays
disagree, so every non-deterministic factor changes. The others stay as they are, deterministically; the summaries
and the readouts report them apart. Planning-time keys are fresh too, so the replays never use the episode's true key.

    python -m k2.oracle --k 1 8        # about a minute on the Mac CPU

Writes `<cache>/oracle_k<k>.npz` (rows, p, p_true, mode, ...) and `oracle_k<k>.md`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import layout, sim
from .cache import CACHE_DIR, full_state_of, load, load_full


LINES = (1.0, 0.9, 0.75)  # predictable at a line c: p >= c (1 = all replays agreed)
FEW = 30  # fewer changing factors than this in a row of a table: marked as unreliable


def predictable(p: np.ndarray, line: float) -> np.ndarray:
    return p >= line - 1e-6


def changing(t: dict, table: dict) -> np.ndarray:
    """(n, 145): the factor changes over the stride: some replay ends elsewhere than the start frame."""
    start = layout.factor_values(layout.take(table, t["rows"]))
    return (t["mode"] != start) | (t["p"] < 1)


def mode_share(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """values (n, R, F) -> the share of the most common value and that value, per (n, F)."""
    counts = np.stack([(values == values[:, r : r + 1]).sum(1) for r in range(values.shape[1])], 1)  # (n, R, F)
    best = counts.argmax(1)
    mode = np.take_along_axis(values, best[:, None], 1)[:, 0]
    return counts.max(1) / values.shape[1], mode


def oracle_table(k: int, split="val", n_keys=64, cache_dir=CACHE_DIR, batch=64, seed=0) -> dict:
    table, full = load(split, cache_dir), load_full(split, cache_dir)
    rows = full["sample"]
    rows = rows[rows + k <= np.repeat(table["ep_offset"] + table["ep_len"] - 1, table["ep_len"])[rows]]
    actions = table["action"][rows[:, None] + np.arange(k)].astype(np.int32)
    if np.any(actions < 0):
        raise ValueError("an action chunk crosses the end of an episode")
    truth = layout.factor_values(layout.take(table, rows + k))
    rng = np.random.default_rng(seed)
    p, p_true, mode = [], [], []
    for a in range(0, len(rows), batch):
        b = slice(a, a + batch)
        states = sim.unpack(full_state_of(full, rows[b]), batched=True)
        obs, _ = sim.sim().rollout_many(states, actions[b], sim.fresh_keys(rng, len(rows[b]), n_keys))
        obs = np.asarray(obs).reshape(-1, layout.STATE_DIM)
        values = layout.factor_values(layout.compress(obs)).reshape(len(rows[b]), n_keys, layout.NUM_FACTORS)
        share, m = mode_share(values)
        p.append(share)
        mode.append(m)
        p_true.append((values == truth[b][:, None]).mean(1))
    p = np.concatenate(p).astype(np.float32)
    return {
        "k": np.int64(k), "split": split, "n_keys": np.int64(n_keys), "rows": rows, "p": p,
        "p_true": np.concatenate(p_true).astype(np.float32), "mode": np.concatenate(mode),
    }  # fmt: skip


def summary(t: dict, table: dict) -> str:
    """Per kind and per class: the factors, how many change, and the share of the changing ones predictable at each line."""
    k, n_keys = int(t["k"]), int(t["n_keys"])
    c = layout.take(table, t["rows"] + k)
    p, chg = t["p"], changing(t, table)
    head = ["factors", "changing"] + [f"predictable, {'all agreed' if line == 1 else f'p >= {line:g}'}" for line in LINES]

    def row(name, mask):
        n = int((mask & chg).sum())
        shares = [f"{predictable(p, line)[mask & chg].mean():.3f}" if n else "-" for line in LINES]
        return f"| {name}{' *' if n < FEW else ''} | {int(mask.sum())} | {n} | " + " | ".join(shares) + " |"

    def block(first, masks, order):
        return ["", f"| {first} | " + " | ".join(head) + " |", "|" + "---|" * (len(head) + 1)] + [
            row(name, masks[name]) for name in order
        ]

    kinds, classes = layout.kind_masks(c), layout.factor_classes(c)
    lines = [
        f"# Oracle table, k = {k}",
        "",
        f"{len(t['rows'])} frames of `{t['split']}`, {n_keys} fresh keys each. A factor is predictable at a line when "
        f"at least that share of the replays agree on its outcome; the main line is all {n_keys} agreed, 0.9 and 0.75 "
        "check that a result does not depend on it. A factor changes when some replay ends elsewhere than the start "
        "frame, so every non-deterministic factor changes; the shares below cover the changing factors only "
        f"(`*`: fewer than {FEW}). Each tile is two factors, its block type and its mobs; tiles are egocentric, so a "
        "step that moves the player changes most block factors. Mob factors of empty tiles count in `Overall` only.",
        "",
        f"Overall: {chg.mean():.3f} of the {chg.size} factors change; of those, "
        + ", ".join(f"{predictable(p, line)[chg].mean():.3f} at {'all agreed' if line == 1 else f'p >= {line:g}'}"
                    for line in LINES) + f" are predictable. The true outcome is the mode in {np.mean(t['p_true'] == p):.3f}.",
    ]  # fmt: skip
    lines += block("kind", kinds, list(kinds))
    lines += block("class", classes, layout.class_order(list(classes)))
    return "\n".join(lines) + "\n"


def load_oracle(k: int, cache_dir=CACHE_DIR) -> dict:
    with np.load(Path(cache_dir) / f"oracle_k{k}.npz") as z:
        return {key: z[key] for key in z.files}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--k", type=int, nargs="+", default=[1, 8])
    p.add_argument("--split", default="val")
    p.add_argument("--keys", type=int, default=64)
    p.add_argument("--cache", default=str(CACHE_DIR))
    args = p.parse_args(argv)
    table = load(args.split, args.cache)
    for k in args.k:
        t = oracle_table(k, args.split, args.keys, args.cache)
        np.savez(Path(args.cache) / f"oracle_k{k}.npz", **t)
        text = summary(t, table)
        (Path(args.cache) / f"oracle_k{k}.md").write_text(text)
        print(text)


if __name__ == "__main__":
    main()
