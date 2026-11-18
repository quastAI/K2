"""The oracle table: what is predictable at stride k given the macro-action, before any model exists.

For every sampled held-out frame t (`<split>_full.npz`, `sample`), the true state is restored from `full_state` and
the k executed actions a_t .. a_{t+k-1} are replayed with N fresh step keys (no auto-reset). Per factor (token) of the
outcome frame, `p` is the share of the N replays that match the most common outcome. Deterministic factors sit at 1.
The cut is read from the data: the least-populated level of p in [0.5, 1) (p takes the values j / N); a factor is
predictable when p lies above it. Planning-time keys are fresh too, so the replays never use the episode's true key.

    python -m k2.oracle --k 1 8        # about a minute on the Mac CPU

Writes `<cache>/oracle_k<k>.npz` (rows, p, p_true, cut, hist, ...) and `oracle_k<k>.md`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import layout, sim
from .cache import CACHE_DIR, full_state_of, load, load_full


def find_cut(p: np.ndarray, n_keys: int) -> tuple[float, np.ndarray]:
    """The cut between predictable and not: the level j / N in [0.5, 1) holding the fewest factors (ties: the highest).
    Returns the cut and the histogram over the levels 0 .. N."""
    hist = np.bincount(np.rint(p.ravel() * n_keys).astype(np.int64), minlength=n_keys + 1)
    levels = np.arange(int(np.ceil(n_keys / 2)), n_keys)
    j = levels[::-1][np.argmin(hist[levels][::-1])]
    return float(j) / n_keys, hist


def mode_share(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """values (n, R, F) -> the share of the most common value and that value, per (n, F)."""
    counts = np.stack([(values == values[:, r : r + 1]).sum(1) for r in range(values.shape[1])], 1)  # (n, R, F)
    best = counts.argmax(1)
    mode = np.take_along_axis(values, best[:, None], 1)[:, 0]
    return counts.max(1) / values.shape[1], mode


def oracle_table(k: int, split="val", n_keys=32, cache_dir=CACHE_DIR, batch=64, seed=0) -> dict:
    table, full = load(split, cache_dir), load_full(split, cache_dir)
    rows = full["sample"]
    rows = rows[rows + k <= np.repeat(table["ep_offset"] + table["ep_len"] - 1, table["ep_len"])[rows]]
    actions = table["action"][rows[:, None] + np.arange(k)].astype(np.int32)
    if np.any(actions < 0):
        raise ValueError("an action chunk crosses the end of an episode")
    truth = layout.token_values(layout.take(table, rows + k))
    rng = np.random.default_rng(seed)
    p, p_true, mode = [], [], []
    for a in range(0, len(rows), batch):
        b = slice(a, a + batch)
        states = sim.unpack(full_state_of(full, rows[b]), batched=True)
        obs, _ = sim.sim().rollout_many(states, actions[b], sim.fresh_keys(rng, len(rows[b]), n_keys))
        obs = np.asarray(obs).reshape(-1, layout.STATE_DIM)
        values = layout.token_values(layout.compress(obs)).reshape(len(rows[b]), n_keys, layout.NUM_TOKENS)
        share, m = mode_share(values)
        p.append(share)
        mode.append(m)
        p_true.append((values == truth[b][:, None]).mean(1))
    p = np.concatenate(p).astype(np.float32)
    cut, hist = find_cut(p, n_keys)
    return {
        "k": np.int64(k), "split": split, "n_keys": np.int64(n_keys), "rows": rows, "p": p,
        "p_true": np.concatenate(p_true).astype(np.float32), "mode": np.concatenate(mode),
        "cut": np.float32(cut), "hist": hist,
    }  # fmt: skip


def summary(t: dict, table: dict) -> str:
    """Per readout class: how many factors, their mean p, the share above the cut, the share at 1, and the p quantiles."""
    k, n_keys, cut = int(t["k"]), int(t["n_keys"]), float(t["cut"])
    classes = layout.factor_classes(layout.take(table, t["rows"] + k))
    p = t["p"]
    lines = [
        f"# Oracle table, k = {k}",
        "",
        f"{len(t['rows'])} frames of `{t['split']}`, {n_keys} fresh keys each. Cut from the data: p > {cut:.3f} "
        f"(the least-populated level in [0.5, 1)). Overall: {np.mean(p > cut):.3f} of the factors predictable, "
        f"{np.mean(p == 1):.3f} deterministic. The true outcome is the mode in {np.mean(t['p_true'] == p):.3f}.",
        "",
        "| class | factors | mean p | predictable | p = 1 | p 5% | p 25% | p 50% |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name in layout.class_order(list(classes)):
        v = p[classes[name]]
        q = np.quantile(v, [0.05, 0.25, 0.5])
        lines.append(f"| {name} | {v.size} | {v.mean():.3f} | {np.mean(v > cut):.3f} | {np.mean(v == 1):.3f} | "
                     f"{q[0]:.2f} | {q[1]:.2f} | {q[2]:.2f} |")  # fmt: skip
    lines += ["", "Histogram of p (level j / N: count): " + ", ".join(
        f"{j}: {c}" for j, c in enumerate(t["hist"]) if c)]  # fmt: skip
    return "\n".join(lines) + "\n"


def load_oracle(k: int, cache_dir=CACHE_DIR) -> dict:
    with np.load(Path(cache_dir) / f"oracle_k{k}.npz") as z:
        return {key: z[key] for key in z.files}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--k", type=int, nargs="+", default=[1, 8])
    p.add_argument("--split", default="val")
    p.add_argument("--keys", type=int, default=32)
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
