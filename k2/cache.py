"""The data cache: AAC's HDF5 episodes in compact form, built once on the Mac and uploaded to the pods.

    python -m k2.cache                     # train, val and test into data/cache (about a minute)
    python -m k2.cache --max-episodes 4    # a smoke build

`<split>.npz` (every frame, every source):
    block, mob, item, vital, facing, light, sleep   the compact `state` (k2.layout)
    action        (n,) int8, the executed action, -1 in the last frame of an episode
    ep_offset, ep_len, ep_seed, ep_source           per episode
    achievements  (n, 22) bool (val and test only: the goal test of the tasks)

`<split>_full.npz` (val and test only), the frames that need the simulator:
    rows          sorted frame indices whose `full_state` is kept
    full_state    (len(rows), 20952) uint8
    sample        the oracle's frames: `max_k` actions follow inside the episode
    task_start, task_goal, task_episode    planning tasks from the expert episodes (source 0), one per episode:
                  the goal frame is `task_offset` frames after the start and has an achievement the start lacks

The HDF5 files are read by path and never copied.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from . import layout

DATA_DIR = Path(__file__).resolve().parents[2] / "AAC/data/hf/craftax-classic-pqn-rnn-v1/data"
CACHE_DIR = Path(__file__).resolve().parents[1] / "data/cache"
SPLITS = ("train", "val", "test")
EXPERT = 0
BLOCK = 50_000


def h5_path(split: str, data_dir=DATA_DIR) -> Path:
    return Path(data_dir) / f"craftax_expert_{split}.h5"


def build(split, data_dir=DATA_DIR, out_dir=CACHE_DIR, max_episodes=None, sample=2048, tasks=128, task_offset=64,
          max_k=16, seed=0) -> dict:  # fmt: skip
    """Write `<split>.npz` (and `<split>_full.npz` for a held-out split); returns the frame table."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path(split, data_dir), "r") as f:
        ep_offset, ep_len = f["ep_offset"][:].astype(np.int64), f["ep_len"][:].astype(np.int64)
        if max_episodes:
            ep_offset, ep_len = ep_offset[:max_episodes], ep_len[:max_episodes]
        if np.any(ep_offset != np.concatenate([[0], np.cumsum(ep_len)[:-1]])):
            raise ValueError(f"{split}: the episodes are not stored back to back")
        n = int(ep_len.sum())
        if f["state"].shape[1] != layout.STATE_DIM:
            raise ValueError(f"{split}: state has {f['state'].shape[1]} columns, expected {layout.STATE_DIM}")
        parts = [layout.compress(f["state"][a : min(a + BLOCK, n)]) for a in range(0, n, BLOCK)]
        table = {k: np.concatenate([p[k] for p in parts]) for k in layout.FIELDS}
        table["action"] = np.nan_to_num(f["action"][:n, 0], nan=-1).astype(np.int8)
        table["ep_offset"], table["ep_len"] = ep_offset, ep_len
        table["ep_seed"] = f["seed"][:n][ep_offset].astype(np.int64)
        table["ep_source"] = f["source"][:n][ep_offset].astype(np.int8)
        last = ep_offset + ep_len - 1
        if np.any(table["action"][last] != -1) or np.sum(table["action"] < 0) != len(ep_len):
            raise ValueError(f"{split}: an action is missing inside an episode or present in a last frame")
        if split != "train":
            table["achievements"] = f["achievements"][:n].astype(bool)
            full = held_out(table, sample, tasks, task_offset, max_k, seed)
            full["full_state"] = f["full_state"][full["rows"]] if len(full["rows"]) else np.zeros((0, 20952), np.uint8)
            np.savez(out_dir / f"{split}_full.npz", **full)
    np.savez(out_dir / f"{split}.npz", **table)
    return table


def episode_of(table: dict) -> np.ndarray:
    """The episode index of every frame."""
    return np.repeat(np.arange(len(table["ep_len"])), table["ep_len"])


def starts(table: dict, steps: int, sources=None) -> np.ndarray:
    """Frames t from which `steps` actions follow inside the episode (t + steps is a frame of the same episode)."""
    last = np.repeat(table["ep_offset"] + table["ep_len"] - 1, table["ep_len"])
    t = np.arange(len(last))
    ok = t + steps <= last
    if sources is not None:
        ok &= np.isin(np.repeat(table["ep_source"], table["ep_len"]), list(sources))
    return t[ok]


def held_out(table, sample, tasks, task_offset, max_k, seed) -> dict:
    """The oracle's frames and the planning tasks of a held-out split (their rows need `full_state`)."""
    rng = np.random.default_rng(seed)
    pool = starts(table, max_k)
    oracle = np.sort(rng.choice(pool, min(sample, len(pool)), replace=False))
    ach = table["achievements"]
    task_start, task_episode = [], []
    expert = np.nonzero(table["ep_source"] == EXPERT)[0]
    for e in rng.permutation(expert):
        o, n = int(table["ep_offset"][e]), int(table["ep_len"][e])
        s = np.arange(o, o + n - task_offset)
        s = s[(ach[s + task_offset] & ~ach[s]).any(1)]
        if len(s):
            task_start.append(int(rng.choice(s)))
            task_episode.append(int(e))
        if len(task_start) == tasks:
            break
    order = np.argsort(task_episode)
    task_start = np.asarray(task_start, np.int64)[order]
    return {
        "rows": np.union1d(oracle, task_start).astype(np.int64),
        "sample": oracle.astype(np.int64),
        "task_start": task_start,
        "task_goal": task_start + task_offset,
        "task_episode": np.asarray(task_episode, np.int64)[order],
        "task_offset": np.int64(task_offset),
    }


def load(split: str, cache_dir=CACHE_DIR) -> dict:
    with np.load(Path(cache_dir) / f"{split}.npz") as z:
        return {k: z[k] for k in z.files}


def load_full(split: str, cache_dir=CACHE_DIR) -> dict:
    with np.load(Path(cache_dir) / f"{split}_full.npz") as z:
        return {k: z[k] for k in z.files}


def full_state_of(full: dict, rows) -> np.ndarray:
    """The kept `full_state` of frames `rows` (each must be in `full['rows']`)."""
    rows = np.atleast_1d(rows)
    i = np.searchsorted(full["rows"], rows)
    if np.any(i >= len(full["rows"])) or np.any(full["rows"][np.minimum(i, len(full["rows"]) - 1)] != rows):
        raise KeyError("a frame without a kept full_state")
    return full["full_state"][i]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits", nargs="+", default=list(SPLITS))
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--out", default=str(CACHE_DIR))
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--sample", type=int, default=2048, help="oracle frames per held-out split")
    p.add_argument("--tasks", type=int, default=128)
    p.add_argument("--task-offset", type=int, default=64, help="goal frame - start frame (H * k = 8 * 8)")
    args = p.parse_args(argv)
    for split in args.splits:
        t = build(split, args.data_dir, args.out, args.max_episodes, args.sample, args.tasks, args.task_offset)
        size = sum(v.nbytes for v in t.values())
        print(f"{split}: {len(t['action'])} frames, {len(t['ep_len'])} episodes, {size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
