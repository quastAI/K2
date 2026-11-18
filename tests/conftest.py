import numpy as np
import pytest

from k2 import layout as L


def synthetic_table(episodes=4, length=40, seed=0, achievements=False) -> dict:
    """A random cache table (frames need not follow the game's rules)."""
    rng = np.random.default_rng(seed)
    n = episodes * length
    t = {
        "block": rng.integers(0, L.NUM_BLOCKS, (n, L.TILES)).astype(np.uint8),
        "mob": (rng.random((n, L.TILES)) < 0.05).astype(np.uint8) << rng.integers(0, 4, (n, L.TILES)).astype(np.uint8),
        "item": rng.integers(0, L.LEVELS, (n, L.NUM_ITEMS)).astype(np.uint8),
        "vital": rng.integers(0, L.LEVELS, (n, L.NUM_VITALS)).astype(np.uint8),
        "facing": rng.integers(0, 4, n).astype(np.uint8),
        "light": rng.random(n).astype(np.float32),
        "sleep": (rng.random(n) < 0.1).astype(np.uint8),
        "action": rng.integers(0, L.NUM_ACTIONS, n).astype(np.int8),
        "ep_offset": np.arange(episodes) * length,
        "ep_len": np.full(episodes, length),
        "ep_seed": np.arange(episodes, dtype=np.int64),
        "ep_source": np.array([0, 3] * (episodes // 2), np.int8),
    }
    t["action"][t["ep_offset"] + length - 1] = -1
    if achievements:
        t["achievements"] = np.zeros((n, L.NUM_ACHIEVEMENTS), bool)
    return t


@pytest.fixture
def cache_dir(tmp_path):
    for i, split in enumerate(("train", "val")):
        np.savez(tmp_path / f"{split}.npz", **synthetic_table(seed=i, achievements=split == "val"))
    return tmp_path
