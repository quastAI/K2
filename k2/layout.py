"""The observation `state` of Craftax-Classic and its compact form. Numpy only: both sides use it.

`state` (1345 floats): the 7 x 9 view, tile by tile, 21 channels per tile (17 block one-hot, then zombie, cow,
skeleton, arrow), then inventory (12, count / 10), vitals (4, value / 10) (both computed in float16), facing (4, one-hot), light, sleeping.
The player stands at view row 3, column 4.

The compact form keeps the same frame exactly in 148 bytes:

    block   (n, 63) uint8    block id per tile
    mob     (n, 63) uint8    bit m set = a mob of type m on the tile
    item    (n, 12) uint8    counts 0 to 9
    vital   (n, 4)  uint8    values 0 to 9
    facing  (n,)    uint8    0 left, 1 right, 2 up, 3 down
    light   (n,)    float32
    sleep   (n,)    uint8

Tokens (the factors of the readouts and the gate) are in this order: 63 tiles, 12 items, 4 vitals, facing, light,
sleeping (82).
"""

from __future__ import annotations

import numpy as np

BLOCKS = [
    "invalid", "out_of_bounds", "grass", "water", "stone", "tree", "wood", "path", "coal", "iron", "diamond",
    "crafting_table", "furnace", "sand", "lava", "plant", "ripe_plant",
]  # fmt: skip
MOBS = ["zombie", "cow", "skeleton", "arrow"]
ITEMS = [
    "wood", "stone", "coal", "iron", "diamond", "sapling",
    "wood_pickaxe", "stone_pickaxe", "iron_pickaxe", "wood_sword", "stone_sword", "iron_sword",
]  # fmt: skip
VITALS = ["health", "food", "drink", "energy"]
ACTIONS = [
    "noop", "left", "right", "up", "down", "do", "sleep", "place_stone", "place_table", "place_furnace",
    "place_plant", "make_wood_pickaxe", "make_stone_pickaxe", "make_iron_pickaxe", "make_wood_sword",
    "make_stone_sword", "make_iron_sword",
]  # fmt: skip
ACHIEVEMENTS = [
    "collect_wood", "place_table", "eat_cow", "collect_sapling", "collect_drink", "make_wood_pickaxe",
    "make_wood_sword", "place_plant", "defeat_zombie", "collect_stone", "place_stone", "eat_plant",
    "defeat_skeleton", "make_stone_pickaxe", "make_stone_sword", "wake_up", "place_furnace", "collect_coal",
    "collect_iron", "collect_diamond", "make_iron_pickaxe", "make_iron_sword",
]  # fmt: skip

ROWS, COLS = 7, 9
TILES = ROWS * COLS
NUM_BLOCKS, NUM_MOBS, NUM_ITEMS, NUM_VITALS = len(BLOCKS), len(MOBS), len(ITEMS), len(VITALS)
NUM_ACTIONS, NUM_ACHIEVEMENTS = len(ACTIONS), len(ACHIEVEMENTS)
CHANNELS = NUM_BLOCKS + NUM_MOBS
VIEW_DIM = TILES * CHANNELS
STATE_DIM = VIEW_DIM + NUM_ITEMS + NUM_VITALS + 4 + 1 + 1
LEVELS = 10  # counts and vital values run 0 to 9
# what the game writes for a count or vital value v: v * 0.1 in float16 (1 is stored as 0.0999756, 3 as 0.2998047)
LEVEL_VALUES = (np.arange(LEVELS, dtype=np.float16) * np.float16(0.1)).astype(np.float32)
PLAYER = (3, 4)

# token index ranges
TILE_TOKENS = slice(0, TILES)
ITEM_TOKENS = slice(TILES, TILES + NUM_ITEMS)
VITAL_TOKENS = slice(TILES + NUM_ITEMS, TILES + NUM_ITEMS + NUM_VITALS)
FACING_TOKEN = TILES + NUM_ITEMS + NUM_VITALS
LIGHT_TOKEN = FACING_TOKEN + 1
SLEEP_TOKEN = FACING_TOKEN + 2
NUM_TOKENS = SLEEP_TOKEN + 1
TOKEN_NAMES = (
    [f"tile_{r}_{c}" for r in range(ROWS) for c in range(COLS)]
    + [f"item_{n}" for n in ITEMS]
    + [f"vital_{n}" for n in VITALS]
    + ["facing", "light", "sleeping"]
)
LIGHT_TOLERANCE = 0.05  # a decoded light level within this counts as correct

# the rare classes of the readouts (the sapling, once placed, is the block `plant`)
RARE_BLOCKS = ["coal", "iron", "diamond", "lava", "crafting_table", "furnace", "plant"]

FIELDS = ("block", "mob", "item", "vital", "facing", "light", "sleep")
_MOB_BITS = 1 << np.arange(NUM_MOBS)


def compress(state: np.ndarray) -> dict:
    """The compact form of `state` (n, 1345). Raises if the frame cannot be rebuilt exactly."""
    state = np.asarray(state, dtype=np.float32).reshape(-1, STATE_DIM)
    tiles = state[:, :VIEW_DIM].reshape(-1, TILES, CHANNELS)
    tail = state[:, VIEW_DIM:]
    o = 0
    item = tail[:, o : o + NUM_ITEMS]
    o += NUM_ITEMS
    vital = tail[:, o : o + NUM_VITALS]
    o += NUM_VITALS
    facing = tail[:, o : o + 4]
    c = {
        "block": tiles[:, :, :NUM_BLOCKS].argmax(2).astype(np.uint8),
        "mob": ((tiles[:, :, NUM_BLOCKS:] > 0.5) * _MOB_BITS).sum(2).astype(np.uint8),
        "item": np.rint(item * LEVELS).astype(np.uint8),
        "vital": np.rint(vital * LEVELS).astype(np.uint8),
        "facing": facing.argmax(1).astype(np.uint8),
        "light": tail[:, -2].astype(np.float32),
        "sleep": (tail[:, -1] > 0.5).astype(np.uint8),
    }
    bad = np.nonzero((expand(c) != state).any(1))[0]
    if len(bad):
        raise ValueError(f"{len(bad)} frames do not rebuild exactly (first: row {bad[0]})")
    return c


def expand(c: dict) -> np.ndarray:
    """The `state` (n, 1345) float32 of a compact batch."""
    n = len(c["block"])
    blocks = np.eye(NUM_BLOCKS, dtype=np.float32)[c["block"]]
    mobs = ((c["mob"][..., None] & _MOB_BITS) > 0).astype(np.float32)
    view = np.concatenate([blocks, mobs], -1).reshape(n, VIEW_DIM)
    return np.concatenate(
        [
            view,
            LEVEL_VALUES[c["item"]],
            LEVEL_VALUES[c["vital"]],
            np.eye(4, dtype=np.float32)[c["facing"]],
            c["light"].astype(np.float32)[:, None],
            c["sleep"].astype(np.float32)[:, None],
        ],
        1,
    )


def take(c: dict, rows) -> dict:
    """The compact frames at `rows`."""
    return {f: c[f][rows] for f in FIELDS}


def token_values(c: dict) -> np.ndarray:
    """One integer per token (n, 82): tile = block * 16 + mob bits, item count, vital value, facing, light in steps of
    1e-4, sleeping. Two frames have the same factor exactly when the integers match."""
    return np.concatenate(
        [
            c["block"].astype(np.int64) * 16 + c["mob"],
            c["item"].astype(np.int64),
            c["vital"].astype(np.int64),
            c["facing"].astype(np.int64)[:, None],
            np.rint(c["light"].astype(np.float64) * 1e4).astype(np.int64)[:, None],
            c["sleep"].astype(np.int64)[:, None],
        ],
        1,
    )


def factor_classes(c: dict) -> dict[str, np.ndarray]:
    """Boolean masks (n, 82) of the factors of every readout class, read from the true frames `c`. A tile counts for its
    block type and for each mob on it; every item, vital, facing, light and sleeping is a class of its own. Classes
    with no factor in `c` are left out."""
    n = len(c["block"])
    out = {}
    for b, name in enumerate(BLOCKS):
        m = np.zeros((n, NUM_TOKENS), bool)
        m[:, TILE_TOKENS] = c["block"] == b
        out[name] = m
    for j, name in enumerate(MOBS):
        m = np.zeros((n, NUM_TOKENS), bool)
        m[:, TILE_TOKENS] = (c["mob"] >> j) & 1 > 0
        out[name] = m
    singles = [f"item_{x}" for x in ITEMS] + VITALS + ["facing", "light", "sleeping"]
    for i, name in enumerate(singles):
        m = np.zeros((n, NUM_TOKENS), bool)
        m[:, ITEM_TOKENS.start + i] = True
        out[name] = m
    return {k: v for k, v in out.items() if v.any()}


def class_order(names) -> list[str]:
    """Readout classes in report order: the rare ones first, then the other blocks, the rest as given."""
    rare = [n for n in RARE_BLOCKS + MOBS if n in names]
    return rare + [n for n in names if n not in rare]
