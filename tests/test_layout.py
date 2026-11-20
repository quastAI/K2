import numpy as np

from k2 import layout as L

from .conftest import synthetic_table


def test_compact_round_trip():
    t = synthetic_table()
    c = L.take(t, slice(None))
    state = L.expand(c)
    assert state.shape == (len(t["block"]), L.STATE_DIM)
    back = L.compress(state)
    for f in L.FIELDS:
        np.testing.assert_array_equal(back[f], c[f])


def test_count_values_are_the_games():
    # the game writes v * 0.1 in float16: 3 is stored as 0.2998047, not as float32(0.3)
    assert L.LEVEL_VALUES[3] == np.float32(0.2998046875)
    assert L.LEVEL_VALUES[5] == np.float32(0.5)


def test_token_values():
    c = L.take(synthetic_table(), slice(0, 10))
    v = L.token_values(c)
    assert v.shape == (10, L.NUM_TOKENS)
    np.testing.assert_array_equal(v[:, L.TILE_TOKENS] // 16, c["block"])
    assert len(L.TOKEN_NAMES) == L.NUM_TOKENS


def test_factor_values_and_classes():
    """Each tile is two factors, block and mobs; every factor belongs to a class except the mob factors of empty tiles."""
    c = L.take(synthetic_table(), slice(0, 10))
    v = L.factor_values(c)
    assert v.shape == (10, L.NUM_FACTORS) and L.NUM_FACTORS == 2 * L.TILES + L.NUM_ITEMS + L.NUM_VITALS + 3
    np.testing.assert_array_equal(v[:, L.BLOCK_FACTORS], c["block"])
    np.testing.assert_array_equal(v[:, L.MOB_FACTORS], c["mob"])
    np.testing.assert_array_equal(v[:, L.ITEM_FACTORS], c["item"])
    assert (v[:, L.SLEEP_FACTOR] == c["sleep"]).all()
    covered = np.logical_or.reduce(list(L.factor_classes(c).values()))
    expect = np.ones_like(covered)
    expect[:, L.MOB_FACTORS] = c["mob"] > 0
    np.testing.assert_array_equal(covered, expect)
    # the goal factors: block types, items, vitals, sleeping; not the mobs, facing or light
    g = L.GOAL_FACTORS
    assert g[L.BLOCK_FACTORS].all() and g[L.ITEM_FACTORS].all() and g[L.VITAL_FACTORS].all() and g[L.SLEEP_FACTOR]
    assert not g[L.MOB_FACTORS].any() and not g[L.FACING_FACTOR] and not g[L.LIGHT_FACTOR]


def test_kinds():
    c = L.take(synthetic_table(), slice(0, 10))
    kinds = L.kind_masks(c)
    assert set(kinds) == set(L.KINDS)
    assert kinds["items"][:, L.ITEM_FACTORS].all() and not kinds["items"][:, L.BLOCK_FACTORS].any()
    assert kinds["mobs"][:, L.MOB_FACTORS].any() and not kinds["mobs"][:, L.BLOCK_FACTORS].any()
    assert kinds["daylight"].sum() == 10 and kinds["daylight"][:, L.LIGHT_FACTOR].all()
    assert kinds["player"][:, [L.FACING_FACTOR, L.SLEEP_FACTOR]].all() and not kinds["player"][:, L.LIGHT_FACTOR].any()
    every = np.logical_or.reduce(list(kinds.values()))
    every[:, L.MOB_FACTORS] |= c["mob"] == 0
    assert every.all()  # every factor is of some kind (the mob factors of empty tiles are in none)
