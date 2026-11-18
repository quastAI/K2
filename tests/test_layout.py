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


def test_token_values_and_classes():
    c = L.take(synthetic_table(), slice(0, 10))
    v = L.token_values(c)
    assert v.shape == (10, L.NUM_TOKENS)
    np.testing.assert_array_equal(v[:, L.TILE_TOKENS] // 16, c["block"])
    classes = L.factor_classes(c)
    covered = np.zeros_like(v, dtype=bool)
    for m in classes.values():
        covered |= m
    assert covered.all()  # every factor belongs to a class
    assert len(L.TOKEN_NAMES) == L.NUM_TOKENS
