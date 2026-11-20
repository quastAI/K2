import numpy as np

from k2 import layout as L
from k2 import sim

from .conftest import synthetic_table
from k2.oracle import changing, mode_share, predictable, summary
from k2.plan import categorical_cem


def test_pack_unpack_round_trip():
    state, key = sim.sim().reset(3)
    buf = sim.pack(state)
    assert buf.size == sim.sim().nbytes == 20952
    np.testing.assert_array_equal(sim.pack(sim.unpack(buf)), buf)
    batched = sim.unpack(np.stack([buf, buf]), batched=True)
    assert batched.player_position.shape == (2, 2)


def test_rollout_depends_only_on_the_key():
    rng = np.random.default_rng(0)
    state, _ = sim.sim().reset(5)
    seqs = rng.integers(0, L.NUM_ACTIONS, (3, 6)).astype(np.int32)
    keys = sim.fresh_keys(rng, 3, 2)
    a, _ = sim.sim().rollout(state, seqs, keys)
    b, _ = sim.sim().rollout(state, seqs, keys)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert a.shape == (3, 2, L.STATE_DIM)
    L.compress(np.asarray(a).reshape(-1, L.STATE_DIM))  # the simulator's frames are exact in compact form


def test_true_key_replay_matches_stepping():
    state, key = sim.sim().reset(7)
    actions = np.array([1, 2, 3, 4, 5, 0], np.int32)
    obs = sim.replay_episode(7, actions)
    env_state = state
    for t, a in enumerate(actions):
        env_state = sim.sim().step(env_state, key, a)
        np.testing.assert_array_equal(np.asarray(sim.sim().observe(env_state)), obs[t + 1])


def test_goal_test():
    start = np.zeros(22, bool)
    goal = start.copy()
    goal[[0, 3]] = True
    cur = start.copy()
    cur[0] = True
    assert not sim.new_achievements(cur, goal, start)
    cur[3] = True
    assert sim.new_achievements(cur, goal, start)
    assert not sim.new_achievements(cur, start, start)  # no progress: never a success


def test_oracle_mode_lines_and_changing():
    values = np.array([[[1], [1], [2], [1]]])  # (n=1, R=4, F=1)
    p, mode = mode_share(values)
    assert p[0, 0] == 0.75 and mode[0, 0] == 1
    p = np.array([1.0, 63 / 64, 58 / 64, 57 / 64, 48 / 64, 47 / 64], np.float32)
    assert predictable(p, 1.0).tolist() == [True, False, False, False, False, False]
    assert predictable(p, 0.9).tolist() == [True, True, True, False, False, False]
    assert predictable(p, 0.75).tolist() == [True, True, True, True, True, False]
    # changing: the mode differs from the start value, or the replays disagree
    table = synthetic_table()
    rows = np.array([0, 1, 2])
    start = L.factor_values(L.take(table, rows))
    mode = start.copy()
    mode[0, 0] += 1  # a deterministic change
    p = np.ones(start.shape, np.float32)
    p[1, 5] = 0.5  # a random factor whose mode is the start value
    t = {"rows": rows, "mode": mode, "p": p}
    chg = changing(t, table)
    assert chg.sum() == 2 and chg[0, 0] and chg[1, 5]


def test_summary_text():
    table = synthetic_table()
    rows = np.arange(0, 20)
    n = len(rows)
    mode = L.factor_values(L.take(table, rows + 2))
    t = {"k": np.int64(2), "split": "val", "n_keys": np.int64(64), "rows": rows, "mode": mode,
         "p": np.ones((n, L.NUM_FACTORS), np.float32), "p_true": np.ones((n, L.NUM_FACTORS), np.float32)}  # fmt: skip
    text = summary(t, table)
    assert "| mobs" in text and "| daylight" in text and "all agreed" in text and "p >= 0.75" in text


def test_categorical_cem_finds_the_minimum():
    target = np.array([3, 1, 4, 1, 5])
    best, cost = categorical_cem(lambda s: (s != target).sum(1), 5, 8, 64, 8, 8, 0.5, 0.05, np.random.default_rng(0))
    np.testing.assert_array_equal(best, target)
