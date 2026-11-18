import numpy as np

from k2 import layout as L
from k2 import sim
from k2.oracle import find_cut, mode_share
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


def test_oracle_mode_and_cut():
    values = np.array([[[1], [1], [2], [1]]])  # (n=1, R=4, F=1)
    p, mode = mode_share(values)
    assert p[0, 0] == 0.75 and mode[0, 0] == 1
    p = np.concatenate([np.ones(100), np.full(30, 31 / 32), np.full(2, 20 / 32), np.full(20, 10 / 32)])
    cut, hist = find_cut(p, 32)
    assert hist[32] == 100 and 16 / 32 <= cut < 31 / 32 and hist[round(cut * 32)] == 0


def test_categorical_cem_finds_the_minimum():
    target = np.array([3, 1, 4, 1, 5])
    best, cost = categorical_cem(lambda s: (s != target).sum(1), 5, 8, 64, 8, 8, 0.5, 0.05, np.random.default_rng(0))
    np.testing.assert_array_equal(best, target)
