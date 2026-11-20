"""The simulator side: Craftax-Classic in JAX on the CPU.

Replay contract (the dataset's): `reset_key, step_key, _ = split(PRNGKey(seed), 3)`; the step taken at
`state.timestep == t` uses `fold_in(step_key, t)`. `full_state` is the flat uint8 dump of the state pytree in leaf
order. Fresh rollouts use the same fold-in with a fresh key in place of `step_key`. There is no auto-reset: a dead
player stays in the rollout as it is.
"""

from __future__ import annotations

import functools
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from craftax.craftax_env import make_craftax_env_from_name  # noqa: E402

# The contract must not depend on the installed jax's default (jax >= 0.5 switched it).
jax.config.update("jax_threefry_partitionable", False)

ENV_NAME = "Craftax-Classic-Symbolic-v1"


class Sim:
    """The jitted functions of one process."""

    def __init__(self):
        env = make_craftax_env_from_name(ENV_NAME, auto_reset=False)
        params = env.default_params
        self.env, self.params = env, params

        def step(state, key, action):
            return env.step(jax.random.fold_in(key, state.timestep), state, action, params)[1]

        @jax.jit
        def reset(seed):
            reset_key, step_key, _ = jax.random.split(jax.random.PRNGKey(seed), 3)
            return env.reset(reset_key, params)[1], step_key

        def run(state, key, actions):
            state, _ = jax.lax.scan(lambda s, a: (step(s, key, a), None), state, actions)
            return state, env.get_obs(state), state.achievements

        def replay(state, key, actions):
            """Every observation along `actions` (n, ...) from `state` with one key."""

            def body(s, a):
                s = step(s, key, a)
                return s, env.get_obs(s)

            return jax.lax.scan(body, state, actions)

        over_keys = jax.vmap(run, (None, 0, None))  # one state, one action sequence, R keys

        self.reset = reset
        self.step = jax.jit(step)
        self.observe = jax.jit(env.get_obs)
        self.replay = jax.jit(replay)
        # one state, S sequences (S, L), keys (S, R, 2): every sequence under its own R keys
        self.rollout = jax.jit(lambda s, acts, keys: _strip(jax.vmap(over_keys, (None, 0, 0))(s, keys, acts)))
        # S states (batched), S sequences, keys (S, R, 2)
        self.rollout_many = jax.jit(lambda s, acts, keys: _strip(jax.vmap(over_keys, (0, 0, 0))(s, keys, acts)))

        leaves, self.treedef = jax.tree_util.tree_flatten(reset(0)[0])
        self.leaf_specs = [(np.shape(x), np.asarray(x).dtype) for x in leaves]
        self.nbytes = sum(int(np.prod(s)) * d.itemsize for s, d in self.leaf_specs)


def _strip(out):
    """(state, obs, achievements) -> (obs, achievements): the end states of a rollout are not needed."""
    return out[1], out[2]


@functools.lru_cache(maxsize=1)
def sim() -> Sim:
    return Sim()


def pack(state) -> np.ndarray:
    """The flat uint8 dump of one state, as the dataset column `full_state`."""
    parts = []
    for leaf in jax.tree_util.tree_leaves(state):
        leaf = np.asarray(leaf)
        if leaf.dtype == np.bool_:
            leaf = leaf.astype(np.uint8)
        parts.append(np.ascontiguousarray(leaf).reshape(-1).view(np.uint8))
    return np.concatenate(parts)


def unpack(full_state: np.ndarray, batched: bool = False):
    """The state of a `full_state` row; with `batched`, rows (n, bytes) give one state with a leading axis n."""
    s = sim()
    buf = np.asarray(full_state, dtype=np.uint8)
    buf = buf.reshape(len(buf), -1) if batched else buf.reshape(1, -1)
    if buf.shape[1] != s.nbytes:
        raise ValueError(f"full_state has {buf.shape[1]} bytes, expected {s.nbytes}: not Craftax 1.6.1")
    leaves, offset = [], 0
    for shape, dtype in s.leaf_specs:
        size = int(np.prod(shape)) * dtype.itemsize
        chunk = np.ascontiguousarray(buf[:, offset : offset + size])
        offset += size
        leaf = chunk.astype(np.bool_) if dtype == np.bool_ else chunk.view(dtype)
        leaf = leaf.reshape(len(buf), *shape)
        leaves.append(jnp.asarray(leaf if batched else leaf[0]))
    return jax.tree_util.tree_unflatten(s.treedef, leaves)


def true_step_key(seed: int):
    """The step key of the episode of world `seed` (the dataset's)."""
    return jax.random.split(jax.random.PRNGKey(int(seed)), 3)[1]


def fresh_keys(rng: np.random.Generator, *shape) -> jnp.ndarray:
    """Keys (*shape, 2) that no episode of the dataset uses: drawn from a numpy generator, not from a world seed."""
    return jnp.asarray(rng.integers(0, 2**32, size=(*shape, 2), dtype=np.uint32))


def replay_episode(seed: int, actions: np.ndarray) -> np.ndarray:
    """The observations (len(actions) + 1, 1345) of world `seed` under `actions` with the true keys."""
    s = sim()
    state, key = s.reset(int(seed))
    _, obs = s.replay(state, key, jnp.asarray(actions, dtype=jnp.int32))
    return np.concatenate([np.asarray(s.observe(state))[None], np.asarray(obs)])


# --- the guard: the replay contract holds on this machine ----------------------------------------------------------

FLOAT_TOLERANCE = 1e-6  # the light level: a GPU (where the data was collected) and a CPU differ in its last bits


def leaf_layout() -> list[tuple]:
    """(name, dtype, byte offset, bytes) of every leaf of `full_state`."""
    state = sim().reset(0)[0]
    out, offset = [], 0
    for path, leaf in jax.tree_util.tree_flatten_with_path(state)[0]:
        leaf = np.asarray(leaf)
        out.append((jax.tree_util.keystr(path), leaf.dtype, offset, leaf.size * leaf.dtype.itemsize))
        offset += out[-1][3]
    return out


def differing_leaves(a: np.ndarray, b: np.ndarray) -> list[str]:
    """The leaves where two `full_state` rows differ beyond the tolerance (exact for integer and boolean leaves)."""
    bad = []
    for name, dtype, offset, size in leaf_layout():
        x, y = a[offset : offset + size], b[offset : offset + size]
        if np.array_equal(x, y):
            continue
        if np.issubdtype(dtype, np.floating):
            d = np.abs(x.view(dtype).astype(np.float64) - y.view(dtype).astype(np.float64)).max()
            if d <= FLOAT_TOLERANCE:
                continue
        bad.append(name)
    return bad


def same_obs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per frame: the observations agree (exact except the light level, within the tolerance)."""
    from .layout import STATE_DIM

    light = STATE_DIM - 2
    rest = np.ones(STATE_DIM, bool)
    rest[light] = False
    return (a[:, rest] == b[:, rest]).all(1) & (np.abs(a[:, light] - b[:, light]) <= FLOAT_TOLERANCE)


def check_episodes(table: dict, episodes) -> list[str]:
    """Replay whole episodes of a cache table from the seed and the executed actions; returns the failures."""
    from .layout import expand, take

    fails = []
    for e in episodes:
        o, n = int(table["ep_offset"][e]), int(table["ep_len"][e])
        obs = replay_episode(int(table["ep_seed"][e]), table["action"][o : o + n - 1].astype(np.int32))
        ok = same_obs(obs, expand(take(table, slice(o, o + n))))
        if not ok.all():
            fails.append(f"episode {e}: frame {int(np.argmin(ok))} of {n} differs")
    return fails


def check_steps(table: dict, full: dict, rows) -> list[str]:
    """One true step from each kept `full_state`: the next observation must be the cached next frame."""
    from .cache import full_state_of
    from .layout import expand, take

    episode = np.repeat(np.arange(len(table["ep_len"])), table["ep_len"])
    fails = []
    for t in rows:
        t = int(t)
        state = unpack(full_state_of(full, t)[0])
        state = sim().step(state, true_step_key(table["ep_seed"][episode[t]]), jnp.int32(table["action"][t]))
        obs = np.asarray(sim().observe(state))[None]
        if not same_obs(obs, expand(take(table, [t + 1])))[0]:
            fails.append(f"frame {t}: the step from full_state differs")
    return fails


def main(argv=None):
    """The guard test: `python -m k2.sim [--split val] [--episodes 8] [--steps 64]` (seconds)."""
    import argparse

    from .cache import load, load_full

    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--split", default="val")
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--cache", default=None)
    args = p.parse_args(argv)
    kw = {"cache_dir": args.cache} if args.cache else {}
    table, full = load(args.split, **kw), load_full(args.split, **kw)
    rng = np.random.default_rng(0)
    episodes = rng.choice(len(table["ep_len"]), min(args.episodes, len(table["ep_len"])), replace=False)
    rows = full["sample"][: args.steps]
    fails = check_episodes(table, episodes) + check_steps(table, full, rows)
    print("\n".join(fails) or f"replay contract holds: {len(episodes)} episodes, {len(rows)} steps from full_state")
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
