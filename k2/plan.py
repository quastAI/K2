"""Planning: level 2 in the latent, level 1 (the executor) in the simulator. JAX on the CPU and torch in one process.

A task (`<split>_full.npz`) starts at an expert frame s_t (restored from `full_state`) with the goal frame
t + H k; success is AAC's `new_achievements` test (every achievement the goal has and the start lacks is unlocked)
within the budget of env steps. The real environment steps with the episode's true keys; every simulated rollout
uses fresh keys, so planning never sees the true future randomness.

Methods:
    level2           CEM over H macro-actions (exact chunks, k x 17 categorical) in the latent: the predictor rolls
                     z out, cost ||z^_H - z_goal||; the executor reaches z^_1, then plan again (receding, H shrinks to 1)
    oracle_subgoals  the expert's frames every k steps, encoded, as the executor's targets: its ceiling, and how often
                     it reaches a subgoal exactly (tiles, inventory, vitals, facing, sleeping; mobs and light ignored)
    flat_latent      CEM in the simulator over the remaining actions toward the goal, cost the encoder distance
    flat_hamming     the same with a fixed distance on the state: mismatching tokens, against the goal frame
    random           uniform random actions

The executor: CEM over k actions; each candidate is rolled out under several fresh keys and scored by
||enc(x_sim) - target|| averaged over the keys; the best sequence is carried out in the real environment.

    python -m k2.plan runs/E2_k8 --split val          # writes <run>/planning.json
    python -m k2.plan --speed                          # simulator steps per second (seconds)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from . import layout as L
from . import sim
from .cache import full_state_of, load, load_full
from .data import from_states
from .train import load_model

METHODS = ("level2", "oracle_subgoals", "flat_latent", "flat_hamming", "random")


@dataclass
class PlanConfig:
    split: str = "val"
    tasks: int = 128
    horizon: int = 8  # H level-2 steps between start and goal
    budget_factor: float = 2.0  # env steps = budget_factor * H * k
    exec_samples: int = 64
    exec_iters: int = 4
    exec_elites: int = 8
    exec_keys: int = 4
    plan_samples: int = 256
    plan_iters: int = 6
    plan_elites: int = 32
    alpha: float = 0.5
    min_prob: float = 0.05
    methods: tuple = METHODS
    seed: int = 0


def categorical_cem(cost, length, classes, samples, iters, elites, alpha, min_prob, rng, init=None):
    """CEM over `length` independent categorical slots: sample, keep the `elites` cheapest, move the distributions toward
    their frequencies (step `alpha`), then spread `min_prob` of the mass uniformly. Returns the best sequence seen and
    its cost. `cost` maps (S, length) int arrays to (S,) costs."""
    probs = np.full((length, classes), 1.0 / classes) if init is None else init
    best, best_cost = None, np.inf
    for _ in range(iters):
        u = rng.random((samples, length, 1))
        seqs = (u > probs.cumsum(-1)[None]).sum(-1).clip(max=classes - 1)
        if best is not None:
            seqs[0] = best
        c = np.asarray(cost(seqs))
        order = np.argsort(c)[:elites]
        if c[order[0]] < best_cost:
            best, best_cost = seqs[order[0]].copy(), float(c[order[0]])
        freq = np.eye(classes)[seqs[order]].mean(0)
        probs = (1 - alpha) * probs + alpha * freq
        probs = (1 - min_prob) * probs + min_prob / classes
    return best, best_cost


class Planner:
    def __init__(self, model, cfg, pc: PlanConfig, device="cpu"):
        self.model, self.cfg, self.pc, self.device = model, cfg, pc, device
        self.rng = np.random.default_rng(pc.seed)

    @torch.no_grad()
    def encode(self, states: np.ndarray) -> torch.Tensor:
        return self.model.encoder(from_states(states, self.device))

    def rollout(self, env, seqs):
        """Observations (S, R, 1345) of action sequences from the env's current state under fresh keys."""
        keys = sim.fresh_keys(self.rng, len(seqs), self.pc.exec_keys)
        obs, _ = sim.sim().rollout(env.state, np.asarray(seqs, np.int32), keys)
        return np.asarray(obs)

    def latent_cost(self, env, target: torch.Tensor):
        def cost(seqs):
            obs = self.rollout(env, seqs)
            z = self.encode(obs.reshape(-1, L.STATE_DIM)).view(len(seqs), -1, target.shape[-1])
            return (z - target).norm(dim=-1).mean(1).cpu().numpy()

        return cost

    def hamming_cost(self, env, goal_values: np.ndarray):
        def cost(seqs):
            obs = self.rollout(env, seqs)
            values = L.token_values(L.compress(obs.reshape(-1, L.STATE_DIM))).reshape(len(seqs), obs.shape[1], -1)
            return (values != goal_values).sum(-1).mean(1)

        return cost

    def cem(self, cost, length):
        pc = self.pc
        return categorical_cem(cost, length, L.NUM_ACTIONS, pc.exec_samples, pc.exec_iters, pc.exec_elites,
                               pc.alpha, pc.min_prob, self.rng)[0]  # fmt: skip

    @torch.no_grad()
    def plan_level2(self, z_now: torch.Tensor, z_goal: torch.Tensor, h: int):
        """The best h macro-actions in the latent; returns the predicted latent after the first of them."""
        k, pc = self.cfg.k, self.pc

        def cost(seqs):
            m = torch.as_tensor(seqs, device=self.device).view(len(seqs), h, k)
            zhat = self.model.predictor.rollout(z_now.expand(len(seqs), 1, -1), m)
            return (zhat[:, -1] - z_goal).norm(dim=-1).cpu().numpy()

        best, _ = categorical_cem(cost, h * k, L.NUM_ACTIONS, pc.plan_samples, pc.plan_iters, pc.plan_elites,
                                  pc.alpha, pc.min_prob, self.rng)  # fmt: skip
        m = torch.as_tensor(best, device=self.device).view(1, h, k)
        return self.model.predictor.rollout(z_now.view(1, 1, -1), m[:, :1])[0, 0]


def reached(obs: np.ndarray, target: np.ndarray) -> bool:
    """The real frame matches the target frame exactly, mobs and light ignored."""
    a, b = (L.compress(x[None]) for x in (obs, target))
    a["mob"][:] = b["mob"][:] = 0
    keep = np.ones(L.NUM_TOKENS, bool)
    keep[L.LIGHT_TOKEN] = False
    return bool((L.token_values(a) == L.token_values(b))[0, keep].all())


def run_task(planner: Planner, method: str, task: dict) -> dict:
    """One task under one method: success, the env steps used, and (oracle subgoals) the share of subgoals reached."""
    k, H, pc = planner.cfg.k, planner.pc.horizon, planner.pc
    budget = int(pc.budget_factor * H * k)
    env = sim.RealEnv(task["full_state"], task["seed"])
    start, goal = env.start_achievements, task["goal_achievements"]
    z_goal = planner.encode(task["goal_state"][None])[0] if method != "random" and method != "flat_hamming" else None
    goal_values = L.token_values(L.compress(task["goal_state"][None]))[0]
    used, hits, j = 0, [], 0
    done = lambda: sim.new_achievements(env.achievements(), goal, start) or env.dead()  # noqa: E731
    while used < budget and not done():
        n = min(k, budget - used)
        h = max(H - j, 1)
        if method == "random":
            seq = planner.rng.integers(0, L.NUM_ACTIONS, n)
        elif method == "level2":
            z_sub = planner.plan_level2(planner.encode(env.obs()[None])[0], z_goal, h)
            seq = planner.cem(planner.latent_cost(env, z_sub), k)
        elif method == "oracle_subgoals":
            target = task["path"][min(j + 1, H)]
            seq = planner.cem(planner.latent_cost(env, planner.encode(target[None])[0]), k)
        elif method == "flat_latent":
            seq = planner.cem(planner.latent_cost(env, z_goal), h * k)
        elif method == "flat_hamming":
            seq = planner.cem(planner.hamming_cost(env, goal_values), h * k)
        else:
            raise ValueError(method)
        for a in seq[:n]:
            env.step(int(a))
            used += 1
            if done():
                break
        if method == "oracle_subgoals" and j < H:
            hits.append(reached(env.obs(), task["path"][j + 1]))
        j += 1
    out = {"success": sim.new_achievements(env.achievements(), goal, start), "steps": used, "dead": env.dead()}
    if hits:
        out["subgoal_reached"] = float(np.mean(hits))
    return out


def load_tasks(split, cache, k, horizon, n):
    """The tasks of a split with the expert's frames every k steps from start to goal (`path`, H + 1 frames)."""
    table, full = load(split, cache), load_full(split, cache)
    offset = int(full["task_offset"])
    if offset != k * horizon:
        raise ValueError(f"the cache's tasks have the goal {offset} steps ahead, planning needs H k = {horizon * k}")
    tasks = []
    for s, g, e in list(zip(full["task_start"], full["task_goal"], full["task_episode"]))[:n]:
        path = L.expand(L.take(table, s + k * np.arange(horizon + 1)))
        tasks.append({"start": int(s), "goal": int(g), "episode": int(e), "seed": int(table["ep_seed"][e]),
                      "full_state": full_state_of(full, s)[0], "goal_state": path[-1], "path": path,
                      "goal_achievements": table["achievements"][g]})  # fmt: skip
    return tasks


def run_planning(run_dir, pc: PlanConfig, device="cpu", checkpoint=None, cache=None) -> dict:
    """Every method on the tasks of `pc.split` with the run's last checkpoint; `cache` overrides the run's cache folder
    (a run trained on a pod, planned on the Mac)."""
    run_dir = Path(run_dir)
    ckpt = checkpoint or sorted(run_dir.glob("ckpt_*.pt"))[-1]
    cfg, model = load_model(ckpt, device)
    if cache:
        cfg.cache = str(cache)
    if model.predictor is None:
        pc = dataclasses.replace(pc, methods=tuple(m for m in pc.methods if m != "level2"))
    tasks = load_tasks(pc.split, cfg.cache, cfg.k, pc.horizon, pc.tasks)
    planner = Planner(model, cfg, pc, device)
    result = {"split": pc.split, "offset": cfg.k * pc.horizon, "budget": int(pc.budget_factor * pc.horizon * cfg.k),
              "checkpoint": str(ckpt), "config": dataclasses.asdict(pc), "methods": {}, "per_task": {}}  # fmt: skip
    for method in pc.methods:
        t0 = time.time()
        rows = [run_task(planner, method, t) for t in tasks]
        result["per_task"][method] = rows
        result["methods"][method] = {
            "tasks": len(rows), "success": float(np.mean([r["success"] for r in rows])),
            "steps": float(np.mean([r["steps"] for r in rows])), "sec": round(time.time() - t0, 1),
            **({"subgoal_reached": float(np.mean([r["subgoal_reached"] for r in rows]))}
               if rows and "subgoal_reached" in rows[0] else {}),
        }  # fmt: skip
        print(f"{method}: success {result['methods'][method]['success']:.3f} ({time.time() - t0:.0f} s)")
        (run_dir / "planning.json").write_text(json.dumps(result, indent=1))
    return result


def speed(batch=256, keys=4, length=8, repeats=3) -> float:
    """Simulator steps per second of a fresh-key rollout batch on this machine's JAX backend."""
    rng = np.random.default_rng(0)
    state, _ = sim.sim().reset(0)
    seqs = rng.integers(0, L.NUM_ACTIONS, (batch, length)).astype(np.int32)
    sim.sim().rollout(state, seqs, sim.fresh_keys(rng, batch, keys))[0].block_until_ready()
    t0 = time.time()
    for _ in range(repeats):
        sim.sim().rollout(state, seqs, sim.fresh_keys(rng, batch, keys))[0].block_until_ready()
    return batch * keys * length * repeats / (time.time() - t0)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", nargs="?")
    p.add_argument("--speed", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--cache", default=None, help="the cache folder, if not the run's")
    for f in dataclasses.fields(PlanConfig):
        if f.name == "methods":
            p.add_argument("--methods", nargs="+", choices=METHODS)
        else:
            p.add_argument(f"--{f.name}", type=type(f.default))
    a = vars(p.parse_args(argv))
    if a.pop("speed"):
        print(f"{speed():.0f} simulator steps per second ({sim.jax.devices()[0]})")
        return
    run, device, ckpt, cache = a.pop("run"), a.pop("device"), a.pop("checkpoint"), a.pop("cache")
    pc = PlanConfig(**{k: (tuple(v) if k == "methods" else v) for k, v in a.items() if v is not None})
    run_planning(run, pc, device, ckpt, cache)


if __name__ == "__main__":
    main()
