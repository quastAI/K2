# CLAUDE.md

K2 studies the level-2 selection problem of a hierarchical JEPA world model on Craftax-Classic. The setting: a level-2 latent that keeps what is predictable at its stride given its macro-action, predicts it and plans with it. The input is the observation `state` (a perfect level 1). The executor is CEM in the JAX simulator. `docs/design.md` is the plan; `docs/aac_findings.md` holds what the predecessor repo found.

## Working rules (from the user)

- Never edit or create files unless told to. A plan or a question is not a go-ahead: explain, give options, ask.
- Never start a run of minutes or more without an explicit go-ahead. Give the command and a time estimate instead. Checks of seconds are fine.
- Commit and push only when asked.
- Unsupervised only. The model's losses see only the input (`state`) and the actions. `full_state`, the replay oracle and the probes are for evaluation only.

## Principles

- **Self-contained.** K2 never imports from or links to AAC at runtime. AAC is a reference to read; what K2 needs is rewritten here, small and adapted. The data files are read by path, never copied.
- **Minimal and clean.** Build only what the current step of `docs/design.md` needs: one encoder, one predictor, one loss module, one readout set. No config framework, no generality "for later", no options nobody runs. Remove what a finished step no longer needs.
- **One notebook per variant.** `notebooks/<ID>.ipynb` (E0, E1, ...) runs that variant end to end: data check, training, readouts, report. The notebooks share the code in the package and differ only in their configuration cell.- **Targeted, clean experiments.** Each experiment answers one open question carried over from AAC (`docs/design.md`, "Order of work"). It is compared against its control and reported with what it does not show.
- **Two sides with a file boundary.**
  - **Simulator side:** JAX on the CPU, run on the Mac. It builds the data cache, runs the replay test and computes the oracle table, and writes plain `.npz` files.
  - **Model side:** PyTorch, trained on a RunPod GPU. It holds the encoder, predictor, losses, training and readouts, and reads those files.
  - **Planning:** the executor puts both in one process. The simulator runs in JAX on the CPU, the encoder in torch, and they exchange numpy arrays.
  - **Smoke tests of the torch code** run on the Mac CPU before any pod.

## Where AAC implemented it (read, then rewrite small)

Paths are relative to `../AAC`.

| Need | AAC file |
|---|---|
| state layout, view/tile helpers | `H-JEPA/h_jepa/probing/targets.py` |
| pack/unpack `full_state`, env, goal test (`new_achievements`) | `H-JEPA/stable_worldmodel/envs/craftax/env.py` |
| replay keys, executed actions, collector | `purejaxql/scripts/collect_craftax_episodes.py` |
| compact state cache (126 bytes per frame) | `H-JEPA/h_jepa/bc.py` (`StateTable`) |
| SIGReg | `H-JEPA/h_jepa/loss.py` |
| the relative floored gate, the `diff` gate | `H-JEPA/h_jepa/sieve/losses.py` |
| VQ latent actions | `H-JEPA/h_jepa/models/vq.py` |
| categorical CEM | `H-JEPA/stable_worldmodel/solver/cem.py` |
| task episodes, planning loop | `H-JEPA/stable_worldmodel/world.py`, `H-JEPA/h_jepa/eval_tasks.py` |
| lagged-CCA basis, change spectrum, decoder as a probe, gate unevenness | `docs/reference/` (copied here, AAC session scripts) |

## Predecessor: `../AAC` (read-only here)

- **Data:** `../AAC/data/hf/craftax-classic-pqn-rnn-v1/data/craftax_expert_{train,val,test}.h5`. Columns: `state` (N, 1345) float32, `action` (executed, 0-16), `full_state` (N, 20952) uint8, `seed`, `source`, and others (AAC README, column table). Never copy them. Read them through a cache.
- **Replay contract:** `reset_key, step_key, _ = split(PRNGKey(seed), 3)`. The step at `state.timestep == t` uses `fold_in(step_key, t)`. `full_state` is the flat uint8 dump of the state pytree in leaf order (`pack_state`/`unpack_state` in `../AAC/H-JEPA/stable_worldmodel/envs/craftax/env.py`). Craftax is pinned at 1.6.1.
- **Randomness:** only the step key carries the dynamics' randomness. `state_rng` feeds only the renderer's night static.
- **Layout of the `state`:** `../AAC/H-JEPA/h_jepa/probing/targets.py`. View 7 x 9 x (17 blocks + 4 mobs), then inventory (12, /10), vitals (4, /10), facing (4), light, sleeping. The player is at row 3, column 4.
- **Environment pitfalls:** `../AAC/CLAUDE.md`, "Pitfalls seen".
- **Old venvs:** the `python3` on PATH may be `Craftax/.venv` (jax 0.11); never use it here.

## Environments

- **One venv per machine,** with the CPU-only JAX wheel next to torch: `jax==0.4.38 jaxlib==0.4.38 flax<=0.10.3 craftax==1.6.1 numpy h5py torch pytest`. Set `JAX_PLATFORMS=cpu`. JAX's own CUDA libraries are never installed, which avoids AAC's clash between JAX and torch CUDA.
- **On the Mac:** install with `PIP_INDEX_URL=https://pypi.org/simple PIP_EXTRA_INDEX_URL=`. The default index is a private registry with expired credentials; do not change the pip config.

## RunPod (from AAC's pod sessions)

- **Local disk for working files.** Keep the clone, the venv, the data cache and the logs on the local disk (e.g. `/root`), not on the network volume `/workspace`. On the volume, `git clone` and `venv` fail (no `chmod`), and frequent writes stall training for about 10 s each. Only checkpoints and results go to the volume.
- **Data travels small.** Upload the compact cache (about 250 MB) built on the Mac; do not download the 7.7 GB HDF5 file onto every pod.
- **Clear `LD_LIBRARY_PATH`** for any JAX process: the image's CUDA path made JAX crash in AAC.
- **CPU quota:** the container may get far fewer CPUs than `nproc` shows (`cat /sys/fs/cgroup/cpu.max`). Limit torch threads and `OMP_NUM_THREADS` to the quota.
- **Before stopping a pod,** wait 10 to 15 minutes after the last write to the volume and run `sync`. Large files written shortly before a stop went missing once.
