# K2

Level 2 of a hierarchical JEPA world model on Craftax-Classic. Can a level-2 latent keep what is determined at its stride given its macro-action, plus what the goals at the next boundary need, predict it and plan with it? Perception and level-1 control are assumed solved: the input is the observation `state`, and level 1 is CEM in the simulator.

- `docs/design.md`: the goal, what we know (findings table), the measurement fixes, the open questions in order, the literature.
- `docs/aac_findings.md`: what the predecessor repo (AAC) found that bears on this.
- `docs/reference/`: analysis scripts from AAC, kept for their method (not runnable here).

## Code

`frame` mode (one frame per level-2 step) and exact action chunks as macro-actions only.

| File | Side | What |
|---|---|---|
| `k2/layout.py` | both | the `state` layout, its exact compact form (148 bytes per frame), tokens and readout classes |
| `k2/sim.py` | JAX | Craftax in JAX on the CPU: replay contract, `full_state` pack/unpack, fresh-key rollouts, the guard |
| `k2/cache.py` | numpy | the data cache built from AAC's HDF5 files (read by path), held-out `full_state`, planning tasks |
| `k2/oracle.py` | JAX | the oracle table: per factor, the share of 64 fresh-key replays matching the most common outcome; predictable at three lines (all agreed, 0.9, 0.75), per kind, for the factors that change |
| `k2/data.py` | torch | the cache on the device, batches of level-2 steps |
| `k2/model.py` | torch | tokenizer and encoder, causal residual predictor, per-token decoder, linear probe |
| `k2/losses.py` | torch | SIGReg, per-token losses, the losses of E0 to E3 (E3: the Grace model, two stages in one run) |
| `k2/train.py` | torch | the config, training, checkpoints, resume |
| `k2/readouts.py` | torch | readouts v2 on whitened z: fresh decoder and linear probe per class, kept against the target D ∪ G, prediction, geometry, the latent's spectrum |
| `k2/report.py` | | the markdown report, read against the control, with what it does not show |
| `k2/plan.py` | both | level-2 CEM in the latent, the level-1 executor CEM in the simulator, the controls |
| `notebooks/E0..E3.ipynb` | | one notebook per variant (written by `notebooks/make_notebooks.py`) |

## Order of work, with commands

Every run of minutes or more waits for a go-ahead. The times are estimates unless marked as measured.

0. **Mac: environment, cache, guard.**
   - `python3.12 -m venv .venv && PIP_INDEX_URL=https://pypi.org/simple PIP_EXTRA_INDEX_URL= .venv/bin/pip install -r requirements.txt`
   - `.venv/bin/python -m pytest tests`: about 20 s (measured). The replay test needs AAC's val file.
   - `.venv/bin/python -m k2.cache`: train, val and test into `data/cache`. About 1 to 2 minutes, about 450 MB (train about 300 MB).
   - `.venv/bin/python -m k2.sim --split val`: the guard (true-key replay reproduces the stored frames). About 10 s.
1. **Mac: the oracle table.** `.venv/bin/python -m k2.oracle --k 1 8`: 2048 val frames x 64 keys, about a minute (measured; the simulator does about 24k steps/s here). Writes `data/cache/oracle_k{1,8}.{npz,md}`.
2. **Done (2026-11-20): the k = 1 series E0 to E3** on a pod. `docs/design.md` has what it showed.
3. **k = 8:** the variant notebooks with `K = 8`. Planning (`python -m k2.plan runs/<run> --split val`, run by the notebook) restores the task starts from `val_full.npz` and needs `achievements` from `val.npz`; its time is still to be measured (`python -m k2.plan --speed` gives the simulator's rate on a machine).

On a pod, unzip `k2_pod.zip` into `/root` first. It holds the code, `data/cache/{train,val,val_full}.npz` and the oracle tables. Each notebook's first cells install `requirements.txt` into the kernel's Python and run the tests.

The JAX processes on a pod run without `LD_LIBRARY_PATH` (the planning cell strips it).
