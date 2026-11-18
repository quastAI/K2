# K2

Level 2 of a hierarchical JEPA world model on Craftax-Classic. Can a level-2 latent keep what is predictable at its stride, predict it and plan with it? Perception and level-1 control are assumed solved: the input is the observation `state`, and level 1 is CEM in the simulator.

- `docs/design.md`: the design note (input, the level-2 step, encoder, variants, the replay oracle, readouts, planning, order of work).
- `docs/aac_findings.md`: what the predecessor repo (AAC) found that bears on this.
- `docs/reference/`: analysis scripts from AAC, kept for their method (not runnable here).

## Code

`frame` mode and stage A (exact chunks) only, as in the design's order of work. Window mode, VQ macro-actions, the stochastic latent, E5 and transfer come later.

| File | Side | What |
|---|---|---|
| `k2/layout.py` | both | the `state` layout, its exact compact form (148 bytes per frame), tokens and readout classes |
| `k2/sim.py` | JAX | Craftax in JAX on the CPU: replay contract, `full_state` pack/unpack, fresh-key rollouts, the guard |
| `k2/cache.py` | numpy | the data cache built from AAC's HDF5 files (read by path), held-out `full_state`, planning tasks |
| `k2/oracle.py` | JAX | the oracle table: per factor, the share of 32 fresh-key replays matching the most common outcome |
| `k2/data.py` | torch | the cache on the device, batches of level-2 steps |
| `k2/model.py` | torch | tokenizer and encoder, causal residual predictor, per-token decoder, linear probe |
| `k2/losses.py` | torch | SIGReg, per-token losses, the losses of E0 to E3, E3's target-gradient ramp |
| `k2/train.py` | torch | the config, training, checkpoints, resume |
| `k2/readouts.py` | torch | fresh decoder and linear probe per class, kept against predictable, prediction, geometry |
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
1. **Mac: the oracle table.** `.venv/bin/python -m k2.oracle --k 1 8`: 2048 val frames x 32 keys, about a minute (the simulator does about 24k steps/s here, measured). Writes `data/cache/oracle_k{1,8}.{npz,md}`.
2. **Pod: E0**, then 3. **E1 against E2**, then **E3 against E2** (E2 with the target's gradient ramped up once the predictor plateaus on every dimension), at k = 1: the notebooks, one epoch (8000 steps) each, about 3 to 5 minutes of training plus readouts on an A40. Upload `data/cache/{train,val,val_full}.npz` and `oracle_k*.npz` first.
5. **k = 8:** the same notebooks with `K = 8`. Planning (`python -m k2.plan runs/<run> --split val`, run by the notebook) restores the task starts from `val_full.npz` and needs `achievements` from `val.npz`; its time is still to be measured (`python -m k2.plan --speed` gives the simulator's rate on a machine).

The JAX processes on a pod run without `LD_LIBRARY_PATH` (the planning cell strips it).
