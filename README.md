# K2

Level 2 of a hierarchical JEPA world model on Craftax-Classic.

## Goal

An encoder `z = E(s_t)` and a predictor `ẑ_{t+k} = P(z_t, m_t)` at stride `k = 4`, with the macro-action `m_t` the 4 executed actions between the two frames, such that

1. **predictable:** P predicts `z_{t+k}` almost without error;
2. **as fine as possible:** `z` holds as much about `s_t` as (1) allows. What is not predictable shall be abstracted until it is, not deleted.

**Frame.** The input is the observation `state` (one frame, 1345-d), a stand-in for a latent we already have (DINOv3, LeVJEPA, ...). Level 1 plans in it, one action per step, with CEM in the JAX simulator; this model is the level above it. The losses see only `state` and the actions; `full_state`, the replay oracle and the readouts are for evaluation only.

## Concepts

**Three things called abstraction.**
1. *Selection:* keep some factors, drop the others. A JEPA does this: it shrinks what it cannot predict toward zero variance; SIGReg only prevents the total collapse.
2. *Coarse-graining:* a many-to-one map `φ`. A mob's tile becomes its quadrant: the content stays, at a coarser resolution. This is the goal's sense.
3. *Predicting expectations:* an MSE predictor outputs `E[z_{t+k}]`, a statement about a distribution, not a coarser state.

In a continuous latent, coarse-graining is invariance: states that differ only below the resolution map to the same point. Selection is invariance by deletion.

**The ideal JEPA** (Tang et al. 2023, "Understanding Self-Predictive Learning for Reinforcement Learning": linear, stop-gradient, a predictor at its optimum): the latent's covariance stays fixed and the latent turns to the top-D eigenvectors of the policy-averaged transition operator, the D most persistent modes of the dynamics (for k steps, of its k-th power). Such modes are nearly constant on sets of states the dynamics rarely leaves and change smoothly between them: a soft coarse-graining. Two departures make it a slow-feature collapse: persistent is not predictable (a mode that changes fast but is set by the action ranks low, because the operator averages over actions), and a nonlinear encoder can fill the D places with smooth functions of the slow content (harmonics) first.

**Closure.** `z = φ(s)` is closed at stride k when the full state adds nothing to the prediction of `z`'s future beyond `z` itself: `p(z_{t+k} | s_t, m_t) = p(z_{t+k} | z_t, m_t)`. Example: inventory alone is not closed, since whether wood is collected depends on the tile the player faces; a mob's quadrant is not closed while a mob near the border may cross it. Closure is the goal's "needed for the prediction"; determinism on top of it is the goal's "predictable". The identity map is closed but not deterministic, a constant map both but empty: closure alone does not pick a level, the goal's "as fine as possible" does.

| Line of work | What it says | For this goal |
|---|---|---|
| State abstraction in RL (Li, Walsh & Littman 2006) | A unified framework for aggregating MDP states; the strictest kind keeps a closed abstract model | closure, there with rewards |
| Markov state abstractions (Allen et al., NeurIPS 2021) | an abstraction that keeps the Markov property, learned without reward (inverse model + temporal contrast) | closure without labels |
| Self-predictive RL (Tang et al., ICML 2023; Ni et al., ICLR 2024) | idealized (linear, tabular, stop-gradient, optimal predictor): the covariance is conserved and the latent turns toward the top eigenvectors of the transition matrix; for n steps, of its n-th power (stated, not proven) | a JEPA selects persistent modes; at stride k fast modes fade like λ^k. Shrinking comes from departures from the idealization |
| Lumpability, closure (Kemeny & Snell; Rosas et al. 2024) | informational closure: `I(X_t; Z_future | Z_past) = 0`; strong lumpability ⇔ `H(Z_{t+1} | Z_t) = H(Z_{t+1} | X_t)`; identity and constant map bound a lattice of closed levels; autonomous processes only | a precise, label-free definition of closure |
| Causal emergence (Hoel, Albantakis & Tononi, PNAS 2013) | a macro scale can carry more effective information (determinism, degeneracy) than the micro scale | why coarsening can raise predictability |
| Molecular kinetics: VAMPnets (Mardt et al. 2018), SPIB (Wang & Tiwary 2021) | coarse states whose dynamics at a lag time Δt is Markov; SPIB: a continuous Gaussian bottleneck predicts the discrete coarse state at t + Δt, the labels refined from its own predictions; a larger Δt merges states (4 to 3 to 2) | the lag is our k; the bottleneck stays continuous, only the target is coarse; no actions |

**Consequences.** The goal is an information trade-off: as much about `s_t` as possible while `z_{t+k}` stays predictable from `z_t` and `m_t`. The prediction loss alone asks for closure but rewards collapse; reconstruction rewards resolution but asks for no closure. Prediction error says nothing alone, since shrinking lowers it; closure is measurable without labels as the gap between predicting `z_{t+k}` from `s_t` and from `z_t`. The stride sets the granularity.

## Results

**R1. Persistence at the stride coarse-grains, but drops what the action sets.** The ideal linear JEPA latent at k = 4, i.e. the most persistent linear modes of the `state` (VAMP: canonical correlation of `s_t` and `s_{t+4}` within episodes, 200k train pairs, no actions; read out by linear regression, R² on val):
- Only 31 modes keep a canonical correlation above 0.9 (119 at k = 1).
- The top 64 hold counts in view, not tiles: iron 0.93 (tile 0.04), table 0.93, zombie 0.87. The top 256 add quadrants (iron 0.79, zombie 0.67). The top 16 hold only inventory and energy.
- The facing, set by the actions, drops out (0.07 at 64).
- PCA with 64 dimensions keeps the facing (0.99) and common tiles, and loses rare content (iron in view 0.17).

## Hints

What Grace, AAC and K2 have suggested so far. One seed each and often a different setting (pixels, a synthetic grid world): starting points, not results. Here everything is proven again. Runs: `runs/E0_k1`; `runs/archive/` and git tag `archive/k1-series` hold the rest.

**What a latent holds**
- Reconstruction makes content available. SIGReg, alone or with prediction, holds almost none and can hurt (Grace: mob 0.39 with it, 0.93 without; AAC A5, SIGReg + inverse dynamics without prediction: mobs and ores lost as with prediction; K2 E1/E2: SIGReg's encoder gradient 10 to 90 times the prediction's, content below the untrained encoder).
- Slow content and its nonlinear copies nearly satisfy SIGReg; prediction alone favours slow content (Grace; AAC B1's latent: inventory, vitals, daylight kept, every block and mob type lost).
- On the `state`, a gradient through the prediction target deleted content for good, while a detached target got it back late (AAC A3 against A4: zombie 0.18 / 0.67, iron 0.10 / 0.57 after 155k steps, close for the first 2k). K2's E1 and E2 ran 8k steps.
- A plain mean reconstruction learns rare content slowly; weighting by error or class speeds it up (Grace; K2 E0: whitened linear probe rare 0.85 at 12k steps and still rising, the error-weighted stage 1 of the archived E3 0.87 at 10k).
- Coarse facts survive where tiles do not: iron in view 0.50 against an iron-tile recall of 0.05, a table adjacent 0.85 (AAC B1).

**What the planner's distance sees**
- Low amplitude is not absence: a readout reads content that the planner's squared distance barely sees (Grace). A readout depends on the latent's shape: whitening lifted K2's E3 from rare 0.56 to 0.82 on the same latent.
- The distance is uneven. In AAC B1, in median one-step distances: sleeping 54, a tool 15 to 23, a pasted table next to the player 2.4, coal 1.5, zombie 0.45, cow 0.06, lava 0.02. In Grace a one-cell move outweighed a turn 6 times.
- Planning with B1: success 0.18 against 0.11 for random (offset 8, 128 tasks); plans were predicted to close 39 % of the goal distance and opened it by 27 %; 0 of 6 zombie and 0 of 6 iron goals.

**Prediction and selection**
- The decoded prediction `D(P(z, a))` keeps what the predictor learned and releases noise; a hold on the real latent is needed because the predictor can draw action-determined content (the facing) from the action alone; fading the reconstruction alone erodes everything (Grace). A per-token gate cannot coarsen, and without a scale term the latent blew up along one direction (K2 E3: participation ratio 23.6 to 1.7).
- Behaviour cloning and inverse dynamics over 8 steps rely on the zombie (0.12) and the near field, hardly on ores or lava (≤ 0.015; AAC).

**Measurement**
- Read with decoders, not probes (Grace: the facing 0.1 by probe, 0.95 by decoder). In K2 the fresh decoder cannot read tools, item_diamond, health, light and ripe_plant even from the raw `state` (0.33 to 0.67; `layout.LOW_CEILING`).
- Oracle (K2, 64 fresh-key replays, each tile's block and mobs apart): at k = 1 everything that changes is determined except the mobs (0.09) and some items (0.84). At k = 8: terrain 0.68, ores 0.69, stations 0.56, items 0.69, vitals 0.71, mobs 0.006; an uncertain player position blurs the egocentric blocks.

## Code

| File | Side | What |
|---|---|---|
| `k2/layout.py` | both | the `state` layout, its compact form (148 bytes per frame), tokens (82), factors (145: each tile's block and mobs apart), readout classes |
| `k2/sim.py` | JAX | Craftax in JAX on the CPU: replay contract, `full_state` pack/unpack, fresh-key rollouts, the guard |
| `k2/cache.py` | numpy | the data cache from AAC's HDF5 files (read by path), held-out `full_state` |
| `k2/oracle.py` | JAX | per factor, the share of 64 fresh-key replays that agree on the outcome at stride k |
| `k2/data.py` | torch | the cache on the device, batches of level-2 steps |
| `k2/model.py` | torch | tokenizer and encoder, causal residual predictor, per-token decoder, linear probe |
| `k2/losses.py` | torch | per-token losses; E0 (reconstruction) |
| `k2/train.py` | torch | the config, training, checkpoints, resume |
| `k2/readouts.py` | torch | what z holds (fresh decoder and linear probe on whitened z), prediction against copying, the raw latent's shape |
| `k2/report.py` | | the markdown report of a run |
| `notebooks/E0.ipynb` | | one notebook per variant (written by `notebooks/make_notebooks.py`) |

## Running

Every run of minutes or more waits for a go-ahead.

- **Mac, once:** `python3.12 -m venv .venv && PIP_INDEX_URL=https://pypi.org/simple PIP_EXTRA_INDEX_URL= .venv/bin/pip install -r requirements.txt`
- **Tests:** `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests`, about 20 s.
- **Cache:** `.venv/bin/python -m k2.cache` (train, val, test into `data/cache`, 1 to 2 minutes), then the guard `.venv/bin/python -m k2.sim --split val` (about 10 s).
- **Oracle:** `.venv/bin/python -m k2.oracle --k 1 8`, about a minute; writes `data/cache/oracle_k{1,8}.{npz,md}`.
- **Pod:** unzip `k2_pod.zip` (code, `data/cache/{train,val,val_full}.npz`, the oracle tables, `runs/E0_k1`) into `/root` and run `notebooks/E0.ipynb`: it resumes E0 from step 12000 to 20000 (about 20 minutes at the measured 0.14 s per step) and reads every checkpoint (about 3 minutes each; estimates). JAX processes run without `LD_LIBRARY_PATH`.
