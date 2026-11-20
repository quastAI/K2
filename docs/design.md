# K2: design note

2026-11-20. Replaces the note of 2026-11-18 (git `fa6b497`), which planned E0 to E5 before anything had run. From here on: one question at a time, its hypotheses, the cheapest check that separates them, then a run.

## The goal

A level-2 world model at stride `k`: an encoder `z = E(s_t)`, a predictor `ẑ_{t+k} = P(z_t, m_t)` with the macro-action `m_t = (a_t .. a_{t+k-1})`, and a planner over `m` in `z` whose subgoals level 1 executes.

**What `z` should hold (decided 2026-11-20):**
- **D_k, the determined factors:** a factor whose value at `t+k` is fixed by `s_t` and `m_t`. Measured by the oracle: all 64 fresh-key replays agree (0.9 and 0.75 as checks).
- **G_k, the goal factors at `t+k`:** the factors the goal test reads at the next level-2 boundary, whether determined or not: inventory, vitals, sleeping, and the block types of the tiles (placed and removed blocks). At `k = 1` they are almost all in D_1, but not all. For example, collecting a sapling succeeds by chance (item_sapling: 0.09 determined).
- **Drop the rest.** That is mostly where mobs are (at `k = 1` only 0.09 of the changing mob factors are determined). A level-2 latent should not spend capacity or planning cost on it.

The latent is good to the degree that planning with it succeeds. The baselines are oracle subgoals (the ceiling), a flat simulator CEM, and a random policy. The readouts below are diagnostics.

**The frame stays as decided:** perfect perception (the input is the observation `state`) and perfect control (level 1 is CEM in the JAX simulator). This removes every pixel confound AAC had (the τ threshold, a decoder that cannot draw rare tiles) and makes an exact oracle possible. Two caveats remain:
- A real level-1 latent will already be pruned. AAC lost arrows and saplings.
- The `state` is egocentric and partial.

**Rules:** unsupervised. The losses see only `state` and the actions. A weight computed from the input's own statistics is allowed and named as such. `full_state`, the oracle and the probes are for evaluation only. Planning never sees the true future keys.

## What we know (k = 1, one seed per run, 2026-11-20)

Readouts are balanced accuracy on val. "Rare" covers the rare blocks and the mobs; for mobs chance is 0.5. "init" is the untrained encoder. The run readouts (v1) read raw z and understate content where the latent is stretched along few directions (F13). "Whitened probe" is the linear probe on whitened z, measured on the Mac. Readouts v2 (Q1) redo all of it.

| # | Finding | Evidence |
|---|---|---|
| F1 | E1 and E2 lose content by step 100 and never get it back. | rare 0.57 (init) → 0.34 at step 100 → 0.37 at step 8000; coal 0.45 → 0.10; facing 0.63 → 0.28 (v1). Whitened probe, rare / blocks / vitals: init 0.54 / 0.48 / 0.59, E1 0.41 / 0.36 / 0.42, E2 0.40 / 0.34 / 0.45. |
| F2 | Detaching the target changes nothing. | E1 and E2 agree within 0.02 at every checkpoint. |
| F3 | **In E1 and E2, SIGReg drives the encoder and the prediction barely does.** | Encoder gradient norm SIGReg / prediction: 2000x at step 0, 30 to 41x at step 100, 7 to 10x at step 8000. SIGReg is about 90% of the loss. |
| F4 | SIGReg does not reach its target. The latent collapses in dimension. | SIGReg 11.8 (E1) and 12.9 (E2), against 1.04 for N(0, I) at B = 512. Rank about 50 of 256; participation ratio 9 to 11 (E0: 233 and 26). |
| F5 | The E1/E2 latent is slow, but no slower than an untrained encoder. | Median slowness per whitened direction: E1 0.05, E2 0.08, init 0.10, E0 0.28, raw state 0.86. Change rate against what E1 keeps over init: Spearman −0.26. |
| F6 | Reconstruction (E0) can hold almost everything but had not converged. | rare 0.46 (step 1000) → 0.81 (10000) → 0.88 (12000), still rising; below init until about step 3000. |
| F7 | Reconstruction does not select. | E0 keeps 0.65 of the determined changing factors and 0.53 of the undetermined ones. |
| F8 | At `k = 1` the undetermined part is the mobs. | Changing factors determined (all 64 replays agree), with each tile's block and mobs apart: terrain, ores, stations, lava and plants 1.00; items 0.84; vitals 0.99; mobs 0.09. With one factor per tile, terrain had read 0.66, because mobs move onto terrain tiles: 2 to 6.5 tiles per frame, more with a mob nearby, independent of the action, never a whole view shift. |
| F9 | At `k = 8` the determined part shrinks well beyond the mobs. | Determined: terrain 0.68, ores 0.69, lava 0.77, stations 0.56, items 0.69, vitals 0.71, mobs 0.006. In egocentric coordinates, an uncertain player position makes the blocks uncertain too. |
| F10 | The geometry readout barely separates the models. | Spearman (distance, steps to go): init 0.71, E0 0.79, E1/E2 0.72 to 0.74. |
| F11 | AAC's A4 result is weak evidence. | One Mac pilot (MLP encoder, BatchNorm projector, lr 1e-3, one epoch), deleted, never repeated on a GPU; the A3/A4 race never ran. |
| F12 | The readout (fresh decoder, 2000 steps) is reliable for tiles, mobs, the resource items and the vitals except health. It is not reliable for the tools, item_diamond, health and light. | Same protocol on the raw `state`: rare 0.99, mobs 1.00, blocks 0.94, coal/iron/diamond 0.97 to 0.98, food/drink/energy 1.00. But tools and item_diamond 0.58 to 0.64, health 0.67, light 0.41, ripe_plant 0.33. E0 scores above this "ceiling" on the tools (0.66 to 0.77), so on those classes the scores are noise. |
| F13 | **The readout depends on the latent's scale and shape, not only on its content.** Whitening z (invertible, so the information stays the same) changes the scores. | Linear probe on E3 at step 16000, rare: 0.56 raw, 0.82 whitened. E0: 0.73 → 0.85. Init: 0.35 → 0.54. Every v1 number from a stretched latent understates it. |
| F14 | E3, stage 1 (to step 10000), is the best latent so far, but it does not select. | Whitened probe: rare 0.87, blocks 0.78, mobs 0.96, items 0.81, vitals 0.76 (E0 at 12000: 0.85 / 0.77 / 0.95 / 0.77 / 0.79). Latent error over copying 0.73 (h1) and 0.61 (h8). It keeps 0.65 of the determined factors and 0.69 of the undetermined ones (v1). |
| F15 | E3, stage 2 (to 16000), keeps nearly everything but blows the latent up along one direction. | Whitened probe: rare 0.82, blocks 0.75, mobs 0.90, items 0.81, vitals 0.75; the mobs lose most, which is the intended direction. z_std 0.25 → 5.3; participation ratio 23.6 → 1.7; one direction holds 0.77 of the variance; the 99th percentile of \|z\| is 61. The v1 "collapse" (fresh rare 0.93 → 0.71) was mostly this (F13). Stage 2 has no term on the scale. |

**What this means for E1 and E2:** they do not test the A3/A4 question (prediction pruning through the target path). Their encoder is shaped by SIGReg, which pushes `z` towards a Gaussian blob without being asked for any content, and does not even reach it. The prediction term's natural scale is tiny: a step moves `z` by 0.014 of its variance, so a residual predictor starts near copying. The weight 0.0225 (AAC's 0.09 at batch 128, scaled by 1/batch) does not balance this. Whether a predictive loss *with an isotropic, full-rank latent* prunes the rare content is still open.

## Measurement: readouts v2 (2026-11-20)

Built and tested on the Mac; the oracle tables are regenerated. Run by `notebooks/Q1.ipynb` on the existing checkpoints.

1. **z is whitened before every readout** (F13).
2. **Each tile is two factors,** its block and its mobs (F8). The oracle and kept-against-target use 145 factors.
3. **Kept against the target D_k ∪ G_k:** in-target kept / outside-target kept, per kind and class. Outside the target are mostly the mob factors.
4. **The summaries leave out the classes the readout cannot read** (`layout.LOW_CEILING`, F12). The tables show them, marked.
5. **The latent:** participation ratio, rank, median slowness, per checkpoint.
6. **Training logs the gradient balance:** the encoder gradient norms of the prediction term and of the rest, and their ratio (F3).
7. **Still open: geometry.** It barely separates the models (F10). Keep it until planning gives a better measure.

## Open questions, in order

| # | Question | Hypotheses | Cheapest separating check |
|---|---|---|---|
| Q1 | What do E0 to E3 keep against D_1 ∪ G_1, read correctly? Does E3's stage 2 drop only the mobs? | stage 2 drops the undetermined (mobs) and keeps the target / it also drops goal factors / no selection beyond stage 1 | `notebooks/Q1.ipynb`: readouts v2 on the checkpoints (no training, about an hour on a pod) |
| Q2 | Does a predictive loss with a working anti-collapse term prune rare content? | SFA: it keeps the slowest directions, and rare blocks are fast in the egocentric view (change rate 0.3 to 0.4) / at full rank it keeps them | first the SIGReg weight at which the gradients balance and the statistic reaches about 1: `notebooks/Q2.ipynb` (six weights x 2000 steps, rule fixed in the notebook, about 35 minutes); then E1 against E2 at that weight |
| Q3 | Does E3's stage 2 need a term on the latent's scale? | the blow-up is harmless (the information stays, F15) / it ruins planning, because the latent distance then sees one direction | after Q1: the geometry and the planning cost on E3 at 10000 against 16000 |
| Q4 | Is 256 x width 128 enough to hold everything? | yes: E0 and E3's stage 1 just need longer / no | E0 to convergence (GPU); E3 at 10000 already reaches whitened rare 0.87 |
| Q5 | Egocentric or not at `k = 8`? | the level-2 latent should hold world content independent of the exact position / position uncertainty is part of what is undetermined | the oracle at k = 8 with the player's displacement factored out (Mac) |

## Literature (from memory, to be checked before citing)

| Line of work | Claim | Bearing here |
|---|---|---|
| Slow feature analysis (Wiskott & Sejnowski 2002) | Minimising `E‖z_{t+1} − z_t‖²` under whitening yields the slowest directions. | Prediction plus isotropy approaches SFA; it predicts that fast egocentric content is dropped (Q2). |
| Implicit bias of JEPA vs MAE (Littwin et al. 2024) | JEPA learns influential, predictable features first; MAE learns high-variance ones. | E0 versus E1/E2, once Q2's balance is right. |
| Dimensional collapse (Jing et al. 2022) | The embedding can collapse to a subspace while the loss looks fine. | F4. |
| LeJEPA (Balestriero & LeCun 2025) | SIGReg alone suffices against collapse, with the right weight. | Our weight is off (F3, F4). |
| Self-supervision without negatives (Tian, Chen & Ganguli 2021) | Stop-gradient and a predictor avoid collapse through the training dynamics. | Why the target path matters only when the prediction term matters (F2, F3). |
| Past-future information bottleneck (Creutzig, Globerson & Tishby 2009); causal states (Shalizi & Crutchfield 2001) | The minimal sufficient statistic of the past for the future; in the linear-Gaussian case, CCA. | Criterion (b) "informative", which we did not choose; it explains AAC's lagged-CCA basis. |
| Multi-step inverse models (Efroni et al. 2022; Lamb et al. 2022, AC-State) | Recover the controllable state and discard exogenous noise. | A candidate route to G_k without labels; AAC found it does not ask for ores. |
| Denoised MDPs (Wang et al. 2022); bisimulation (Zhang et al. 2021) | Keep what is controllable and relevant to the reward. | The goal plays the reward's role (G_k). |
| Options and multi-time models (Sutton, Precup & Singh 1999) | An option model predicts the distribution of the end state. | The level-2 step with `m` as the option. |
| Value equivalence (Grimm et al. 2020) | A model only needs to be right where planning needs it. | Planning success is the final measure. |
