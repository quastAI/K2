# What AAC found that matters here

AAC (`../AAC`, frozen) trained flat JEPA world models on Craftax-Classic pixels. Its CLAUDE.md has the dated record. This page keeps only what bears on K2. One seed throughout; the thresholds were heuristics.

## Pruning

- **The target path prunes early.** The Mac pilot before A3/A4 used an MLP encoder on the `state`, prediction plus SIGReg, with the gradient through the target. Its latent lost the mobs, the ores and the neighbouring tiles within about 100 steps. With the target detached (A4) it kept them and recovered. The pixel model B1 had lost the same things before its first checkpoint (epoch 1); after that, the prediction loss put little pressure on the latent.
- **B1's latent was a slow-feature latent.** SIGReg took its variance from slow, predictable quantities (inventory, vitals, the view's composition). Nothing demanded the rest of the view.
- **Expensive and rare things were both lost.** In that data the two are confounded.

## Reading what a latent holds

- **Probes undersell rare classes badly.** V4 stage 1 at epoch 8 drew iron at 0.86 against a ceiling of 0.93 (decoded-tile identity), while an MLP probe scored iron at 0.09. Use a decoder (fresh, class-balanced) as the readout, not a probe.
- **Saplings (plants)** were lost in every pixel model (0.33 against 1.00).

## Gating (the Sieve, V4 and V5)

- **The absolute gate pruned for the wrong reason.** It compared the decoded prediction with the pixels and shut every patch the decoder could not draw within τ, even where the prediction was right. Rare things were the casualties (`own_ok` about `sharp`: iron 0.03/0.03, cow 0.03/0.01). V4 stage 2 drew iron as another tile (0.86 → 0.00) and kept coal (0.92). In state space, decoding is exact, so this confound disappears.
- **The relative gate has no threshold.** V5 compared the decoded prediction with the decoded target (`diff`), as `G = floor + (1 - floor) exp(-g / (β mean g))` with floor 0.1 and β 1. The error was heavy-tailed: the worst 1% of patches carried 60% of the total, so the batch mean closes only clear outliers (mean gate 0.84). This was kept on purpose.
- **V5 trailed V4 stage 1 at epoch 3** (probes: coal 0.04 against 0.27, rank 17 against 45). Its suspects were the missing error weights and a decoded prediction active from step 0. It was not resolved.

## Geometry

- **Not axis-aligned.** Factors are linearly readable but spread over 64 to 128 standard axes. PCA concentrates them (median k90 of 32 components). The lag-1 canonical basis of `(z_t, z_{t+1})` concentrates them best (median k90 of 14). A gate per dimension would cut across factors.
- **Facing, mobs, furnace and iron sit in the low-variance tail;** daylight and tools sit on the top PCs. A raw MSE latent cost therefore sees mostly the slow subspace.
- **The predictor was worse than copying along slow directions.** This is the case for a residual predictor.
- **A predictability-only criterion would prune goal events.** Inventory and vitals changes were not foreseen better than copying and sat 15% to 70% on copy-level directions (diamond 38%). Safer: predictable OR needed by the actions or goals.

## What actions and distances need (analysis on the state, `bc.py`)

- **Behaviour cloning** (the next 4, 8 and 16 actions) needs the four neighbouring tiles, facing, zombies and the vitals. It needs the near field less the longer the chunk.
- **Multi-step inverse dynamics** (1, 4 and 8 steps) needs the near field less and the outer view more as k grows.
- **Neither task asks for ores, diamond, lava, cows or skeletons** (importance 0.02 or below). The action-based terms will not demand the rare things.

## Other

- **LeVJEPA's augmentations do not fit Craftax.** Crops, flips, colour jitter and grayscale change the game's semantics, and 95% token dropping is too much on 81 patches. The only fitting idea was resampling the night static, and the symbolic state has none.
- **Infrastructure (RunPod):**
  - Keep the venv, the data and the logs on the local disk, not the network volume.
  - Strip `LD_LIBRARY_PATH` for JAX.
  - Limit torch threads to the container's CPU quota.
  - Wait 10 to 15 minutes and `sync` before stopping a pod.
