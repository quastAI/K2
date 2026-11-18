# K2: design note

2026-11-18. A head start, not a plan of record: every choice marked *open* is still to be settled, and nothing here has run.

**Decided on 2026-11-18:**
- **Two sides with a file boundary.**
  - **Simulator side:** JAX on the CPU, run on the Mac. It covers the data cache, the replay test and the oracle table, and writes `.npz` files.
  - **Model side:** PyTorch, trained on a RunPod GPU. It covers the encoder, predictor, losses, training and readouts.
  - **The executor** joins them in one process through numpy arrays.
- **`frame` mode first.** `window` mode, and with it goals in window mode, comes later.
- **Thresholds come from the data,** not fixed in advance (see Readouts).
- **The work continues AAC's open questions as targeted experiments** (see "Order of work").

## The question

Can a learned level-2 latent

1. keep what is predictable at its stride given its macro-action (and what the actions and goals need),
2. predict it well,
3. and plan well with it?

The question is the same one the Sieve (AAC, V4/V5) asked of pixels at level 1. Here it is asked where it matters for the hierarchy, one level up, with everything below made perfect:

- **Perception is assumed solved.** The input is the observation `state` (1345-d, one frame), a stand-in for a perfect level-1 latent.
- **Control is assumed solved.** Level 1 is CEM in the JAX simulator, which executes level-2 subgoals.

Moving to pixel latents is a separate, later step ("Transfer").

## Rules

- **The model is unsupervised.** Its losses see only its input (the `state`, as pixels would be) and the actions. Reconstructing the input is allowed. A weight computed from the input's own statistics is allowed and is named as such. Weights or targets chosen by hand per factor are not.
- **Evaluation-only material:** `full_state`, the replay oracle and the probes. None of them reaches a loss.
- **Planning never sees the true future randomness.** Rollouts in the simulator use fresh keys, never the key the real environment will use.
- **No minutes-long run starts without an explicit go-ahead.**

## Input

The observation `state`, which is what one frame shows:

| Part | Size | Values |
|---|---|---|
| view: 7 x 9 tiles x (17 block one-hot + 4 mob bits) | 1323 | 0/1 |
| inventory (12 items) | 12 | count / 10 (counts 0 to 9) |
| vitals (health, food, drink, energy) | 4 | value / 10 |
| facing | 4 | one-hot |
| light level | 1 | 0 to 1 |
| sleeping | 1 | 0/1 |

- **No night static.** Light is a scalar here, so the only random part of a pixel frame is absent.
- **Not `full_state`.** It holds the global map and position, which the agent never sees.
- **Imbalance:** the view has 1323 dimensions against 22 for everything else. The token encoder below removes it; in a flat MLP or a flat loss the view dominates about 60 to 1.

## The level-2 step

Stride `k`. Boundary frames are at `t_j = t_0 + j*k`. The macro-action `m_j = (a_{t_j}, ..., a_{t_j + k - 1})` is the actions that lead from boundary j to boundary j+1.

Two input modes (setting `window`):

- **`frame`:** `x_j = s_{t_j}`. One frame per level-2 step (t, t+8, t+16 at k = 8). The frames in between are never encoded.
- **`window`:** `x_j = (s_{t_j - k + 1}, ..., s_{t_j})`, the k frames *ending* at the boundary. These are the outcome of the previous macro-action `m_{j-1}`.

The window ends at the boundary on purpose. A window that started there would contain the effects of `m_j`, the macro-action the predictor is conditioned on, so the target would leak into the input. With windows ending at the boundary, both modes have the same alignment: predict `x_{j+1}` from `x_j` and `m_j`.

```
frames   ... s_{t-7} ... s_t | s_{t+1} ... s_{t+8} | s_{t+9} ... s_{t+16}
frame       x_j = s_t        |     x_{j+1} = s_{t+8}  |  x_{j+2} = s_{t+16}
window   x_j = s_{t-7..t}    | x_{j+1} = s_{t+1..t+8} | x_{j+2} = s_{t+9..t+16}
actions                 m_j = a_t..a_{t+7}  m_{j+1} = a_{t+8}..a_{t+15}
```

**Context `C`:** the predictor sees `(z_{j-C+1..j}, m_{j-C+1..j})`. The encoder never sees `C`.

**Ladder:** `k = 1` in `frame` mode is level 1 on the state, the setting of AAC's A3/A4 pilot, where the pruning showed up within about 100 steps. That makes it the first case. Then `k` = 2, 4, 8, 16, each in both modes. Start with `C = 1` (the observation is nearly Markov at one step), and use `C = 3` as the first variation.

**Goals in `window` mode** are the window ending at the goal frame. The expert episodes provide one. A goal given as a single state would need `frame` mode or a padded window (*open*).

## Macro-actions

- **Stage A, exact chunks.** The k one-hot actions are embedded by an MLP. This isolates the selection problem from action learning. A level-2 planner over exact chunks is only a coarse flat planner (17^k choices), so stage A is the scaffold for the latent question, not yet the hierarchy.
- **Stage B, latent macro-actions.** A VQ code (AAC's convention: groups x codes) from an inverse-dynamics encoder `q(m | x_j, x_{j+1})` or from the chunk. Level 2 plans over codes, and level 1 grounds them. Starts once stage A's latent works.
- **Stage C, the stochastic part.** Some outcomes are not set by the macro-action: mob moves, spawns, combat, growth. A sampled latent variable with a prior at test time covers these (the user's plan of mob positions as a stochastic input). Designed later.

## Encoder

**A shared frame tokenizer** (width 128, about 82 tokens):

- **63 tile tokens:** embed(block type) + sum of embed(mob type) over the mobs present + embed(row, col). The positions are relative to the player, who is always at the centre.
- **12 item tokens:** embed(item) + embed(count, categorical 0 to 9).
- **4 vital tokens:** embed(vital) + embed(value, 0 to 9).
- **One token each** for facing, light (embed + linear in the value) and sleeping.

**Frame encoder:** about 3 transformer layers, then cross-attention pooling with `Q` learned queries into the latent (start: Q = 4 x 64 = 256-d), then a linear projector.

**Window mode is factorised.** Each frame is pooled to P tokens by the frame encoder. A temporal transformer (time embedding 1..k) then attends over the k x P tokens and pools them into the latent. `frame` mode is the same code path with a window of one frame.

**Baseline encoder:** A3's flat MLP (1345 → 1024 → 256), used in `frame` mode only. The pilot results came from it.

Why tokens:

- **One architecture for every level.** "Tokens in, one latent out" fits state tiles now, pixel patch tokens later and level-1 latents one level up. The transfer check then swaps the token source and keeps the rest.
- **Equal standing.** Each item and vital gets a token of its own instead of being outweighed by the view.
- **The egocentric shift.** A move shifts all 63 tiles. Position embeddings plus attention follow content, not slots.

**The final layer is *open*.** No final LayerNorm in the SIGReg variants: LeWorldModel keeps the projector because a final LayerNorm stops SIGReg from being optimised.

## Predictor

- **Architecture:** a causal transformer over the C context steps of `(z, m)`.
- **Output:** residual, `ẑ_{j+1} = z_j + f(...)`. In AAC the predictor was worse than copying along the slow directions, so the residual form is the default and a direct output is the variant.
- **Rollouts:** feed back `ẑ`.

## Variants

All variants use the next-latent loss `||ẑ_{j+1} - sg(z_{j+1})||²`, with the target detached (in A4 the detached target kept the content). The one exception is E1, the control. Every variant also needs an anti-collapse term: SIGReg or reconstruction.

| Id | Loss | Role |
|---|---|---|
| E0 | autoencoder only: `D(z_j) → x_j` | **capacity check**: can this encoder hold every factor, rare ones included? |
| E1 | prediction, gradient through the target, + SIGReg | the pruning reference (A3 analogue, B1-like) |
| E2 | E1 with a detached target | A4 analogue |
| E3 | detached prediction + reconstruction of `x_j` + decoded prediction `D(ẑ_{j+1}) → x_{j+1}` | Sieve stage 1 analogue: keep everything |
| E4 | E3 with the reconstruction gated per token | V5 analogue: keep what is predictable |
| E5 | E4 with the gate on latent directions (lagged-CCA basis) instead of tokens | later |

**Decoder:**
- **Per token:** cross-entropy over block types, binary cross-entropy per mob bit, cross-entropy over counts and vital values, and cross-entropy for facing. Light uses MSE and sleeping uses BCE.
- **In window mode:** it reconstructs every frame of the window.
- **No precision threshold.** The state is decoded exactly, so the τ problem of V4 does not exist here. That is one less confound.

**The gate (E4):** `G = floor + (1 - floor) * exp(-g / (β * mean g))`.
- `g` is the per-token divergence between `D(ẑ_{j+1})` and `D(sg(z_{j+1}))`: V5's `diff` gate with tiles in place of patches.
- Floor 0.1 and β 1 come from V5. The batch mean was kept on purpose.
- Tokens are the factor grid (tile, item, vital), so the gate cuts along factors, which pixel patches never did.

**An *open* extension.** In AAC, gating on predictability alone would prune goal events: inventory and vitals changes were not foreseen better than copying. Candidates:
- a floor per token type (a structural choice, not a label);
- an inverse-dynamics term on the macro-action, so that what the actions need is demanded;
- keeping changes that the macro-action explains.
The oracle below shows which of these the data asks for.

**Training:** Adam, constant learning rate after a warmup, batch 256. PyTorch on a RunPod GPU, after a CPU smoke test on the Mac.

## Readouts

### 1. What the latent keeps

- **Fresh decoder:** for every checkpoint, a decoder `D'` trained on frozen `z` (train split) and scored on held-out worlds per class.
- **Rare classes:** coal, iron, diamond, lava, table, furnace, sapling, and the four mobs.
- **Scores:** recall and balanced accuracy, with a class-balanced loss. AAC showed that MLP probes undersell rare classes; a decoder of the full state is the readout.
- **A linear probe as well**, for the geometry.

### 2. What is predictable: the oracle

1. Sample `(s_{t_j}, m_j)` from the held-out worlds and restore `full_state` (an unpack function rewritten in K2 from AAC's `unpack_state`).
2. Replay the k steps with N = 32 fresh step keys through `craftax_step`, without an auto-reset.
3. Per factor, score `p_f`: the share of replays that match the most common outcome (entropy as an alternative).
4. A factor is *predictable at stride k given m* if `p_f` is above a cut found from the data. Report the full curve per factor: deterministic factors sit at 1.0, and the cut goes in the gap below them, not at a value fixed in advance.

What the oracle gives:
- **The target map** for every variant, per stride, before any model exists. The predictable set shrinks as k grows; how fast is the first result.
- **In window mode** it covers every frame of the window.
- **Coarser macro-actions (stage B)** condition on less than the exact chunk. The predictable set then needs chunks sampled within a code (later).
- **One step in the simulator is fully determined by `(state, action, step key)`.** The randomness lives only in the key (`state_rng` only feeds the renderer's night static), so key resampling is the whole of the stochasticity.

### 3. Kept against predictable

Per factor, a 2 x 2 table:
- kept and predictable: good;
- dropped but predictable: over-pruning;
- kept but unpredictable: noise;
- dropped and unpredictable: good.

**Headlines:** the share of the predictable factors kept, and the share of the unpredictable factors kept. Both are reported per class, so that rare classes are not drowned out.

### 4. Prediction

Per factor, the error of the decoded prediction relative to copying. Rollouts over H level-2 steps.

### 5. Geometry for planning

- latent distance against steps along expert paths (rank correlation);
- smoothness along a path.

## Planning

**Tasks** come from AAC's expert episodes (`source` 0): validation worlds for tuning, test worlds for the headline. A task starts at `s_t` with the goal at frame `t + H*k` (e.g. H = 8, k = 8: offset 64), and later the diamond. The goal test is AAC's `new_achievements`, plus the decoded-state distance (*open*).

**Level 2** is CEM over macro-actions in the latent: categorical over `k x 17` for exact chunks, over codes in stage B. The cost is `||ẑ_H - z_goal||`, receding: carry out one level-2 step, then plan again.

**Level 1, the executor**, is CEM in the simulator:
1. Restore the true current state.
2. Sample action sequences of length k.
3. Roll each out with several fresh keys, to average over the randomness.
4. Score `||enc(x^{sim}_{j+1}) - ẑ_{j+1}||`; the encoder reads simulated states directly.
5. Carry out the best sequence in the real environment with its true key.

With exact chunks the executor could just replay the chunk; replanning per level-2 step is what tests the latent.

**Controls:**
- **A random policy.**
- **Oracle subgoals:** the expert's frames every k steps, encoded, given to the executor. This is the executor's ceiling, and it tests the assumption of "90% of subgoals reached" at k = 8 directly.
- **A flat simulator CEM** toward the goal with the same budget. It uses the same encoder distance, and a fixed distance on the state (a Hamming distance per token) as the non-learned reference.

**Cost:** the simulator's steps per second without rendering, on CPU and GPU (a seconds-long check).

## Transfer (later)

1. **Noise on the state input** that copies the failures of AAC's V4 decoder: saplings missing, mobs faded or dropped, tiles shifted by one, worse at night. Then measure how the kept-against-predictable table changes.
2. **Swap the tokenizer** for frozen pixel patch tokens or a frozen level-1 latent (e.g. V4 stage-1 epoch 8). The predictor, gate and readouts stay the same.

## Order of work

These are targeted experiments that continue where AAC stopped, in `frame` mode only. Each answers one open question against its control. Steps 0 and 1 run on the Mac and the trainings on a pod. Every run waits for an explicit go-ahead.

0. **Data and correctness (Mac, JAX).**
   - **The cache:** `state` in compact form (block id and mob bits per tile plus scalars, about 126 bytes per frame; about 250 MB for the 2M training frames), `action`, and the episode boundaries. `full_state` is kept for the held-out sample only. The cache is built once and uploaded to the pods.
   - **A guard test:** replay with the true keys reproduces the stored next states exactly (AAC's replay contract).
1. **The oracle table at k = 1 and k = 8:** what is predictable at all. AAC never had this ground truth. It needs no model, and every later result is read against it.
2. **E0, the capacity control.** Its per-class scores are the reference for every variant; there is no fixed pass mark.
3. **E1 against E2 at k = 1:** the A3/A4 race (gradient through the target against a detached target). It only ever ran as AAC's Mac pilot.
4. **E3 against E4 at k = 1:** the Sieve's stage 1 (keep everything) against V5's relative gate, without the pixel and τ confounds. Does the gate keep the predictable content and drop the unpredictable? This was V5's open question.
5. **The same at k = 8 in `frame` mode,** then the executor check and level-2 planning.

Later, after these: `window` mode, latent macro-actions (VQ), the stochastic latent, E5, transfer.

## Compute (estimates, to be measured)

- **Mac (M1 Max, 32 GB), JAX on the CPU:**
  - the oracle table: about 2M simulator steps, seconds to a minute vmapped;
  - the executor CEM over 128 tasks: minutes.
  - Training on the Mac CPU would be too slow (the width-128 transformer takes about 1 s per step), so the Mac only runs smoke tests of the torch code.
- **RunPod GPU (e.g. A40), PyTorch:**
  - the width-128, 3-layer tile transformer at batch 256: about 10 to 20 ms per step;
  - one epoch at k = 1 (about 7.8k steps): 2 to 3 minutes;
  - 10 epochs: about 30 minutes.
  - The data cache sits in RAM, so the loader is not expected to be the limit.
- **How many steps:** the AAC pilot showed losses within about 100 steps and recovery within one epoch, so the targeted questions should show within the first epoch.

## Open decisions

- **The criterion for what actions and goals need** (see Variants). Decided after the oracle table and E3/E4 show what is lost.
- **The encoder size:** start at width 128 with 3 layers. E0 shows whether it holds the rare classes.
- **Training data:** all sources or the expert only. At level 1 AAC trained on all sources.
- **Latent size** and the number of queries `Q`; the final normalisation.
- **Goals in window mode;** the goal test for level-2 tasks.
