# PROBE-D Finding — d=2 divergence is partition-state class, not precision class

**Date**: 2026-04-24
**Anchor**: `np.random.default_rng(42)`, N=50000, 20 features,
y = 0.5·X[0] + 0.3·X[1] + 0.1·noise, ST/Cosine/RMSE, depth=6, bins=128, l2=3, lr=0.03
**Build**: `csv_train_probe_d` with `-DCOSINE_RESIDUAL_INSTRUMENT -DPROBE_D_ARM_AT_ITER=1`
**Kernel md5**: `9edaef45b99b9db3e2717da93800e76f` (unchanged)

## Why this probe

PROBE-C (FINDING.md) narrowed iter=2 tree[1] divergence to depth=2: CPU re-picks
feat=0 (border=-0.946874), MLX picks feat=10 (a noise feature). Stage 3
(STAGE3_FINDING.md) tested the precision-class hypothesis indirectly via an
f10-zeroed retrain and saw winner-vs-runner-up gaps of ~0.002 — argued for
precision-class arguments. PROBE-D directly compares fp32 vs fp64 gain at every
(feature, bin) at iter=2 d=0..5 to settle whether the d=2 winner is precision-
sensitive.

## Result — precision class falsified

| depth | rows | winner (feat,bin) | gain_f32 | 2nd_global g32 | max abs(f32−f64) | argmax(g32)==argmax(g64) |
|---|---|---|---|---|---|---|
| 0 | 2540 | (0, 64)  | 87.1806  | 87.1784  | 9.66e-06 | **yes** |
| 1 | 2540 | (1, 58)  | 101.9421 | 101.8957 | 1.74e-05 | **yes** |
| 2 | 2540 | (10, 79) | 101.9539 | 101.9532 | 2.23e-05 | **yes** |
| 3 | 2540 | (14, 109)| 101.9692 | 101.9688 | 2.50e-05 | **yes** |
| 4 | 2540 | (12, 10) | 101.9890 | 101.9887 | 3.54e-05 | **yes** |
| 5 | 2540 | (13, 101)| 102.0183 | 102.0183 | 3.89e-05 | **yes** |

**fp32 max residual at d=2: 2.23e-5** — three orders of magnitude smaller than
the gain gap of interest (see below). Switching the entire FindBestSplit
accumulation to fp64 would not flip the d=2 argmax. **Precision class is closed.**

## The smoking gun — signal/noise gain inversion at d=2

Per-feature winners at d=2 (MLX evaluation, fp32):

| rank | feat | bin | gain_f32 | role |
|---|---|---|---|---|
| 1  | 10 |  79 | 101.953865 | noise |
| 2  | 19 |  37 | 101.952454 | noise |
| 3  | 12 |  82 | 101.951881 | noise |
| 4  | 16 |  13 | 101.951157 | noise |
| 5  |  3 |  15 | 101.950462 | noise |
| ...| ...| ... |    ...     | (13 more noise features here) |
| 18 | 11 |   1 | 101.946304 | noise |
| **19** |  **0** |  **20** | **81.892723**  | **signal (×0.5)** |
| **20** |  **1** | 106 | **77.767220**  | **signal (×0.3)** |

Compare to d=0 (where MLX and CPU agree, gain bit-exact):

| feat | best gain | role |
|---|---|---|
| 0 | 87.18 | signal |
| 1 | 53.30 | signal |
| all 18 noise | 0.58–2.30 | noise |

**At d=0, signal scores 30–80× higher than noise. At d=2, signal scores LOWER
than noise.** That is the structural inversion. CPU's d=2 pick (feat=0,
border=-0.946874, MLX bin=21) is evaluated by MLX at gain=**81.887** — gap
to MLX's winner (feat=10, gain=101.954) is **20.07 gain units**, ~3 orders
of magnitude bigger than fp32 noise.

## What this rules out

- **Precision class.** fp32 vs fp64 residuals are 1e-5 absolute; the gap is
  20+. No reasonable accumulation-order or fp-widening change can move that.
- **Per-feature single-bin precision drift.** The argmax bin for every
  feature is fp32/fp64-identical at every depth.
- **Cosine-numerator sign cancellation.** Both shadows agree to 4–5 decimals;
  the formula is computing what it claims to compute.

## What this points to — partial-coverage / degenerate-split semantics

CPU's d=2 pick is **feat=0 at border=-0.946874**. After d=0 (feat=0, border=
0.014169) and d=1 (feat=1, border=-0.092413), the docs are partitioned into
4 leaves by (X[0] > 0.014, X[1] > -0.092). The two leaves with X[0] > 0.014
**cannot be split by X[0] < -0.946874** — every doc in those leaves is on
the same side of -0.947, so the candidate produces degenerate (empty/full)
children in 2 of 4 partitions.

The 18 noise features split all 4 partitions roughly in half — no degenerate
children. They cluster within ~0.007 of each other (101.946–101.954),
because the joint denominator is dominated by a partition-global term that
is identical across noise features (residuals haven't changed; noise just
re-shuffles docs).

The mechanism is therefore **how empty/degenerate child partitions enter the
joint cosNum/cosDen sum at depth ≥ 1**, not the floating-point computation
itself. Specifically:

- If MLX's joint-denominator includes empty-child weight (or equivalent)
  but CPU's skips it, MLX's gain for splits with degenerate children is
  artificially deflated → exactly matches the observation that feat=0
  scores 81.89 at d=2 (down from 87.18 at d=0 where it had no degeneracy).
- 4-decimal monotonic trickle up the depths (87.18 → 101.94 → 101.95 →
  101.97 → 101.99 → 102.02) is consistent with each noise feature adding
  a tiny per-leaf cosine term to a numerator that's already saturated by
  the d=0+d=1 signal extraction.

## What this rules in (next probe)

**PROBE-E candidate**: dump per-leaf (cosNum, cosDen, leaf-doc-count, child-
doc-count L/R) for the d=2 candidate (feat=0, bin=21) and compare to a
non-degenerate candidate (feat=10, bin=79) on both MLX and CPU. The
expected finding is that 2 of 4 leaves have one zero-doc child for feat=0,
and that MLX vs CPU differ specifically in the contribution of those
zero-doc children to the joint sums. This is partition-state class —
likely a one-line fix in the score reducer once located.

## Implication for DEC-036

DEC-036's 52.6% drift is **the cumulative effect of degenerate-split
mishandling** at every iteration once any tree reaches depth ≥ 2 with a
re-pickable signal feature. The drift is structural (partition-state),
not precision-related. A fix is plausibly local to the score reducer or
the partition iteration in `score_calcer.cpp` / equivalent MLX kernel
path. DEC-038 (allVals fp64) and DEC-039 (cap-127) are NOT load-bearing
for this drift — they are independent precision items.

## Implication for DEC-041

DEC-041 (static-vs-dynamic quantization border drift) was already
INVALIDATED by the L4-FIX result (S33 path, MEMORY.md `project_sprint33_l4_fix.md`).
This probe further reduces the likelihood of any quantization-grid mechanism
contributing to the d=2 divergence: the MLX 127-grid is a strict subset of
CPU's 128-grid (PROBE-C #1), and CPU's d=2 border (-0.946874) maps to MLX
bin=21 with ULP-identical physical value. Both sides see the same border;
they evaluate it to different gain.

## What this changes for the successor probe

The original C plan (per-bin dump at iter=2 d=0) was already pivoted to
the d=2 dump executed here. The next-step pivot:

- **Original PROBE-D plan**: top-k gain dump at iter=2 d=2 to find
  precision-class disagreements. Done — falsified.
- **Successor PROBE-E plan**: per-leaf cosNum/cosDen with leaf doc counts
  and L/R child doc counts for two specific candidates at d=2:
  (feat=0, bin=21) [degenerate] and (feat=10, bin=79) [non-degenerate].
  Compare to CPU's per-leaf accounting at the same iteration.

## Artifacts

- `data/cos_accum_seed42_depth{0..5}.csv` — fp32/fp64 gain shadow at each
  d=0..5 of iter=2 (1-indexed iter=2), 2540 rows each
- `data/mlx_anchor_iter2.json` — full MLX 2-tree model JSON
- `data/leaf_sum_seed42.csv`, `data/approx_update_seed42.csv` — auxiliary
  per-iteration leaf and approx-update dumps
- `scripts/build_probe_d.sh` — build script
- (no new analysis script — analysis is single-pass Python in this finding)

## Build invariants

- `kernel_sources.h` md5 = `9edaef45b99b9db3e2717da93800e76f` (unchanged).
- `csv_train_probe_d` built with `-DCOSINE_RESIDUAL_INSTRUMENT
  -DPROBE_D_ARM_AT_ITER=1` to arm the cosDen/cosNum double-shadow at iter=1
  (1-indexed iter=2). The instrument bypasses the ST+Cosine guard at
  csv_train.cpp line 595 via the existing `COSINE_RESIDUAL_INSTRUMENT`
  short-circuit — same pattern PROBE-C used.
