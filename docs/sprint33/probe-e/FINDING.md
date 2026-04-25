# PROBE-E Finding — partition-state class CONFIRMED, mechanism is degenerate-child skip

**Date**: 2026-04-24
**Anchor**: `np.random.default_rng(42)`, N=50000, 20 features,
y = 0.5·X[0] + 0.3·X[1] + 0.1·noise, ST/Cosine/RMSE, depth=6, bins=128, l2=3, lr=0.03
**Build**: `csv_train_probe_e` with `-DCOSINE_RESIDUAL_INSTRUMENT
-DPROBE_E_INSTRUMENT -DPROBE_D_ARM_AT_ITER=1`
**Kernel md5**: `9edaef45b99b9db3e2717da93800e76f` (unchanged)

## Why this probe

PROBE-D (FINDING.md) closed the precision class — fp32 vs fp64 max gain
residual at d=2 = 3.89e-5, three orders smaller than the 20+ unit gain gap
between MLX's pick and CPU's pick. That same finding identified a
candidate mechanism: at iter=2 d=2, CPU's pick (feat=0, bin=21) is a
constrained signal-feature split that produces empty-child partitions in 2
of 4 leaves after d=0/d=1 partitioning, while MLX's pick (feat=10, bin=79)
is a noise-feature split with no degenerate children.

PROBE-E adds per-(feat, bin, partition) capture of the joint cosNum/cosDen
contribution under both rules — MLX's "skip when one child has zero weight"
(`continue` at `csv_train.cpp:1980`) and CPU's "mask each side
independently and add zero from the masked side" (SSE2 path at
`short_vector_ops.h:155+`, scalar fallback at the same file). The capture
fires inside FindBestSplit BEFORE the existing skip-continue, so MLX's
emitted gain values are unchanged and the CPU rule is computed
counter-factually.

## Result — partition-state class CONFIRMED

### Hypothesis anchors at d=2

| candidate | role | parts | skips | MLX gain | CPU gain | Δ |
|---|---|---|---|---|---|---|
| (feat=0, bin=21)  | signal × constrained (CPU's pick) | 4 | **2** | 81.83  | 108.32 | **+26.49** |
| (feat=10, bin=79) | noise (MLX's pick)                | 4 | 0     | 101.79 | 101.79 | 0.000  |

### Per-partition smoking gun — (feat=0, bin=21)

| part | sumL | sumR | wL | wR | skip | mlxTermN | mlxTermD | cpuTermN | cpuTermD |
|---|---|---|---|---|---|---|---|---|---|
| 0 |  4290.19 |  3568.35 | 4489 |  7854 | no  | 5718.05 | 5714.69 | 5718.05 | 5714.69 |
| 1 |     0.00 | -1949.57 |    **0** | 13828 | **YES** |    0.00 |    0.00 |  274.81 |  274.75 |
| 2 |  1997.71 |  -119.11 | 4104 |  6990 | no  |  973.74 |  973.03 |  973.74 |  973.03 |
| 3 |    -0.02 | -7787.80 |    **0** | 12735 | **YES** |    0.00 |    0.00 | 4761.33 | 4760.21 |

Sum across 4 partitions:
- MLX: cosNum = 6691.79, cosDen = 6687.72 → gain = 6691.79 / √6687.72 = **81.83**
- CPU: cosNum = 11727.93, cosDen = 11722.68 → gain = 11727.93 / √11722.68 = **108.32**

The 2 skipped partitions contribute exactly Δnum=5036.14, Δden=5034.96 to
CPU's sum but zero to MLX's. That delta is the entire 26.49 gain unit gap.

### Per-partition control — (feat=10, bin=79)

All 4 partitions have non-zero weight on both sides; MLX and CPU
contributions are bit-identical per partition; both gains = 101.79.

### Top-5 by CPU gain at d=2 (signal feature elevated)

| feat | bin | role | CPU gain | MLX gain | skips |
|---|---|---|---|---|---|
| **0** | 105 | **signal** | **109.51** |  81.67 | **2/4** |
| **0** | 104 | **signal** | **109.50** |  81.67 | **2/4** |
| **0** | 106 | **signal** | **109.49** |  81.65 | **2/4** |
| **0** | 107 | **signal** | **109.49** |  81.65 | **2/4** |
| **0** | 108 | **signal** | **109.47** |  81.63 | **2/4** |

Compare to top-5 by MLX gain at d=2 — 5 of 5 are noise (feat=12 bins
75–82, all gain ≈ 101.79). **Under CPU's rule, the signal feature outranks
noise; under MLX's rule, every signal-feature candidate scores 26 units
below the noise floor and loses.**

### Coverage of degenerate splits

- d=0: 0/2540 partitions skipped
- d=1: 128/5080 (2.5%)
- d=2: 512/10160 (5.0%)
- d=3: 1553/20320 (7.6%)
- d=4: 4289/40640 (10.6%)
- d=5: 11872/81280 (14.6%)

Skip rate grows monotonically with depth — exactly the structural pattern
expected when more partitions are constrained by upstream splits, making
degenerate child-splits more common per candidate.

At d=2, **every one of the 127 non-trivial bins on feat=0 has skips=2** —
because the d=0 split on feat=0 fixed every doc's relationship to feat=0's
range, leaving any further feat=0 split candidate degenerate in exactly
the 2 leaves where d=0 partitioned all docs to one side.

## Mechanism (fully specified)

In `catboost/mlx/tests/csv_train.cpp:1980` (FindBestSplit, partition loop
inside the per-bin sweep):

```cpp
if (weightLeft < 1e-15f || weightRight < 1e-15f) continue;
```

This `continue` skips the entire partition's contribution to both
`cosNum` and `cosDen`. The reducer's joint-denominator sum is therefore
**under-attributed** for any candidate whose split is degenerate in some
partitions.

CPU's reference path (`catboost/libs/helpers/short_vector_ops.h:155+`,
SSE2 `UpdateScoreBinKernelPlain`) computes `average = mask·sum/(w+λ)`
where the mask zeros the average when `w ≤ 0`. The result:

- non-degenerate side contributes: `sumX² / (wX + λ)` to num,
  `sumX² · wX / (wX + λ)²` to den
- degenerate side contributes 0 to both

Both sides are added. The non-degenerate side's contribution is **not
discarded** — only the empty side is masked out.

## What this means for DEC-036

- DEC-036's 52.6% drift on ST+Cosine at N=50k is the cumulative effect of
  this skip — every iteration whose tree reaches d ≥ 2 with any signal
  feature already split has 5–15% of partitions silently dropped from the
  joint sum, and every depth where one child is degenerate (which is
  topology-driven, not noise-driven) systematically deflates gains
  for signal-correlated split candidates.
- The drift is **structural, partition-state class**, not precision
  class (PROBE-D), not quantization class (DEC-041 INVALIDATED), not
  histogram-fidelity class (S33-L4 PRODUCTION-BROKEN, retracted).
- Fix is plausibly local: **either** (a) replace the skip-continue with a
  per-side mask in FindBestSplit's per-partition update, **or** (b) at
  the score reducer, accumulate `sumX²/(wX+λ)` from the non-empty side
  instead of dropping the whole partition. The CPU formula is the
  reference; MLX needs to match it.

## What this RULES IN as the next step

**S33-L4-FIX (task #123, REOPENED)**: implement per-side mask in
FindBestSplit's per-partition update, validate parity on the same
50k/Cosine/RMSE/d6 anchor, then run the formal four-gate sweep.

Predicted result on the anchor:
- MLX d=2 pick flips from (feat=10, bin=79) to (feat=0, bin ≈ 102–108)
- iter=2 tree[1] d=2 split matches CPU
- DEC-036 drift collapses below the 1.07× R8 threshold

If the fix is bigger than expected (e.g. requires reducer-level join with
empty-leaf handling), it splits cleanly into a separate task.

## Artifacts

- `data/cos_leaf_seed42_depth{0..5}.csv` — per-(feat, bin, partition) leaf
  records with MLX's actual contribution and CPU's counterfactual
  contribution; 14 columns including `mlx_skipped`, `mlx_termN/D`,
  `cpu_termN/D`, `term_num_diff`, `term_den_diff`. Row counts:
  2540 / 5080 / 10160 / 20320 / 40640 / 81280.
- `data/cos_accum_seed42_depth{0..5}.csv` — PROBE-D-style per-bin gain
  shadow; the PROBE-E binary regenerates these alongside leaf records
  since `-DCOSINE_RESIDUAL_INSTRUMENT` is still on. Values match
  PROBE-D's run-to-run (float-summation noise; max delta < 1.5e-3 abs
  cosNum, max delta < 4e-3 abs gain).
- `data/leaf_sum_seed42.csv`, `data/approx_update_seed42.csv` — auxiliary
  per-iter dumps (unchanged behavior from PROBE-D).
- `data/mlx_anchor_iter2.json` — not regenerated (PROBE-D's copy at
  `docs/sprint33/probe-d/data/mlx_anchor_iter2.json` is canonical for
  iter=2 model state).
- `scripts/build_probe_e.sh` — build script.

## Build invariants

- `kernel_sources.h` md5 = `9edaef45b99b9db3e2717da93800e76f` (unchanged
  from PROBE-D and from S33 sprint-open).
- All instrumentation gated under `#ifdef PROBE_E_INSTRUMENT`. Production
  builds (no flag) compile to bit-identical machine code as before.
- Capture fires only when `g_cosInstr.dumpCosDen && Cosine && K==1` —
  same arming as PROBE-D's per-bin shadow, no new arming logic.

## Reproducibility

```bash
bash docs/sprint33/probe-e/scripts/build_probe_e.sh
COSINE_RESIDUAL_OUTDIR=docs/sprint33/probe-e/data \
  DYLD_LIBRARY_PATH=/opt/homebrew/opt/mlx/lib \
  ./csv_train_probe_e docs/sprint33/probe-c-borders/data/anchor.csv \
  --iterations 2 --depth 6 --bins 128 --l2 3 --lr 0.03 --seed 42 \
  --loss RMSE --score-function Cosine --grow-policy SymmetricTree
```

Anchor.csv md5: `9137a36d991e7b620a07b2fb1d49ed0d`, 50000 docs.
