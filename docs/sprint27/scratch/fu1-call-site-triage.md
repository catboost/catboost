# S27-FU-1 Call-Site Triage: `ComputeLeafIndicesDepthwise`

**Date**: 2026-04-22
**Branch**: `mlx/sprint-27-correctness-closeout`
**Purpose**: Scope-gate — determine whether `ComputeLeafIndicesDepthwise` bugs affect
the training hot path, the approximation-update path, or validation only.

---

## Section 1 — Call Site Enumeration

### Grep results across the full repo

```
catboost/mlx/tests/csv_train.cpp:1751   — function definition
catboost/mlx/tests/csv_train.cpp:4040   — sole call site
```

No other `.cpp`, `.h`, `.py`, or `.metal` file contains the symbol.

| # | File:Line | Enclosing construct | Calling context |
|---|-----------|---------------------|-----------------|
| 1 | `csv_train.cpp:1751` | `ComputeLeafIndicesDepthwise(...)` | **Definition** — free function |
| 2 | `csv_train.cpp:4040` | `RunTraining(...)` training loop, inside `if (valDocs > 0)` guard | **Validation-only path** |

### Call site in full context (lines 4032–4052)

```cpp
// Apply tree to validation data
if (valDocs > 0) {
    mx::array valLeafIndices = mx::array(0, mx::uint32);
    if (isLossguide) {
        valLeafIndices = ComputeLeafIndicesLossguide(...);
    } else if (isDepthwiseTree) {
        valLeafIndices = ComputeLeafIndicesDepthwise(   // line 4040
            valCompressedData, splits, valDocs, actualTreeDepth);
    } else {
        valLeafIndices = ComputeLeafIndices(...);
    }
    auto valDocLeafValues = mx::take(leafValues, mx::astype(valLeafIndices, mx::int32), 0);
    ...
    valCursor = mx::add(..., valDocLeafValues);
    mx::eval(valCursor);
}
```

The `if (valDocs > 0)` guard makes the entire block dead when no validation set is
provided. `valCursor` feeds only `result.EvalLossHistory` and validation RMSE reporting
downstream — it does not alter `cursor`, `partitions`, `leafValues`, or any object
used by the next training iteration.

---

## Section 2 — Training-Path Determination

**VERDICT: NO — `ComputeLeafIndicesDepthwise` is NOT on the training hot path.**

Evidence: the training-path approximation update (Step 4, lines 3990–4003) uses
`partitions` directly, with no call to `ComputeLeafIndicesDepthwise`:

```cpp
// Step 4: Apply tree to training data
// For depthwise trees, partitions is already the correct leaf index (same bit-encoding
// as oblivious trees — the partition update loop sets bit `depth` based on per-node splits).
auto docLeafValues = mx::take(leafValues, mx::astype(partitions, mx::int32), 0);  // line 3996
```

`partitions` is maintained entirely by the bit-accumulation loop (lines 3660–3683):

```cpp
partitions = mx::bitwise_or(partitions, bits);  // line 3683
mx::eval(partitions);
```

`ComputeLeafIndicesDepthwise` is never referenced anywhere in `GreedyTensorSearch`,
the histogram build loop, the partition update loop, `CalcLeafValues`, or the training
cursor update. Its only call site is behind `if (valDocs > 0)`.

---

## Section 3 — Approximation-Update Path

**VERDICT: NO — the DW approx-update path does NOT touch `ComputeLeafIndicesDepthwise`.**

The DW `UpdateApproximations` equivalent is the `mx::take` at line 3996 shown in
Section 2. Input is `partitions` (a bit-packed MLX array maintained on-GPU throughout
the depth loop). `ComputeLeafIndicesDepthwise` recomputes leaf indices from scratch via
CPU tree traversal — the training path does not use this recomputation at all. The
comment on line 3991 makes this explicit:

```cpp
// For depthwise trees, partitions is already the correct leaf index (same bit-encoding
// as oblivious trees — the partition update loop sets bit `depth` based on per-node splits).
```

---

## Section 4 — `FindBestSplitPerPartition` Coupling Check

**VERDICT: NO — `FindBestSplitPerPartition` does NOT depend on `ComputeLeafIndicesDepthwise`.**

`FindBestSplitPerPartition` (defined at line 1281) takes pre-built histogram data
(`perDimHistData`) and partition statistics (`perDimPartStats`) as plain C++ vectors.
These are populated from `gradSumArrays` and `hessSumArrays` (lines 3558–3568) derived
from `layout` (a `TPartitionLayout` computed from the `partitions` bit array, not from
`ComputeLeafIndicesDepthwise`).

Call site at line 3602:

```cpp
auto perPartSplits = FindBestSplitPerPartition(
    perDimHistData, perDimPartStats,
    packed.Features, packed.TotalBinFeatures,
    config.L2RegLambda, numPartitions, featureMask,
    config.RandomStrength, gradRms, &rng
);
```

None of these inputs pass through `ComputeLeafIndicesDepthwise`. The histogram kernel
(`DispatchHistogram`) is dispatched using `layout.DocIndices`, `layout.PartOffsets`, and
`layout.PartSizes` — all derived from the `partitions` bit array via
`ComputePartitionLayout(partitions, ...)`.

Therefore FU-3 (if it concerns a gain/score asymmetry in `FindBestSplitPerPartition`)
is **independent** of FU-1. If there is a DW N=1000 parity asymmetry related to gain
computation, its root cause must be traced to `ComputePartitionLayout`, histogram
accumulation, or the scoring formula — not to this function.

---

## Section 5 — FU-3 Hypothesis Implication

Because `ComputeLeafIndicesDepthwise` is validation-only, **Bug A and Bug B have zero
effect on training RMSE, leaf values, or split selection**. Consequently they cannot
directly cause the DW N=1000 parity asymmetry visible in the S26-FU-2 gate data (MLX
achieves 13–17% lower RMSE than CPU at N=1000). However, the asymmetry still warrants
explanation. The S26-FU-2 gate report attributed it to "pre-existing small-N Depthwise
overfitting" and confirmed it was identical pre- and post-FU-2 at DW N=1000 seed=1337
rs=0 (MLX RMSE = 0.17972 in both cases). If FU-3 is investigating this divergence,
the causal chain is: at small N, the DW `partitions`-based split selection
(`FindBestSplitPerPartition` with per-partition histograms) over-fits more aggressively
than CatBoost CPU's equivalent, because (a) MLX's float32 histogram accumulation
order differs from CPU's double-precision accumulation, producing slightly different
split thresholds, and/or (b) per-partition gradient variance is larger at small N so
even small floating-point differences select different splits, compounding across 50
iterations. Bug A/B play no role in this. FU-3 should look at
`ComputePartitionLayout` output ordering and float32 vs float64 histogram accumulation.

---

## Section 6 — S26-FU-2 Gate Report Exposure

All 18 Depthwise cells in the S26-FU-2 gate sweep used `d=6` (fixed config header in
`g1-results.md`: "Fixed: d=6, 128 bins, LR=0.03, 50 iters, RMSE, 20 features"). Depth 6
is ≥ 2, so all 18 DW cells were run with a configuration where both Bug A and Bug B
can fire during validation-path execution.

**Affected DW cells: 18 / 18** — all Depthwise rows in `g1-results.md`.

The bugs corrupt `valCursor` (validation predictions), which means:
- `result.EvalLossHistory` values are incorrect for DW at depth≥2.
- Any `use_best_model=True` selection driven by these histories would pick a wrong
  iteration.
- The gate metric `delta% = (MLX_RMSE − CPU_RMSE) / CPU_RMSE` is computed from
  **final training RMSE** (`result.FinalTrainLoss` from `cursor`, not `valCursor`),
  so the gate **pass/fail decisions are unaffected** by the bug.
- The `pred_std_R` and Pearson metrics in `g1-results.md` are computed from
  **training-set predictions** (also from `cursor`), so those are also unaffected.

Summary: the S26-FU-2 gate report's correctness conclusions are sound. The 18 DW cells'
validation RMSE tracking was corrupted by FU-1 bugs, but the gate metrics themselves
were drawn from the training cursor, which is clean.

| Impact surface | Affected? | Notes |
|---------------|-----------|-------|
| Gate pass/fail | No | Gate metric uses training RMSE (`cursor`) |
| `pred_std_R` | No | Computed from training-set preds |
| `EvalLossHistory` | Yes | Computed from `valCursor` via `ComputeLeafIndicesDepthwise` |
| `use_best_model` selection | Yes | Would select wrong iteration if eval set provided |
| Training model weights | No | `cursor` and `partitions` unaffected |
