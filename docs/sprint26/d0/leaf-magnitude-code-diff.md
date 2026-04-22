# S26 D0-1/2/3 — Leaf-Magnitude Code Diff & Instrumentation Plan

**Date**: 2026-04-22
**Branch**: `mlx/sprint-26-python-parity`
**Triage scope**: Read-only. No code changes. No benchmark runs.
**Status**: Root cause NOT eliminated by static analysis — formulas are algebraically
equivalent. Instrumentation required.

---

## Section 1 — D0-1: Leaf-Estimation Algebra Diff

### CPU reference path (Newton step)

| Step | File | Lines | Expression |
|---|---|---|---|
| Accumulate `SumDer` and `SumDer2` per leaf | `catboost/private/libs/algo/approx_calcer.cpp` | ~147-162 | `bucket->AddDerDer2(der1, der2)` |
| Newton delta (body) | `catboost/private/libs/algo_helpers/online_predictor.h` | 162-169 | `sumDer / (-sumDer2 + l2 * (sumAllWeights / allDocCount))` |
| Scale l2 reg | `catboost/private/libs/algo_helpers/online_predictor.h` | 121-127 | `l2 * (sumAllWeights / allDocCount)` — equals `l2` for unweighted |
| Apply learning rate | `catboost/private/libs/algo_helpers/approx_updater_helpers.cpp` | 33-37 | `leafValue *= learningRate` — after Newton step |
| Caller that invokes NormalizeLeafValues | `catboost/private/libs/algo/train.cpp` | 323-353 | `NormalizeLeafValues(isPairwise, learningRate, sumLeafWeights, &treeValues)` |

Expanded for RMSE (unweighted, `SumDer2 = -N` because `RMSE_DER2 = -1.0`):

```
rawLeaf = SumDer / (-SumDer2 + l2 * (W/N))
        = Σ(target - approx) / (N + l2)        [for unweighted: W = N so W/N = 1]
finalLeaf = rawLeaf * LR
```

### MLX RunBoosting path (nanobind / `mlx_boosting.cpp`)

| Step | File | Lines | Expression |
|---|---|---|---|
| Compute grad/hess | `catboost/mlx/targets/pointwise_target.h` | 27-33 | `grad = pred - target; hess = weights (=1)` |
| Accumulate per leaf via Metal kernel | `catboost/mlx/methods/leaves/leaf_estimator.cpp` | 51-79 | `leaf_accumulate` Metal kernel: `gradSums[leaf] += grad[doc]` etc. |
| Newton + LR (in-formula) | `catboost/mlx/methods/leaves/leaf_estimator.cpp` | 29-31 | `rawValues = -gradSums / (hessSums + l2); return rawValues * LR` |
| Caller | `catboost/mlx/methods/mlx_boosting.cpp` | 207-208 | `leafValues = ComputeLeafValues(gradSumsGPU, hessSumsGPU, config.L2RegLambda, config.LearningRate)` |

Expanded for RMSE (unweighted, `hess = 1.0`, `grad = pred - target`):

```
rawValues = -gradSums / (hessSums + l2)
          = -Σ(pred - target) / (N + l2)
          = Σ(target - pred) / (N + l2)
finalLeaf = rawValues * LR
```

### MLX csv_train subprocess path (`csv_train.cpp`)

| Step | File | Lines | Expression |
|---|---|---|---|
| Compute grad/hess | `catboost/mlx/tests/csv_train.cpp` | 2783-2785 | `dimGrads[0] = pred - target; dimHess[0] = ones` |
| Accumulate via `scatter_add_axis` | `catboost/mlx/tests/csv_train.cpp` | 3480-3481 | `gSumsArr = scatter_add_axis(leafTarget, partitions, dimGrads[0], 0)` |
| Newton + LR (in-formula) | `catboost/mlx/tests/csv_train.cpp` | 3492-3493 | `leafValues = -lr * gSumsArr / (hSumsArr + l2)` |

Expanded for RMSE:

```
leafValues = -LR * Σ(pred - target) / (N + l2)
           = LR * Σ(target - pred) / (N + l2)
```

### Side-by-side algebra table

| Path | Numerator | Sign correction | Denominator | LR application |
|---|---|---|---|---|
| CPU (correct) | `Σ(target - approx)` | none needed: CalcDer = target-approx | `N + l2` | `*LR` after Newton |
| MLX RunBoosting | `Σ(pred - target)` | negation applied by `mx::negative` | `N + l2` | `*LR` inside ComputeLeafValues |
| MLX csv_train | `Σ(pred - target)` | negation applied by `mx::negative` | `N + l2` | `*LR` inside formula |

**Static-analysis verdict: the three formulas are algebraically equivalent for unweighted
RMSE.** No factor-of-2 or sign bug is visible in the Newton step algebra at rest.

### Structural difference between paths

The `RunBoosting` path uses a custom `leaf_accumulate` Metal kernel
(`leaf_estimator.cpp:51-79`) for grad/hess accumulation. The `csv_train` subprocess
path uses MLX's built-in `scatter_add_axis` (`csv_train.cpp:3480-3481`). Both should
produce identical sums, but the Metal kernel has not been independently validated
against `scatter_add_axis` at the sum-statistics level. If the kernel has an
off-by-one in its partition indexing or a coverage hole (docs not visited, docs
double-counted), the grad/hess sums will be wrong and the Newton step will produce
wrong leaf values even though its formula is correct.

The python wrapper (`python/catboost_mlx/core.py:1322`) uses the RunBoosting path when
nanobind is available and falls back to csv_train subprocess otherwise. The empirical
shrinkage (~0.69×) is observed on the python path but it is not yet known which of the
two code paths is active during the test runs. This must be determined during
instrumentation (see Section 3).

---

## Section 2 — D0-2: RMSE Target Grad/Hessian Diff

### CPU `TRMSEError` definition

| Field | File | Line | Value |
|---|---|---|---|
| `RMSE_DER2` | `catboost/private/libs/algo_helpers/error_functions.h` | 381 | `-1.0` (class constant) |
| `CalcDer` | `catboost/private/libs/algo_helpers/error_functions.h` | 392-394 | `target - approx` |
| `CalcDer2` | `catboost/private/libs/algo_helpers/error_functions.h` | 396-398 | `RMSE_DER2 = -1.0` |

CPU convention: the second derivative is stored as `-1.0` because CatBoost's Newton
formula is `SumDer / (-SumDer2 + l2)`. The per-leaf hessian sum is therefore
`SumDer2 = -N` for N unweighted docs; `-SumDer2 = N`. Denominator = `N + l2`.

### MLX `TRMSETarget` definition

| Field | File | Lines | Value |
|---|---|---|---|
| `gradients` | `catboost/mlx/targets/pointwise_target.h` | 27 | `pred - target` |
| weight multiplication | `catboost/mlx/targets/pointwise_target.h` | 30 | `gradients *= weights` (=1 for unweighted) |
| `hessians` | `catboost/mlx/targets/pointwise_target.h` | 33 | `weights` (=1 for unweighted) |

MLX convention: hessian is `+1.0` (positive). Newton formula uses `hessSums + l2`.
For unweighted docs: `hessSums = N`. Denominator = `N + l2`.

### Equivalence proof (unweighted)

```
CPU:  SumDer / (-SumDer2 + l2) = Σ(target-approx) / (N + l2)
MLX: -gradSums / (hessSums + l2) = -Σ(pred-target) / (N + l2) = Σ(target-pred) / (N + l2)
```

Both resolve to `Σ(target - pred) / (N + l2)`. **No discrepancy.**

### `ScaleL2Reg` check

CPU scales l2 as `l2 * (sumAllWeights / allDocCount)` (online_predictor.h:126). For
unweighted data `sumAllWeights == allDocCount`, so the ratio is 1.0 and the effective l2
equals the raw `l2` parameter. MLX passes `config.L2RegLambda` directly without scaling.
Both use the same value for unweighted data.

### Weighted case (future risk)

If weights are non-uniform, the CPU `ScaleL2Reg` produces a value different from the
raw `l2`, but MLX does not replicate this scaling. This is a latent discrepancy for
weighted training, but it does not explain the current bug because the test data is
unweighted.

---

## Section 3 — D0-3: Instrumentation Plan

The static analysis failed to isolate the root cause. The following probes are required.
All logging is temporary (not committed). The goal is to print raw numbers that confirm
or falsify each hypothesis in Section 4.

### Probe P1 — Determine which Python path is active

**Goal**: Establish whether the empirical shrinkage is coming from the RunBoosting
(Metal kernel) path or the csv_train (scatter_add_axis) path.

**Location**: `python/catboost_mlx/core.py:1322`

```python
# In CatBoostMLXRegressor.fit(), just before the branch:
import os
_using_nanobind = _HAS_NANOBIND  # check the module-level flag
print(f"[S26-P1] using_nanobind={_using_nanobind}", flush=True)
```

**Exit criterion**: If `using_nanobind=True`, Probes P3-P5 target `leaf_estimator.cpp`.
If `using_nanobind=False`, Probes P3-P5 target `csv_train.cpp`.

### Probe P2 — Log raw grad/hess sums at depth 0 leaf 0 for iter 0

**Goal**: Compare `Σgrad` and `Σhess` values between MLX and CPU for the exact same
training data at iteration 0. If `Σhess` is approximately `2N` instead of `N`, the
denominator is doubled and the leaf value is halved — this is the classic hessian=2
hypothesis.

**Location A (RunBoosting path)**: `catboost/mlx/methods/mlx_boosting.cpp`, after line 197
(after `ComputeLeafSumsGPU` call), inside `iter == 0` guard:

```cpp
if (iter == 0) {
    mx::eval(gradSumsGPU, hessSumsGPU);
    const float* gs = gradSumsGPU.data<float>();
    const float* hs = hessSumsGPU.data<float>();
    CATBOOST_INFO_LOG << "[S26-P2] iter=0 numLeaves=" << numLeaves
        << " gradSum[0]=" << gs[0] << " hessSum[0]=" << hs[0]
        << " total_grad=" << [&]{ float s=0; for(ui32 i=0;i<numLeaves;i++) s+=gs[i]; return s; }()
        << " total_hess=" << [&]{ float s=0; for(ui32 i=0;i<numLeaves;i++) s+=hs[i]; return s; }()
        << Endl;
}
```

**Location B (csv_train path)**: `catboost/mlx/tests/csv_train.cpp`, after line 3481
(after `gSumsArr`/`hSumsArr` assignment), inside `iter == 0` guard:

```cpp
if (iter == 0) {
    mx::eval(gSumsArr, hSumsArr);
    const float* gs = gSumsArr.data<float>();
    const float* hs = hSumsArr.data<float>();
    float totalG=0, totalH=0;
    for (ui32 i=0; i<numLeaves; i++) { totalG+=gs[i]; totalH+=hs[i]; }
    fprintf(stderr, "[S26-P2] iter=0 numLeaves=%u gradSum[0]=%.6f hessSum[0]=%.6f"
                    " totalGrad=%.6f totalHess=%.6f\n",
            numLeaves, gs[0], hs[0], totalG, totalH);
}
```

**What to compare**: `totalHess` should equal `trainDocs` (N) for unweighted RMSE.
If it is `2*N`, hypothesis H1 is confirmed. If it equals N, H1 is falsified.

**CPU reference**: for N=10000 unweighted RMSE, `SumDer2 = -10000`, `-SumDer2 = 10000`.
So the denominator is `10000 + 3.0 = 10003` (at default l2=3). Compare directly.

### Probe P3 — Log raw leaf values at iter 0

**Goal**: Directly compare the first iteration's leaf values between MLX and CPU. If
MLX leaf values are uniformly ~0.51× of CPU, the magnitude bug is confirmed at this
level. If they match, the bug is in the update/apply step, not leaf estimation.

**Location A (RunBoosting path)**: `catboost/mlx/methods/mlx_boosting.cpp`, after line 208
(after `ComputeLeafValues` call), inside `iter == 0` guard:

```cpp
if (iter == 0) {
    mx::eval(leafValues);
    const float* lv = leafValues.data<float>();
    CATBOOST_INFO_LOG << "[S26-P3] iter=0 leaf[0]=" << lv[0]
        << " leaf[1]=" << lv[1] << Endl;
}
```

**Location B (csv_train path)**: `catboost/mlx/tests/csv_train.cpp`, after line 3493
(after `leafValues` assignment):

```cpp
if (iter == 0) {
    mx::eval(leafValues);
    const float* lv = leafValues.data<float>();
    fprintf(stderr, "[S26-P3] iter=0 leaf[0]=%.8f leaf[1]=%.8f\n",
            lv[0], lv[1]);
}
```

**CPU reference**: add to approx_calcer.cpp after `CalcLeafDeltasSimple` returns or
add a Python-side print: `print(f"CPU leaf[0]={model.get_leaf_values()[0]:.8f}")`.

### Probe P4 — Validate Metal leaf_accumulate kernel against scatter_add_axis

**Goal**: Run both the Metal kernel path and the `scatter_add_axis` path on the same
inputs and compare. Isolates whether the Metal kernel is the source of the discrepancy.

**Location**: `catboost/mlx/methods/leaves/leaf_estimator.cpp`, in `ComputeLeafSumsGPU`,
after the single-pass kernel call returns `outGradSums`/`outHessSums`:

```cpp
// Reference: scatter_add_axis on same inputs (CPU path for validation)
auto leafTargetRef = mx::zeros({static_cast<int>(numLeaves)}, mx::float32);
auto refGradSums = mx::scatter_add_axis(leafTargetRef, flatParts, flatGrads, 0);
auto refHessSums = mx::scatter_add_axis(leafTargetRef, flatParts, flatHess, 0);
mx::eval(outGradSums, outHessSums, refGradSums, refHessSums);
const float* kg = outGradSums.data<float>();
const float* kh = outHessSums.data<float>();
const float* rg = refGradSums.data<float>();
const float* rh = refHessSums.data<float>();
float maxGDiff = 0, maxHDiff = 0;
for (ui32 i = 0; i < numLeaves; i++) {
    maxGDiff = std::max(maxGDiff, std::abs(kg[i] - rg[i]));
    maxHDiff = std::max(maxHDiff, std::abs(kh[i] - rh[i]));
}
CATBOOST_INFO_LOG << "[S26-P4] kernel_vs_scatter: maxGradDiff=" << maxGDiff
    << " maxHessDiff=" << maxHDiff
    << " kernel_hessSum[0]=" << kh[0]
    << " scatter_hessSum[0]=" << rh[0] << Endl;
```

**Exit criterion**: If `maxGDiff` and `maxHDiff` are both < 1e-4 (float32 rounding
noise), the Metal kernel is correct. If they differ substantially, the Metal kernel is
the bug site.

### Probe P5 — Check cursor (base prediction) initialization

**Goal**: Verify the starting cursor value is correct. If the base prediction is wrong,
all subsequent gradient/leaf computations will be wrong in a systematic way.

**Location (RunBoosting path)**: `catboost/mlx/methods/mlx_boosting.cpp`, before the
loop begins (before `iter == 0`):

```cpp
mx::eval(trainData.GetCursor());
const float* c = trainData.GetCursor().data<float>();
float minC=c[0], maxC=c[0], sumC=0;
for (ui32 i=0;i<numDocs;i++) { minC=std::min(minC,c[i]); maxC=std::max(maxC,c[i]); sumC+=c[i]; }
CATBOOST_INFO_LOG << "[S26-P5] initial cursor: min=" << minC << " max=" << maxC
    << " mean=" << sumC/numDocs << Endl;
```

**Location (csv_train path)**: `catboost/mlx/tests/csv_train.cpp`, after
`CalcBasePrediction` and before the iteration loop:

```cpp
mx::eval(cursor);
const float* c = cursor.data<float>();
float minC=c[0],maxC=c[0],sumC=0;
for(ui32 i=0;i<trainDocs;i++){minC=std::min(minC,c[i]);maxC=std::max(maxC,c[i]);sumC+=c[i];}
fprintf(stderr,"[S26-P5] initial cursor: min=%.6f max=%.6f mean=%.6f\n",
        minC, maxC, sumC/trainDocs);
```

**CPU reference**: the initial cursor should equal the weighted mean of targets (for RMSE).
For the test dataset, `mean(y) ≈ 0.0001`. If MLX starts at zero instead of the mean,
the first-iteration gradient magnitude will be larger than expected but subsequent
corrections should absorb this. If it starts at the wrong constant, investigate
`CalcBasePrediction` (csv_train.cpp:2518-2531).

### Probe execution order

Execute P1 first (path disambiguation). Then P5 (base pred sanity). Then P2 and P3
together (leaf sums and leaf values). Then P4 only if P2/P3 show `hessSum ≈ N` (i.e.,
the formula inputs look correct but the output is still wrong — implying kernel error).

---

## Section 4 — Ranked Hypotheses

### H1 — Metal `leaf_accumulate` kernel double-counts docs (MOST LIKELY)

**Probability**: High.

The analytic derivation in the triage brief gives `α ≈ 0.51` — leaf values are
computed at roughly half magnitude. The Newton denominator is `Σhess + l2`. For
unweighted RMSE `Σhess = N`. If the kernel processes each doc twice (e.g., all-threads
iterate over all docs and each doc falls into two thread-blocks' ranges), then
`Σhess ≈ 2N` and the denominator doubles, halving the leaf value. The single-pass
`leaf_accumulate` kernel dispatches `grid=(256,1,1), threadgroup=(256,1,1)` — 256
threads total over N=10000 docs. If the doc-iteration stride is computed incorrectly so
that multiple threads cover the same docs, this would produce double-counting.

This hypothesis is specific to the RunBoosting (nanobind) path. If P1 shows
`using_nanobind=False`, this hypothesis shifts to H4.

**Probe to confirm**: P2 (totalHess ≈ 2N?) and P4 (Metal kernel vs scatter_add_axis
divergence).

### H2 — Partitions array wrong for SymmetricTree at leaf-sum time (MODERATE)

**Probability**: Moderate.

In `mlx_boosting.cpp:182-184`, for SymmetricTree the partition array is
`trainData.GetPartitions()`. These are bit-encoded leaf IDs (depth-d doc has
`partitionBits` = the path bits through the tree). At depth `d`, the leaf IDs range
`[0, 2^d)` in bit-encoded form. If the partition encoding uses a different convention
than what `leaf_accumulate` expects (e.g., bit-reversed order, or pre-shift vs
post-shift of bits), docs could be assigned to wrong leaves, producing incorrect
gradient sums. The leaf values would still have the right total magnitude but would be
assigned to wrong leaves — this would produce the observed shrinkage if many leaves end
up with near-zero sums.

**Probe to confirm**: P3 — if leaf[0] and leaf[1] have equal magnitudes but
inconsistent signs compared to CPU, partition mismatch is the cause.

### H3 — Base prediction initialization (cursor) wrong (LOW-MODERATE)

**Probability**: Low-moderate.

If the cursor (base prediction) is initialized to all-zeros instead of `mean(y)`,
the initial gradient `Σ(pred - target)` at iter=0 has larger magnitude than expected,
but this does not mechanically produce a consistent 0.69× shrinkage factor across all
iterations. The GBDT loss history in `dissect.py` shows monotone decrease starting from
0.5825 — consistent with a non-zero base pred. However if the base prediction in MLX
is initialized to 0 vs CPU's `mean(y) ≈ 0.0001`, the difference is negligible at
`σ(y) = 0.59`.

**Probe to confirm**: P5 — initial cursor mean should equal `mean(y)` for RMSE.

### H4 — csv_train subprocess partition array wrong for Depthwise (secondary)

**Probability**: High (for Depthwise/Lossguide only).

`csv_train.cpp:3368-3370` encodes partitions using bit-shifting:
`partitions = partitions | (updateBits << depth)`. At depth d, the current leaf IDs
range over `[0, 2^d)` as bit-packed paths. At depth d+1, a new bit is OR'd at
position `depth`. The leaf-value scatter at line 3480 uses these bit-encoded IDs
directly as scatter indices. This is correct for SymmetricTree (leaf ID = all d path
bits, range `[0, 2^d)`). However for Depthwise, splits are per-partition so the
resulting partition IDs are identical in structure. This should still be correct.

The larger delta for Depthwise (561%) vs SymmetricTree (68%) may instead be caused by
the leaf-value scatter receiving `numLeaves = 1 << actualTreeDepth` at line 3461-3462.
If `actualTreeDepth` is less than `config.MaxDepth` because `anyValid` becomes false
early for some leaf partitions, fewer leaves are produced. This would not explain the
systematic magnitude bug but would explain why Depthwise compounds the error.

**Probe to confirm**: P2 applied to a Depthwise run — check `numLeaves` printed vs
expected.

### H5 — LR applied twice in RunBoosting fused multi-pass path (LOW)

**Probability**: Low for SymmetricTree at depth 6.

`ComputeLeafSumsGPUMultiPass` in `leaf_estimator.cpp:92-164` applies for
`numLeaves > 64` (depth > 6). The test uses `depth=6` so `numLeaves = 64`, which
routes through the single-pass path. This hypothesis applies only at depth >= 7.
Not the primary cause for the current test config but worth verifying is not
accidentally triggered.

---

## Section 5 — Secondary Bugs Flagged

### B1 — Depthwise/Lossguide extra divergence (561%/598%)

The SymmetricTree delta is 68%. Depthwise is 561% and Lossguide is 598%. Even if the
primary leaf-magnitude bug is fixed, Depthwise and Lossguide will likely still exceed
the G2 gate (5% delta) due to additional bugs.

**Lossguide-specific issue**: `evalLeafLossguide` (`csv_train.cpp:3020-3088`) builds a
2-partition sub-histogram for each leaf being evaluated. The leaf grad/hess sums at
lines 3069-3072 are read from partition-0 of this 2-partition array. If the
`leafPartArr` encoding `(lossguideLeafDocVec[d] == leafId) ? 0u : 1u` is reversed in
one of the paths (i.e., in-leaf docs get ID=1 instead of ID=0), then `partStats[0]`
contains out-of-leaf docs and `partStats[1]` contains the target leaf. This would make
split scoring use the wrong partition's stats, producing systematically bad splits. The
final leaf-value scatter at `csv_train.cpp:3143-3145` uses `lossguideLeafDocVec`
(dense leaf IDs 0..numLeaves-1) which looks correct.

**Depthwise + SymmetricTree in mlx_boosting.cpp**: The Depthwise path in `mlx_boosting.cpp`
passes `trainData.GetPartitions()` as the partition array to `ComputeLeafSumsGPU`. The
partition array is updated by `SearchDepthwiseTreeStructure`. If that function encodes
leaf IDs differently from what `leaf_accumulate` expects (e.g., using the first `depth`
bits of a 32-bit integer vs a simple integer index), the leaf sums will be wrong.
This is independent of H1 but would amplify the error.

### B2 — Weighted-case ScaleL2Reg discrepancy (latent)

As noted in Section 2, the CPU path scales l2 by `sumAllWeights / allDocCount` while
MLX uses raw l2. For non-uniform weights this produces a different effective
regularization strength. Not triggered by the current unweighted test but will manifest
if any of the 18-config G1 sweep uses weights.

### B3 — Multi-pass leaf accumulation not exercised by test config

The test uses `depth=6`, `numLeaves=64`, which routes through the single-pass Metal
kernel. The multi-pass path (`leaf_estimator.cpp:92-164`) has never been exercised in
the S24 ULP=0 record. Any depth-7+ config will hit this path; it has not been validated.

---

## Summary: what the code inspection did and did not resolve

**Ruled out by static analysis:**
- Factor-of-2 hessian in the Newton formula algebra (CPU `SumDer2=-1` vs MLX `hess=1`
  are different representations of the same quantity; both give denominator = N+l2)
- LR applied twice (each path applies LR exactly once)
- Gradient sign error (both produce `Σ(target-pred)` in the numerator)
- `ScaleL2Reg` discrepancy for unweighted case (ratio = 1.0)
- Predict-time bug (train_loss history == predict-based RMSE, confirmed empirically)

**Not resolved — requires instrumentation:**
- Whether the Metal `leaf_accumulate` kernel double-counts docs (P2, P4)
- Whether the partition array encoding is correct at leaf-sum time (P3)
- Which Python path (RunBoosting vs csv_train) is actually active (P1)
- Whether `Σhess` at iter=0 equals N or 2N (P2 direct measurement)

**Most likely root cause** (H1): the `leaf_accumulate` Metal kernel produces
`hessSums ≈ 2N` because of incorrect thread-doc iteration in the single-pass dispatch,
causing the Newton denominator to be 2× larger than correct and the leaf values to be
2× smaller. This produces the observed `α ≈ 0.51` shrinkage.

**Verification strategy**: run P1 to confirm the active path, then P2 to measure
`totalHess` directly. If `totalHess ≈ 2*N`, H1 is confirmed and the fix is in the
Metal kernel's iteration logic. If `totalHess ≈ N`, the sums are correct and the bug
is elsewhere — proceed to P4 (kernel vs scatter_add comparison) and P3 (leaf value
comparison).
