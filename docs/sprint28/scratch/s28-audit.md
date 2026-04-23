# S28-AUDIT: score_function Grep Audit — MLX Backend

**Branch**: `mlx/sprint-28-score-function-fidelity`
**Date**: 2026-04-23
**Authored by**: @ml-engineer (S28-AUDIT, task #70)

---

## 1. Context

DEC-032 (Sprint 27, S27-FU-3) established that `FindBestSplitPerPartition` in the
MLX backend hardcodes the L2 Newton gain formula throughout the Depthwise and
Lossguide code paths. CPU CatBoost dispatches on a `score_function` hyperparameter
whose default is `Cosine`, not `L2`. The two gain functions are not equivalent; at
small N (depth=6, ~15 docs/partition) they diverge by 14–17% in RMSE on the DW path.
S28 is charged with (a) formally confirming the absence of any `score_function`
plumbing in `catboost/mlx/`, (b) porting Cosine gain, and (c) wiring enum dispatch.
This document covers step (a) — the grep audit formality.

---

## 2. Grep — zero hits confirmed

Command run against the entire MLX backend subtree:

```
grep -r "score_function" catboost/mlx/
```

Output: **(no output — zero matches)**

The pattern `score_function` does not appear in any file under `catboost/mlx/`.
This covers all subdirectories: `kernels/`, `methods/`, `train_lib/`, `targets/`,
`gpu_data/`, and `tests/`.

---

## 3. Hardcoded L2 gain call site — `FindBestSplitPerPartition`

File: `catboost/mlx/tests/csv_train.cpp`

The function `FindBestSplitPerPartition` begins at **line 1281**. The L2 Newton
gain formula is computed inline — hardcoded — at two places within the function body:

**One-hot branch (lines 1325–1327):**
```cpp
// lines 1314–1327
                    for (ui32 k = 0; k < K; ++k) {
                        const float* histData = perDimHist[k].data()
                            + static_cast<size_t>(p) * 2 * totalBinFeatures;
                        float totalSum    = perDimPartStats[k][p].Sum;
                        float totalWeight = perDimPartStats[k][p].Weight;
                        float sumRight    = histData[feat.FirstFoldIndex + bin];
                        float weightRight = histData[totalBinFeatures + feat.FirstFoldIndex + bin];
                        float sumLeft    = totalSum - sumRight;
                        float weightLeft = totalWeight - weightRight;
                        if (weightLeft < 1e-15f || weightRight < 1e-15f) continue;
                        gain += (sumLeft * sumLeft) / (weightLeft + l2RegLambda)
                              + (sumRight * sumRight) / (weightRight + l2RegLambda)
                              - (totalSum * totalSum) / (totalWeight + l2RegLambda);
                    }
```

**Ordinal branch (lines 1377–1379):**
```cpp
// lines 1367–1380
                    for (ui32 k = 0; k < K; ++k) {
                        float totalSum    = perDimPartStats[k][p].Sum;
                        float totalWeight = perDimPartStats[k][p].Weight;
                        size_t base = (k * numPartitions + p) * stride;
                        float sumRight    = suffGrad[base + binOffset];
                        float weightRight = suffHess[base + binOffset];
                        float sumLeft    = totalSum - sumRight;
                        float weightLeft = totalWeight - weightRight;
                        if (weightLeft < 1e-15f || weightRight < 1e-15f) continue;
                        gain += (sumLeft * sumLeft) / (weightLeft + l2RegLambda)
                              + (sumRight * sumRight) / (weightRight + l2RegLambda)
                              - (totalSum * totalSum) / (totalWeight + l2RegLambda);
                    }
```

Both blocks implement the identical formula:

```
gain += G_left² / (W_left + λ)
      + G_right² / (W_right + λ)
      − G_total² / (W_total + λ)
```

This is the **L2 Newton gain** (split improvement in squared-gradient over hessian
regularized by `l2RegLambda`). There is no conditional, no enum argument, and no
alternative gain path anywhere in the function.

---

## 4. Absence of dispatch layer in `catboost/mlx/methods/`

`catboost/mlx/methods/score_calcer.h` / `score_calcer.cpp` expose two overloads
of `FindBestSplitGPU` (for the SymmetricTree GPU path). Neither overload accepts a
`score_function` parameter. The kernel sources dispatched by `score_calcer.cpp`
(`kScoreSplitsSource`, `kScoreSplitsLookupSource`) also implement L2 Newton gain
only — no Cosine branch exists in the Metal shader layer.

The DW/LG call path runs through `FindBestSplitPerPartition` in `csv_train.cpp`
entirely on CPU (suffix sums + sequential scan), not through `score_calcer.cpp`.
There is therefore no pre-existing dispatch layer anywhere in the MLX backend —
only the hardcoded L2 call to replace.

---

## 5. Verdict

**AUDIT CONFIRMED — S28-COSINE scope bounded to `FindBestSplitPerPartition` in
`catboost/mlx/tests/csv_train.cpp`; no pre-existing dispatch layer to refactor,
only hardcoded call to replace.**
