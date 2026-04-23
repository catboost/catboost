# S28-COSINE: CPU Cosine Gain Formula — Source Audit

**Date**: 2026-04-23

---

## CPU source location

- `catboost/private/libs/algo/score_calcers.h` — `TCosineScoreCalcer` class (lines 47–74)
- `catboost/private/libs/algo/score_calcers.cpp` — `AddLeafPlain` dispatch (line 11)
- `catboost/libs/helpers/short_vector_ops.h` — `NGenericSimdOps::UpdateScoreBinKernelPlain` (lines 61–81)
- `catboost/private/libs/algo_helpers/online_predictor.h` — `CalcAverage` (lines 112–119)

---

## Cosine gain formula (Plain/non-ordered boosting, per split)

For a candidate split at bin `b` on a partition `p` with gradient histogram `G` and
hessian histogram `H`:

```
sumLeft   = G[0..b]          sumRight   = G[b+1..end]
weightLeft = H[0..b]         weightRight = H[b+1..end]

leafApprox_left  = sumLeft  / (weightLeft  + λ)    // CalcAverage
leafApprox_right = sumRight / (weightRight + λ)

numerator   += leafApprox_left  * sumLeft   + leafApprox_right  * sumRight
denominator += leafApprox_left² * weightLeft + leafApprox_right² * weightRight

score(b, p) = numerator / sqrt(denominator)
```

Where `denominator` is initialized to `1e-100` (guard against sqrt(0)).

Equivalently expanded (substituting CalcAverage inline):

```
numerator   = sumLeft²  / (weightLeft  + λ)  +  sumRight²  / (weightRight + λ)
denominator = sumLeft²  / (weightLeft  + λ)²  * weightLeft
            + sumRight² / (weightRight + λ)²  * weightRight
            = sumLeft²  * weightLeft  / (weightLeft  + λ)²
            + sumRight² * weightRight / (weightRight + λ)²
```

Note the numerator is identical to the L2 gain! The difference is entirely in the
denominator: Cosine normalizes by `sqrt(Σ leafApprox² × hessSum)`, which is a
norm of the leaf-update vector weighted by hessian. The final score is a cosine
similarity between the "proposed update vector" and the "gradient direction."

---

## Comparison: L2 vs Cosine

| Property | L2 | Cosine |
|---|---|---|
| Formula | `G_L²/(W_L+λ) + G_R²/(W_R+λ) - G_T²/(W_T+λ)` | `numerator / sqrt(denominator)` |
| Baseline subtraction | Yes (subtracts parent term) | No (score is absolute, not differential) |
| Scale invariant | No — grows with N | Yes — normalized |
| Favors tiny leaves | Yes — micro-leaves with high G²/W | No — sqrt(W) denominator penalizes |

Key insight: L2 computes differential gain (split improvement over parent node). Cosine
computes a scale-invariant score. They rank splits the same in the limit but diverge
at small N where individual partitions have very few documents.

---

## Numerical behavior notes

1. **No baseline subtraction in Cosine.** CPU's `TCosineScoreCalcer::AddLeafPlain`
   accumulates `score[0]` and `score[1]` for BOTH left and right leaves. There is no
   parent-node subtraction. The final `score[0]/sqrt(score[1])` is an absolute, not
   differential, metric. This means Cosine score values are not directly comparable to
   L2 gain values — they live on different scales.

2. **Guard value `1e-100`.** `Scores.resize(splitsCount, {0, 1e-100})` in
   `TCosineScoreCalcer::SetSplitsCount`. This guards against `sqrt(0)` when a
   partition is empty. In our float32 implementation we use `1e-20f` (the smallest
   representable positive float32 that stays above zero under 1 ULP rounding).

3. **CPU uses double precision.** `TCosineScoreCalcer` accumulates in `double`.
   Our MLX port uses `float32`. Expected ULP drift vs CPU double: empirically
   determined from the gate runs (see t2-gate-report.md). Theoretical upper bound:
   FP32 relative error ≈ 6e-8; for accumulated sums of ~15 terms, ~1e-6 relative.

4. **Zero-weight guard.** Same as L2: `if (weightLeft < 1e-15f || weightRight < 1e-15f) continue;`
   This prevents division by zero before the Cosine formula is even evaluated.

5. **Single-doc leaf.** When a partition has only 1 document after a split, `weightLeft`
   or `weightRight` will be 1 (or the document's weight). Cosine handles this correctly
   — the denominator's `weightLeft/(weightLeft+λ)²` term is non-zero for any positive
   weight. No special case needed.

---

## Implementation approach chosen: Option A

Add `ComputeCosineGain` as a parallel helper function next to the existing L2 inline
computation in `FindBestSplitPerPartition`. The main dispatch path (one-hot and ordinal
branches) keeps calling L2 by default. A local `bool useCosine = false;` flag
(commented with the S28-L2-EXPLICIT TODO) can be flipped for local validation runs.

Rationale: DEC-012 atomicity. This commit adds the Cosine function only. S28-L2-EXPLICIT
will add the `EScoreFunction` enum parameter + call-site replacement.
