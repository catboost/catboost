# S32-T1-CODEPATH Verdict

**Status:** G3-T1 PASS — code path resolved: SAME-PATH  
**Date:** 2026-04-24  
**Config:** iter=1, depth=6, bins=128, loss=RMSE, SymmetricTree, Cosine, seed=42  
**Task:** Reconcile `ComputeCosineGainKDim` (the helper T1-PRE mapped) against the live
code path at `FindBestSplit` / `S28-OBLIV-DISPATCH`.

---

## Verdict: SAME-PATH

`FindBestSplit`'s ordinal branch does **NOT** call `ComputeCosineGainKDim`. It inlines
the Cosine formula directly, using `double` accumulators throughout. The inline
expression is algebraically identical to CPU CatBoost's `UpdateScoreBinKernelPlain`.

Hypothesis H1 (code-path skew — bug is in a different expression than T1-PRE audited)
is **ELIMINATED**.

---

## Evidence

### 1. `ComputeCosineGainKDim` — defined but not called from the hot path

**File:** `catboost/mlx/tests/csv_train.cpp`

```
line 1478: inline float ComputeCosineGainKDim(
line 1479:     float totalNum,   // Σ_k sumLeft_k²/(wLeft_k+λ) + sumRight_k²/(wRight_k+λ)
line 1480:     float totalDen    // Σ_k (sumLeft_k²·wLeft_k/(wLeft_k+λ)² + ...) + guard
line 1481: ) {
```

The helper accepts and returns `float`. The comments at line 1562–1565 say
"Convert back to float only at `ComputeCosineGainKDim` call", but the subsequent
code never calls it. The comment is stale from before the K4 double-widening (S30).

Grep for all call sites of `ComputeCosineGainKDim` in `csv_train.cpp` returns zero
hits in any scoring loop. The function is defined but inert in the hot path.

### 2. Live ordinal branch — `S28-OBLIV-DISPATCH`

**Label:** `S28-OBLIV-DISPATCH` (line 1712)  
**Accumulator initialization (lines 1714–1715):**
```cpp
double cosNum_d = 0.0;
double cosDen_d = 1e-20;  // guard against sqrt(0), mirrors float guard
```

**Per-partition-per-k accumulation (lines 1788–1791):**
```cpp
cosNum_d += dSL * dSL * dInvL + dSR * dSR * dInvR;
cosDen_d += dSL * dSL * dWL * dInvL * dInvL
          + dSR * dSR * dWR * dInvR * dInvR;
```
where `dInvL = 1.0 / (dWL + dL2)`, `dInvR = 1.0 / (dWR + dL2)`, all widened
to `double` before the arithmetic (lines 1782–1788).

**Finalization (lines 1813–1815):**
```cpp
if (scoreFunction == EScoreFunction::Cosine) {
    totalGain = cosNum_d / std::sqrt(cosDen_d);
}
```
No cast back to `float`. `totalGain` is `double` (line 1709).

### 3. CPU reference — `UpdateScoreBinKernelPlain`

**File:** `catboost/libs/helpers/short_vector_ops.h`

```cpp
const double trueAvrg  = CalcAverage(trueStatsPtr[0],  trueStatsPtr[1],  scaledL2Regularizer);
const double falseAvrg = CalcAverage(falseStatsPtr[0], falseStatsPtr[1], scaledL2Regularizer);
scoreBinPtr[0] += trueAvrg  * trueStatsPtr[0];          // num += gR²/(wR+λ)
scoreBinPtr[1] += trueAvrg  * trueAvrg  * trueStatsPtr[1];  // den += gR²·wR/(wR+λ)²
scoreBinPtr[0] += falseAvrg * falseStatsPtr[0];         // num += gL²/(wL+λ)
scoreBinPtr[1] += falseAvrg * falseAvrg * falseStatsPtr[1]; // den += gL²·wL/(wR+λ)²
```
where `CalcAverage(sum, weight, λ) = sum / (weight + λ)`.

This expands to:
```
num += gL²/(wL+λ) + gR²/(wR+λ)
den += gL²·wL/(wL+λ)² + gR²·wR/(wR+λ)²
```

MLX inline is identical. Both use `double` for all intermediate values.

### 4. numPartitions equivalence

**MLX** (line 4255): `ui32 numPartitions = 1u << depth;`  
**CPU** (`scoring.cpp`, line 348): `int leafCount = 1 << depth;`

At depth=0: `numPartitions = 1`. The ordinal branch loops `for (ui32 p = 0; p < numPartitions; ++p)` — a single iteration. The scan is not a multi-partition composition issue at depth=0.

### 5. Suffix sum semantics

`suffGrad[base + b]` (built at lines 1684–1686) = `Σ_{i=b}^{folds-1} hist[firstFold + i]` = sum of gradient for docs with bin_value in `[b, folds-1]`. With `folds = 128` (post-DEC-037), this correctly represents the right-child statistics for a split at threshold `b`.

---

## Formula Algebraic Identity — Confirmed

From T1-PRE's 11-row audit (verdict at `docs/sprint31/t1-pre/verdict.md`, rows F1–F11):
all formula rows are algebraically equivalent between CPU and MLX for the `S28-OBLIV-DISPATCH`
anchor. T1-CODEPATH confirms the live code path executes the same expression.

The only caveat from T1-PRE — "T1-PRE mapped `ComputeCosineGainKDim` but noted it might be
on an INERT code path" — is now resolved definitively: the helper is inert.

---

## Remaining Hypothesis Space

H1 eliminated. The formula and code path are correct.

The 5.4% stable deficit (ratio 0.946, stable across seeds 42/43/44 and depths 0–5) must
originate in the **input values** `(sumLeft, sumRight, weightLeft, weightRight)` passed to
the formula. These come from the histogram suffix sums (`suffGrad`, `suffHess`) and
partition stats (`perDimPartStats`).

Candidate mechanisms for T2:

| Layer | Variable | What to check |
|-------|----------|---------------|
| Histogram writeback | `hist[firstFold + b]` | 1-indexed bin convention (bin+1 offset in kernel) — docs with bin_value=0 NOT written |
| Partition stats | `perDimPartStats[k][p].Sum` / `.Weight` | totalSum / totalWeight used for `sumLeft = total - sumRight` |
| suffGrad sentinel | `suffGrad[base + folds] = 0` | Correct — sentinel is zero, so `bin = folds-1` gets only the last bucket |
| wL/wR P11 risk | `weightLeft`, `weightRight` | For RMSE hessian=1 per doc; `wL = count_left`, `wR = count_right`; matches CPU `SumWeight` |

T2-INSTRUMENT must dump `(gL, gR, wL, wR, λ, cosNum_term, cosDen_term, gain)` at bin scope
for depth=0, seed=42, feature=0 and compare term-by-term against CPU's dump at the same
(feature, bin) coordinate. The first diverging quantity names the bug layer.

---

## G3-T1 Gate

G3-T1 criterion: *code path resolved (SAME-PATH or DIFFERENT-PATH with file:line evidence).*

- Code path: `FindBestSplit` ordinal branch at `csv_train.cpp:1708–1815` (S28-OBLIV-DISPATCH)
- Formula: algebraically identical to CPU `UpdateScoreBinKernelPlain` at `short_vector_ops.h`
- `ComputeCosineGainKDim` (line 1478): defined, inert, never called from hot path
- H1 eliminated

**G3-T1: PASS**

---

## Next Step: T2-INSTRUMENT (#116)

Add a compile-time flag `COSINE_TERM_AUDIT` enabling per-bin dumps of
`(feat, bin, p, k, sumLeft, sumRight, weightLeft, weightRight, λ, cosNum_term, cosDen_term)`
at depth=0 seed=42. Add a matching dump to `cpu_dump.py` at the same scope.
Align by `(feat_idx, bin_idx)` and report the first term where MLX ≠ CPU.
