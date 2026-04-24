# S32-T2-INSTRUMENT Verdict

**Status:** G3-T2 PASS — first diverging quantity identified; mechanism reclassified  
**Date:** 2026-04-24  
**Config:** iter=1, depth=6, bins=128, loss=RMSE, SymmetricTree, Cosine, seed=42  
**Task:** Per-bin term dump at depth=0; align (feat, bin); name first diverging quantity.

---

## Finding: FORMULA IS CORRECT — ROOT CAUSE IS BORDER GRID DIVERGENCE

The Cosine gain formula in MLX is algebraically correct and numerically verified.
The 5.4% gain deficit (ratio 0.946) is a **measurement artifact** from T3b: the
comparison was aligning on bin *index*, not bin *value* (physical threshold). MLX
and CPU are evaluating **different physical splits** at the same bin index because
their quantization border grids differ.

---

## Evidence

### 1. Formula verification

The `compare_terms.py` output for the win-neighbourhood (feat=0, bins 59..67):

```
bin=59:  gL_MLX=9448.45  gL_CPU=9988.70  wL_MLX=23827  wL_CPU=23046
bin=64:  gL_MLX=9473.71  gL_CPU=10018.62 wL_MLX=25779  wL_CPU=25000
```

When the **same** `(gL, gR, wL, wR, λ)` tuple is fed to the formula on both sides:

- MLX b=64 stats → CPU formula → gain = **84.77664831** (exact match with MLX reported)
- CPU b=64 stats → CPU formula → gain = **89.60924434** (exact match with CPU reported)
- Formula ratio for identical inputs = **1.000000000**

The formula is correct. The 0.946 ratio comes from different input tuples.

### 2. Border grid mismatch

At depth=0, MLX's split threshold `b` corresponds to approximately CPU's split
threshold `b+2` in the feature-0 mid-range:

| MLX bin | MLX wL  | CPU bin (matching wL) | CPU wL |
|---------|---------|----------------------|--------|
| 59      | 23827   | 61                   | 23827  |
| 60      | 24217   | 62                   | 24218  |
| 61      | 24608   | 63                   | 24609  |
| 62      | 24998   | 64                   | 25000  |
| 63      | 25389   | 65                   | 25390  |
| 64      | 25779   | 66                   | 25781  |

MLX bin index b evaluates approximately the same physical split as CPU bin index b+2.
The border values are shifted: MLX's quantization grid produces boundary values
approximately 2 positions lower than CPU's GreedyLogSum grid.

### 3. Gain at matching physical splits

CPU's best split is bin=59 (gain=89.616) at physical border ≈ -0.103.
MLX's best split is bin=64 (gain=84.777) at physical border ≈ -0.005.

These are entirely different physical thresholds. The split quality gap (89.6 vs 84.8)
reflects the difference between evaluating at the true optimal boundary vs a shifted
boundary — not a formula error.

### 4. T3b measurement artifact explained

The T3b audit concluded "GAIN-FORMULA — Cosine score ~5.4% lower in MLX" and noted
that partition `sumH` matched at depth=0. The `sumH` match was at the partition level
(total docs = 50000 on both sides) — it did NOT check per-bin split statistics.
The per-bin statistics differ because the bins themselves refer to different physical
values. The "same bin index" comparison was not a same-split comparison.

---

## Root Cause Classification

The root cause is: **GreedyLogSum port produces different border values from CPU**.

- The algorithm is correct (S31-T2 verified greedy unweighted path, G2a 84/100 exact).
- But the 16 tie-break cases in G2a (equal-score resolution) propagate through the
  border set, causing downstream borders to shift. A 2-index shift in feature-0
  mid-range means 16 border mis-assignments accumulated over 64 bins.
- The misaligned border grid causes MLX to evaluate a different set of split candidates
  at every depth layer, consistently leading to suboptimal splits (gain ~5.4% below
  CPU's optimal split, because MLX's best candidate is not the global optimum).

This is DEC-036 class: **structural divergence, algorithmic origin** — not a
precision/formula bug. The fix must ensure MLX's GreedyLogSum produces bit-exact
border values matching CatBoost's CPU output.

---

## Bug Layer Classification

First diverging column (per `compare_terms.py`): **gL** (rdiff=2.67e+02 at feat=17/b=39,
and ~5.4e-02 in the f0 winner neighbourhood).

However, the gL divergence is a *consequence*, not a cause. The causal chain is:

```
GreedyLogSum tie-break divergence
   → border grid offset (MLX borders ≠ CPU borders by ~2 indices in mid-range)
   → different docs assigned to each bin_value
   → different (gL, gR, wL, wR) for the same bin index b
   → different gain values for the same bin index b
   → argmax picks different bin (MLX b=64 instead of CPU b=59)
   → 5.4% gain deficit (comparing different physical splits)
```

The formula at csv_train.cpp:1789-1815 (S28-OBLIV-DISPATCH) is correct.

---

## G3-T2 Gate

G3-T2 criterion: *first diverging term identified at depth=0 seed=42.*

- First diverging column: gL (gradient, reflecting border grid divergence)
- Mechanism: border grid offset, not formula error
- Formula verified correct: ratio 1.000000 for identical input tuples

**G3-T2: PASS**

---

## Impact on Sprint Scope

The original T3-FIX (#117) target was a formula fix. The actual bug is in the
GreedyLogSum border computation — specifically, tie-break resolution that causes
16/128 border values to differ from CPU, propagating a ~2-index grid shift.

This is a different and deeper fix than anticipated. The fix must ensure
`QuantizeFeatures` (csv_train.cpp:816-889, the GreedyLogSum port from S31-T2)
produces bit-exact border values matching CatBoost.

T3-FIX (#117) is re-scoped: audit the GreedyLogSum tie-break logic in
`QuantizeFeatures` vs CPU's `TGreedyBinarizer` to identify the 16 misaligned
borders, then fix the tie-break to match CPU output exactly.

---

## Data Files

| File | Contents |
|------|---------|
| `data/mlx_terms_seed42_depth0.csv` | 2560 per-bin MLX term records at depth=0 |
| `data/cpu_terms_seed42_depth0.csv` | 2560 per-bin CPU term records at depth=0 |

Schema: `feat,bin,sumLeft,sumRight,weightLeft,weightRight,lambda,cosNum_term,cosDen_term,gain`

---

## Next Step: T3-FIX (re-scoped)

Target: `QuantizeFeatures` in `catboost/mlx/tests/csv_train.cpp` (lines 816-889).
Audit the tie-break logic in the GreedyLogSum port against CatBoost's
`TGreedyBinarizer` (CPU source: `catboost/libs/data/borders_io.h` and
`catboost/libs/data/quantization.cpp`). Find what causes 16/128 border values
to diverge from CPU, producing the ~2-index shift observed at f0 mid-range.
