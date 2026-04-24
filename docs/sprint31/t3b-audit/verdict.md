# S31-T3b-T1-AUDIT Verdict

**Status:** G1 PASS — divergence localized, mechanism class named  
**Date:** 2026-04-24  
**Config:** iter=1, depth=6, bins=128, loss=RMSE, SymmetricTree, Cosine, seed={42,43,44}, N=50k

---

## First Diverging Layer

Across all 3 seeds, the first diverging layer is at **depth=0** (seeds 42 and 44) or
**depth=2** (seed 43, after two coincidental matches at depth 0-1).

The nominal first divergence is **depth=0** — the very first split selection.

---

## Mechanism Class: GAIN-FORMULA

**DEC-036 classification: GAIN-FORMULA**

The Cosine score formula in MLX's `FindBestSplit` produces gain values that are
systematically ~5.4% lower than CPU CatBoost. This shifts the argmax to the wrong
bin, causing split divergence.

Evidence:
- seed=42: CPU picks f0/b59 (gain=89.616), MLX picks f0/b64 (gain=84.777).
  Gain rdiff = 5.40e-02. Both select feature f0 (same feature), but the gain
  magnitude is wrong in MLX, altering the argmax.
- seed=43: Depth=0 coincidentally matches (both pick f0/b64), but depth=2
  diverges. Same GAIN-FORMULA pattern.
- seed=44: CPU picks f0/b62 (gain=89.098), MLX picks f0/b61 (gain=84.140).
  Gain rdiff = 5.56e-02.

**Gain ratio: MLX/CPU ≈ 84.8/89.6 ≈ 0.946** — consistent across seeds and depths.

---

## Co-Fix: DEC-037 Border Count (Applied This Sprint)

Before this audit, MLX used `maxBordersCount = maxBins - 1 = 127` borders, while
CPU CatBoost uses `border_count = 128`. This caused systematic MLX_bin = CPU_bin - 1
offset (confirmed in previous runs). The fix was applied in this sprint:

**File:** `catboost/mlx/tests/csv_train.cpp`  
**Change:** `maxBordersCount = maxBins` (was `maxBins - 1`)  
**Algorithm restored:** Greedy unweighted GreedyLogSumBestSplit (TGreedyBinarizer unweighted
path — count of unique values, not document counts). The earlier DP rewrite was incorrect
(used document-count weights instead of unique-value counts).

After the DEC-037 fix, seeds 42 and 43 at depth=0 now both select feature f0 (matching
the CPU feature), confirming the border alignment is correct. The bin index divergence
is entirely attributable to the gain formula.

---

## Root Cause: Cosine Accumulation Divergence

**File:** `catboost/mlx/tests/csv_train.cpp`  
**Function:** `FindBestSplit` (labelled `S28-OBLIV-DISPATCH`)  
**Reference:** `catboost/private/libs/algo/score_calcers.cpp`, `CosineScoreCalcer`

The CPU Cosine gain formula (from `score_calcers.cpp`):
```
cosNum = sum_partitions [ gL^2/(wL+lambda) + gR^2/(wR+lambda) ]
cosDen = sum_partitions [ gL^2 * wL/(wL+lambda)^2 + gR^2 * wR/(wR+lambda)^2 ] + 1e-20
gain   = cosNum / sqrt(cosDen)
```

MLX's gain = ~0.946 * CPU gain. The 5.4% deficit is consistent across all splits and
depths. Likely causes:
1. **lambda (L2 regularization)** applied differently — missing from denominator or
   applied to wrong term.
2. **wL/wR** (sum of Hessians per partition side) accumulated incorrectly.
3. **Multi-partition accumulation** — CPU sums cosNum/cosDen across all active
   partitions for SymmetricTree; MLX may be missing some partitions.

The Sprint 30 K4 fp64 fix (DEC-036 K4) narrowed this residual to ~5%, but did not
eliminate it. The accumulation divergence is structural.

---

## File:Line Pointers

| Component | File | Search Label |
|-----------|------|--------------|
| FindBestSplit (MLX) | `catboost/mlx/tests/csv_train.cpp` | `S28-OBLIV-DISPATCH` |
| Cosine score (CPU ref) | `catboost/private/libs/algo/score_calcers.cpp` | `CosineScoreCalcer` |
| Border selection fix | `catboost/mlx/tests/csv_train.cpp` | `GreedyLogSumBestSplit`, `maxBordersCount` |
| Audit data | `docs/sprint31/t3b-audit/data/` | `cpu_splits_seed{42,43,44}.json`, `mlx_splits_seed*.json` |

---

## G1 Gate Status

G1 criterion: *divergence localized with mechanism class named.*

- First diverging layer: depth=0 (seeds 42, 44) / depth=2 (seed 43)
- Mechanism: GAIN-FORMULA (Cosine score ~5.4% lower in MLX)
- Co-fix shipped: DEC-037 border count (maxBins, not maxBins-1)

**G1: PASS**

---

## Next Steps (S32)

1. Instrument `FindBestSplit` to dump `cosNum`, `cosDen`, `wL`, `wR`, `gL`, `gR`
   per partition per bin at depth=0 for seed=42.
2. Compare against CPU instrumentation (add analogous dump to `cpu_dump.py`).
3. Identify the exact term causing the 5.4% deficit.
4. Fix and verify: gain ratio must be 1.000 ± 1e-4 at depth=0 for all 3 seeds.
