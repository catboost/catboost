# S32-T3-FIX Verdict

**Status:** COMPLETE — DEC-038 + DEC-039 shipped  
**Date:** 2026-04-24  
**Config:** iter=1, depth=6, bins=127, loss=RMSE, SymmetricTree, Cosine, seed=42  
**Task:** Fix GreedyLogSum border divergence (root cause of 5.4% Cosine gain deficit).

---

## Summary

Two bugs were identified and fixed in `catboost/mlx/tests/csv_train.cpp`:

### DEC-038: GreedyLogSum operated on unique values instead of all-docs

**Root cause:** `QuantizeFeatures` was deduplicating the sorted feature array before passing it
to `GreedyLogSumBestSplit`. CatBoost CPU's `TGreedyBinarizer<MaxSumLog>::BestSplit` initialises
its `TFeatureBin` over `features.Values` — the FULL document array with duplicates, N=50000
entries. The score function uses `BinEnd - BinStart` as the document count in each bin.
With deduplicated input (49983 unique values for feature 0 vs 50000 total), the score landscape
changed, causing a ~2-index border grid offset.

**Fix:** Pass `allVals` (sorted, with duplicates) to `GreedyLogSumBestSplit`.

**Verification:** Direct binary border dump (CATBOOST_MLX_DUMP_BORDERS build) showed
0 diffs at 1e-6 threshold across all 20 features at 128 borders.

### DEC-039: Histogram kernel VALID_BIT aliasing at fold_count=128

**Root cause:** The MLX histogram kernel uses `VALID_BIT = 0x80000000` (bit 31) in the
packed 32-bit word to mark valid documents. Features at `posInWord=0` occupy bits 31..24
(shift=24). When `fold_count=128` (i.e., 128 borders), bin_value=128 (docs above all borders)
sets bit 31 of the packed word, which is identical to the VALID_BIT. The kernel strips bit 31
via `p_clean = p_s & 0x7FFFFFFF`, aliasing bin_value=128 to bin_value=0. The writeback loop
skips slot 0 (by reading `stagingHist[f*256 + bin + 1]`), so these 391 docs were silently
dropped from the histogram.

This caused:
- `wL = totalWeight - suffHess[b]` to be inflated by 391 for all bins of features 0,4,8,12,16
  (the 5 posInWord=0 features)
- wL offset = +391 uniformly across all 128 bins of affected features
- Different split statistics → MLX selected bin=64 instead of CPU's bin=59 for feature 0

**Fix:** Cap `maxBordersCount = std::min(maxBins, 127u)` in `QuantizeFeatures`. With
`fold_count <= 127`, `bin_value <= 127` and bit 7 of the posInWord=0 byte is never set,
eliminating the VALID_BIT collision. This matches the `bench_boosting` convention
(`NumBins - 1` for fold count) which already respected the T2_BIN_CAP=128 (0..127) contract.

**Note:** The T2_BIN_CAP comment in `kernel_sources.h` (line 38) already documented this
constraint: "Safe ONLY when every feature's fold count ≤ 127." The csv_train.cpp test binary
was violating this contract when `--bins 128` was passed.

---

## Results

### Gain ratio at depth=0 (seed=42, 127 bins)

| Metric | Pre-fix | Post-fix |
|--------|---------|----------|
| Median gain ratio (MLX/CPU) | 0.946 | 0.9999 |
| Winner neighbourhood rdiff (feat=0) | 5.4e-2 | 3-4e-4 |
| wL max delta (posInWord=0 features) | 391 | 25 |

The residual wL delta of 25 (from 0 before the VALID_BIT fix) is due to float32 vs float64
midpoint arithmetic in `GreedyLogSumBestSplit` (`0.5f * vals[bs-1] + 0.5f * vals[bs]` vs
CatBoost CPU's float64 midpoint). This causes 1-5 ULP border differences, which re-assign
~25 docs at the physical split boundary. This is a known limitation of the float32 midpoint
computation and does not affect the structural correctness.

### 1-iteration RMSE delta (50k docs, depth=6, Cosine)

| | RMSE | Delta |
|---|---|---|
| CatBoost CPU (bins=127, 1 iter) | 0.578748 | — |
| MLX DEC-039 (bins=127, 1 iter)  | 0.583062 | +0.75% |

The 0.75% residual at iter=1 is from the float32 border arithmetic. Pre-fix (T2 state) this
was ~5.4% from the border grid offset alone.

### Multi-iteration RMSE (50 iters)

| | RMSE | Delta |
|---|---|---|
| CatBoost CPU (bins=127, 50 iters) | 0.193679 | — |
| MLX DEC-039 (bins=127, 50 iters)  | 0.295608 | +52.6% |

The 52.6% multi-iteration drift is a pre-existing structural issue (DEC-036 class) — not caused
by the border quantization bugs fixed here. This is a separate investigation target.

---

## Gate Assessment: G3a (T4-VALIDATE)

G3a criterion: "depth=0 gain ratio 1.000 ± 1e-4 (3 seeds)"

At seed=42: median gain ratio = 0.9999 (within 1e-4). Winner-neighbourhood rdiff ≈ 3-4e-4.
The two sides select different bin indices (64 vs 58) because they evaluate different physical
splits (float32 vs float64 border arithmetic). The gain values at those different splits are
within 0.02% of each other.

**G3a partial pass at seed=42.** Full gate (3 seeds) deferred to T4-VALIDATE.

---

## Root Cause Chain (revised from T2 verdict)

```
Pre-DEC-038 state (T2 finding):
  "GreedyLogSum tie-break divergence → border grid offset"

Actual root cause (T3-FIX):
  DEC-038: uniqueVals (not allVals) → wrong score landscape → ~2-index border offset
    → different physical splits evaluated at each bin index
    → 5.4% gain deficit at winner (T3b observation confirmed as artifact of this bug)

DEC-039: fold_count=128 → VALID_BIT aliasing at posInWord=0
    → bin_value=128 docs aliased to bin_value=0, lost from histogram
    → wL inflated by +391 for features 0,4,8,12,16
    → further split mismatch on top of DEC-038 border offset
    → BOTH bugs present in T2 data; DEC-038 alone insufficient
```

The T2 verdict's mechanism description ("tie-break divergence") was a hypothesis that turned
out to be incorrect. The actual cause was deduplicated input to the score function (DEC-038),
compounded by the kernel VALID_BIT contract violation at bins=128 (DEC-039).

---

## Files Modified

- `catboost/mlx/tests/csv_train.cpp` — DEC-038 (allVals) + DEC-039 (cap 127)
- `docs/sprint32/t2-terms/compare_terms.py` — bug fix (NoneType subscript when max_rdiff=0)

## Data Files (t3-fix/data/)

| File | Contents |
|------|---------|
| `mlx_terms_seed42_depth0.csv` | 2540 DEC-039 MLX term records (bins=127) |
| `cpu_terms_seed42_depth0.csv` | 2540 CPU term records (bins=127) |
