# S30-D2-REDUX — Corrected L3/L4 Instrumentation Verdict

**Verdict: L3 RULED OUT, L4 RULED OUT (with honest methodology).**

D2's original "0 / 18 argmax flips" and "3.81e-6 gain residual" conclusions survive the V2
methodology correction.  With a genuine pre-K4 fp32 shadow gain path, L3 residuals are 5–13x
larger than D2's biased cast-ULP floor — but still below the split-selection flip threshold at
every depth level across all 18 decisions.  K4's accumulation-widening fix is confirmed to have
eliminated the only meaningful source of pre-K4 gain path error, and that error was never
large enough to flip a split at this cell.

**The 53% trajectory drift does NOT originate from L3 (gain-cast quantization) or L4 (split
argmax instability).  S31 should target L0 (histogram kernel output divergence).**

---

## Methodology Change (V2 Audit Correction)

### What D2 did (biased)

```cpp
// csv_train.cpp (pre-D2-redux, BIASED):
rec.gain_f32 = totalGain;  // = float(cosNum_d / sqrt(cosDen_d))
rec.gain_f64 = cosNum_d / std::sqrt(cosDen_d);
```

Both fields were derived from the same fp64 accumulators (`cosNum_d`, `cosDen_d`).
The residual `|gain_f32 - gain_f64|` was literally `|float(x) - x|` — the fp32 cast ULP of a
single fp64 value, approximately `gain × eps32 / 2 ≈ 3.81e-6` at gain ≈ 80.  This was
seed-independent and measured nothing about the pre-K4 fp32 accumulation path.

### What D2-redux does (corrected)

```cpp
// csv_train.cpp (D2-redux, CORRECTED — behind COSINE_RESIDUAL_INSTRUMENT):
float gain_f32_path = static_cast<float>(
    cosNum_f32_shadow / std::sqrtf(cosDen_f32_shadow));   // true pre-K4 fp32 path
double gain_f64_ref = cosNum_d / std::sqrt(cosDen_d);     // true fp64 path
rec.gain_f32 = gain_f32_path;
rec.gain_f64 = gain_f64_ref;
```

`cosNum_f32_shadow` and `cosDen_f32_shadow` are the parallel float32 accumulators that have
been maintained since T2 — they accumulate the same per-term expressions as `cosNum_d` /
`cosDen_d` but in float32 arithmetic, never touching the double values.  The division and
`sqrtf` are also float.  This is the code path that ran before K4.

The residual `|gain_f32_path - gain_f64_ref|` now measures fp32 accumulation path divergence
vs the fp64 reference — the quantity V2 required.

Gate: change is compile-time-gated behind `COSINE_RESIDUAL_INSTRUMENT` (implied by
`-DCOSINE_D2_INSTRUMENT`).  Release builds are unaffected.

---

## Cell Parameters (identical to D2)

| Parameter | Value |
|-----------|-------|
| N | 50,000 |
| Features | 20 (S26 canonical: f0×0.5 + f1×0.3 + noise×0.1) |
| Depth | 6 |
| max_leaves | 64 (2^6, SymmetricTree) |
| Bins (max_bin) | 128 |
| Learning rate | 0.03 |
| L2 reg | 3.0 |
| Loss | RMSE |
| score_function | Cosine |
| grow_policy | SymmetricTree |
| Iterations | 1 (iter=0 internally) |
| Seeds | 42, 43, 44 |
| Binary | `csv_train_d2_redux` (-DCOSINE_D2_INSTRUMENT) |

---

## M1 — L3 Gain Residual (CORRECTED)

`|gain_f32_path - gain_f64|` where `gain_f32_path = float(cosNum_f32_shadow / sqrtf(cosDen_f32_shadow))`.

| seed | depth | n | max residual | mean residual | p99 residual |
|------|-------|---|-------------|---------------|--------------|
| 42 | 0 | 2540 | 1.19e-05 | 2.75e-07 | 5.41e-06 |
| 42 | 1 | 2540 | 1.70e-05 | 3.67e-06 | 1.18e-05 |
| 42 | 2 | 2540 | 2.07e-05 | 4.73e-06 | 1.55e-05 |
| 42 | 3 | 2540 | 2.80e-05 | 5.79e-06 | 1.87e-05 |
| 42 | 4 | 2540 | 2.96e-05 | 5.82e-06 | 1.87e-05 |
| 42 | 5 | 2540 | 2.51e-05 | 5.93e-06 | 1.96e-05 |
| 43 | 0 | 2540 | 1.17e-05 | 2.69e-07 | 5.50e-06 |
| 43 | 1 | 2540 | 1.60e-05 | 3.67e-06 | 1.13e-05 |
| 43 | 2 | 2540 | 2.02e-05 | 4.61e-06 | 1.46e-05 |
| 43 | 3 | 2540 | 2.09e-05 | 4.83e-06 | 1.55e-05 |
| 43 | 4 | 2540 | 1.69e-05 | 4.60e-06 | 1.39e-05 |
| 43 | 5 | 2540 | **5.03e-05** | 9.14e-06 | 3.16e-05 |
| 44 | 0 | 2540 | 1.02e-05 | 2.61e-07 | 5.10e-06 |
| 44 | 1 | 2540 | 1.61e-05 | 3.39e-06 | 1.10e-05 |
| 44 | 2 | 2540 | 1.94e-05 | 4.62e-06 | 1.46e-05 |
| 44 | 3 | 2540 | 2.30e-05 | 5.82e-06 | 1.82e-05 |
| 44 | 4 | 2540 | 4.08e-05 | 7.81e-06 | 2.56e-05 |
| 44 | 5 | 2540 | 3.72e-05 | 7.73e-06 | 2.62e-05 |

**Global M1 max: 5.03e-05 (seed=43, depth=5)**
**Global M1 mean-of-means: 4.61e-06**

Comparison with D2's biased measurement: D2 reported a uniform 3.81e-6 across all 18 cells
(seed-independent, depth-independent), which was the fingerprint of a cast-ULP floor rather
than signal.  D2-redux shows:
- The max is now **5.03e-05** — 13.2x larger than D2's floor.
- The values are **seed-variable and depth-variable** as expected for real accumulation error
  (deeper partitions aggregate fewer docs per bin, changing the cancellation pattern).
- Depth-0 (single partition, full 50k docs) shows lower residual (~1.1e-05) consistent with
  better fp32 summation over a single large uniform partition.  Deeper levels (partitions have
  ~1500 docs average at depth 5 with 32 partitions) show higher residual (~5e-05).

The V2 prediction of "1e-3 to 1e-4" did not materialize: T1's ~4e-3 cosNum/cosDen residuals
do NOT propagate to a proportional gain error because the gain formula is a ratio that partially
cancels correlated errors in cosNum and cosDen.  The actual gain residuals are in the ~1e-5
range, roughly 2 orders of magnitude below the V2 prediction.

---

## M2 — L5 Leaf-Value Residual (unchanged methodology)

| seed | n_leaves | gSum_max | gSum_mean | leafVal_max | leafVal_mean |
|------|----------|----------|-----------|-------------|--------------|
| 42 | 64 | 9.94e-04 | 1.01e-04 | 1.22e-08 | 2.25e-09 |
| 43 | 64 | 3.33e-03 | 1.86e-04 | 2.29e-08 | 2.96e-09 |
| 44 | 64 | 5.31e-03 | 2.80e-04 | 2.69e-08 | 2.74e-09 |

**Global M2 max leafVal residual: 2.69e-08** (consistent with D2's 4.79e-08; minor seed
variation from different partition assignments).

---

## M3 — L4 Argmax Flip Count (CORRECTED — now meaningful)

`gain_f32` column is now the true pre-K4 fp32 path.  Argmax flip = pre-K4 fp32 path and fp64
path disagree on which (feat, bin) pair is the winning split at this depth level.

| seed | depth | feat_f32 | bin_f32 | feat_fp64 | bin_fp64 | flipped |
|------|-------|----------|---------|-----------|----------|---------|
| 42 | 0 | 0 | 58 | 0 | 58 | no |
| 42 | 1 | 1 | 58 | 1 | 58 | no |
| 42 | 2 | 19 | 37 | 19 | 37 | no |
| 42 | 3 | 19 | 34 | 19 | 34 | no |
| 42 | 4 | 5 | 112 | 5 | 112 | no |
| 42 | 5 | 5 | 69 | 5 | 69 | no |
| 43 | 0 | 0 | 63 | 0 | 63 | no |
| 43 | 1 | 1 | 68 | 1 | 68 | no |
| 43 | 2 | 12 | 1 | 12 | 1 | no |
| 43 | 3 | 12 | 1 | 12 | 1 | no |
| 43 | 4 | 12 | 44 | 12 | 44 | no |
| 43 | 5 | 5 | 15 | 5 | 15 | no |
| 44 | 0 | 0 | 61 | 0 | 61 | no |
| 44 | 1 | 1 | 69 | 1 | 69 | no |
| 44 | 2 | 11 | 68 | 11 | 68 | no |
| 44 | 3 | 11 | 63 | 11 | 63 | no |
| 44 | 4 | 2 | 22 | 2 | 22 | no |
| 44 | 5 | 2 | 21 | 2 | 21 | no |

**Flip count: 0 / 18 decisions (rate = 0.0%)**

This is now a genuine measurement.  gain_f32 at seed=43 depth=5 has max residual 5.03e-05.
The winning split at that depth still agrees between the pre-K4 fp32 path and the fp64 path.
This confirms the top-vs-runner-up gain margin exceeds 5.03e-05 at every decision point — the
fp32 accumulation error is real but sub-threshold for split selection on this canonical cell.

---

## Layer Verdict Table

| Layer | D2 methodology | D2-redux methodology | D2 max | D2-redux max | Verdict |
|-------|---------------|---------------------|--------|-------------|---------|
| L1 | cosNum_f32_shadow vs cosNum_d | unchanged | ~4.07e-3 (T1) | ~4.22e-3 (cos_accum CSVs) | RULED IN for K4 motivation; K4 suppressed it |
| L2 | cosDen_f32_shadow vs cosDen_d | unchanged | ~4.07e-3 (T1) | ~4.13e-3 (cos_accum CSVs) | same |
| L3 | float(cosNum_d/sqrt(cosDen_d)) − fp64 | float(cosNum_f32_shadow/sqrtf(cosDen_f32_shadow)) − fp64 | 3.81e-6 (cast ULP) | **5.03e-05** (real path error) | **RULED OUT** — residual real but sub-flip |
| L4 | argmax(cast-ULP gain) | argmax(true fp32 path gain) | 0/18 (trivial) | **0/18 (genuine)** | **RULED OUT** — no split flips |
| L5 | fp64 scatter + Newton | unchanged | 4.79e-8 | 2.69e-8 | RULED OUT |

---

## Why 0 Flips Despite ~5e-05 Gain Error

At depth 5 (the worst case), the maximum gain residual across all 2540 candidates is 5.03e-05.
This means the pre-K4 fp32 path assigned gain values that differed from the fp64 values by at
most 5.03e-05.  A flip would require the top-two candidates to be within 5.03e-05 of each
other.  On this canonical cell (50k docs, 20 features, depth 6, SymmetricTree with `randomStrength=0`),
the winning margin appears to be larger than this at all 18 depth levels.

This does NOT mean L3 is harmless at all cells or all configs:
- Smaller N (fewer docs per partition) increases per-term fp32 error and reduces top-two margins.
- Different feature sets with near-tied splits could expose flips.
- This measurement covers exactly one iter-1 pass; iter-50 compound effects are not modeled here.

---

## Implications for S31

1. **L3 and L4 are RULED OUT as the primary source of the 53% trajectory drift** at the
   canonical cell.  D2's original conclusion survives with corrected methodology.

2. **The drift source remains unlocated.**  L1/L2 error (~4e-3 pre-K4) was the largest raw
   signal; K4 suppressed it.  All downstream layers (L3–L5) are sub-threshold for split
   selection error.  The remaining candidate is **L0: histogram kernel output divergence**.

3. **S31 should target histogram instrumentation.**  Dump `(sumGrad, sumHess)` per bin per
   feature from the Metal kernel at iter=1 and compare against a CPU double-precision
   scatter-add reference.  If Metal's histogram bins differ from the CPU reference — even by
   a few float ULPs — split thresholds will differ before the gain formula runs, and the
   cascade will compound across 50 iterations.

4. **K4's value is confirmed but narrowly scoped.**  K4 (T2) fixed a real ~4e-3 accumulation
   error in the gain computation, but that error was never the dominant source of the 53% drift.
   K4 is correct to keep (it makes the gain formula more precise) but it did not address the
   root cause.

---

## Limitations

1. **Single-config coverage.** All 18 cells use the identical canonical config (ST, Cosine,
   50k, depth 6, 128 bins).  Other configs (different N, different grow policies, LossGuide)
   may show different results.  In particular, cells with smaller N or tight near-tied splits
   could exhibit L4 flips even on the corrected path.

2. **Iter-1 only.**  The instrumentation runs at iter=0 internal (the first tree).  Cascade
   effects beyond iter=1 are not measured here; D2-redux does not rule out interaction between
   the ~5e-05 gain error and later-iteration drift.

3. **Pre-K4 fp32 path, not current MLX kernel path.**  The `cosNum_f32_shadow` /
   `cosDen_f32_shadow` accumulators mirror what the code did before K4 in scalar C++.  The
   Metal GPU kernel may have different rounding behaviour (thread-reduction order, SIMD
   shuffle) — those errors are in L0, not L3.

4. **No absolute reference.**  Gain residuals are measured against the K4 fp64 path, not
   against a CPU CatBoost reference.  If K4 itself has residual error (unlikely but not ruled
   out), this measurement would not capture it.

---

## Artifacts

| File pattern | Contents |
|---|---|
| `data/gain_scalar_seedN_depthD.csv` | feat_idx, bin, gain_f32 (true fp32 path), gain_f64, gain_abs_residual |
| `data/cos_accum_seedN_depthD.csv` | Full cosNum/cosDen/gain f32+f64+residual |
| `data/leaf_sum_seedN.csv` | gSum/hSum/leafVal f32 vs f64 per leaf |
| `data/argmax_flip_seedN.csv` | per depth: chosen split f32 vs fp64, flipped flag |

Runner: `docs/sprint30/d2-redux/run_d2_redux.py`

Build command:
```
clang++ -std=c++17 -O2 -DCOSINE_D2_INSTRUMENT \
  -I. -I/opt/homebrew/opt/mlx/include \
  -L/opt/homebrew/opt/mlx/lib -lmlx \
  -framework Metal -framework Foundation -Wno-c++20-extensions \
  catboost/mlx/tests/csv_train.cpp -o csv_train_d2_redux
```

Code change: `catboost/mlx/tests/csv_train.cpp` lines 1523–1548 (ordinal branch instrument
block) — `rec.gain_f32` now uses `cosNum_f32_shadow / sqrtf(cosDen_f32_shadow)` instead of
`totalGain`.  Stale comment at line 135 (`TBinRecord::gain_f32`) also updated.  Release builds
unaffected (both behind `COSINE_RESIDUAL_INSTRUMENT`).
