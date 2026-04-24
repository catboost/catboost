# S30-D2-FULL-STACK-INSTRUMENT — Verdict

**Q2: L3 is NOT dominant — M3 flip rate 0/18 (0.0%); the 3.81e-6 gain residual never changes the winning split; the 53% trajectory drift has a different origin than gain-cast quantization.**

---

## Cell Parameters (audit continuity with T1/T2/T3)

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
| Binary | `csv_train_d2` (-DCOSINE_D2_INSTRUMENT, implies -DCOSINE_RESIDUAL_INSTRUMENT) |

---

## M1 — Gain Scalar Residual Post-Cast (L3)

Post K4: gain is computed as `float(cosNum_d / sqrt(cosDen_d))`. The residual is the float32 quantization of this cast.

| seed | depth | max gain residual | mean gain residual | p99 gain residual |
|------|-------|-------------------|--------------------|-------------------|
| 42 | 0 | 3.808e-6 | 1.37e-7 | 2.60e-6 |
| 42 | 1–5 | 3.814e-6 | 1.89e-6 | 3.78e-6 |
| 43 | 0 | 3.780e-6 | 1.35e-7 | 2.73e-6 |
| 43 | 1–5 | 3.814e-6 | 1.89e-6 | 3.78e-6 |
| 44 | 0 | 3.783e-6 | 1.40e-7 | 2.74e-6 |
| 44 | 1–5 | 3.814e-6 | 1.89e-6 | 3.78e-6 |

**Global M1 max: 3.814e-6** (consistent across all depths and seeds after depth 0).

This is the irreducible fp32 quantization of the gain scalar — approximately `gain × eps32 / 2` where `eps32 ≈ 1.19e-7` and gain ≈ 80. The uniformity across all seed-depth pairs (all tightly at 3.81e-6) confirms this is not variable cancellation error but the fp32 ULP floor of the cast itself.

---

## M2 — Leaf-Value Sum Residual (L5)

gSum is the per-leaf scatter_add of float32 gradients. leafVal is the Newton step: `-lr × gSum / (hSum + l2)`.

| seed | n_leaves | gSum_max | gSum_mean | leafVal_max | leafVal_mean |
|------|----------|----------|-----------|-------------|--------------|
| 42 | 64 | 4.41e-3 | 2.17e-4 | 3.35e-8 | 3.52e-9 |
| 43 | 64 | 7.17e-3 | 2.72e-4 | 4.79e-8 | 3.82e-9 |
| 44 | 64 | 5.07e-3 | 2.29e-4 | 2.74e-8 | 2.82e-9 |

**Global M2 max leafVal residual: 4.79e-8.**

gSum absolute residuals are large (~7e-3 at seed=43) but attenuation through the Newton step (`lr / (hSum + l2) ≈ 0.03 / 783 ≈ 3.8e-5`) suppresses them to sub-1e-7 in leaf values. The L5 layer contributes a maximum of 4.8e-8 per leaf to the cursor, 3 orders below L3. Consistent with T1 measurement (3.96e-8 at seed=43).

---

## M3 — Argmax Flip Count

For each of the 6 depth levels per seed, the fp32 argmax (what MLX uses) and the fp64 argmax (shadow re-computation) are compared. A flip occurs when they select different (feat, bin) pairs.

| seed | depth | feat_mlx | bin_mlx | feat_fp64 | bin_fp64 | flipped |
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

**Flip count: 0 / 18 decisions across 3 seeds (rate = 0.0%).**

*Argmax shadow method*: The argmax is computed by post-processing the dumped `gain_scalar_seedN_depthD.csv` files, independently maximizing over `gain_f32` and `gain_f64` columns. This is exact because FindBestSplit (SymmetricTree with `randomStrength=0`) selects the pure gain argmax with no noise perturbation.

---

## Layer Ranking

| Rank | Layer | Location | Max residual (this cell) | Role in trajectory drift |
|------|-------|----------|--------------------------|--------------------------|
| 1 (largest raw) | L1/L2 | cosNum/cosDen accumulators | ~4.07e-3 (T1 pre-K4) | Suppressed to L3 floor by K4 |
| 2 | L4/gSum | Leaf gradient scatter_add | 7.17e-3 (absolute) | Newton-step-attenuated; ~4.8e-8 in leafVal |
| 3 | L3 | Gain scalar post-cast | 3.81e-6 | fp32 ULP of cast; **0 split flips** |
| 4 | L5 | leafVal Newton step | 4.79e-8 | Negligible |
| — | approxUpdate | mx::take + cursor add | 0.0 | Exact (integer-indexed lookup) |

**None of the measured layers are causing split-selection flips at iter=1 on this cell.**

---

## Hypothesis Test Results

| Hypothesis | Evidence | Verdict |
|------------|----------|---------|
| **L3-dominant**: gain-cast + argmax flip drives 53% drift | 0/18 argmax flips; gain_f32 and gain_f64 agree on all 18 split decisions | **FALSIFIED** |
| **L5-dominant**: leaf value errors drive trajectory compounding | leafVal max 4.79e-8; approxUpdate 0.0 | **FALSIFIED** |
| **Multi-layer**: several layers interact | All measured layers are below the flip threshold; the dominant source is not in the gain or leaf computation | **OPEN — source not yet located** |

---

## Why T2's 12.5x Gain Reduction Did Not Fix G3a

The T2 K4 fix eliminated cosDen/cosNum accumulation error (~4e-3 → ~0) and reduced the gain residual 12.5x (4.75e-5 → 3.81e-6). D2 now shows that even the *pre-K4* 4.75e-5 gain residual was not causing split flips on this cell — the winning candidate's margin over the second-best candidate was already larger than the residual at every depth. K4 reduced the residual below a flip threshold that was never active.

The 53% trajectory drift is not in `FindBestSplit` (ST path). The correct mechanism must be either:

1. **Histogram kernel quantization**: The fp32 histogram values (`sumLeft`, `weightLeft`, etc.) that feed into FindBestSplit carry quantization noise from the Metal reduction. If the histogram sums diverge from the CPU reference, split thresholds will be wrong before the gain formula even runs. This has not yet been instrumented.

2. **Partition update error**: After each split the partition assignment is recomputed. Differences in rounding of the feature column threshold comparison could accumulate across 6 depth levels. This is also un-instrumented.

3. **Gradient computation divergence**: The initial gradients (residuals = y - pred) are computed in fp32. If the base prediction or the first tree's contribution differs from CPU, every subsequent gradient is wrong. The iter=1 approxUpdate residual is 0 (exact lookup), but this does not rule out the *gradient values themselves* being off relative to CPU due to different leaf assignments.

4. **Cumulative approximation divergence across iterations**: At iter=1 there is only 1 tree; the 0.74% drift at iter=1 is small. The 53% at iter=50 comes from this 0.74% being amplified. Each wrong split propagates to wrong gradients in the next iteration; the cascade is arithmetically correct given the first tree's leaf assignments — but the first tree's leaf assignments differ from CPU.

The root cause is most likely **histogram divergence** (the fp32 histogram values differ from what CPU computes for the same split threshold, causing the winning split to be at a different threshold even if the gain formula is applied correctly). This requires S31 instrumentation of the histogram kernel output vs a CPU reference at iter=1.

---

## Next Step Recommendation

**S31 target: L0 — histogram kernel output residual.**

Instrument `computeHistograms` (Metal kernel) by dumping the per-bin `(sumGrad, sumHess)` histogram arrays at iter=1 for each feature and comparing against a CPU double-precision scatter-add reference. If any feature's histogram bins differ between MLX and CPU — even by a single float ULP — the split thresholds can differ, causing cascading partition errors independent of the gain formula.

Expected impact if the histogram is the source: fixing histogram precision (e.g., fp32→fp64 histogram accumulation, or atomic-safe reduction) should collapse the iter-1 drift and the 50-iter cascade simultaneously. This is a larger change than K4 but has clear mechanical justification given D2's null result on all downstream layers.

---

## Data Artifacts

All CSV files under `docs/sprint30/d2-stack-instrument/data/`:

| File pattern | Contents | Rows per file |
|---|---|---|
| `gain_scalar_seedN_depthD.csv` | feat_idx, bin, gain_f32, gain_f64, gain_abs_residual | 2540 |
| `leaf_sum_seedN.csv` | gSum/hSum/leafVal f32 vs f64 per leaf | 64 |
| `argmax_flip_seedN.csv` | per depth: chosen split MLX vs fp64 shadow, flipped flag | 6 |
| `cos_accum_seedN_depthD.csv` | Full cosNum/cosDen/gain f32+f64+residual | 2540 |

Runner/post-processor: `docs/sprint30/d2-stack-instrument/run_d2_instrument.py`
Build command:
```
clang++ -std=c++17 -O2 -DCOSINE_D2_INSTRUMENT \
  -I. -I/opt/homebrew/opt/mlx/include \
  -L/opt/homebrew/opt/mlx/lib -lmlx \
  -framework Metal -framework Foundation -Wno-c++20-extensions \
  catboost/mlx/tests/csv_train.cpp -o csv_train_d2
```
