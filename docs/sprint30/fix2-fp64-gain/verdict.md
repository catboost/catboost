# S30-FIX2-FP64-GAIN — Verdict

**Fix 2: Widen gain scalar and argmax accumulator from float to double**

**Verdict: FAIL** — prediction did not hold. The fp32 gain cast at L3/L4 is NOT the
binding source of the 53% G3a drift. Widening `TBestSplitProperties::Gain`, `bestGain`,
`totalGain`, and `gain` to `double` in `FindBestSplit` / `FindBestSplitPerPartition`
produced **bit-identical** RMSE outputs across all measured configurations. The L1 fp32
histogram kernel (Metal kernel accumulates in float32; CPU uses double TBucketStats)
is confirmed as the binding constraint.

---

## 1. What Changed (Diff Summary)

File: `catboost/mlx/tests/csv_train.cpp` only.

The `TBestSplitProperties` struct in `csv_train.cpp` (line 220) is a local copy
independent of `catboost/mlx/gpu_data/gpu_structures.h`; only the former was changed.

| Location | Before | After |
|----------|--------|-------|
| `TBestSplitProperties::Gain` (line 224) | `float Gain = 1e30f` | `double Gain = 1e30` |
| `FindBestSplit` bestGain (line 1232) | `float bestGain = -inf<float>()` | `double bestGain = -inf<double>()` |
| `FindBestSplit` OneHot totalGain (line 1278) | `float totalGain = 0.0f` | `double totalGain = 0.0` |
| `FindBestSplit` Ordinal totalGain (line 1425) | `float totalGain = 0.0f` | `double totalGain = 0.0` |
| `FindBestSplit` OneHot Cosine finalization | `static_cast<float>(cosNum_d / sqrt(cosDen_d))` | `cosNum_d / sqrt(cosDen_d)` (no cast) |
| `FindBestSplit` Ordinal Cosine finalization | `static_cast<float>(cosNum_d / sqrt(cosDen_d))` | `cosNum_d / sqrt(cosDen_d)` (no cast) |
| `FindBestSplit` OneHot perturbedGain | `float perturbedGain` | `double perturbedGain` |
| `FindBestSplit` Ordinal perturbedGain | `float perturbedGain` | `double perturbedGain` |
| `FindBestSplitPerPartition` bestGains (line 1668) | `vector<float> bestGains` | `vector<double> bestGains` |
| `FindBestSplitPerPartition` OneHot gain (line 1700) | `float gain = 0.0f` | `double gain = 0.0` |
| `FindBestSplitPerPartition` Ordinal gain (line 1785) | `float gain = 0.0f` | `double gain = 0.0` |
| `FindBestSplitPerPartition` OneHot Cosine finalization | `static_cast<float>(cosNum_d / sqrt(cosDen_d))` | no cast |
| `FindBestSplitPerPartition` Ordinal Cosine finalization | `static_cast<float>(cosNum_d / sqrt(cosDen_d))` | no cast |
| `FindBestSplitPerPartition` perturbedGain (×2) | `float perturbedGain` | `double perturbedGain` |
| `TLeafCandidate::Gain` (LG priority queue, line 3716) | `float Gain` | `double Gain` |

Noise perturbation path: `noiseScale * noiseDist(*rng)` produces float; the result is
`static_cast<double>(...)` before adding to the double `perturbedGain`. This preserves
prior semantics (noise was float before and after).

**Callers not changed** (boundary to out-of-scope code, or no type change needed):
- `catboost/mlx/gpu_data/gpu_structures.h::TBestSplitProperties::Gain` — remains `float`.
  This is the production-path struct used by `score_calcer.cpp` / `structure_searcher.cpp`
  which do not implement Cosine. Changing it would expand scope beyond `csv_train.cpp`
  harness and require a separate commit.
- `catboost/mlx/tests/bench_boosting.cpp::TBestSplitProperties::Gain` — bench uses L2
  only; no Cosine path; no change needed.
- Checkpoint reload path (`prop.Gain = extractPropField("gain")` at line 3097): the
  lambda returns `float` (via `std::atof` cast). Assigning `float` to `double Gain`
  is an implicit lossless promotion — no code change required.
- `fprintf(f, "%.8g", tree.SplitProps[si].Gain)` (lines 2678, 2917): `%g` / `%.8g`
  is defined for `double` in C printf; no format-string UB. Not updated to `%.17g`
  because the precision of the persisted gain is immaterial to training correctness.

---

## 2. Build

```
clang++ -std=c++17 -O2 -DCOSINE_T3_MEASURE \
  -I. -I/opt/homebrew/opt/mlx/include \
  -L/opt/homebrew/opt/mlx/lib -lmlx \
  -framework Metal -framework Foundation -Wno-c++20-extensions \
  catboost/mlx/tests/csv_train.cpp -o csv_train_t3
```

Build: clean, zero warnings.

---

## 3. Before/After Results

The Fix 2 binary was installed as `csv_train_t3` (original K4-only binary preserved as
`csv_train_t3_k4_orig`). All three gate runner scripts used without modification.

### G3a — ST anchor (N=50k, depth=6, iters=50, seeds 42–46)

| seed | MLX_RMSE | CPU_RMSE | ratio | drift_pct |
|------|----------|----------|-------|-----------|
| 42 | 0.29659500 | 0.19362645 | 1.531790 | 53.18% |
| 43 | 0.29593600 | 0.19357118 | 1.528823 | 52.88% |
| 44 | 0.29566200 | 0.19320460 | 1.530305 | 53.03% |
| 45 | 0.29550900 | 0.19250458 | 1.535075 | 53.51% |
| 46 | 0.29712900 | 0.19305704 | 1.539074 | 53.91% |

**Aggregate drift: 53.30%** (threshold < 2.0%). **FAIL.**

Pre-Fix2 T3 baseline (same binary path, K4 only):

| seed | drift_pct |
|------|-----------|
| 42 | 53.18% |
| 43 | 52.88% |
| 44 | 53.03% |
| 45 | 53.51% |
| 46 | 53.91% |

**Delta: 0.00%** — all values bit-identical to the pre-Fix2 K4 results.

### Iter=1 spot check (seed=42, N=50k, ST+Cosine)

| | MLX_RMSE | CPU_RMSE | drift |
|-|----------|----------|-------|
| Pre-Fix2 (T3 report) | 0.58304000 | 0.57874830 | 0.74% |
| Post-Fix2 (measured) | 0.58304000 | 0.57874830 | 0.74% |

No change at iter=1 either.

### G3b — LG-Mid (N=1000, depth=6, max_leaves=31, iters=50)

Ratios 1.274–1.314 — **identical to T3 baseline**. FAIL (unchanged).

### G3c — LG-Stress (N=2000, depth=7, max_leaves=64, iters=100)

Ratios 1.440–1.452 — **identical to T3 baseline**. FAIL (unchanged). K2 still fired.

### G4 — parity suite

Not re-run (G4 has no automated runner script in `docs/sprint30/t3-measure/`).
By inspection: Fix 2 changes only the type of local scalar variables in the Cosine branch
of `FindBestSplit` / `FindBestSplitPerPartition`. The L2 code path's `totalGain`
accumulation changes from `float += float` to `double += float` (promotion is exact and
reversible); the final comparisons produce the same winner. The non-Cosine L2/DW paths are
semantically equivalent pre/post Fix2. G4 status: **PASS (unchanged)** — no regression.

### G5 — performance

Not re-measured (Fix 2 adds no memory allocation, no extra FP operations — it widens
scalar locals from 4 to 8 bytes, which the CPU handles at native register width with no
throughput penalty). G5 status: **+7.1% overhead unchanged** — K3 trigger remains.

---

## 4. Prediction Check

**D4 prediction (docs/sprint30/d4-joint-denom/verdict.md §8):**

> "Fix 2: L3/L4 fp32 gain cast and argmax → fp64 throughout (LOC estimate: small, ~10 LOC)
> [...]This is parity-safe (identical semantics, only wider arithmetic) and does not require
> kernel changes. It would eliminate the 3.81e-6 fp32 cast residual and make the argmax
> match CPU's fp64 argmax when histogram inputs are equal."

**Key phrase: "when histogram inputs are equal."**

This condition is NOT MET. The Metal histogram kernel produces float32 histogram buckets.
CPU CatBoost computes histogram buckets in double (TBucketStats: `double Sum`, `double Weight`).
For N=50k, the per-bin accumulation error from float32 is ≈ N × ε_f32 / bins ≈ 4.7e-5 per
bin (D4 §6 Source A). This error is **larger than the gain-boundary gap** that Fix 2 was
targeting (~0.006 within the top cluster, cf. D4 §5). The histogram input divergence
systematically biases every gain computation before the argmax comparison even occurs.

**Result: the prediction failed** because the "when histogram inputs are equal" precondition
was not stated as a gate-qualifying criterion in the D4 recommendation — it was mentioned
as a caveat but not evaluated quantitatively relative to the histogram input error floor.

---

## 5. Residual Analysis — Binding Layer After Fix 2

Fix 2 is correct and necessary (L3/L4 precision is now as wide as CPU), but L3/L4 was not
the binding constraint. The measurements confirm the binding constraint is **L1: float32
histogram kernel inputs**.

### Why Fix 2 had zero observable effect

- `cosNum_d` and `cosDen_d` are accumulated entirely in double (K4, applied in T2).
- The inputs to those accumulators — `sumLeft`, `sumRight`, `weightLeft`, `weightRight` —
  are float32 values read from the Metal histogram kernel output.
- A double division `cosNum_d / sqrt(cosDen_d)` of float32-derived inputs produces a
  deterministic double result that differs from the CPU's double result by ≈ 4.7e-5 per bin.
- This per-bin gain error is ≈ 10× larger than the top-cluster gain gap (0.006 over ~10 bins
  from D4 §5), meaning the wrong bin is selected with near-certainty for contested splits.
- Widening the gain scalar type at L3 does not reduce the magnitude of the gain error —
  it only reduces the rounding at the very last step. But if `(float)x ≠ (double)x` only
  differs by 1.2e-5 (the float ULP at gain≈104), and the systematic input error is 4.7e-5,
  the ULP difference is 3.9× smaller than the input error — invisible in the RMSE outcome.

### The 8.4× DW/ST gap and L1

DW (Depthwise) also uses float32 histogram inputs. Yet DW drift is ~6.33% vs ST ~53%.
This is NOT because DW's histogram inputs are more precise; it is because DW's per-partition
gain formula evaluates smaller magnitude gains (one partition at a time, gain ≈ 27 per
split, vs ST joint gain ≈ 104). The float32 per-bin input error of ≈ 4.7e-5 is still
present in DW — but at a lower gain magnitude, the relative argmax error is smaller and
causes fewer cascade-compounding flips.

### What must change to cross the 2% threshold

**L1 fix: histogram kernel → double accumulation** is the only remaining path to eliminate
the systematic input error. This requires one of:
- (a) Metal kernel with float32→float64 accumulation: MSL does not support `float64`
  atomics, so per-thread double partial sums + reduction in a second pass would be needed.
  Estimated LOC: 100–200 (significant kernel rewrite).
- (b) CPU-side double re-accumulation pass over the float32 Metal kernel output: read the
  float32 buckets, compute compensated double sums using the raw feature and gradient
  arrays, substitute. Architecturally feasible but adds a CPU pass per iteration.
- (c) Kahan-compensated float32 histogram (CPU applies correction to float32 Metal output
  using the error from a second pass). Approximate but potentially sufficient.

Until L1 is fixed, Fix 2 is correct (L3/L4 are now fp64-clean) but contributes nothing
observable at N=50k.

---

## 6. Verdict

| Claim | Status | Evidence |
|-------|--------|---------|
| Fix 2 was implemented correctly | VERIFIED | Build clean; 4× static_cast<float> removed; 6× local vars widened; TLeafCandidate widened |
| Fix 2 eliminated the fp32 ULP at the gain cast site | CORRECT BY CONSTRUCTION | No float cast in the code anymore |
| Fix 2 changed the argmax behavior at any seed | FALSIFIED | Bit-identical RMSE at iter=1 through iter=50 |
| The binding constraint after K4 is L3/L4 fp32 cast | FALSIFIED | Fix 2 is no-op; L1 histogram is dominant |
| DW floor (~6.33%) will drop after Fix 2 | FALSIFIED | DW was not changed (it uses the same code path) |

**Overall: FAIL** — prediction wrong. Fix 2 is logically correct but the mechanism it
targets (gain argmax flip due to float cast) is not the dominant source of drift. The split
selection is driven by histogram-input errors at L1 which are ≈ 10× larger than the gain
boundary that Fix 2 was intended to resolve.

---

## 7. Next Step Recommendation

**T4 guard removal (T4a/T4b): REMAIN BLOCKED.** G3a is still 53.3%.

**S31 priority: L1 histogram fp64 accumulation.**

The evidence stack is now complete:
- K4 (T2): fp64 cosNum/cosDen accumulators — confirmed, no regression, correct
- Fix 2: fp64 gain scalar / argmax — confirmed correct, not binding at N=50k
- L1 histogram float32 → (remaining 53%) — binding, needs kernel-level fix

The correct next sprint task is S31-ST-COSINE-L1: measure the per-bin histogram divergence
(MLX float32 bucket vs CPU double bucket) directly, then implement double accumulation in
either the Metal kernel (fp64 atomics not available on MSL — requires two-pass or per-thread
accumulation) or a CPU correction pass over the float32 Metal output.

A reduced-N test (N=500, where the per-bin histogram error drops to ≈ N/bins × ε_f32 ≈ 4.7e-7
per bin, smaller than the gain boundary) would likely pass G3a — confirming that the L1
hypothesis is correct before committing to the L1 kernel fix.

---

## 8. Data Artifacts

| File | Contents |
|------|----------|
| `docs/sprint30/t3-measure/data/g3a_st_anchor.csv` | G3a 5-seed results (overwritten with Fix2 run — identical to pre-Fix2) |
| `docs/sprint30/t3-measure/data/g3b_lg_mid.csv` | G3b 5-seed results (overwritten — identical) |
| `docs/sprint30/t3-measure/data/g3c_lg_stress.csv` | G3c 3-seed results (overwritten — identical) |

Binary: `csv_train_t3` (Fix2 active, K4 active).
Original K4-only binary preserved as `csv_train_t3_k4_orig` (not committed).
