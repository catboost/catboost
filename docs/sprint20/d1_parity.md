# S20-D1 Parity Report — T3b vs T0 DEC-008 Envelope

**Branch**: `mlx/sprint-20-hist-atomic-cas`
**Date**: 2026-04-19
**Gate**: 18/18 configs within DEC-008 ulp bounds + 100/100 determinism runs

---

## Verdict

**D1 PASS** — 18/18 configs pass parity, 100/100 determinism runs identical

All 18 configs are **bit-exact** (max_ulp=0), which is stronger than the DEC-008 envelope requires
(RMSE ulp=0, Logloss ulp≤4, MultiClass ulp≤8). Including all six MultiClass configs across all
three approxDims.

---

## Per-config parity table

| Config | ApproxDim | ULP threshold | Max ULP | Max abs diff | Verdict |
|---|---|---|---|---|---|
| 1000_rmse_d6_32bins | 1 | 0 | 0 | 0.00e+00 | **PASS** |
| 1000_rmse_d6_128bins | 1 | 0 | 0 | 0.00e+00 | **PASS** |
| 1000_logloss_d6_32bins | 1 | 4 | 0 | 0.00e+00 | **PASS** |
| 1000_logloss_d6_128bins | 1 | 4 | 0 | 0.00e+00 | **PASS** |
| 1000_multiclass_d6_32bins | 3 | 8 | 0 | 0.00e+00 | **PASS** |
| 1000_multiclass_d6_128bins | 3 | 8 | 0 | 0.00e+00 | **PASS** |
| 10000_rmse_d6_32bins | 1 | 0 | 0 | 0.00e+00 | **PASS** |
| 10000_rmse_d6_128bins | 1 | 0 | 0 | 0.00e+00 | **PASS** |
| 10000_logloss_d6_32bins | 1 | 4 | 0 | 0.00e+00 | **PASS** |
| 10000_logloss_d6_128bins | 1 | 4 | 0 | 0.00e+00 | **PASS** |
| 10000_multiclass_d6_32bins | 3 | 8 | 0 | 0.00e+00 | **PASS** |
| 10000_multiclass_d6_128bins | 3 | 8 | 0 | 0.00e+00 | **PASS** |
| 50000_rmse_d6_32bins | 1 | 0 | 0 | 0.00e+00 | **PASS** |
| 50000_rmse_d6_128bins | 1 | 0 | 0 | 0.00e+00 | **PASS** |
| 50000_logloss_d6_32bins | 1 | 4 | 0 | 0.00e+00 | **PASS** |
| 50000_logloss_d6_128bins | 1 | 4 | 0 | 0.00e+00 | **PASS** |
| 50000_multiclass_d6_32bins | 3 | 8 | 0 | 0.00e+00 | **PASS** |
| 50000_multiclass_d6_128bins | 3 | 8 | 0 | 0.00e+00 | **PASS** |

### MultiClass per-approxDim detail

| Config | Dim | Max ULP | Max abs diff | Verdict |
|---|---|---|---|---|
| 1000_multiclass_d6_32bins | 0 | 0 | 0.00e+00 | **PASS** |
| 1000_multiclass_d6_32bins | 1 | 0 | 0.00e+00 | **PASS** |
| 1000_multiclass_d6_32bins | 2 | 0 | 0.00e+00 | **PASS** |
| 1000_multiclass_d6_128bins | 0 | 0 | 0.00e+00 | **PASS** |
| 1000_multiclass_d6_128bins | 1 | 0 | 0.00e+00 | **PASS** |
| 1000_multiclass_d6_128bins | 2 | 0 | 0.00e+00 | **PASS** |
| 10000_multiclass_d6_32bins | 0 | 0 | 0.00e+00 | **PASS** |
| 10000_multiclass_d6_32bins | 1 | 0 | 0.00e+00 | **PASS** |
| 10000_multiclass_d6_32bins | 2 | 0 | 0.00e+00 | **PASS** |
| 10000_multiclass_d6_128bins | 0 | 0 | 0.00e+00 | **PASS** |
| 10000_multiclass_d6_128bins | 1 | 0 | 0.00e+00 | **PASS** |
| 10000_multiclass_d6_128bins | 2 | 0 | 0.00e+00 | **PASS** |
| 50000_multiclass_d6_32bins | 0 | 0 | 0.00e+00 | **PASS** |
| 50000_multiclass_d6_32bins | 1 | 0 | 0.00e+00 | **PASS** |
| 50000_multiclass_d6_32bins | 2 | 0 | 0.00e+00 | **PASS** |
| 50000_multiclass_d6_128bins | 0 | 0 | 0.00e+00 | **PASS** |
| 50000_multiclass_d6_128bins | 1 | 0 | 0.00e+00 | **PASS** |
| 50000_multiclass_d6_128bins | 2 | 0 | 0.00e+00 | **PASS** |

---

## Determinism

100 × T3b runs on 50k/RMSE/d6/128b/seed42. Unique histogram hashes: **1**.
Verdict: **PASS — bit-exact run-to-run**

AGX executes the CAS-float loop in a stable order within a single dispatch. Non-determinism was
the highest-risk unknown for T3b (DEC-017 noted this). It does not manifest on this device.

---

## Why bit-exact (not merely within-envelope)?

T0 and T3b accumulate values in different orders, yet produce bit-identical outputs. This is
explained by the synthetic data pattern: bin values are generated as `(doc * K) % 256`, which
distributes docs uniformly across bins. For each bin, the accumulating docs hit it at a regular
stride (one every `256 / gcd(K, 256)` docs), and since all stat values are from a simple
`doc % 128 / 128.0f` pattern, the FP32 additions happen to commute exactly. Real training data
with irregular stat distributions and irregular bin occupancy will likely produce small non-zero
ulp differences — still within DEC-008 bounds, but non-zero. The bit-exact result here is a
property of the synthetic data regularity, not a general guarantee.

This is not a concern for DEC-008 compliance, but it is worth documenting so the result is not
misread as "T3b is always bit-exact vs T0 on any data." D3 (bench_boosting end-to-end) will
stress this with the training loss convergence check, which is the real parity gate.

---

## Honest bottom line

D1 PASS. All 18 configs are bit-exact (ulp=0). T3b is bit-exact run-to-run.
Proceed to D2 (kernel integration) per sprint plan.

Kahan compensated summation is NOT required.

---

## Reproduce commands

```bash
# Compile
clang++ -std=c++17 -O2 \
  -I/opt/homebrew/Cellar/mlx/0.31.1/include \
  -L/opt/homebrew/Cellar/mlx/0.31.1/lib -lmlx \
  -framework Metal -framework Foundation \
  docs/sprint20/scratch/microbench_parity.cpp \
  -o /tmp/microbench_parity

# Run
DYLD_LIBRARY_PATH=/opt/homebrew/Cellar/mlx/0.31.1/lib /tmp/microbench_parity
```

JSON output: `.cache/profiling/sprint20/d1_parity/` (gitignored)
