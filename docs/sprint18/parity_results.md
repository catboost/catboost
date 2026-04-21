# Sprint 18 Parity Results (S18-04b — re-run on fixed kernel)

**Verdict: PASS** — S18-G3 hard merge gate CLEARED.

Reference binary: `bench_boosting_s17` (D1c `simd_shuffle_xor` + 8-term cross-SIMD fold, kernel `ed0ec8221b`, Sprint 17 final).
Subject binary: `bench_boosting_s18_fixed` (L1a cooperative-broadcast accumulation, BUG-S18-001 fixed, kernel commit `19fa5ce6cc`).
Both binaries compiled from `catboost/mlx/tests/bench_boosting.cpp` (identical data path) with only `kernel_sources.h` differing.
Each config: 50 iterations, depth 6, lr 0.10, l2 3.0, 50 features, seed 42.
Checkpoints reported: iter ∈ {0, 10, 20, 30, 40, 49} (bench_boosting reports iter=0, every 10, and final).

## Prior run (S18-04, commit `abc4c229f9`)

**FAIL** — BUG-S18-001: intra-SIMD butterfly read shared memory slot from all 32 lanes (same address), producing 32× amplification. All 18 configs failed with 4–20 million ULP. See `docs/sprint18/parity_results_s18_04_fail.md` for the original FAIL report and root-cause analysis.

## Fixed kernel (commit `19fa5ce6cc`)

Fix: replaced broken per-SIMD accumulation with cooperative 32-doc batch loop using `simd_shuffle` broadcast. Lane `src`'s packed/stat broadcast to all 32 lanes; only the bin-owner lane writes (`bin & 31 == lane`). Every doc contributes exactly once with zero atomics. Intra-SIMD D1c butterfly deleted entirely — `simdHist[g][bin]` is already a full per-SIMD-group sum. Only the cross-SIMD 8-term linear fold (DEC-009) remains.

## Max ULP per loss type

| loss       | max ulp | tolerance | status |
|:-----------|--------:|----------:|:-------|
| RMSE       |       0 |         4 | PASS   |
| Logloss    |       0 |         4 | PASS   |
| MultiClass |       0 |         8 | PASS   |

## Full 18-config table (18 × 6 checkpoints)

| N      | loss       | bins | ref_loss   | sub_loss   | max_ulp | final_ulp | tol | status |
|-------:|:-----------|-----:|-----------:|-----------:|--------:|----------:|----:|:-------|
|   1000 | RMSE       |   32 | 0.44764100 | 0.44764100 |       0 |         0 |   4 | PASS   |
|   1000 | RMSE       |  128 | 0.46951200 | 0.46951200 |       0 |         0 |   4 | PASS   |
|   1000 | Logloss    |   32 | 0.34807300 | 0.34807300 |       0 |         0 |   4 | PASS   |
|   1000 | Logloss    |  128 | 0.62872700 | 0.62872700 |       0 |         0 |   4 | PASS   |
|   1000 | MultiClass |   32 | 0.59497100 | 0.59497100 |       0 |         0 |   8 | PASS   |
|   1000 | MultiClass |  128 | 0.98377200 | 0.98377200 |       0 |         0 |   8 | PASS   |
|  10000 | RMSE       |   32 | 0.45874700 | 0.45874700 |       0 |         0 |   4 | PASS   |
|  10000 | RMSE       |  128 | 0.48016100 | 0.48016100 |       0 |         0 |   4 | PASS   |
|  10000 | Logloss    |   32 | 0.29658200 | 0.29658200 |       0 |         0 |   4 | PASS   |
|  10000 | Logloss    |  128 | 0.60814900 | 0.60814900 |       0 |         0 |   4 | PASS   |
|  10000 | MultiClass |   32 | 0.57257200 | 0.57257200 |       0 |         0 |   8 | PASS   |
|  10000 | MultiClass |  128 | 0.96393000 | 0.96393000 |       0 |         0 |   8 | PASS   |
|  50000 | RMSE       |   32 | 0.45872100 | 0.45872100 |       0 |         0 |   4 | PASS   |
|  50000 | RMSE       |  128 | 0.48047800 | 0.48047800 |       0 |         0 |   4 | PASS   |
|  50000 | Logloss    |   32 | 0.29455800 | 0.29455800 |       0 |         0 |   4 | PASS   |
|  50000 | Logloss    |  128 | 0.60131500 | 0.60131500 |       0 |         0 |   4 | PASS   |
|  50000 | MultiClass |   32 | 0.56760600 | 0.56760600 |       0 |         0 |   8 | PASS   |
|  50000 | MultiClass |  128 | 0.94424900 | 0.94424900 |       0 |         0 |   8 | PASS   |

**18 / 18 configs PASS. 0 / 18 configs FAIL.**

## Drift analysis

No non-zero ULP at any of the 108 measured checkpoints (18 configs × 6 checkpoints each). The fix restores bit-exactness between the L1a kernel and the Sprint 17 D1c reference at every checkpoint.

| N | loss | bins | iter=0 | iter=10 | iter=20 | iter=30 | iter=40 | iter=49 |
|--:|:-----|-----:|-------:|--------:|--------:|--------:|--------:|--------:|
| (all 18 configs) | — | — | 0 | 0 | 0 | 0 | 0 | 0 |

The Sprint 17 transient 17-ULP spike at iter=10 (N=10k, MultiClass, 32 bins) does not appear in this run. This is expected: both binaries use `bench_boosting` with identical data-synthesis code, whereas the Sprint 17 transient arose from a specific tree geometry in the `csv_train` data path that happened to expose FP32 cross-SIMD fold associativity at that iteration.

## Determinism fixture

100-run determinism check: gate config (N=10000, RMSE, bins=128, depth=6, 50 iter, seed=42).

**PASS — 100/100 runs bit-exact at gate config.**

Reference final loss (all 100 runs): `BENCH_FINAL_LOSS=0.48016092`

The cooperative-broadcast accumulation pattern (stride-partition, single writer per bin) is fully deterministic: each bin slot has exactly one writing lane per SIMD group per batch pass. The cross-SIMD fold runs in fixed `simd_id=0..7` order (DEC-009). No non-deterministic ordering source exists in the fixed kernel.

## Gate summary

| Gate | Criterion | Result |
|------|-----------|--------|
| S18-G3 (RMSE ulp ≤ 4) | max ulp = 0 | **PASS** |
| S18-G3 (Logloss ulp ≤ 4) | max ulp = 0 | **PASS** |
| S18-G3 (MultiClass ulp ≤ 8) | max ulp = 0 | **PASS** |
| Determinism fixture (100 runs, bit-exact) | 100/100 PASS | **PASS** |

**S18-G3 hard merge gate: CLEARED. Kernel commit `19fa5ce6cc` is merge-ready.**

## Methodology note

The Sprint 17 parity report (`docs/sprint17/parity_results.md`) used `csv_train_sprint16` (reference) vs `csv_train_profiled` (subject) on CSV files, producing the reference loss values in that report (e.g. RMSE N=10k: 0.496241). This S18-04b re-run uses `bench_boosting` binaries with synthetic data. The numeric reference values differ because `bench_boosting` uses a different data-synthesis path than `csv_train` + CSV files. The parity comparison remains valid: both reference and subject use the identical `bench_boosting.cpp` data path (same seed, same generator), differing only in `kernel_sources.h`. The 0-ULP result confirms the fixed kernel is numerically identical to the D1c reference on the full 18-config DEC-008 grid.
