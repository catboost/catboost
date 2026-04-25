# S33-L1-DETERMINISM Verdict

**Date:** 2026-04-24
**Branch:** `mlx/sprint-33-iter2-scaffold`
**Tip at measurement:** `a356109ec6`
**Task:** #120 S33-L1-DETERMINISM
**Anchor:** N=50000, grow_policy=SymmetricTree, score_function=Cosine, loss=RMSE, depth=6,
bins=128 (DEC-039 cap: 127 borders), iter=50, seeds={42,43,44}
**Binary:** `csv_train_t4` (built 2026-04-24 17:55; DEC-038 + DEC-039 fixes active)
**kernel_sources.h md5:** `9edaef45b99b9db3e2717da93800e76f` — VERIFIED (v5, unchanged)

---

## Determinism Stack Applied

Both CPU and MLX sides ran with identical settings:

| Parameter | CPU value | MLX value | Source |
|-----------|-----------|-----------|--------|
| `bootstrap_type` | `No` | `--bootstrap-type no` | L0-confirmed NO-DIFF |
| `subsample` | 1.0 (implicit; CatBoost rejects explicit subsample when bootstrap=No) | `--subsample 1.0` | L0-confirmed NO-DIFF |
| `random_strength` | `0.0` | `--random-strength 0.0` | L0-confirmed NO-DIFF |
| `has_time` | `True` | N/A (MLX processes in file order — has_time=True equivalent) | NEW — only lever in L1 stack not confirmed at L0 |
| `sampling_unit` | `Object` (explicit) | N/A (default) | L0-confirmed NO-DIFF |
| `random_seed` | `seed` parameter (42/43/44) | `--seed seed` | L0-confirmed NO-DIFF |
| `l2_leaf_reg` | `3.0` | `--l2 3.0` | L0-confirmed NO-DIFF |
| `border_count` | `127` (max_bin=128) | `--bins 127` | L0-confirmed NO-DIFF |
| `score_function` | `Cosine` | `--score-function Cosine` | L0-confirmed NO-DIFF |
| `grow_policy` | `SymmetricTree` | `--grow-policy SymmetricTree` | L0-confirmed NO-DIFF |
| `iterations` | `50` | `--iterations 50` | L0-confirmed NO-DIFF |

Note: CatBoost raises `CatBoostError: you shouldn't provide bootstrap options if bootstrap is
disabled` when `subsample` is passed alongside `bootstrap_type='No'`. The `subsample=1.0`
default is implicit and identical to the MLX side's explicit `--subsample 1.0`.

The only genuinely new setting in L1 versus the S32 G3b baseline is `has_time=True` on the
CPU side. All other fields were already set to maximally deterministic values in the S32 G3b
measurement (rs=0, bootstrap=No). Setting `has_time=True` removes the fold permutation
randomness for Ordered boosting — but both sides are in Plain boosting mode where
`has_time` is structurally inactive (as documented in the L0 verdict).

---

## Results

### Per-seed RMSE table

| seed | CPU RMSE | MLX RMSE | drift% | vs S32 baseline |
|------|----------|----------|--------|-----------------|
| 42   | 0.19362645 | 0.29562600 | +52.679% | ~same (S32: 52.6%) |
| 43   | 0.19357118 | 0.29512200 | +52.462% | ~same |
| 44   | 0.19320460 | 0.29491300 | +52.643% | ~same |
| **Median** | | | **+52.643%** | |
| **Range** | | | 52.462%–52.679% | |

### Comparison against S32 baseline

| Config | Median drift | Notes |
|--------|-------------|-------|
| S32 G3b (seed=42 only, DEC-039 fix, rs=0, bootstrap=No) | ~52.6% | Single-seed measurement; bootstrap=No, rs=0 already active |
| L1 determinism stack (seeds 42/43/44, + has_time=True) | **52.643%** | No change |

The drift is within 0.18 percentage points of the S32 baseline across all three seeds. This
is well within run-to-run variation and is "roughly the same" — not a SOFT-PARTIAL reduction.

---

## Class Call

**FALSIFIED**

The iter=50 drift is 52.643% (median) under the maximally deterministic configuration. This
is statistically indistinguishable from the S32 baseline of 52.6%. K6 did not fire.

### Why no toggle ablation is needed

The `has_time=True` flag is the only new setting in L1 that was not already confirmed NO-DIFF
at L0. Since the drift is unchanged (~52.6% vs ~52.6%), `has_time=True` has zero effect. No
further ablation is warranted — the null result is unambiguous.

### Frame C-RNG status

**Frame C-RNG is fully falsified.** Combined with L0's Frame C-config falsification, Frame C
(Config/RNG mismatch) is **closed** as an explanation for the 52.6% drift. The drift is not
caused by bootstrap sampling, row subsampling, score perturbation, temporal ordering, or
any other stochastic configuration difference between the two sides.

---

## Decision

**Proceed to #121 L2-GRAFT.** DEC-036 remains OPEN; Frame C is eliminated. Frame A
(trajectory cascade) vs Frame B (per-iter persistent bug) is the remaining discrimination.

L2 GRAFT: inject the CPU iter=1 tree into MLX, run 49 more MLX iterations. If drift drops
≥80% → Frame A. If drift unchanged → Frame B.

---

## Data Files

| File | Contents |
|------|---------|
| `data/cpu_rmse_seed42.txt` | CPU RMSE seed=42: 0.1936264503 |
| `data/mlx_rmse_seed42.txt` | MLX RMSE seed=42: 0.2956260000 |
| `data/cpu_rmse_seed43.txt` | CPU RMSE seed=43: 0.1935711763 |
| `data/mlx_rmse_seed43.txt` | MLX RMSE seed=43: 0.2951220000 |
| `data/cpu_rmse_seed44.txt` | CPU RMSE seed=44: 0.1932046050 |
| `data/mlx_rmse_seed44.txt` | MLX RMSE seed=44: 0.2949130000 |
| `data/l1_drift_summary.csv` | All seeds, CPU/MLX RMSE, drift%, class |
| `run_l1_determinism.py` | Harness script |
