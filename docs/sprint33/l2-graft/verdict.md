# S33-L2-GRAFT Verdict

**Date:** 2026-04-24
**Branch:** `mlx/sprint-33-iter2-scaffold`
**Tip at measurement:** `bceaff4655`
**Task:** #121 S33-L2-GRAFT
**Anchor:** N=50000, grow_policy=SymmetricTree, score_function=Cosine, loss=RMSE, depth=6,
bins=128 (DEC-039 cap: 127 borders), iter=50, seeds={42,43,44}
**Determinism stack:** bootstrap_type=No, random_strength=0.0, has_time=True, random_seed=0
**Binary:** `csv_train_t4` (built 2026-04-24 17:55; DEC-038 + DEC-039 fixes active)
**kernel_sources.h md5:** `9edaef45b99b9db3e2717da93800e76f` — VERIFIED (v5, unchanged)

---

## Graft Mechanism

**Option (b) — manual approx-init via the existing snapshot mechanism.**

Rationale: `csv_train_t4` already has a complete snapshot save/restore path (SaveSnapshot /
LoadSnapshot, JSON format). Resuming from a snapshot sets `startIteration = snap.Iteration + 1`,
restoring the cursor (per-doc accumulated predictions), the trees list, and the RNG state.
No new C++ code required; no guarded env-var patch needed.

Steps:
1. Run CPU CatBoost iter=1 → capture per-doc `RawFormulaVal` predictions (float32).
2. Run MLX iter=1 with `--snapshot-path` → capture the MLX snapshot (tree structure,
   RNG state, iteration=0, num_trees=1).
3. Overwrite `train_cursor` in the MLX snapshot with the CPU iter=1 predictions.
   All other fields (tree structure, RNG state) preserved from the MLX snapshot so the
   snapshot validity check (`snap.Trees.size() == num_trees && !snap.TrainCursor.empty()`)
   passes cleanly.
4. Resume MLX from the grafted snapshot with `--iterations 50` → runs iterations 1..49
   (49 MLX iterations from the CPU starting point).

The graft is clean: bootstrap is disabled and colsample=1.0, so the RNG state in the
snapshot is never consumed by the training loop in iterations 1..49. The cursor drives
gradient computation; the trees list only accumulates for model serialization.

---

## Sanity Check

Seed=42. Grafted cursor RMSE vs CPU iter=1 RMSE.

| Quantity | Value |
|----------|-------|
| CPU iter=1 RMSE | 0.5787483020 |
| RMSE(cpu_preds array, y) | 0.5787482858 |
| RMSE(graft JSON cursor, y) | 0.5787482858 |
| Sanity drift | −0.000003% |
| Verdict | **PASS** |

The 3e-6% discrepancy is pure float32 serialization rounding (%.10g format). The graft
mechanism is valid; the L2 result is not contaminated by the graft procedure itself.

---

## Drift Table

| seed | CPU iter=50 | MLX iter=50 (L1) | MLX iter=50 grafted | drift_grafted% | drift_L1% | frame |
|------|-------------|------------------|---------------------|----------------|-----------|-------|
| 42   | 0.19362645  | 0.29562600        | 0.29299000          | +51.317%       | +52.679%  | FRAME-B |
| 43   | 0.19357118  | 0.29512200        | 0.29257700          | +51.147%       | +52.462%  | FRAME-B |
| 44   | 0.19320460  | 0.29491300        | 0.29230100          | +51.291%       | +52.643%  | FRAME-B |
| **Median** | | | | **+51.291%** | **+52.643%** | **FRAME-B** |

**Ratio (grafted median / ungrafted median):** 51.291 / 52.643 = **0.974**

For Frame-A the expected ratio would be ≤ 0.20 (80% collapse). For Frame-B the expected
ratio is ≥ 0.75. Observed ratio 0.974 is far into the Frame-B territory.

---

## Class Call

**FRAME-B**

Forcing MLX to start from a bit-exact CPU iter=1 starting point (cursor = CPU iter=1
predictions, sanity drift 3e-6%) reduces the final iter=50 drift by only 1.35 percentage
points (52.643% → 51.291%), a 2.6% relative reduction. This is within run-to-run noise.

The iter=1 epsilon is **not** the root cause. There is a **per-iter persistent mechanism**
that re-injects divergence regardless of the starting point. Every MLX iteration from 2..50
adds its own increment of divergence, independent of where iteration 1 landed.

**Frame A is falsified.** The trajectory cascade hypothesis (tiny iter=1 ε amplified via
greedy argmax flips) would require drift to collapse to ≤ 10% when the iter=1 state is
CPU-exact. The observed 51.3% rules this out conclusively across all three seeds.

---

## Decision

**Next task: #122 L3-ITER2** — per-iteration instrumentation.

The per-iter persistent bug must be localized. L3 strategy: run with shared CPU history for
the first N iterations, dump per-partition split candidates, leaf values, and cursor deltas at
iteration N+1. Compare MLX vs CPU at each stage:
- Gradient computation (cursor → gradients)
- Split selection (FindBestSplit: gains, chosen feature/bin)
- Leaf value estimation (CalcLeafValues: sumGrad, sumHess, lambda)
- Approximation update (cursor += lr * leaf_value[doc_leaf])

One of these four stages has a systematic error that fires at every iteration. The gain
computation discrepancy found in S31 (T3b: GAIN-FORMULA, ~5.4% low in MLX at iter=1) is a
strong prior — the cosNum/cosDen terms have known precision issues (DEC-036, K4+Fix2 shipped
but insufficient). L3 should start with iter=2 split-selection dump.

**Do NOT claim #122; leave pending for orchestrator.**

---

## Data Files

| File | Contents |
|------|----------|
| `data/sanity_check_seed42.txt` | Sanity: graft cursor drift vs CPU iter=1 (-0.000003%) |
| `data/cpu_iter1_state_seed42.json` | CPU iter=1 RMSE + prediction stats, seed=42 |
| `data/cpu_iter1_state_seed43.json` | CPU iter=1 RMSE + prediction stats, seed=43 |
| `data/cpu_iter1_state_seed44.json` | CPU iter=1 RMSE + prediction stats, seed=44 |
| `data/graft_snapshot_seed42.json` | MLX snapshot with CPU iter=1 cursor, seed=42 |
| `data/graft_snapshot_seed43.json` | MLX snapshot with CPU iter=1 cursor, seed=43 |
| `data/graft_snapshot_seed44.json` | MLX snapshot with CPU iter=1 cursor, seed=44 |
| `data/mlx_grafted_rmse_seed42.txt` | MLX grafted iter=50 RMSE, seed=42: 0.29299000 |
| `data/mlx_grafted_rmse_seed43.txt` | MLX grafted iter=50 RMSE, seed=43: 0.29257700 |
| `data/mlx_grafted_rmse_seed44.txt` | MLX grafted iter=50 RMSE, seed=44: 0.29230100 |
| `data/l2_drift_summary.csv` | Full matrix: all seeds, CPU/L1/grafted RMSE, drifts, frame call |
| `run_l2_graft.py` | Harness script |
