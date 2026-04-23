# S26-D0-6 Post-Fix Residual Deltas: Depthwise / Lossguide

**Date**: 2026-04-22  
**Branch**: `mlx/sprint-26-python-parity`  
**After**: DEC-028 RandomStrength fix (S26-D0-6 commit)

## Summary

The DEC-028 fix correctly resolves the SymmetricTree noise formula. Depthwise and Lossguide
grow policies show large residual deltas that are **not caused by the RandomStrength bug** and
**not affected by the DEC-028 fix**.

## localize.py results (post-fix)

| policy        | CPU RMSE   | MLX RMSE   | delta %  |
|---------------|------------|------------|----------|
| SymmetricTree | 0.2010     | 0.1948     | 3.10%    |
| Depthwise     | 0.1950     | 1.2888     | 560.89%  |
| Lossguide     | 0.1970     | 1.3754     | 598.15%  |

## Why D0-6 does not fix Depthwise/Lossguide

- `FindBestSplit` (the function where the noise formula lived) is only called from the
  **SymmetricTree path** (`config.GrowPolicy == "SymmetricTree"`).
- Depthwise and Lossguide grow policies use `FindBestSplitPerPartition`, which has **no noise
  path** (RandomStrength has never applied to these policies in MLX — this is pre-existing
  behavior, not a regression introduced by D0-6).
- The Depthwise/Lossguide deltas of ~560-598% existed before this fix (they were visible in
  localize.py pre-fix as well; the noise formula was not their root cause).

## Root cause (Depthwise/Lossguide — not yet diagnosed)

Depthwise RMSE = 1.29 vs CPU 0.195. Lossguide RMSE = 1.38 vs CPU 0.197. Both are > 6× worse.
Possible causes (not yet investigated):
1. Leaf value computation for non-oblivious trees — different leaf-estimation path.
2. Partition assignment for multi-leaf trees — BFS node routing bug.
3. Model export of non-symmetric trees — `TNonSymmetricTreeModelBuilder` (ADR-006) untested.
4. Per-leaf histogram aggregation in `FindBestSplitPerPartition` — suffix-sum or partition
   index bug.

## Recommended follow-up

Open a new sprint diagnostic (S26-D1 or equivalent) targeting Depthwise/Lossguide parity.
Start with `localize.py` on a single-tree Depthwise run with `random_strength=0` to isolate
whether the gap is in tree structure search or leaf estimation.

## Files referenced

- `benchmarks/sprint26/d0/localize.py` — policy sweep driver
- `catboost/mlx/tests/csv_train.cpp` — `FindBestSplitPerPartition` (no noise; lines ~1285+)
- `catboost/mlx/tests/csv_train.cpp` — Depthwise path (lines ~3488-3530)
- `catboost/mlx/tests/csv_train.cpp` — Lossguide path (lines ~3179-3333)
