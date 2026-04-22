# S26 D0 — Pre-Sprint Triage Results

Captured 2026-04-22 before sprint kickoff. All scripts run from project root with activated conda env.

## smoke.py — CPU determinism + MLX feasibility

```
=== D0-b: CPU CatBoost Depthwise determinism (3x same seed) ===
  run0: loss=0.1950051039  hex=3e47af5f  t=0.340s
  run1: loss=0.1950051337  hex=3e47af6f  t=0.259s
  run2: loss=0.1950051337  hex=3e47af6f  t=0.268s
  deterministic (ULP=0): NO
  max(loss) - min(loss) = 2.980e-08

=== D0-c: CatBoost-MLX Depthwise feasibility (1x) ===
  run0: pred-based RMSE=1.2887209654  hex=3fa4f649  t=3.310s
  MLX vs CPU(run0) delta: 1.094e+00  (560.89%)
```

Note: CPU Depthwise shows 1-ULP drift (max-min = 2.98e-08). Not our concern — well within
float32 noise. What matters: MLX deviates by 561% at the same config.

## localize.py — 3-policy sanity

```
policy         | CPU RMSE          | MLX pred RMSE     | delta       | delta %   | MLX time(s)
----------------------------------------------------------------------------------------------------
SymmetricTree  | 0.2010103464  hex=3e4dd5ac | 0.3381237686  hex=3ead1c6d | 1.371e-01 |   68.20% | 3.30
Depthwise      | 0.1950051337  hex=3e47af6d | 1.2887209654  hex=3fa4f649 | 1.094e+00 |  560.89% | 3.31
Lossguide      | 0.1970180720  hex=3e49bd83 | 1.3754267693  hex=3fb00e4b | 1.178e+00 |  598.15% | 3.15
```

All 3 policies show divergence. SymmetricTree "only" 68% is because it has simpler
structure; Depthwise/Lossguide compound additional bugs likely related to DEC-024
readback overhead or BFS tree build.

## seedsweep.py — SymmetricTree × {seeds, sizes}

```
      N  seed     CPU RMSE     MLX RMSE      delta   delta%
--------------------------------------------------------------
   1000  1337     0.238306     0.344937  1.066e-01   44.75%
   1000    42     0.245491     0.341913  9.642e-02   39.28%
   1000     7     0.238478     0.337522  9.904e-02   41.53%
   1000    99     0.230945     0.344098  1.132e-01   49.00%
  10000  1337     0.201010     0.338107  1.371e-01   68.20%
  10000    42     0.202941     0.327965  1.250e-01   61.61%
  10000     7     0.200976     0.325011  1.240e-01   61.72%
  10000    99     0.201393     0.333352  1.320e-01   65.52%
  50000  1337     0.196418     0.337933  1.415e-01   72.05%
  50000    42     0.196025     0.325938  1.299e-01   66.27%
  50000     7     0.195946     0.324842  1.289e-01   65.78%
  50000    99     0.195953     0.330250  1.343e-01   68.54%
```

MLX RMSE clamps to ~0.33 regardless of N or seed. CPU improves from 0.24 → 0.20 → 0.196.
Delta % grows because CPU improves; MLX is stuck. Pattern: **not seed-sensitive, not
N-sensitive — a systematic magnitude bug**.

## dissect.py — train vs predict localization

```
y stats: mean=0.0001 std=0.5919 min=-2.8174 max=2.3295

=== CPU CatBoost SymmetricTree ===
  train loss history: [0]=0.5772 [10]=0.4525 [25]=0.3208 [-1]=0.2010
  preds stats: mean=0.0001 std=0.4220 min=-1.1459 max=1.1135
  pred-based RMSE: 0.2010

=== MLX CatBoost SymmetricTree ===
  train loss history length: 50
  train loss: [0]=0.5825 [10]=0.4915 [25]=0.4067 [-1]=0.3381
  preds stats: mean=0.0003 std=0.2887 min=-0.6193 max=0.5826
  pred-based RMSE: 0.3381

=== Prediction correlation (CPU vs MLX) ===
  Pearson(CPU_pred, MLX_pred): 0.9664
  Pearson(CPU_pred, y):        0.9769
  Pearson(MLX_pred, y):        0.9345
  ratio MLX_std / CPU_std:     0.6841
  bias (MLX_mean - CPU_mean):  0.0002
```

Smoking gun. MLX training DOES decrease monotonically (it's not stuck at iter 0); train
loss history equals predict() RMSE (0.3381 == 0.3381) so predict is faithful; trees
learn the right shape (Pearson 0.9664); predictions are shrunken by 0.68× on std; no
mean bias. **Leaf-magnitude bug.**

## bootstrap.py — rule out bootstrap mismatch

```
CPU bootstrap_type=Bayesian   final RMSE=0.2025  pred std=0.4204
CPU bootstrap_type=No         final RMSE=0.2031  pred std=0.4206
```

CPU with `bootstrap_type=No` (matching MLX wrapper's default) converges identically to
Bayesian default. **Bootstrap mismatch is NOT the cause.**

## Verdict

Leaf-magnitude bug. Trees structurally correct; leaf values ~0.69× too small on
prediction std; compounded over 50 iters this is consistent with effective LR ≈ LR/2
(leaf values computed at roughly half-magnitude — classic candidate: RMSE hessian=2
vs hess=1 confusion in Newton denominator).

Bug lives in MLX backend code touched by production Python path but bypassed by
`bench_boosting.cpp:899`'s standalone pipeline. v5's ULP=0 parity record does NOT
transitively cover this path.
