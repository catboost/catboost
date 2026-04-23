# Sprint 26 D0 — Pre-Sprint Triage: Python-Path Leaf-Magnitude Regression

**Date**: 2026-04-22
**Branch**: `mlx/sprint-26-python-parity`
**Sprint framing**: correctness-first (option α). Depthwise/Lossguide benchmark harness
deferred to a later sprint pending resolution of this gap.

## Problem statement

MLX Python subprocess path (`csv_train` binary) produces systematically shrunken
predictions vs CPU CatBoost reference at identical hyperparameters. Empirical shrinkage
factor ≈ 0.69× on prediction standard deviation. Trees have correct split structure
(Pearson correlation with CPU predictions = 0.9664) but leaf values are too small in
magnitude. Compounded over 50 iterations, this manifests as RMSE 68% above CPU.

Pattern reproducible across:
- Grow policies (SymmetricTree 68% / Depthwise 561% / Lossguide 598% delta — latter two
  compound additional BFS-specific bugs)
- Dataset sizes (N ∈ {1k, 10k, 50k})
- Random seeds ({1337, 42, 7, 99})

## Why bench_boosting did not catch this

From `catboost/mlx/tests/bench_boosting.cpp:899`:

> `bench_boosting uses its own standalone pipeline, not RunBoosting/structure_searcher.`
> `Fine-grained depth-level stages 3/4/5 are only available via the real RunBoosting`
> `path (csv_train compiled with -DCATBOOST_MLX_STAGE_PROFILE).`

bench_boosting exercises histogram + score kernels directly. It does NOT exercise:
- `csv_train` CLI → `TrainOptions` dispatch
- `RunBoosting()` orchestration in `catboost/mlx/train_lib/`
- `structure_searcher.cpp` BFS tree build
- Leaf estimation in `catboost/mlx/methods/leaves/`
- Model serialization → predict round trip

v5's ULP=0 parity record (S24 D0) is valid for histogram kernel output only. It does
NOT transitively guarantee parity on any code path that `structure_searcher.cpp` or
leaf estimation touches.

## Empirical evidence

All artifacts under `benchmarks/sprint26/d0/`. Full raw output in
`benchmarks/sprint26/d0/RESULTS.md`.

### Summary table (SymmetricTree, N=10k, d=6, 128 bins, LR=0.03, 50 iters)

| Metric | CPU | MLX | Ratio |
|---|---:|---:|---:|
| final train RMSE | 0.2010 | 0.3381 | 1.68× |
| predict-based RMSE | 0.2010 | 0.3381 | 1.68× (agrees with train) |
| prediction std | 0.4220 | 0.2887 | **0.686×** |
| prediction mean | +0.0001 | +0.0003 | bias ≈ 0 |
| Pearson(CPU_pred, MLX_pred) | — | 0.9664 | shape correct |

### Rule-outs (see `bootstrap.py`, `dissect.py`)

| Candidate | Ruled out by |
|---|---|
| predict-time scaling bug | `train_loss_history[-1] == predict-based RMSE` exactly |
| CPU Bayesian vs MLX no-bootstrap mismatch | CPU with `bootstrap_type=No` still converges to 0.2031 |
| grow-policy-specific | all 3 policies affected (SymmetricTree 68%) |
| seed sensitivity | 61–72% delta across 4 seeds at N=10k |
| parameter-default offset | delta grows with N; constant offset wouldn't |

## Analytic hypothesis

Observation: prediction-std shrinkage = 0.686 after 50 iterations of gradient boosting
at LR=0.03.

Model: if MLX leaf values are uniformly scaled by factor α vs correct, the effective
learning rate is α·LR. Since GBDT fits residuals iteratively, predictions asymptote to
`(1 − (1 − α·LR)^K) · true_signal`.

Solving the observed 0.686 ratio:
```
(1 − (1 − α·0.03)^50) / (1 − 0.97^50) = 0.686
                       → α ≈ 0.51
```

**MLX leaves appear to be computed at roughly half the correct magnitude.** "Half" is
suggestive. Top candidates:

1. **RMSE hessian treated as 2 instead of 1** in Newton denominator
   `leaf = −Σgrad / (Σhess + l2)` → if hess=2, denom doubles → leaf halves
2. **RMSE gradient using `pred − y` (correct) but hessian compensation missing a factor**
   at the point of accumulation into leaf-stat buffer
3. **Leaf value halved by double-application of a 0.5 damping factor** somewhere in the
   Newton step (less likely)
4. **Learning rate applied on half the tree** in some BFS/depthwise traversal (plausibly
   why Depthwise is 8×+ worse — stacking bugs)

## Code zones to investigate

Primary suspects:
- `catboost/mlx/methods/leaves/` — leaf value computation (Newton step formula)
- `catboost/mlx/targets/` — RMSE gradient/hessian definitions
- `catboost/mlx/train_lib/train.cpp` — LR composition, base prediction, initialization
- `catboost/mlx/tools/csv_train/` — CLI flag propagation (LR, L2, iterations, depth)

CPU reference (correct implementation):
- `catboost/private/libs/algo/approx_calcer.cpp` — Newton step reference
- `catboost/private/libs/algo/approx_calcer_multi.cpp` — multi-target version
- `catboost/private/libs/algo_helpers/error_functions.h` — RMSE/MSE target definitions
- `catboost/private/libs/options/` — default parameter semantics

## D0 triage plan

Per Ramos directive (2026-04-22):
- **No DEC record opened yet** — wait until root cause is identified before formalizing
- **Sprint scope expands if multiple bugs surface** — do not backlog secondary issues
- **Delegate code read to @ml-engineer** — triage report only, NOT a code fix

Phase plan:

| Phase | Owner | Deliverable | Exit |
|---|---|---|---|
| D0-1 | @ml-engineer | MLX vs CPU leaf-estimation algebra diff + file:line table | Written report |
| D0-2 | @ml-engineer | MLX vs CPU RMSE target definitions + file:line table | Written report |
| D0-3 | @ml-engineer | Proposed instrumentation plan (Σgrad, Σhess, l2, leaf@depth-0) | Written plan |
| D0-4 | TBD based on D0-1/2/3 | Root cause identified and fix landed | Fix commit |
| D0-5 | @qa-engineer | Python-path parity sweep 18-config | MLX RMSE within 2% of CPU |

## Acceptance criteria (sprint exit gates)

- **G0**: Root cause identified and documented in `DECISIONS.md` as DEC-028 (reserved)
- **G1**: Python-path parity — 18-config RMSE delta ≤ 2% vs CPU at SymmetricTree
- **G2**: Python-path parity — Depthwise and Lossguide each within 5% of CPU (may not be
  as tight as SymmetricTree due to DEC-024 readback differences)
- **G3**: Regression test added: `tests/test_python_path_parity.py` (new) catches the
  bug class if reintroduced
- **G4**: No regression in bench_boosting ULP=0 record (v5 kernel output unchanged)

## Standing orders

- DEC-012 one-structural-change-per-commit
- No `Co-Authored-By: Claude` trailer
- RR-AMATOK fork only
- Honest parity — 2% RMSE tolerance, not ULP=0 (CSV round-trip is not bit-exact)
- Ultrathink on the algebra — scaling bugs hide in non-obvious places
