# DEC-042 Commit 2 — Four-Gate Parity Validation Report

**Date**: 2026-04-25
**Branch**: `mlx/sprint-33-iter2-scaffold`
**Fix commits validated**: `10c72b4e96` (Cosine per-side mask), `e98c6725cd` (L2 per-side mask)
**Kernel md5**: `9edaef45b99b9db3e2717da93800e76f` (unchanged throughout S33)
**Reference DEC**: DEC-042

---

## Executive Summary

All five DEC-042 formal gates PASS. The per-side mask fix (Commits 1 + 1.5) resolves
DEC-036 completely: ST+Cosine drift collapsed from **52.6% pre-fix to 0.027% post-fix** at
iter=50 — a 1941x reduction. Iter=1 drift is 0.0001% (effectively zero). The L2 parity
sweep shows no regression (18/18, ratios [0.9991, 1.0008]). The kernel is byte-identical to
v5 (ULP=0 confirmed). DW+Cosine sanity is intact (5/5 seeds within [0.98, 1.02], deltas
from S28 baseline are noise-level). Guard removal (Commit 3, #93/#94) is unblocked.

---

## Gate Results

| Gate | Criterion | Result | Key Numbers |
|------|-----------|--------|-------------|
| G4a | iter=1 ST+Cosine drift <= 0.1% | **PASS** | drift = 0.0001%, ratio = 0.999999 |
| G4b | iter=50 ST+Cosine drift <= 2% | **PASS** | drift = 0.027%, ratio = 1.000271 |
| G4c | v5 kernel ULP=0 (md5 invariant) | **PASS** | `0.48231599` = AN-009, ULP=0 |
| G4d | 18-config L2 parity [0.98, 1.02] | **PASS** | 18/18, range [0.9991, 1.0008] |
| G4e | DW+Cosine sanity at S28 anchor | **PASS** | 5/5 seeds [0.98, 1.02] |

---

## G4a: iter=1 ST+Cosine Drift

Criterion: drift <= 0.1%
Anchor: np.random.default_rng(42), N=50000, 20 features, ST/Cosine/RMSE, d=6, bins=128, l2=3, lr=0.03
Binary: csv_train_g4_cosine (built with -DCOSINE_T3_MEASURE to bypass guard for measurement)

  MLX RMSE (iter=1): 0.57874800
  CPU RMSE (iter=1): 0.57874830
  Ratio:             0.999999
  Drift:             0.0001%   (threshold: <=0.1%)

G4a: PASS

---

## G4b: iter=50 ST+Cosine Drift

Criterion: drift <= 2%
Anchor: same as G4a, iter=50

  Pre-fix (DEC-036):   MLX ~0.2956, CPU ~0.1937, drift ~52.6%
  Post-fix:            MLX  0.19367900, CPU  0.19362645, ratio 1.000271
  Drift:               0.027%   (threshold: <=2%)
  Improvement:         1941x

G4b: PASS

---

## G4c: v5 Kernel ULP=0

  kernel_sources.h md5:  9edaef45b99b9db3e2717da93800e76f  (MATCH)
  BENCH_FINAL_LOSS:       0.48231599  (= AN-009, ULP=0)
  Binary:                 bench_boosting_t4
  Command:                ./bench_boosting_t4 --rows 10000 --features 50 --classes 1
                          --depth 6 --iters 50 --bins 128 --seed 42

G4c: PASS

---

## G4d: 18-Config L2 Parity

Criterion: rs=0 ratio in [0.98, 1.02]; rs=1 MLX <= CPU*1.02
Grid: N in {1k, 10k, 50k} x seed in {1337, 42, 7} x rs in {0.0, 1.0}

      N  seed   rs |  CPU_RMSE   MLX_RMSE  ratio | gate
   1000  1337  0.0 | 0.202116   0.202126  1.0001 | PASS
   1000  1337  1.0 | 0.203738   0.203614  0.9994 | PASS
   1000    42  0.0 | 0.205187   0.205187  1.0000 | PASS
   1000    42  1.0 | 0.206865   0.206712  0.9993 | PASS
   1000     7  0.0 | 0.202568   0.202703  1.0007 | PASS
   1000     7  1.0 | 0.204226   0.204034  0.9991 | PASS
  10000  1337  0.0 | 0.194626   0.194567  0.9997 | PASS
  10000  1337  1.0 | 0.194639   0.194754  1.0006 | PASS
  10000    42  0.0 | 0.194793   0.194873  1.0004 | PASS
  10000    42  1.0 | 0.194965   0.195030  1.0003 | PASS
  10000     7  0.0 | 0.194371   0.194375  1.0000 | PASS
  10000     7  1.0 | 0.194339   0.194264  0.9996 | PASS
  50000  1337  0.0 | 0.194132   0.194072  0.9997 | PASS
  50000  1337  1.0 | 0.194036   0.194060  1.0001 | PASS
  50000    42  0.0 | 0.193702   0.193706  1.0000 | PASS
  50000    42  1.0 | 0.193758   0.193744  0.9999 | PASS
  50000     7  0.0 | 0.193298   0.193304  1.0000 | PASS
  50000     7  1.0 | 0.193180   0.193337  1.0008 | PASS

18/18 PASS. Ratio range: [0.9991, 1.0008].
No movement from the @ml-engineer preliminary check (preliminary: [0.9991, 1.0008]).

G4d: PASS

---

## G4e: DW+Cosine Sanity (S28 Anchor)

Criterion: 5/5 seeds in [0.98, 1.02]
Config: N=1000, DW, Cosine, 50 iters, d=6, bins=128, lr=0.03, l2=3, rs=0, seeds 42-46

 seed |   MLX_RMSE |  CPU_Cosine |  ratio | S28_ratio |  delta | gate
   42 | 0.21382041 | 0.21067719  | 1.0149 |    1.0159 | -0.0010 | PASS
   43 | 0.21297749 | 0.20896760  | 1.0192 |    1.0160 | +0.0032 | PASS
   44 | 0.21067910 | 0.21015602  | 1.0025 |    1.0023 | +0.0002 | PASS
   45 | 0.21359787 | 0.21317366  | 1.0020 |    1.0076 | -0.0056 | PASS
   46 | 0.21960599 | 0.21957084  | 1.0002 |    0.9950 | +0.0052 | PASS

5/5 PASS. Ratio range: [1.0002, 1.0192]. Delta from S28 baseline: [-0.0056, +0.0052].
DW+Cosine path is unaffected (fix is in FindBestSplit / ST path only).

G4e: PASS

---

## Kernel md5 Invariant

Verified before running any gate:
  kernel_sources.h md5: 9edaef45b99b9db3e2717da93800e76f
  Expected:             9edaef45b99b9db3e2717da93800e76f
  MATCH: YES

Fix is exclusively host-side (csv_train.cpp). No kernel sources modified.

---

## Pre/Post Summary (DEC-036)

  iter=1 drift:  pre ~0% (d=0 has 0% skip rate)  -> post 0.0001%  (no change)
  iter=50 drift: pre 52.6%                        -> post 0.027%   (1941x reduction)
  L2 parity:     pre 18/18 [0.9991, 1.0008]       -> post 18/18 [0.9991, 1.0008]  (no regression)
  DW+Cosine:     pre 5/5 [0.9950, 1.0159]         -> post 5/5 [1.0002, 1.0192]    (no regression)
  Kernel ULP=0:  pre 0.48231599                    -> post 0.48231599              (unchanged)

---

## Verdict: ALL GATES PASS

DEC-042 fix validated. Commit 3 (guard removal, #93/#94) is unblocked.
DEC-042 status: RESOLVED.
