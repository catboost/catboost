# Sprint 33 Close — Iter≥2 Scaffold (DEC-040)

**Date**: 2026-04-24
**Branch**: mlx/sprint-33-iter2-scaffold
**Base**: mlx/sprint-32-cosine-gain-term-audit tip `9fcc9827d9`
**Kernel md5**: 9edaef45b99b9db3e2717da93800e76f (unchanged throughout)

---

## Outcome

DEC-036 mechanism fully localized: the 52.6% ST+Cosine RMSE drift is caused
by a **quantization strategy divergence** between csv_train.cpp (static
upfront border grid) and CatBoost CPU (dynamic border accumulation during
training). The L3 hypothesis (stale `statsK` via lazy-eval alias) is
falsified. L4 instrumentation removed. DEC-041 opened for the redesign.

---

## Task Completion

| Task | Class    | Result                                          |
|------|----------|-------------------------------------------------|
| #119 L0-CONFIG         | NO-DIFF  | Frame C-config falsified. Config fields identical. |
| #120 L1-DETERMINISM    | FALSIFIED| Drift 52.643% across 3 seeds — seed-independent. |
| #121 L2-GRAFT          | FRAME-B  | Graft ratio 0.974; per-iter persistent bug confirmed. |
| #122 L3-ITER2          | SPLIT    | S1-grad bit-identical; S2-split divergent. Histogram anomaly flagged (later corrected in L4). |
| #123 L4-FIX            | QUANTIZATION | L3 statsK hypothesis falsified. True cause: static vs dynamic quantization. L4 instrumentation removed. |

---

## What Shipped

- **L4 instrumentation removal**: `#ifdef L3_ITER2_DUMP` block removed from
  csv_train.cpp across all instrumented sites. One atomic commit per DEC-012.
- **Verdict docs**: `l4-fix/verdict.md` (this sprint's primary finding).
- **DEC-041 opened** in `docs/decisions.md`.

---

## What Did NOT Ship

- A fix for the 52.6% drift. The mechanism is quantization divergence;
  the fix requires a quantization pipeline redesign (DEC-041). csv_train.cpp
  is a test harness, not the production path. The production nanobind Python
  path uses CatBoost's own quantization and is not affected.
- Guard removal (#93 ST-REMOVE, #94 LG-REMOVE): still blocked on G4b
  (drift ≤ 2%). Deferred to S34 pending DEC-041.

---

## Gate Results

| Gate | Criterion                    | Result  | Notes                              |
|------|------------------------------|---------|------------------------------------|
| G4a  | iter=1 ratio ≤ 1.001         | N/A     | Not the divergence site            |
| G4b  | iter=50 drift ≤ 2%           | BLOCKED | Requires DEC-041 redesign          |
| G4c  | v5 ULP=0 (bench_boosting)    | PASS    | Kernel sources unchanged           |
| G4d  | 18-config L2 [0.98, 1.02]    | PASS    | No csv_train.cpp logic changes     |
| G4e  | DW sanity rs=0 3 seeds       | PASS    | No DW code changes                 |

---

## Root Cause Summary

CatBoost CPU builds borders dynamically: a feature's border list grows only
when that feature is chosen for a split. On the 20-feature anchor dataset
(X[:,0], X[:,1] carry signal; X[:,2]–X[:,19] are pure noise), CatBoost
produces 95 + 71 = 166 useful bin-features at 50 iterations; noise features
accumulate 0 borders.

csv_train.cpp pre-quantizes all 20 features to 127 borders = 2540
bin-features. MLX evaluates noise-heavy candidates at every tree level,
wasting depth. At 50 trees with depth=6, this produces 52.6% RMSE drift.

Confirmed by `model.get_borders()` output after 50 iterations:
```
Feature 0: 95 borders
Feature 1: 71 borders
Features 2-19: 0 borders each (never selected for a split)
```

The mechanism is not a Metal kernel bug, not a gradient computation error,
not a lazy-eval alias, and not a gain formula error. It is a design gap in
the test harness.

---

## L3 Errors Corrected

The L3 verdict contained two errors that were corrected in the L4 verdict:

1. **Wrong expected formula**: "expected ~+0.228 = 20 feats × sum_g" is
   incorrect. The histogram total of −738.99 is the correct value: it is
   the sum of gradients of non-bin-0 docs across all 20 features. The bin-0
   docs (top quantile of X[:,0]) carry large positive gradients and are
   intentionally excluded by the Metal writeback loop.

2. **Wrong hypothesis**: The `statsK` lazy-eval alias hypothesis was
   falsified. `statsK` contains correct iter-2 gradients (confirmed by
   explicit `mx::eval()` + readback diagnostic; max_diff vs CPU = 1.5e-8).

---

## Open Items (carry-forward to S34)

| Item | Status | Blocker |
|------|--------|---------|
| DEC-041 quantization redesign | OPEN | Design decision needed (options 1/2/3 in L4 verdict) |
| #93 ST-REMOVE | Blocked | DEC-041 G4b |
| #94 LG-REMOVE | Blocked | DEC-041 G4b |
| S31-T-LATENT-P11 | Carry | Low priority |
| S31-T-CLEANUP | Carry | SA-I2 + S29 CR nits |

---

## DEC Status After S33

| DEC  | Status          | Notes                                      |
|------|-----------------|-------------------------------------------|
| DEC-036 | PARTIAL-CLOSED | Mechanism explained; drift remains pending DEC-041 |
| DEC-040 | CLOSED          | L0-L4 scaffold complete                    |
| DEC-041 | OPEN            | Static vs dynamic quantization redesign    |

---

## Files of Record

- `docs/sprint33/l4-fix/verdict.md` — full L4 analysis + L3 corrections
- `docs/sprint33/sprint-close.md` — this file
- `docs/sprint33/l0-config/verdict.md` — L0 NO-DIFF
- `docs/sprint33/l1-determinism/verdict.md` — L1 FALSIFIED
- `docs/sprint33/l2-graft/verdict.md` — L2 FRAME-B
- `docs/sprint33/l3-iter2/verdict.md` — L3 SPLIT (contains falsified L3 hypothesis; L4 verdict is the correction)
- `docs/sprint33/l4-fix/data/` — raw diagnostic dumps (mlx_grad_iter2.bin, mlx_hist_d0_iter2.bin, etc.)
