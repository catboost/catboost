# A1 (BATCH_DOCS=64) — empirical drop note

**Owner:** Main thread (Sprint 19 close-out, Commit 3 candidate)
**Captured:** 2026-04-19
**Disposition:** DROPPED — production regression, not noise

## Context

Original plan (cozy-watching-sunset.md): ship `DEC-014 (A1) BATCH_DOCS=64 wider-batch`
as Commit 3 after re-measurement. Standing order: "if not reproducible, drop."

## Toy micro-bench result (post-T1 baseline)

Harness: `microbench_algorithmic.cpp` (added `kA1Source`, registered `a1Kernel`).
Config: N=50,000, NUM_BINS=128, 1 TG × 256 threads, warm=5, timed=5. 3 runs.

| Variant | Run 1 | Run 2 | Run 3 | Mean  | vs T1 |
|---------|------:|------:|------:|------:|------:|
| T0      | 2.450 | 2.433 | 2.487 | 2.457 | —     |
| T1      | 2.438 | 2.409 | 2.385 | 2.411 | —     |
| **A1**  | 2.363 | 2.363 | 2.365 | 2.364 | **−1.9%** |

A1 showed a small but reproducible toy-kernel improvement over T1.
Per standing order ("empirical micro-bench backing required"), toy backing
was satisfied — proceeded to production port.

## Production e2e result

Edit: `catboost/mlx/kernels/kernel_sources.h` — lane holds lo/hi slab in
registers, outer loop halved, inner shuffle loop runs 2×32 passes.
Parity harness: bit-exact at 50k/RMSE/d6/128b/seed42 (0.48047778, 3/3 runs).

| Build | warm_mean_ms (3-run) | Δ vs T1-only |
|-------|---------------------:|-------------:|
| T1-only (Commit 2 tip) | 31.73 | — |
| **T1 + A1**            | **34.73** | **+9.4% REGRESSION** |

## Root cause (hypothesis)

Toy kernel has no partition/multi-stat context, so the added per-lane register
pressure from holding two docs (packed_lo/hi, stat_lo/hi, d_lo/hi, valid_lo/hi)
is absorbed. In the production kernel this register pressure composes with
existing live-ness (statIdx loop, partOffset, docIndices gather) and likely
pushes VGPR allocation over the spill threshold. The halved outer-loop count
does not offset the spill cost.

**This mirrors the Sprint 19 pattern: analytical model-driven projections
continue to falsify against production measurement.** DEC-013 (writeback
plurality), DEC-014 original (gather sub-phase), DEC-015 (col-major), and now
DEC-014 (A1) — four models falsified by direct measurement.

## Disposition

- A1 dropped from Sprint 19.
- A1 variant kept in `microbench_algorithmic.cpp` for future reference.
- T1 ships alone as DEC-016 (Commit 2, SHA `92f3832169`).
- R8 gate (≥1.07× e2e) is NOT met by Sprint 19 on 50k/RMSE/d6/128b
  (actual delivered: 1.023× on this config). R8 is deferred to Sprint 20
  via T3b atomic-CAS (DEC-017, draft) pending full DEC-008 parity sweep.

## Lesson

"If not reproducible, drop" was the right guardrail. The toy-kernel signal
was real but did not survive production integration. The plan's pre-commit
re-measurement clause prevented shipping a regression.
