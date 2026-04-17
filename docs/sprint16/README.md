# Operation Verstappen — Sprint 16

## What this is

Operation Verstappen is a multi-sprint performance campaign (Sprints 16–24) to make CatBoost-MLX competitive with, and eventually faster than, CPU CatBoost and CUDA CatBoost on Apple Silicon. Sprint 16 is the first sprint: diagnosis and the first confirmed free-move fix.

See [`docs/operation-verstappen.md`](../operation-verstappen.md) for the full campaign roadmap and scoreboard.

---

## Why we are doing this

Sprint 15 shipped a correct, feature-complete GBDT implementation. It is not fast.

Baseline measurements from `.cache/benchmarks/baseline_results.json` (100 iterations, depth 6, 50 features, ~default bins):

| Task | N | CatBoost-MLX (s) | CPU CatBoost (s) | Slowdown |
|------|---|-----------------|-----------------|---------|
| Regression | 1k | 3.40 | 0.27 | 12.4x |
| Regression | 10k | 5.04 | 0.32 | 15.9x |
| Regression | 50k | 9.58 | 1.35 | 7.1x |
| Binary | 10k | 12.25 | 1.10 | 11.2x |
| Binary | 50k | 14.50 | 2.16 | 6.7x |
| Multiclass | 10k | 39.99 | 2.57 | 15.6x |
| Multiclass | 50k | 51.53 | 4.08 | 12.6x |

The gap is 7–16x depending on task and N. At least six structural bottlenecks have been identified by static code analysis (see [`docs/sprint16/bottlenecks.md`](bottlenecks.md)). Sprint 16 measures all of them before fixing more than one.

---

## Championship targets (Sprint 24 exit)

All benchmarks report two columns: 32-bin and 128-bin (see DEC-006 in `.claude/state/DECISIONS.md`).

### Primary dominance targets

| Benchmark | Target time | vs CPU |
|-----------|------------|--------|
| 10k regression | ≤ 0.20s | 1.35x faster |
| 50k regression | ≤ 0.25s | 1.68x faster |
| 50k binary | ≤ 0.40s | 1.85x faster |
| 50k multiclass | ≤ 0.60s | 2.2x faster |
| 500k regression | ≤ 0.3x CPU time | — |
| 2M regression | ≤ 0.2x CPU time | 5x faster |
| Beat XGBoost + LightGBM | all 50k+ benchmarks | — |
| Airline 10M | ≤ 15s | matches CatBoost CUDA A100 (szilard/GBM-perf 2024) |

### Secondary targets

- Histogram SIMD occupancy ≥ 70%
- Sync points per training iteration ≤ 2
- CI regression gate: no PR lands with >5% wall-clock regression on the 50k regression anchor

---

## Small-N policy

N < 5k: CPU fallback is acceptable. GPU kernel launch overhead (~50–200 µs per kernel) creates a floor that GPU throughput cannot overcome on tiny datasets. The championship push targets N ≥ 10k exclusively. Runtime dispatch threshold will be determined from Sprint 16 profiling data and implemented Sprint 22–23.

---

## Sprint 16 scope

Two workstreams in parallel:

**Diagnosis (S16-01 through S16-05, S16-10):** Full per-stage wall-clock breakdown for all 9 pipeline stages × {depth 6, 10} × {oblivious, depthwise, lossguide} × {32 bins, 128 bins}, plus a Metal System Trace and a complete sync-point inventory. Output goes to `.cache/profiling/sprint16/` and feeds the diagnosis report at [`docs/sprint16/diagnosis.md`](diagnosis.md).

**Sync-storm fix (S16-07, S16-08, S16-12):** Remove the 18 `EvalNow` calls scattered throughout `pointwise_target.h`. These force a CPU-GPU round-trip after every gradient and hessian computation and after every loss evaluation. Replace with a single `EvalAtBoundary` at the top of the boosting iteration in `mlx_boosting.cpp`. The fix is the only change that lands in this sprint; all other bottlenecks are measured but deferred to Sprint 17+.

**Infrastructure (S16-05, S16-06):** `bench_boosting` and `bench_mlx_vs_cpu.py` extended with `--stage-profile`; CI gate added so a >5% regression on the 50k anchor fails a PR.

---

## Active task tracker

See `.claude/state/TODOS.md` — Sprint 16 section — for all acceptance criteria and assignees.

---

## References

- Bottleneck inventory: [`docs/sprint16/bottlenecks.md`](bottlenecks.md)
- Diagnosis report skeleton: [`docs/sprint16/diagnosis.md`](diagnosis.md)
- Campaign roadmap: [`docs/operation-verstappen.md`](../operation-verstappen.md)
- Sync-point architecture: [`catboost/mlx/ARCHITECTURE.md`](../../catboost/mlx/ARCHITECTURE.md) — "Sync Boundaries" section
- Decisions: DEC-005, DEC-006, DEC-007 in `.claude/state/DECISIONS.md`
