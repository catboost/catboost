# Operation Verstappen — Multi-Sprint Performance Campaign

CatBoost-MLX shipped correct and feature-complete at Sprint 15 exit. It is 7–16x slower than CPU CatBoost. This campaign is the plan to close that gap, beat CPU on all benchmarks at N ≥ 10k, and match CatBoost CUDA A100 on the Airline 10M dataset.

Sprint range: 16–24. Status updates below are filled at each sprint exit.

---

## Sprint Roadmap

| Sprint | Branch | Focus | Expected gain | Status |
|--------|--------|-------|--------------|--------|
| 16 | `mlx/sprint-16-perf-diagnosis` | Diagnosis + sync-storm fix (`pointwise_target.h` 18 EvalNow → 1 EvalAtBoundary) | ≥10% e2e on N=100k RMSE | In progress |
| 17 | `mlx/sprint-17-histogram-occupancy` | `maxBlocksPerPart` tuning; eliminate B2 occupancy bottleneck | 4–8x histogram stage at N≥100k | Planned |
| 18 | `mlx/sprint-18-multiclass-fusion` | Fuse multiclass per-dim histogram loop into single dispatch | 2–4x multiclass | Planned |
| 19 | `mlx/sprint-19-depth-pipelining` | Eliminate per-depth CPU readbacks (B5) at structure_searcher.cpp:252,550 | 2–3x depthwise/oblivious | Planned |
| 20 | `mlx/sprint-20-quantization-fastpath` | GPU quantization on ingest; persistent device-resident datasets | Eliminate pre-training latency | Planned |
| 21 | `mlx/sprint-21-leaf-apply-fusion` | Fuse leaf estimation + tree apply into one command buffer | 5–15% iteration time | Planned |
| 22 | `mlx/sprint-22-kernel-specialization` | Depth- and dtype-specialized kernels via MLX JIT templates; small-N CPU dispatch | Varies | Planned |
| 23 | `mlx/sprint-23-large-scale-tiling` | Datasets exceeding Metal buffer limits; async CPU→GPU streaming | Enables 10M+ rows | Planned |
| 24 | `mlx/sprint-24-championship` | Final tuning; full dominance benchmark suite vs CPU + CUDA; release polish | Championship targets | Planned |

Expected gain estimates are pre-measurement heuristics. Sprint 17+ sequencing and gain estimates will be revised after Sprint 16 diagnosis data is available.

---

## Championship Scoreboard

Updated at each sprint exit. All times in seconds. All benchmarks report 32-bin and 128-bin columns (see DEC-006).

### Regression (RMSE)

| N | Bins | CatBoost-MLX | CPU CatBoost | XGBoost | LightGBM | Target | Status |
|---|------|-------------|-------------|---------|---------|--------|--------|
| 10k | — | 5.04 | 0.32 | 0.63 | 1.26 | ≤ 0.20s (1.35x faster than CPU) | Baseline |
| 50k | — | 9.58 | 1.35 | 2.40 | 3.89 | ≤ 0.25s (1.68x faster than CPU) | Baseline |
| 500k | 32 | — | — | — | — | ≤ 0.3x CPU | Not yet benchmarked |
| 2M | 32 | — | — | — | — | ≤ 0.2x CPU | Not yet benchmarked |
| Airline 10M | 128 | — | — | — | — | ≤ 15s | Not yet benchmarked |

### Binary classification (Logloss)

| N | Bins | CatBoost-MLX | CPU CatBoost | XGBoost | LightGBM | Target | Status |
|---|------|-------------|-------------|---------|---------|--------|--------|
| 10k | — | 12.25 | 1.10 | 1.54 | 2.57 | — | Baseline |
| 50k | — | 14.50 | 2.16 | 2.14 | 3.10 | ≤ 0.40s (1.85x faster than CPU) | Baseline |

### Multiclass (K=10)

| N | Bins | CatBoost-MLX | CPU CatBoost | XGBoost | LightGBM | Target | Status |
|---|------|-------------|-------------|---------|---------|--------|--------|
| 10k | — | 39.99 | 2.57 | 7.89 | 12.93 | — | Baseline |
| 50k | — | 51.53 | 4.08 | 10.39 | 16.12 | ≤ 0.60s (2.2x faster than CPU) | Baseline |

Baseline numbers from `.cache/benchmarks/baseline_results.json`. 100 iterations, depth 6, 50 features. Bin count not controlled in baseline run — to be stratified at Sprint 17 exit per DEC-006.

---

## CUDA Reference Targets

These are the external published numbers that define "championship":

| Reference | Dataset | Hardware | Time | Source |
|-----------|---------|---------|------|--------|
| CatBoost CUDA | Airline 10M | A100 | ~15s | szilard/GBM-perf 2024 |
| CatBoost CUDA | Epsilon 400k | V100 | ~49s | arXiv 1810.11363 |
| CatBoost CUDA | Higgs 10M | V100 | ~77s | arXiv 1810.11363 |

A100 and V100 are not Apple Silicon — the comparison is performance-class parity, not architectural equivalence. Matching a datacenter GPU class on a laptop SoC is the championship standard.

---

## Secondary Infrastructure Targets

These gate championship readiness regardless of raw speed numbers:

| Target | Current | Goal | Sprint |
|--------|---------|------|--------|
| Sync points per iteration | ~20+ (pre-S16 fix) | ≤ 2 | 16–19 |
| Histogram SIMD occupancy | ~5% (maxBlocksPerPart=1) | ≥ 70% | 17 |
| CI regression gate | None | >5% regression fails PR | 16 |
| Correctness vs CPU | Bit-exact (RMSE), ulp≤4 (Logloss) | Maintained | Ongoing |

---

## Decision Log

All campaign architecture decisions are in `.claude/state/DECISIONS.md`:

- **DEC-005** — Measurement before optimization rationale (why Sprint 16 is diagnosis first)
- **DEC-006** — Dual bin count (32 + 128) for all championship benchmarks
- **DEC-007** — Small-N CPU fallback policy (N < 5k)

---

## References

- Sprint 16 detail: [`docs/sprint16/README.md`](sprint16/README.md)
- Bottleneck inventory: [`docs/sprint16/bottlenecks.md`](sprint16/bottlenecks.md)
- Diagnosis report: [`docs/sprint16/diagnosis.md`](sprint16/diagnosis.md)
- Sync boundary architecture: [`catboost/mlx/ARCHITECTURE.md`](../catboost/mlx/ARCHITECTURE.md) — "Sync Boundaries" section
- Active tasks: `.claude/state/TODOS.md`
