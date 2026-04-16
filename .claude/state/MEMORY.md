# Project Memory — CatBoost-MLX

> Coverage: Sprints 0–15 reconstructed from git/agent-memory on 2026-04-15. Items marked [reconstructed] are inferred from commit messages and code reading. Sprint 16+ is source of truth.

## MLX / Metal gotchas

- **`EvalNow` = `mx::eval` = full CPU↔GPU sync.** Every call drains the entire command buffer. Minimize these in hot paths. [reconstructed from mlx_device.h:31-37]
- **`.item<float>()` forces sync.** Any scalar read from an MLX array blocks the CPU. Use sparingly, only for metric reporting at `MetricPeriod` intervals.
- **`maxBlocksPerPart = 1` in histogram.cpp:105.** Catastrophic under-occupancy on M-series GPUs. At depth 0, only 2 threadgroups launched. Sprint 17 target. [reconstructed]
- **MLX lazy eval is well-exploited in histogram accumulation** — per-feature-group dispatch uses `mx::add` without intermediate syncs. The accumulation graph is flushed together. [reconstructed]
- **Metal threadgroup memory limit**: 32KB per threadgroup on M-series. Histogram bins × stats × sizeof(float) must fit. Large bin counts may need tiling.
- **Float32 for accumulation, always.** Metal half-precision introduces unacceptable drift in histogram gradient sums. [from CLAUDE.md convention]

## Performance baseline (2026-04-06)

| Task | N | CPU CatBoost | CatBoost-MLX | Gap |
|------|---|-------------:|-------------:|----:|
| regression | 50k | 0.42s | 8.56s | 20.4x slower |
| binary | 50k | 0.74s | 8.61s | 11.6x slower |
| multiclass | 50k | 1.33s | 31.84s | 23.9x slower |

Source: `.cache/benchmarks/baseline_results.json`, `.cache/benchmarks/phase_a_results.json`

Gap widens with N — per-iteration fixed cost, not data-parallel.

## CatBoost-MLX conventions

- Python bindings call `csv_train` binary via subprocess, not a shared library [from ml-engineer memory]. Nanobind bindings (Sprint 11) provide in-process alternative.
- `bench_boosting` C++ binary must be rebuilt from source before QA; always check binary timestamp vs commit [from qa-engineer feedback].
- EvalNow count before Sprint 16: ~21 per binary iteration, ~28 per multiclass K=3 iteration [from mlops-engineer Sprint 6 audit].

## Known bugs (unresolved)

- **BUG-007**: nanobind path doesn't sort group_ids — silent divergence on unsorted ranking input [from qa-engineer Sprint 12 review]
- **bench_boosting K=10 anchor**: expected 2.22267818, measured 1.78561831 — flagged Sprint 7, unresolved [from qa-engineer Sprint 7 review]

## CUDA reference targets (from researcher, 2026-04-15)

- **szilard/GBM-perf 2024**: A100 SXM4, Airline 10M, 100 iters, depth 10 → CatBoost CUDA **15s**. CPU (Xeon E5-2686 v4): 70s. Citable: https://github.com/szilard/GBM-perf issue #57.
- **arXiv 1810.11363 Table 2**: V100-PCI, Epsilon 400K, depth 6 → CatBoost CUDA **49s (32 bins) / 77s (128 bins)**. CPU: 653s / 713s.
