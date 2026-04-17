# Developer Changelog — CatBoost-MLX

> Coverage: Sprints 0–15 reconstructed from git log on 2026-04-15. Sprint 16+ is source of truth.

## Sprint 17 — Histogram Tree Reduction (D1c) (2026-04-17)

**Branch**: `mlx/sprint-17-hist-tree-reduce`
**Campaign**: Operation Verstappen — headline performance lever
**Verdict**: **All gates PASS.** Cleared for merge.

- S17-00: Branch cut from master; 18 Sprint 16 baselines copied to `.cache/profiling/sprint17/` as before-snapshots.
- S17-01 (`5b4a8206bc`): D1c kernel — replaced 255-step serial threadgroup reduction in `catboost/mlx/kernels/kernel_sources.h:160–181` with 5-round `simd_shuffle_xor` intra-SIMD butterfly (xor 16/8/4/2/1) + 8-term linear cross-SIMD fold. Barriers 255 → 8, threadgroup memory 12KB (25% of 32KB limit). 95 lines changed.
- S17-02 (`1ce1ea6ee1`): Ablation verdict D1c over D1a (D1a structurally infeasible — ~9,216 barriers from orthogonal axes). Higham γ_8 FP32 bound analysis documented in `docs/sprint17/ablation.md`. Sprint 18 prior in `docs/sprint18/plan_prior.md`.
- S17-03 (`26fbabe932`): 18-config perf capture. `histogram_ms` reduced **89.4–93.0%** (308.20→28.75 ms on gate config, -90.7%). `iter_total_ms` reduced 84.4–92.4%. Secondary stages (suffix_scoring, leaf_sums, cpu_readback) improved 10–30% from pipeline backpressure unblocking. Full table in `docs/sprint17/results.md`.
- S17-04 (`26fbabe932`): Parity matrix — 35/36 checkpoints bit-exact across 18 configs × 6 checkpoints. Final-iteration ulp=0 for all 18 configs. One transient 17-ulp spike at iter=10 of 10k/MultiClass/32 healed to 0 by iter=20 — within Higham γ_8 bound. See `docs/sprint17/parity_results.md`.
- S17-05 (`afded6c4e5`): CI gate `benchmarks/check_histogram_gate.py` (15 tests, all pass). `.github/workflows/mlx-perf-regression.yaml` wired to block >5% histogram regression.
- S17-06: Code review PASS. Three should-fix items addressed in a follow-up: (1) stale "left for S17-06 code review" comment → "deferred to Sprint 18"; (2) scope caveats added to results.md and parity_results.md bounding findings to `approxDim ∈ {1,3}`, `N ≤ 50k`; (3) DECISIONS.md updated with DEC-008 (parity envelope), DEC-009 (linear 8-term choice), DEC-010 (Sprint 18 L1 lever).
- S17-07: Security audit PASS — no exploitable findings, 2 info-level hardening suggestions (SHA-pin actions, add `permissions: read`). Metal shader bounds provable from compile-time constants; CI gate parser uses only argparse+json.load; workflow is `pull_request` (safe) with no secret interpolation.
- **Sprint 18 headline lever identified**: steady-state histogram is still ~175× above memory-bandwidth floor. `privHist[1024]` register spill is the top ceiling. Tiled accumulation (256-lane × 4-pass fold) is the Sprint 18 L1.

## Sprint 16 — Performance Diagnosis & First Cut (2026-04-15, in progress)

**Branch**: `mlx/sprint-16-perf-diagnosis`
**Campaign**: Operation Verstappen — performance domination push
- Restored `.claude/state/` files (HANDOFF, TODOS, MEMORY, DECISIONS, CHANGELOG-DEV)
- S16-05: Extended `benchmarks/bench_mlx_vs_cpu.py` with `--bins`, `--mlx-stage-profile`, `--save-baseline` flags; CPU-parity runner with side-by-side JSON; new `ParityResult` data class; JSON schema with `meta`+`runs[]` including `bins`, `stage_timings`, `cpu_baseline`, `mlx_baseline`
- S16-06: Created `.github/workflows/mlx-perf-regression.yaml` — CI gate on 50k RMSE 128-bin benchmark, 5% threshold, step summary table, `macos-14` only
- S16-02 (baseline support): Regenerated `.cache/benchmarks/sprint16_baseline.json` with accurate Sprint 15 numbers — old phase_a data was stale (from early-sprint code). True MLX/CPU gap is 100–300x, not 10–24x
- S16-07: Sync-storm elimination — removed all 18 `EvalNow` from `pointwise_target.h`, 3 per-depth `EvalNow` from `structure_searcher.cpp`, added `EvalAtBoundary` at iteration boundary. Validated: bit-exact loss across 9 test combos, zero perf regression
- S16-08: Numerical parity validated — RMSE/Logloss/MultiClass × 1k/10k/50k all bit-exact between Sprint 15 and Sprint 16 binaries
- Fixed `bench_mlx_vs_cpu.py` bug: `n_bins=` → `bins=` (API param name mismatch)
- Key finding: per-iteration cost barely scales with N (300ms at 1k, 323ms at 10k, 487ms at 50k with 50 features) — confirms histogram occupancy (`maxBlocksPerPart=1`) as dominant bottleneck
- Stage profiler code drafted by @performance-engineer (pending write to disk)

---

## Sprint 15 — Upstream Submission Prep and Release Packaging [from git log]

**Commit**: `74f2ba63d4` | **Merge**: `165f2bc706`
- Upstream submission preparation
- Release packaging

## Sprint 14 — CI/CD Workflows and Performance Benchmarks [from git log]

**Commit**: `7b36f60a82` | **Merge**: `97a069c93a`
- CI/CD workflow setup
- Performance benchmark infrastructure

## Sprint 13 — Library Path Feature Parity [from git log]

**Commit**: `f1d6b00b20` | **Merge**: `1a2dd61ea2`
- Library path feature parity with CPU CatBoost

## Sprint 12 — Docs Refresh, Ranking Hardening, Upstream Prep [from git log]

**Commit**: `0ec8754c82` | **Merge**: `46ba563172`
- Documentation refresh
- Ranking hardening
- Upstream prep
- BUG-007 found: nanobind path doesn't sort group_ids

## Sprint 11 — Nanobind Python Bindings [from git log]

**Commit**: `3722eb9f95` | **Merge**: `7f7d540276`
- Nanobind in-process GPU training bindings
- CUDA coexistence specification

## Sprint 10 — Lossguide, Model Versioning, PyPI 0.3.0 [from git log]

**Commit**: `d8e3e7ba7b` | **Merge**: `8641eee078`
- Lossguide grow policy (best-first leaf-wise construction)
- Model format versioning (format_version=2)
- PyPI packaging
- User-facing README and quickstart
- `bench_mlx_vs_cpu.py` benchmark script
- BUG-006 fix: scope max_leaves validation to Lossguide only

## Sprint 9 — Depth>6, Depthwise Policy, MLflow, 16M Fix [from git log]

**Commit**: `b8a0ab258a` | **Merge**: `445f55c20a`
- `max_depth > 6` via chunked multi-pass leaf accumulation
- Depthwise grow policy (per-leaf splits at each depth level)
- Deferred histogram EvalNow — reduced CPU-GPU syncs to 5 remaining
- Optional MLflow logging
- bench_boosting CI regression check
- int32 accumulator in ComputePartitionLayout (DEC-003)
- BUG-005 fix: validate grow_policy in _validate_params
- 66 new tests, 789 total

## Sprint 8 — Housekeeping, Poisson/Tweedie/MAPE Losses [from git log]

**Commit**: `1d1e25321f` | **Merge**: `9d9d645430`
- Poisson, Tweedie, MAPE loss functions (library path)
- BUG-004 fix: strip variance_power= prefix in loss param validation
- 39 QA tests for new losses

## Sprint 7 — Multiclass Fuse, Partition Kernel Output, BUG-002 [from git log]

**Commit**: `cd239c84d1` | **Merge**: `7b483ad631`
- Fused multiclass leaf computation — eliminated K EvalNow calls per iteration
- Partitions output from tree_applier kernel — deleted O(depth) recompute
- BUG-002 fix: threshold comparison in bench_boosting

## Sprint 6 — CI Infra, bench --onehot, Tree Applier Metal Kernel [from git log]

**Commit**: `44ac16d66d` | **Merge**: `c7b478f352`
- Tree applier ported to Metal kernel dispatch
- bench_boosting `--onehot` flag
- CI workflow: bench_boosting compile step
- ARCHITECTURE.md deep-dive added
- CONTRIBUTING.md, CHANGELOG.md, Known Limitations docs

## Sprint 5 — BUG-001 Fix + Lint Cleanup [from git log]

**Commit**: `ee617527e3` | **Merge**: `0d2e97f914`
- Deterministic suffix-sum scan (BUG-001 fix)
- Ruff lint cleanup across test and source files
- Parallel SIMD scan for suffix_sum_histogram
- bench_boosting library-path harness
- 16M-row float32 limit documented in DECISIONS.md

## Sprint 4 — GPU Partition Layout [from git log]

**Commit**: `591822a51e` | **Merge**: `fff9f02b7b`
- ComputePartitionLayout ported to GPU
- 16M-row float32 safety guard
- Sprint branch convention established (DEC-004)

## Sprint 3 — Leaf Estimation, Score Splits, Loss Functions [from git log]

**Commits**: `928c7ff4d1` through `38f963cd4a`
- MAX_LEAVES=64 runtime enforcement
- Bin-to-feature lookup table for score_splits
- Fused leaf sum dispatch
- MAE/Quantile/Huber losses wired into dispatch
- Loss function validation tests

## Sprints 0–2 — Foundation [from git log]

**Commits**: `b78d428f58` through `edf8a97ba5`
- Initial Metal kernels for histogram, scoring, leaf accumulation
- Multi-block histogram dispatch (1.2x speedup)
- Feature group batching (1.6x speedup)
- In-process tree evaluation for predict (5–25x faster)
- CBMX binary format (200x faster I/O)
- MVS sampling, base prediction (boost from average)
- Input validation, accuracy bug fixes
- Multiclass fix (off-by-one, 2-class crash)
- random_strength, performance profiling
