# Project Memory — CatBoost-MLX

> Coverage: Sprints 0–15 reconstructed from git/agent-memory on 2026-04-15. Items marked [reconstructed] are inferred from commit messages and code reading. Sprint 16+ is source of truth.

## MLX / Metal gotchas

- **`EvalNow` = `mx::eval` = full CPU↔GPU sync.** Every call drains the entire command buffer. Minimize these in hot paths. [reconstructed from mlx_device.h:31-37]
- **`.item<float>()` forces sync.** Any scalar read from an MLX array blocks the CPU. Use sparingly, only for metric reporting at `MetricPeriod` intervals.
- **`maxBlocksPerPart = 1` in histogram.cpp:105.** Catastrophic under-occupancy on M-series GPUs. At depth 0, only 2 threadgroups launched. Sprint 17 target. [reconstructed]
- **MLX lazy eval is well-exploited in histogram accumulation** — per-feature-group dispatch uses `mx::add` without intermediate syncs. The accumulation graph is flushed together. [reconstructed]
- **Metal threadgroup memory limit**: 32KB per threadgroup on M-series. Histogram bins × stats × sizeof(float) must fit. Large bin counts may need tiling.
- **Float32 for accumulation, always.** Metal half-precision introduces unacceptable drift in histogram gradient sums. [from CLAUDE.md convention]

## Performance baseline (2026-04-16, Sprint 16 accurate measurement)

| Task | N | CPU CatBoost | CatBoost-MLX | Gap |
|------|---|-------------:|-------------:|----:|
| regression | 1k | 0.09s | 30.1s | 334x slower |
| regression | 10k | 0.16s | 32.3s | 202x slower |
| regression | 50k | 0.26s | 48.8s | 188x slower |
| multiclass | 1k | 0.16s | 59.0s | 369x slower |
| multiclass | 10k | 0.27s | 63.7s | 236x slower |
| multiclass | 50k | 0.48s | 95.9s | 200x slower |

Source: `.cache/benchmarks/sprint16_baseline.json` (regenerated 2026-04-16 with `bench_mlx_vs_cpu.py --save-baseline`)

**STALE DATA WARNING**: Previous baseline from `phase_a_results.json` (2026-04-06) showed 10–24x gaps — those numbers came from an early-sprint codebase and are not representative of Sprint 15+ performance.

Gap narrows slightly with N — per-iteration fixed cost dominates:
- RMSE per-iter: 300ms (1k) → 323ms (10k) → 487ms (50k) with 50 features
- Tree search is 99%+ of per-iteration time
- Confirms `maxBlocksPerPart=1` as #1 bottleneck

## CatBoost-MLX conventions

- Python bindings call `csv_train` binary via subprocess, not a shared library [from ml-engineer memory]. Nanobind bindings (Sprint 11) provide in-process alternative.
- `bench_boosting` C++ binary must be rebuilt from source before QA; always check binary timestamp vs commit [from qa-engineer feedback].
- EvalNow count before Sprint 16: ~21 per binary iteration, ~28 per multiclass K=3 iteration [from mlops-engineer Sprint 6 audit].

## Known bugs (unresolved)

- **Sibling S-1**: `kHistOneByte` cross-TG writeback race on `atomic_float` when `maxBlocksPerPart > 1`. Latent; guarded by compile-time `static_assert` at `catboost/mlx/methods/histogram.cpp:126`. Fix when multi-block dispatch is needed. Authoritative record: `docs/sprint23/d0_bimodality_verification.md §C`.

## Resolved / mitigated (historical)

- **BUG-007** (MITIGATED 2026-04-22): nanobind path — Python wrapper sorts group_ids (`python/catboost_mlx/core.py:1131-1137`); C++ `BuildDatasetFromArrays` now CB_ENSUREs sortedness as defense-in-depth. See `KNOWN_BUGS.md`.
- **bench_boosting K=10 anchor** (3rd lifetime as of S27-AA-T4): value is now `1.85752499` at `20k × 30 × depth5 × 50iters × seed42`. History: `2.22267818` (mismatched-param, S8 TODO-022) → `1.78561831` (corrected S8) → `1.85752499` (S27 T4, after S18/S19/S22/S23/S24 kernel changes). Docs-only in `CHANGELOG.md` — repeat-offender under DEC-031 Rule 5; must be promoted to live pytest on next update. **DEC-032 note**: K=10 multiclass runs use CPU default `score_function` (probably Cosine); if MLX only implements L2, this anchor is coincidental-not-structural per DEC-032 and requires explicit `score_function` labeling before it can be re-blessed as a parity claim.

## RandomStrength noise is a global scalar shared across all CatBoost grow policies (from S26-FU-2 T1 triage, 2026-04-22)

CPU `CalcDerivativesStDevFromZeroPlainBoosting` returns a single `double` scalar from the full fold — no partition or leaf subset. It is called exactly once per tree (before any depth or partition loop) in all three grow-policy functions: `GreedyTensorSearchOblivious` (:1186), `GreedyTensorSearchDepthwise` (:1480), `GreedyTensorSearchLossguide` (:1776). The scalar is declared `const` and passed unchanged into every per-partition and per-leaf candidate evaluation.

**MLX implication**: any future code path that touches RandomStrength noise must thread `gradRms` as a single per-tree scalar — not recomputed per partition, per leaf, or per depth. This applies to both `FindBestSplit` (SymmetricTree, fixed S26 D0) and `FindBestSplitPerPartition` (DW/LG, fixed S26-FU-2).

Source: `catboost/private/libs/algo/greedy_tensor_search.cpp:92–107, :1186, :1480, :1776`. Triage: `docs/sprint26/fu2/d0-triage.md`.

## Parity-gate methodology (from Sprint 26 D0, 2026-04-22)

- **Kernel-ULP=0 ≠ full-path parity.** v5's `bench_boosting` 18/18 ULP=0 record coexisted with a Python-path 0.69× leaf-magnitude collapse for multiple sprints because `bench_boosting` exercises only the histogram kernel — not `FindBestSplit` (noise), nanobind orchestration, quantization borders, or `methods/leaves/`. **New standing order**: every parity gate must label which path it covers. Python-path / `FindBestSplit` / leaf-estimation parity requires its own harness (`tests/test_python_path_parity.py`).
- **Segmented parity gate beats strict symmetric** for stochastic branches. CPU and MLX use independent RNGs; at the same seed they draw different noise realizations. Strict `ratio ∈ [0.98, 1.02]` false-fails cells where MLX is better. Use: (a) rs=0 tight symmetric (no PRNG divergence to explain away), (b) rs=1 one-sided `MLX_RMSE ≤ CPU_RMSE × 1.02` AND `pred_std_R ∈ [0.90, 1.10]`.
- **`pred_std_R = std(MLX_preds) / std(CPU_preds)`** is the primary signal for leaf-magnitude bugs — orthogonal to RMSE (which can be dominated by irreducible noise at small N). DEC-028's signature was `pred_std_R ≈ 0.69`. Keep it in any parity harness touching leaf values.
- **Gradient RMS, not hessian sum, scales RandomStrength noise.** Dimensional check: noise that scales with `N` (hessian sum for RMSE) grows without bound as dataset size increases; noise that scales with `sqrt(sum(g²)/N)` (gradient RMS) shrinks as residuals shrink over boosting iters. If a parity gap grows with N or grows across iters, suspect the wrong scale.
- **Non-oblivious tree serialization requires explicit BFS node index.** Depthwise/Lossguide training cursors are keyed by bit-packed `partitions` (bit k = direction at depth k), but model JSON consumers walk trees in BFS order. Emit `bfs_node_index` per split and `leaf_bfs_ids` inverse map for Lossguide. Don't rely on the consumer to re-derive the mapping.

## Anchor hygiene (DEC-031, S27-AA-T5, 2026-04-22)

S27 audit found 18 anchors: 4 drifted > 1e-4, 3 structurally dead, 9 docs-only (no live enforcement). AN-008 hit its third numeric lifetime. DEC-031 codifies five standing rules:

1. **No new docs-only canonical values** — every committed anchor needs a live pytest assertion.
2. **Anchor-change-on-path-change** — any commit touching histogram/kernel/accumulation/leaf/gain must re-run and update affected anchors atomically or open an immediate audit.
3. **Sprint-close drift check** — re-run anchors touched by that sprint's code changes at close.
4. **Dead anchors removed or wired** — class-d anchors must resolve within the sprint found; never leave unreachable "canonical" values in docs.
5. **Repeat-offender promotion** — if an anchor has been value-updated more than once, the next update must also wire it to a live test (AN-008 is the pending case).

Full taxonomy (class-a/b/c/d) and enforcement details: `DECISIONS.md §DEC-031`. Next audit: Sprint 31 or first kernel/accumulation change, whichever is sooner. T4 landing SHAs: `adce339b56`–`62f17df7a9`.

## Score function fidelity (DEC-032, S27-FU-3-T1, 2026-04-22)

- **MLX `FindBestSplitPerPartition` implements L2 Newton gain only (hardcoded).** No Cosine, no NewtonCosine, no dispatch on `score_function`.
- **CPU CatBoost DW default is Cosine.** `score_function` hyperparameter (`{Cosine, L2, NewtonL2, NewtonCosine}`) is likely silently ignored on the MLX path today — unverified until S28-AUDIT confirms.
- **Any DW (or LG) parity test MUST set CPU `score_function='L2'` explicitly until S28 closes.** Tests that compare MLX against CPU default (Cosine) are measuring algorithmic divergence, not parity. The divergence is not noise: per-partition gain ratios of 0.82–0.86 at N=1000, forced `score_function='L2'` reproduces MLX to ±0.11%.
- **v5 `bench_boosting` ULP=0 record is unaffected** — covers histogram kernel output only; `FindBestSplitPerPartition` gain computation is not in scope.
- **Aggregate RMSE parity claims (Python-path harness, S26-FU-2 gate numbers) are coincidental-not-structural** at N≥10k. Both Cosine (CPU default) and L2 (MLX) happen to agree within gate tolerance at larger N; this is aggregation smoothing, not algorithmic equivalence. Re-blessing with explicit `score_function` labeling is S28-REBLESS.
- Source: S27-FU-3-T1 commit `0931ad6e9c`; triage doc `docs/sprint27/scratch/fu3-t1-triage.md`; DEC-032.

## CUDA reference targets (from researcher, 2026-04-15)

- **szilard/GBM-perf 2024**: A100 SXM4, Airline 10M, 100 iters, depth 10 → CatBoost CUDA **15s**. CPU (Xeon E5-2686 v4): 70s. Citable: https://github.com/szilard/GBM-perf issue #57.
- **arXiv 1810.11363 Table 2**: V100-PCI, Epsilon 400K, depth 6 → CatBoost CUDA **49s (32 bins) / 77s (128 bins)**. CPU: 653s / 713s.
