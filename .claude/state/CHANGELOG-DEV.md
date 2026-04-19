# Developer Changelog — CatBoost-MLX

> Coverage: Sprints 0–15 reconstructed from git log on 2026-04-15. Sprint 16+ is source of truth.

## Sprint 19 — Accumulation Redesign (PIVOTED from Two-Phase Writeback) (2026-04-17, in progress)

**Branch**: `mlx/sprint-19-hist-writeback` (name reflects original scope — history over cosmetics)  
**Campaign**: Operation Verstappen — battle 4 of 9 — L_accum lever (pivoted from L_writeback)  
**Verdict**: IN PROGRESS

### Day 3 (2026-04-18) — DEC-015 col-major layout: correct, but performance gate not met (BLOCKER)

**S19-03 Commit 1 (DEC-015) — BLOCKED, NOT COMMITTED**

Implementation completed across 5 files. Status: parity-clean, determinism-clean, performance gate not met.

**Changes in working tree (not committed):**
- `catboost/mlx/gpu_data/compressed_index.h` — Added `CompressedDataTransposed_` member (`[numUi32PerDoc * numDocs]` uint32, col-major). Built in `Build()` via `mx::copy(mx::transpose(CompressedData_, {1,0}))` → `mx::reshape(..., {-1})` → `EvalAtBoundary`. One-time materialisation at load time. Added `GetCompressedDataTransposed()` accessor.
- `catboost/mlx/kernels/kernel_sources.h` — Changed compressedIndex load address from row-major `compressedIndex[docIdx * lineSize + featureColumnIdx]` to col-major `compressedIndex[featureColumnIdx * totalNumDocs + docIdx]`.
- `catboost/mlx/methods/histogram.cpp` — Rewrote `DispatchHistogramGroup()` (scalar per-group dispatch with broken variable name mismatch) to `DispatchHistogramBatched()` (correct batched dispatch matching `bench_boosting.cpp`/`build_verify_test.cpp`). Input names now match kernel body: `featureColumnIndices` (array) + `numGroups` (scalar). Passes `compressedDataTransposed` from `GetCompressedDataTransposed()`.
- `catboost/mlx/tests/bench_boosting.cpp` — Pre-computes `compressedDataTransposed` once before training loop; passes as parameter to `RunIteration()` → `DispatchHistogram()`.
- `catboost/mlx/tests/csv_train.cpp` — Pre-computes `compressedDataTransposed` once in `RunBoosting()` before training loop; passes to all 3 `DispatchHistogram()` call sites.

**Bugs fixed along the way (would have shipped regardless):**
- Pre-existing `histogram.cpp` kernel variable name mismatch: old code used `featureColumnIdx` (scalar 0-dim) but kernel body referenced `featureColumnIndices[groupIdx]` (array). Metal compile would have errored. Fixed as part of DEC-015 rewrite.
- Stale S18 parity reference (8/18 FAIL on first run): S18 parity table was from older D1c binary. Rebuilt reference binary from pre-DEC-015 stash. Result: 18/18 PASS, 0 ULP.
- Per-call transpose overhead: initial attempt placed `mx::copy(mx::transpose(...))` inside `DispatchHistogram()`, causing 6× GPU copies per iteration. Moved to pre-training-loop.

**Gate measurements (50k/RMSE/d6/128b, 5 warm runs each):**
- `bench_boosting_ref` warm mean: 33.7–34.2 ms (5 runs, σ ≈ 0.3 ms)
- `bench_boosting` (DEC-015) warm mean: 34.3–35.7 ms (5 runs, σ ≈ 0.5 ms)
- Speedup: **~0.98× e2e** (effectively 0, within noise)
- Expected from S19-01b model: 2.13× e2e (`histogram_ms` 15.43 → 4.17 ms)
- **Gate: NOT MET. BLOCKER.**

**Implication for S19-01b:** The analytical model (25 CL per 32-doc batch → 4 L2 stall rounds → 12.78 ms CI gather latency) is not validated by direct measurement. The DEC-015 layout change is the most direct test of that model's core prediction. The 0.98× result implies the model's latency estimate or access-pattern description is incorrect for this hardware. A hardware-controlled micro-benchmark (isolated kernel, swept N/lineSize, both layouts) is needed before the next intervention.

### Day 2 (2026-04-17) — Ground-truth falsifies writeback hypothesis; pivot to accumulation redesign

- **S19-01** (commit `d7ea14e28c`, `docs/sprint19/attribution.md`): Ground-truth Metal System Trace attribution on 50k/RMSE/d6/128b gate config. **Writeback = 0.79 ms (5%)** of steady-state `histogram_ms`. **Accumulation = 14.30 ms (93%)**. The "~15 ms writeback floor" from S18 was a mis-scaling of N=10k numbers to N=50k. R8 fired: writeback elimination projects 1.02–1.04× e2e (below the 1.5× aggressive target). Evidence correct; premise (writeback as plurality) falsified.
- **S19-02** (commit `fb05205ec0`, `docs/sprint19/ablation.md`): @research-scientist wrote a clean DEC-013 draft for two-phase writeback reduction. Variant (c) projected 3.0 ms reduction. Premise immediately invalidated by S19-01 — secondary effects ground truth does not support the projection. DEC-013 draft stands as historical artifact; not implemented.
- **R8 result**: writeback elimination → 1.02–1.04× e2e. Does not meet the 1.5× aggressive gate.
- **Ramos decision**: Option 2 — pivot Sprint 19 to accumulation redesign. Option 1 (ship weak writeback) and Option 3 (cleanup-only demote) rejected.
- **DEC-013 SUPERSEDED** by DEC-014 (see `.claude/state/DECISIONS.md`). DEC-013 entry preserved as audit trail.
- **DEC-014 DRAFT added**: accumulation redesign over writeback rewrite. 4 candidate variants (A: wider batch, B: coalesced TG staging, C: per-feature specialization, D: different ownership granularity). Projection: 30–50% `histogram_ms` reduction → 1.25–1.50× e2e. Locks at S19-02b close.
- **Day 2 kickoff**: @performance-engineer running S19-01b (accumulation sub-phase attribution); @research-scientist running S19-02b (accumulation redesign ablation + DEC-014 lock). Both in parallel.
- Sprint length bumped Day 5 → **Day 6** (pivot cost one day).
- G1 gate revised: `histogram_ms` −40% → **−30% min** (accumulation = 93%; 32% accumulation reduction ≈ 30% histogram_ms).

### Day 0 (2026-04-17) — Branch cut and scaffold

- S19-00: Branch cut from `mlx/sprint-18-hist-privhist-tile@463de74efa`. Sprint 18 after-profiles copied to `.cache/profiling/sprint19/baseline/` (18 JSONs, identical to S18 after). Gate config shift: 10k/RMSE/128b → **50k/RMSE/128b** (writeback lever has force at large N). Steady-state baselines — gate config: `histogram_ms` 15.52 ms (mean), `iter_total_ms` 21.12 ms. State files scaffolded (HANDOFF S19 rewrite, TODOS S19 section, DECISIONS DEC-013 placeholder, CHANGELOG S19 header). `docs/sprint19/README.md` scaffold created with campaign context, lever description, gates table, and projection table. DEC-013 DRAFT: two-phase on-chip reduction over batched-atomic (Ramos: "whatever is more robust"). PR #10 (Sprint 18) remains OPEN, unblocked.

---

## Sprint 18 — Histogram Accumulator Re-architecture (L1a) (2026-04-17)

**Branch**: `mlx/sprint-18-hist-privhist-tile`  
**Campaign**: Operation Verstappen — second structural kernel rewrite  
**Verdict**: **All gates PASS.** Cleared for merge.

- S18-00: Branch cut from `mlx/sprint-17-hist-tree-reduce`; Sprint 17 after-profiles copied to `.cache/profiling/sprint18/` as baselines.
- S18-01 (`attribution.md`): Ground-truth post-S17 attribution by linear regression on steady-state per-depth `histogram_ms` breakdown. Accumulation = 6.4 ms (27% of SS), zero-init = 4.0 ms (17%), D1c reduction = 3.0 ms (13%), writeback = 5.0 ms (21%), JIT = 5.3 ms. Plan's 52–59% accumulation estimate refuted (actual 27%); D1c had already eliminated the device-memory re-read cost conflated in the Sprint 16 baseline. Gate revised from ≥50% to ≥35% (≤18.7 ms) with Ramos Day-1 approval.
- S18-02: Ablation sweep L1a / L1b / L1c / L1d. L1a is the only variant with error-envelope gate clearance (worst case 17.3 ms vs 18.7 ms gate; L1b/c miss upper bounds). Ramos approved L1a Day 2. See `docs/sprint18/ablation.md`.
- S18-03 (`abc4c229f9` → `19fa5ce6cc`): L1a implementation. **Pivot**: initial kernel (commit `abc4c229f9`) failed all 18 parity configs by 6 orders of magnitude (BUG-S18-001). Two compounding structural flaws: (1) 1/32 doc-inclusion rate from stride/ownership mismatch; (2) 32× butterfly amplification from applying D1c's intra-SIMD `simd_shuffle_xor` butterfly to shared `simdHist` slots. Fixed at commit `19fa5ce6cc`: replaced accumulation with cooperative 32-doc batch loop using `simd_shuffle` broadcast (every doc contributes exactly once, no atomics); removed intra-SIMD butterfly entirely (`simdHist[g][bin]` is already the full per-SIMD-group sum). See `docs/sprint18/bug_s18_001.md` for post-mortem.
- S18-04a (initial, commit `abc4c229f9`): Parity FAIL — 4–20M ULP all 18 configs. Determinism PASS (consistent wrong answer).
- S18-04b (`7ab4e8e804`): Parity re-run on fixed kernel. **108/108 checkpoints bit-exact (ULP = 0 all loss types). 100/100 determinism runs bit-exact.** Cleaner than Sprint 17's 35/36 outcome. S18-G3 hard merge gate CLEARED.
- S18-05b (`da303866ef`): 18-config stage-profiler delta. Gate config (N=10k, RMSE, d6, 128b): **28.75 → 9.56 ms (-66.8%)**. S18-G1 (≥35%) **PASS** — 9.1 ms margin above target. Full range: -56.6% to -85.5%. All 18 configs improved, no regressions. Non-histogram stages all improved or unchanged. S18-G2, S18-G4 PASS. Sprint 19 floor visible: N=50k configs converge to ~15 ms (writeback-dominated). See `docs/sprint18/results.md`.
- S18-06: CI gate `benchmarks/check_histogram_gate.py` baseline updated to Sprint 17 after-JSON. S18-G5 PASS.
- S18-07: Code review PASS — barrier correctness, threadgroup-memory bound, stride-partition ownership.
- S18-08: Security audit PASS — no new exploitable surfaces.
- S18-09: Metal System Trace re-capture confirms `simdHist` on-chip residency; accumulation phase below 5 ms target. Appendix in `docs/sprint18/results.md`.
- S18-10: Docs — `bug_s18_001.md` post-mortem, `design.md` updated with final kernel structure and BUG-S18-001 root cause diagram, `ablation.md` post-ship actual vs projected section, `README.md` verdict banner, DEC-011 + DEC-012 in `DECISIONS.md`, `ARCHITECTURE.md` histogram section refreshed, `CHANGELOG.md` user-facing entry.

**Kernel change summary** (`catboost/mlx/kernels/kernel_sources.h`, commit `19fa5ce6cc`):
- `float privHist[HIST_PER_SIMD]` (4 KB/thread, 1 MB/threadgroup device-memory spill) → `threadgroup float simdHist[8][1024]` (32 KB, on-chip, at Apple Silicon limit).
- Zero-init loop eliminated (implicit for threadgroup memory).
- Per-thread stride accumulation → cooperative 32-doc batch loop with `simd_shuffle` broadcast and stride-partition ownership.
- D1c intra-SIMD butterfly removed (DEC-012). Cross-SIMD 8-term linear fold (DEC-009) unchanged.
- Barriers: 9 → 6 per dispatch.
- Reduction depth: γ_12 (S17) → γ_7 (S18). Higham bound improves ~7.2e-7 → ~4.2e-7.

**Sprint 19 carry-forward lever**: writeback (global-atomic) phase at ~15 ms for N=50k configs is now the floor. Batched-atomic writeback or shared-memory prefix-scan reduction of per-SIMD histograms before global writeback is the likely S19 L1. Scope constraint: results bounded to DEC-008 envelope (`approxDim ∈ {1, 3}`, `N ≤ 50k`, depth 6, 50 iterations).

---

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
