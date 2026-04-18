# Operation Verstappen — Sprint 18: Histogram Accumulator Re-architecture

## What this is

Sprint 18 is the second **structural kernel rewrite** of Operation Verstappen. Sprint 17 eliminated the serial 255-step threadgroup reduction and delivered an 89–93% `histogram_ms` reduction across all 18 configs. Sprint 18 rewrites the accumulator phase that now dominates: the per-thread `privHist[1024]` array that spills 4 KB per thread to device memory, consuming an estimated 52–59% of the post-S17 histogram kernel.

**Sprint 18 verdict: SHIPPED.** L1a fixed kernel (commit `19fa5ce6cc`). All five gates PASS. `histogram_ms` reduced 56.6–85.5% across all 18 configs. Gate config -66.8% (28.75 → 9.56 ms). Parity: 108/108 checkpoints bit-exact, 100/100 determinism runs bit-exact. The sprint included a significant pivot: the initial L1a kernel (commit `abc4c229f9`) failed all parity configs by 6 orders of magnitude (BUG-S18-001). See `docs/sprint18/bug_s18_001.md` for the post-mortem.

See [`docs/operation-verstappen.md`](../operation-verstappen.md) for the full campaign roadmap.  
Plan of record: `/Users/ramos/.claude/plans/sprint18-hist-privhist-tile.md`

---

## The lever

DEC-010: replace `float privHist[HIST_PER_SIMD]` (1024 floats × 4 B = 4 KB per thread, 1 MB per threadgroup) with a **SIMD-group-local shared histogram in threadgroup memory**. Each SIMD group owns one 1024-float histogram. Stride partitioning assigns each bin to exactly one lane within its SIMD group — zero atomics, zero contention during accumulation. The existing D1c reduction block (DEC-009, Sprint 17) folds the per-SIMD histograms downstream.

Source anchor: `catboost/mlx/kernels/kernel_sources.h:115–148` (accumulation loop, Sprint 18 L1 target).  
Source of truth for post-S17 bottleneck attribution: `docs/sprint18/attribution.md` (filled by @performance-engineer, S18-01).

---

## Sprint 18 perf gate (committed)

**S18-G1 (headline).** `histogram_ms` reduction **≥35%** at the gate config — N=10k, RMSE, depth=6, 128 bins — before/after on same hardware, mean of 5 runs (revised Day 1 per S18-01 attribution).

| metric | S17 after (baseline) | S18 target |
|--------|---------------------:|-----------:|
| `histogram_ms` | 28.75 ms | **≤ 18.7 ms** |
| `iter_total_ms` | 34.94 ms | — (secondary) |

**S18-G2.** No config in the 18-grid (N ∈ {1k, 10k, 50k} × {RMSE, Logloss, MultiClass} × {32, 128 bins}) regresses `histogram_ms` by >5% vs the Sprint 17 after-JSON.

**S18-G3 (parity, hard merge gate).** RMSE ulp ≤ 4, Logloss ulp ≤ 4, MultiClass ulp ≤ 8 across DEC-008 envelope (approxDim ∈ {1, 3}, N ∈ {1k, 10k, 50k}, 32/128 bins, 50 iter, d6). 100-run determinism fixture on 10k/RMSE/d6/128 bins (BUG-001 regression guard).

**S18-G4.** No non-histogram stage regresses >10% on the gate config.

**S18-G5.** `benchmarks/check_histogram_gate.py --18config` and `.github/workflows/mlx-perf-regression.yaml` continue to pass with the updated baseline threshold. No new benchmark infrastructure introduced.

---

## Variants under ablation (S18-02)

Chosen variant: **L1a (Ramos Day-2 approval, 2026-04-17)**. See `docs/sprint18/ablation.md` for full projection and gate-clearance analysis. @ml-engineer is implementing S18-03.

| variant | description | peak threadgroup_mem (KB) | accumulation passes | primary risk |
|---------|-------------|------------------------:|--------------------:|--------------|
| **L1a** | Full 32 KB per-SIMD histogram; reduction operates in place on the same buffer | 32 (at limit) | 1 | Hits threadgroup-memory ceiling; no headroom for Sprint 19+ geometry changes |
| **L1b** | Tiled: 256-bin × 8-SIMD × 4 tiles; existing D1c reduction per tile | 12 | 4 | 4× doc-loop DRAM re-reads; may be bandwidth-bound at N=50k |
| **L1c** | Hybrid stride-partition: 256-float SIMD slab, 8 bins/lane, 4 tiles | 12 | 4 | Same bandwidth risk as L1b; stride pattern is more complex |
| **L1d** | Control — D1c baseline (Sprint 17 after) | 12 | 1 | Reference; should match Sprint 17 after-JSONs |

Default if ablation is inconclusive: **L1a** — single accumulation pass, strongest parity story, simplest reduction path.

Full variant designs and analytical projections: `docs/sprint18/ablation.md` (TBD per S18-02).

---

## Sub-tasks

| ID | Task | Owner | Depends on | Status |
|----|------|-------|------------|--------|
| S18-00 | Branch creation off `mlx/sprint-17-hist-tree-reduce`; baseline JSON refresh to `.cache/profiling/sprint18/baseline_*.json`; PR #9 status check | @ml-product-owner | PR #9 status | **COMPLETE** |
| S18-01 | Ground-truth post-S17 attribution via Metal System Trace on gate config; output: `docs/sprint18/attribution.md` with ±1 ms error bars; Ramos checkpoint if attribution diverges from plan estimates | @performance-engineer | S18-00 | **COMPLETE** |
| S18-02 | Ablation sweep L1a / L1b / L1c / L1d × {BLOCK_SIZE 128, 256} × {bins 32, 128} at N=10k RMSE d6; PROPOSE → CRITIQUE → IMPLEMENT-draft → VERIFY-project → REFLECT; output: `docs/sprint18/ablation.md` | @research-scientist | S18-00, S18-01 | **COMPLETE — L1a chosen** |
| S18-03 | Implement chosen L1 variant at `kernel_sources.h:115–225`; PROPOSE → CRITIQUE → IMPLEMENT → VERIFY → REFLECT harness; single-file kernel-string change; preserve D1c reduction block | @ml-engineer | S18-02 + Ramos approval | **COMPLETE — `19fa5ce6cc` (initial `abc4c229f9` failed BUG-S18-001)** |
| S18-04 | Parity tests across DEC-008 envelope (approxDim ∈ {1,3}, N ∈ {1k,10k,50k}, all losses, 32/128 bins, 50 iter, d6); 100-run determinism fixture on 10k/RMSE/d6/128 | @qa-engineer | S18-03 | **COMPLETE — S18-04b: 108/108 PASS, 100/100 determinism PASS** |
| S18-05 | Stage-profiler before/after on 18-config grid; output: `.cache/profiling/sprint18/{before,after}_*.json` + `docs/sprint18/results.md` delta table | @performance-engineer | S18-03 | **COMPLETE — S18-05b: -56.6% to -85.5% all configs** |
| S18-06 | Update `benchmarks/check_histogram_gate.py` reference baseline to Sprint 17 after-JSON; verify CI gate continuity; intentional-regression dry-run | @mlops-engineer | S18-05 | **COMPLETE** |
| S18-07 | Code review: barrier correctness, threadgroup-memory bound, BUG-001 stride-partition ownership, D1c reduction integrity | @code-reviewer | S18-03, S18-06 | **COMPLETE — PASS** |
| S18-08 | Security pass on S18-03: kernel string injection surface; no new externally-controlled buffer sizes | @security-auditor | S18-03 | **COMPLETE — PASS** |
| S18-09 | Metal System Trace re-capture on gate config; confirm accumulation phase <5 ms, `simdHist` residency on-chip; output: appendix in `docs/sprint18/results.md` | @performance-engineer | S18-05 | **COMPLETE** |
| S18-10 | `docs/sprint18/` full population, `CHANGELOG-DEV.md` entry, `ARCHITECTURE.md` update, DEC-011/DEC-012, `bug_s18_001.md` post-mortem; same PR as S18-03 | @technical-writer | S18-02, S18-05, S18-09 | **IN PROGRESS** |

---

## Same-PR docs standing order

All Sprint 18 source changes land in a single PR. That PR must include:

- `docs/sprint18/{README, design, attribution, ablation, results, non_goals}.md`
- `CHANGELOG-DEV.md` Sprint 18 entry
- `catboost/mlx/ARCHITECTURE.md` histogram-accumulation section updated
- `.claude/state/DECISIONS.md` DEC-011 (chosen L1 variant) + DEC-012 (if 32 KB ceiling-trade adopted)
- `docs/sprint19/plan_prior.md` draft

No doc, no merge.

---

## Sprint 17 starting point

Sprint 17 after (full table): [`docs/sprint17/results.md`](../sprint17/results.md).  
Representative anchor: **N=10k, RMSE, depth=6, 128 bins → `histogram_ms` = 28.75 ms (82.3% of iter time)**.
