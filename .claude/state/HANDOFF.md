# Handoff — CatBoost-MLX

> Last updated: 2026-04-18 by @ml-engineer (S19-03 DEC-015 implementation — BLOCKED: performance gate not met)

## Current state

- **Branch**: `mlx/sprint-19-hist-writeback`
- **Last commit**: `dccb7ec0a2` (S16 sync-storm fix — unchanged; S19-03 work is uncommitted)
- **Working tree**: DIRTY — 5 files modified (DEC-015 implementation, NOT committed — gate blocker)
- **Campaign**: Operation Verstappen — multi-sprint performance domination push (Sprints 16–24), battle 4 of 9

## Sprint 18 — CLOSED (PR #10 OPEN, awaiting merge)

Sprint 18 delivered the L1a `simdHist` accumulator re-architecture. All gates passed. The sprint included a significant pivot (BUG-S18-001) — the initial kernel failed all parity configs by 6 orders of magnitude before the fixed kernel landed at commit `19fa5ce6cc`.

**Final result**: `histogram_ms` -56.6% to -85.5% across all 18 configs. Gate config (N=10k, RMSE, 128b): 28.75 → 9.56 ms (-66.8%). Parity 108/108 bit-exact (ULP=0). Barriers reduced 9 → 6. Reduction depth γ_12 → γ_7.

**PR #10**: `mlx/sprint-18-hist-privhist-tile` → `master` on `RR-AMATOK/catboost-mlx`. OPEN. No blockers.

**Sprint 18 carry-forward (now Sprint 19 starting point)**: N=50k configs converge to ~15 ms `histogram_ms` in steady state. The writeback (global-atomic) phase is the plurality cost at large N — the S18-05b profile identified this as the floor that L1a could not eliminate. Sprint 19 targets this floor directly.

## Sprint 19 — ACTIVE (PIVOTED: accumulation redesign)

**Branch**: `mlx/sprint-19-hist-writeback` (cut from `mlx/sprint-18-hist-privhist-tile@463de74efa`; name reflects original scope — history over cosmetics)  
**Lever**: Accumulation redesign (L_accum) — **pivot from L_writeback after S19-01 ground-truth falsified writeback-as-plurality**  
**Gate config**: 50k/RMSE/128b (unchanged)

**Pivot summary**: S19-01 attribution (commit `d7ea14e28c`) showed writeback = 0.79 ms (5%), accumulation = 14.30 ms (93%). The S18 "~15 ms writeback floor" was a mis-scaling from N=10k. R8 fired at 1.02–1.04× e2e. Ramos chose Option 2: pivot to accumulation redesign. DEC-013 SUPERSEDED. DEC-014 DRAFT.

**Projection (pending S19-01b/S19-02b; aggressive target still live)**:
- `histogram_ms`: **−30% min** on 50k/RMSE/128b (32% accumulation reduction = ~30% histogram_ms, since accumulation = 93%; target range 8–11 ms from 15.46 ms baseline)
- `iter_total_ms`: −30% min target (1.5× aggressive still live; baseline 21.12 ms → target 14–17 ms)
- Championship 50k exit: pending S19-02b numbers

**Baselines**: S18 after-JSONs copied to `.cache/profiling/sprint19/baseline/` (18 configs). Gate config steady-state: `histogram_ms` 15.46 ms (ground-truth, S19-01), `iter_total_ms` 21.12 ms.

**DEC-014**: DRAFT — accumulation redesign. 4 candidate variants under ablation (S19-02b). Locks at S19-02b close (end of Day 2).

**Accumulation variants under ablation (S19-02b)**:
- (A) Wider batch accumulation — under ablation
- (B) Coalesced threadgroup staging — under ablation
- (C) Per-feature specialization — under ablation
- (D) Different ownership granularity — under ablation

## Sprint 19 task table

| ID | Task | Owner | Day/Phase | Status |
|----|------|-------|-----------|--------|
| S19-00 | Branch cut from S18 tip; baseline JSON copy; state file scaffold; docs/sprint19/README scaffold | @ml-product-owner / @technical-writer | Day 0 | **DONE** |
| S19-01 | Ground-truth writeback attribution via Metal System Trace on gate config (50k/RMSE/128b); output: `docs/sprint19/attribution.md` with ±1 ms error bars; R8 trigger check | @performance-engineer | Day 1 | **COMPLETE-BUT-SUPERSEDED** (evidence correct, premise falsified — see DEC-014) |
| S19-02 | Ablation sweep (writeback variants a/b/c); DEC-013 draft written; premise invalidated by S19-01; output: `docs/sprint19/ablation.md` | @research-scientist | Day 1–2 | **COMPLETE-BUT-SUPERSEDED** (variant (c) 3.0 ms projection not supported by ground truth) |
| S19-01b | Accumulation sub-phase attribution: re-attribute 14.30 ms accumulation across sub-phases; ±1 ms error bars; output appended to `docs/sprint19/attribution.md` | @performance-engineer | Day 2 | IN PROGRESS |
| S19-02b | Accumulation redesign ablation: variants A/B/C/D × {bins 32,128} × {N 10k,50k}; PROPOSE → CRITIQUE → IMPLEMENT-draft → VERIFY-project → REFLECT; DEC-014 lock; output appended to `docs/sprint19/ablation.md` | @research-scientist | Day 2 | IN PROGRESS |
| S19-03 | Implement winning accumulation redesign from DEC-014 at `kernel_sources.h`; preserve DEC-011 32 KB ceiling | @ml-engineer | Day 3 | PENDING |
| S19-04 | Parity sweep: DEC-008 envelope (approxDim ∈ {1,3}, N ≤ 50k, all losses, 32/128 bins, 50 iter, d6); 100-run determinism on 50k/RMSE/d6/128b | @qa-engineer | Day 3–4 | PENDING |
| S19-05 | Stage-profiler delta on 18-config grid; output: `.cache/profiling/sprint19/{before,after}_*.json` + `docs/sprint19/results.md` delta table | @performance-engineer | Day 4 | PENDING |
| S19-06 | Update `benchmarks/check_histogram_gate.py` reference baseline to S18 after-JSON; verify CI gate continuity; intentional-regression dry-run | @mlops-engineer | Day 4 | PENDING |
| S19-07 | Code review: accumulation phase correctness, DEC-011 ceiling, barrier count | @code-reviewer | Day 4–5 | PENDING |
| S19-08 | Security pass: kernel string injection surface; no new externally-controlled buffer sizes | @security-auditor | Day 4–5 | PENDING |
| S19-09 | Metal System Trace re-capture on gate config; confirm accumulation improvement; output: appendix in `docs/sprint19/results.md` | @performance-engineer | Day 5 | PENDING |
| S19-10 | `docs/sprint19/` full population, `CHANGELOG-DEV.md` entry, `ARCHITECTURE.md` update, DEC-013 (SUPERSEDED) + DEC-014 (lock from DRAFT), `docs/sprint19/` subdocs | @technical-writer | Day 5–6 | IN PROGRESS (Day 0 scaffold done; pivot updates applied) |
| S19-11 | In-sprint cleanup: 6 EvalAtBoundary CPU readbacks in `structure_searcher.cpp` — "fix properly always" (carry-forward from S18 non-goals) | @ml-engineer | Day 1–3 | PENDING |
| S19-12 | In-sprint cleanup: VGPR confirmation + S18 deferred code-review items | @performance-engineer / @code-reviewer | Day 1–2 | PENDING |

## Acceptance gates (revised)

| Gate | Criterion |
|------|-----------|
| G1 | **`histogram_ms` −30% min** on 50k/RMSE/128b (accumulation phase = 93%; 32% accumulation reduction ≈ 30% histogram_ms) |
| G2 | No 18-config regression >5% |
| G3 | Parity 108/108 bit-exact across DEC-008 envelope |
| G4 | `iter_total_ms` −30% min on 50k/RMSE/128b (aggressive 1.5× target still live pending S19-01b/S19-02b) |
| G5 | No non-histogram stage regresses >10% |
| G6 | CI green |

## Blockers

**ACTIVE BLOCKER — S19-03 DEC-015 performance gate not met.**

DEC-015 (column-major compressedIndex transposed view) implementation is complete, builds clean, passes parity and determinism. The S19-01b attribution model predicted 2.13× e2e speedup. Direct measurement shows ~0.98× (essentially no improvement).

**Measured gate results (50k/RMSE/d6/128b, 5 runs each):**

| Binary | warm mean | warm mean runs | Result |
|--------|-----------|----------------|--------|
| `bench_boosting_ref` (pre-DEC-015) | 33.7–34.2 ms | 5× | baseline |
| `bench_boosting` (DEC-015) | 34.3–35.7 ms | 5× | ~0.98× |

**Parity:** 18/18 PASS, 0 ULP vs current production reference (bit-exact). BENCH_FINAL_LOSS=0.48042428 identical across all configs.

**Determinism:** 100/100 runs identical.

**Why the S19-01b prediction failed:** The analytical model assumed AGX processes the row-major `compressedIndex[docIdx * lineSize + col]` gather as 25 sequential L2 requests with 4 stall-rounds per batch. The measured speedup near 1.0× implies one or more of these is true:
- AGX hardware prefetcher / L2 hardware coalescer handles the strided gather better than the model assumed (effectively reducing 25-CL cost toward 1–2 rounds)
- `docIdx` values within a sorted partition are clustered in a small physical address range, causing L2 cache re-use between SIMD groups
- The transposed layout introduces a new bottleneck: with `totalNumDocs = 50,000`, adjacent feature columns are 200 KB apart in `CompressedDataTransposed_`, potentially causing increased L2 thrashing when multiple groups dispatch simultaneously
- The actual bottleneck is not compressedIndex gather latency — the S19-01b latency model needs re-attribution with a more controlled micro-benchmark

**The DEC-015 5-file change set is NOT committed.** Per the task spec: "do not commit broken work."

## Next actions

1. **REQUIRED — Re-attribution micro-benchmark:** Build an isolated Metal kernel that sweeps row-major vs col-major `compressedIndex` access patterns at N = {10k, 50k} with varying lineSize. Measure wall time per kernel call. This will confirm whether AGX hardware is hiding the scatter cost or whether the bottleneck is elsewhere entirely.
2. **REQUIRED — Decide on DEC-015 disposition:** If micro-benchmark confirms ~0 speedup from layout change, DEC-015 should be marked REJECTED (layout change has no cost but also no benefit; no reason to keep the extra 5 MB buffer). If micro-benchmark shows speedup at a different N or lineSize, narrow the scope.
3. **S19-03 re-spec needed from @ml-product-owner:** The S19-01b model is falsified by measurement. A new sub-phase attribution pass (instrumented micro-benchmarks, not analytical model) is needed before choosing the next kernel intervention.
4. S19-11 and S19-12 are independent — can proceed now.
5. PR #10 (Sprint 18) merge is independent — unblock with Ramos review.

## Carry-forward to Sprint 20+

- L2 (pre-permute stats + compressedIndex gather removal — 2–4 ms headroom) deferred from S18, still valid.
- L3 (MultiClass per-dim dispatch fusion — 15–25 ms on MultiClass configs) deferred from S18; zero effect on RMSE gate config; Sprint 20 candidate.
- `maxBlocksPerPart` retuning (library-path dead code for csv_train) — Sprint cleanup candidate.
- DEC-011 occupancy constraint (1 tg/SM from 32 KB ceiling): Sprint 19 writeback phase reuses `simdHist` post-barrier-6 as on-chip staging; if this forces 2-pass structure at large N, Sprint 20 must re-negotiate buffer geometry.
