# Handoff — CatBoost-MLX

> Last updated: 2026-04-17 by @technical-writer (Sprint 19 S19-00 kickoff scaffold)

## Current state

- **Branch**: `mlx/sprint-19-hist-writeback`
- **Last commit**: S19-00 kickoff (branch cut + baselines + state files + README scaffold)
- **Campaign**: Operation Verstappen — multi-sprint performance domination push (Sprints 16–24), battle 4 of 9

## Sprint 18 — CLOSED (PR #10 OPEN, awaiting merge)

Sprint 18 delivered the L1a `simdHist` accumulator re-architecture. All gates passed. The sprint included a significant pivot (BUG-S18-001) — the initial kernel failed all parity configs by 6 orders of magnitude before the fixed kernel landed at commit `19fa5ce6cc`.

**Final result**: `histogram_ms` -56.6% to -85.5% across all 18 configs. Gate config (N=10k, RMSE, 128b): 28.75 → 9.56 ms (-66.8%). Parity 108/108 bit-exact (ULP=0). Barriers reduced 9 → 6. Reduction depth γ_12 → γ_7.

**PR #10**: `mlx/sprint-18-hist-privhist-tile` → `master` on `RR-AMATOK/catboost-mlx`. OPEN. No blockers.

**Sprint 18 carry-forward (now Sprint 19 starting point)**: N=50k configs converge to ~15 ms `histogram_ms` in steady state. The writeback (global-atomic) phase is the plurality cost at large N — the S18-05b profile identified this as the floor that L1a could not eliminate. Sprint 19 targets this floor directly.

## Sprint 19 — ACTIVE

**Branch**: `mlx/sprint-19-hist-writeback` (cut from `mlx/sprint-18-hist-privhist-tile@463de74efa`)  
**Lever**: Two-phase histogram writeback reduction (L_writeback)  
**Gate config shift**: S18 used 10k/RMSE/128b (where accumulation dominated). S19 shifts to **50k/RMSE/128b** — the config where the writeback lever has force.

**Projection (aggressive, Ramos-approved)**:
- `histogram_ms`: 1.7–2.2× improvement on gate config (15.52 ms baseline → ~7–9 ms target)
- `iter_total_ms`: 1.5–1.8× improvement on gate config (21.12 ms baseline → ~12–14 ms target)
- Championship 50k exit: revised from 0.75–0.85 s conservative to **0.55–0.70 s**
- **R8 constraint**: if S19-01 attribution doesn't justify 1.5×+, projection revises DOWN before S19-03 commits

**Baselines**: S18 after-JSONs copied to `.cache/profiling/sprint19/baseline/` (18 configs). Gate config steady-state: `histogram_ms` 15.52 ms (mean), `iter_total_ms` 21.12 ms.

**DEC-013**: PLACEHOLDER — Two-phase writeback reduction over batched-atomic. Status: DRAFT, locks at S19-02 ablation close.

## Sprint 19 task table

| ID | Task | Owner | Day/Phase | Status |
|----|------|-------|-----------|--------|
| S19-00 | Branch cut from S18 tip; baseline JSON copy; state file scaffold; docs/sprint19/README scaffold | @ml-product-owner / @technical-writer | Day 0 | **DONE** |
| S19-01 | Ground-truth writeback attribution via Metal System Trace on gate config (50k/RMSE/128b); output: `docs/sprint19/attribution.md` with ±1 ms error bars; R8 trigger check | @performance-engineer | Day 1 | PENDING |
| S19-02 | Ablation sweep: (a) two-phase reduction, (b) batched-atomic, (c) CHOSEN two-phase + prefix-scan × {BLOCK_SIZE 128,256} × {bins 32,128} at N=50k RMSE d6; PROPOSE → CRITIQUE → IMPLEMENT-draft → VERIFY-project → REFLECT; output: `docs/sprint19/ablation.md` | @research-scientist | Day 1–2 | PENDING |
| S19-03 | Implement chosen variant at `kernel_sources.h`; preserve DEC-011 32 KB ceiling; reuse `simdHist[0..1023]` as on-chip staging post-barrier-6 | @ml-engineer | Day 2 | PENDING |
| S19-04 | Parity sweep: DEC-008 envelope (approxDim ∈ {1,3}, N ≤ 50k, all losses, 32/128 bins, 50 iter, d6); 100-run determinism on 50k/RMSE/d6/128b | @qa-engineer | Day 2–3 | PENDING |
| S19-05 | Stage-profiler delta on 18-config grid; output: `.cache/profiling/sprint19/{before,after}_*.json` + `docs/sprint19/results.md` delta table | @performance-engineer | Day 3 | PENDING |
| S19-06 | Update `benchmarks/check_histogram_gate.py` reference baseline to S18 after-JSON; verify CI gate continuity; intentional-regression dry-run | @mlops-engineer | Day 3 | PENDING |
| S19-07 | Code review: writeback phase correctness, two-phase reduction ordering, DEC-011 ceiling, barrier count | @code-reviewer | Day 3–4 | PENDING |
| S19-08 | Security pass: kernel string injection surface; no new externally-controlled buffer sizes | @security-auditor | Day 3–4 | PENDING |
| S19-09 | Metal System Trace re-capture on gate config; confirm writeback phase <5 ms; output: appendix in `docs/sprint19/results.md` | @performance-engineer | Day 4 | PENDING |
| S19-10 | `docs/sprint19/` full population, `CHANGELOG-DEV.md` entry, `ARCHITECTURE.md` update, DEC-013 (lock from DRAFT), `docs/sprint19/` subdocs | @technical-writer | Day 4–5 | IN PROGRESS (Day 0 scaffold done) |
| S19-11 | In-sprint cleanup: 6 EvalAtBoundary CPU readbacks in `structure_searcher.cpp` — "fix properly always" (carry-forward from S18 non-goals) | @ml-engineer | Day 1–3 | PENDING |
| S19-12 | In-sprint cleanup: VGPR confirmation + S18 deferred code-review items | @performance-engineer / @code-reviewer | Day 1–2 | PENDING |

## Blockers

None at kickoff.

## Next actions

1. Parent dispatches **@performance-engineer** (S19-01 — writeback attribution on gate config, R8 trigger check).
2. Parent dispatches **@research-scientist** (S19-02 — ablation sweep, variant selection).
3. S19-11 and S19-12 can begin in parallel once S19-03 branch point is stable.
4. PR #10 (Sprint 18) merge is independent — unblock with Ramos review.

## Carry-forward to Sprint 20+

- L2 (pre-permute stats + compressedIndex gather removal — 2–4 ms headroom) deferred from S18, still valid.
- L3 (MultiClass per-dim dispatch fusion — 15–25 ms on MultiClass configs) deferred from S18; zero effect on RMSE gate config; Sprint 20 candidate.
- `maxBlocksPerPart` retuning (library-path dead code for csv_train) — Sprint cleanup candidate.
- DEC-011 occupancy constraint (1 tg/SM from 32 KB ceiling): Sprint 19 writeback phase reuses `simdHist` post-barrier-6 as on-chip staging; if this forces 2-pass structure at large N, Sprint 20 must re-negotiate buffer geometry.
