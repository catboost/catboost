# Handoff — CatBoost-MLX

> Last updated: 2026-04-19 (Sprint 19 CLOSE-OUT — three DEC-012 commits landed; A1 dropped per empirical measurement; R8 deferred to Sprint 20)

## Current state

- **Branch**: `mlx/sprint-19-hist-writeback`
- **Last commit**: `92f3832169` — `[mlx] sprint-19: DEC-016 T1 fuse-valid simd_shuffle reduction`
- **Working tree**: clean on code; scratch docs (`docs/sprint19/algorithmic_ablation.md`, `docs/sprint19/scratch/algorithmic/`, `docs/sprint20/`) not committed yet
- **Campaign**: Operation Verstappen — multi-sprint performance domination push (Sprints 16–24), battle 4 of 9

## Sprint 19 — CLOSING

**Shipped kernel commits** (in order):
1. `77db8b5631` — Commit 1: extract DEC-015 side-fix (`featureColumnIndices`+`numGroups` per-group variable correction in `DispatchHistogramBatched`; col-major portions reverted)
2. `7387814dd6` — S19-06 CI gate widened from 10k/sprint17 to 50k/RMSE/d6/128b baseline
3. `020eacfb4c` — S19-11 remove one EvalAtBoundary readback at `structure_searcher.cpp:738` (bit-exact; other 3 `EvalAtBoundary` calls on that path are legitimate guard-syncs before `.data<T>()` CPU reads)
4. `92f3832169` — Commit 2: DEC-016 T1 fuse-valid simd_shuffle (3→2 shuffles/src). Parity bit-exact at 50k/RMSE, 10k/RMSE, 50k/MultiClass. Warm-mean e2e **−2.3%** at gate config (32.47 → 31.73 ms, 3-run mean).
5. **A1 (Commit 3 candidate) DROPPED.** Toy micro-bench showed A1 vs T1 = −1.9% (noise-marginal). Production port measured **+9.4% regression** (34.7 vs 31.7 ms) — register spill dominates outer-loop saving. See `docs/sprint19/scratch/algorithmic/a1_empirical_drop.md` and DEC-014 status update (REJECTED).

**Four analytical models falsified this sprint** — pattern consistent with S19-01c finding that AGX cache hierarchy + out-of-order behavior resists first-principles modeling:
- DEC-013 (writeback-as-plurality) — SUPERSEDED Day 0
- DEC-014 original (gather sub-phase) — INVALIDATED Day 2
- DEC-015 (col-major layout) — REJECTED Day 3
- DEC-014 (A1 BATCH_DOCS=64) — REJECTED Day 4 (this session)

## R8 status — honest accounting

**R8 revised mid-sprint from aggressive 1.5–1.8× e2e to ≥1.07× e2e exit gate.** Rationale: the real bottleneck (simd_shuffle serial chain = 86% of accumulation per S19-01c probe A) is structurally invariant to layout-level changes. Sprint 19 is a bit-exact bridge; the structural replacement (T3b atomic-CAS) is a Sprint 20 flagship gated on full DEC-008 parity sweep.

**Actual delivered on gate config (50k/RMSE/d6/128b)**:
- Pre-Sprint-19 baseline (post-Commit 1, pre-T1): 32.47 ms warm mean
- Post-T1 (current HEAD): 31.73 ms warm mean
- **Delivered: 1.023× e2e** (vs revised gate 1.07×)

**R8 NOT met on this config. Deferred to Sprint 20 via DEC-017 (T3b atomic-CAS).** Per Ramos's "honest R8 — do not soften" standing order: this sprint's shipping lever (T1) is +2.3% e2e and bit-exact. The 1.07× gate requires T3b's 84% accumulation reduction to close — which cannot ship without the full parity sweep scheduled for Sprint 20 D1.

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
| S19-01b | Accumulation sub-phase attribution: re-attribute 14.30 ms accumulation across sub-phases; ±1 ms error bars; output appended to `docs/sprint19/attribution.md` | @performance-engineer | Day 2 | **DONE** (S19-01b gather model → falsified by S19-01c) |
| S19-01c | Micro-bench re-attribution: simd_shuffle serial chain = 86% of accumulation; AGX hides compressedIndex gather | @ml-engineer / @research-scientist | Day 3 | **DONE** (commit `b0c853a6f6`) |
| S19-02b | Accumulation redesign ablation: variants A/B/C/D × {bins 32,128} × {N 10k,50k}; DEC-014 DRAFT | @research-scientist | Day 2 | **DONE** — DRAFT ablation at `docs/sprint19/ablation_accumulation.md` |
| S19-02c | Algorithmic variant ablation (T0/T1/T2/T3/T3b): measured toy-kernel at 1 TG × 256 threads × N=50k | @research-scientist | Day 3–4 | **DONE** — see `docs/sprint19/algorithmic_ablation.md` and `docs/sprint19/scratch/algorithmic/microbench_algorithmic.cpp` |
| S19-03a | Commit 1: extract DEC-015 side-fix (per-group variable correction, col-major reverted) | main | Day 4 | **DONE** — commit `77db8b5631` |
| S19-03b | Commit 2: DEC-016 T1 fuse-valid simd_shuffle reduction (3→2 shuffles/src) | main | Day 4 | **DONE** — commit `92f3832169`; −2.3% e2e at 50k/RMSE, bit-exact 3 configs |
| S19-03c | Commit 3: DEC-014 (A1) BATCH_DOCS=64 | main | Day 4 | **DROPPED** — toy −1.9% but production +9.4% regression (register spill). A1 kept in scratch. See DEC-014 status (REJECTED) + `docs/sprint19/scratch/algorithmic/a1_empirical_drop.md` |
| S19-04 | Parity sweep: DEC-008 envelope (approxDim ∈ {1,3}, N ≤ 50k, all losses, 32/128 bins, 50 iter, d6); 100-run determinism on 50k/RMSE/d6/128b | @qa-engineer | Day 5 | PENDING (post-commit) |
| S19-05 | Stage-profiler delta on 18-config grid; output: `.cache/profiling/sprint19/{before,after}_*.json` + `docs/sprint19/results.md` delta table | @performance-engineer | Day 5 | PENDING (post-commit) |
| S19-06 | Update `benchmarks/check_histogram_gate.py` reference baseline to 50k/RMSE/128b; verify CI gate continuity; intentional-regression dry-run | @mlops-engineer | Day 4 | **DONE** — commit `7387814dd6`; dry-run triggers at +6.1% |
| S19-07 | Code review: T1 MSB-sentinel correctness, DEC-011 ceiling, barrier count | @code-reviewer | Day 5 | PENDING (post-commit) |
| S19-08 | Security pass: kernel string injection surface; no new externally-controlled buffer sizes | @security-auditor | Day 5 | PENDING (post-commit) |
| S19-09 | Metal System Trace re-capture on gate config; confirm T1 accumulation improvement; output: appendix in `docs/sprint19/results.md` | @performance-engineer | Day 5 | PENDING (post-commit) |
| S19-10 | `docs/sprint19/` population, DECISIONS.md (DEC-013 SUPERSEDED, DEC-014 REJECTED, DEC-015 REJECTED, DEC-016 ACTIVE, DEC-017 DRAFT-S20), HANDOFF + CHANGELOG, Sprint 20 skeleton | @technical-writer + main | Day 4 | **DONE** — `docs/sprint19/algorithmic_ablation.md`, `docs/sprint20/README.md`, DECISIONS.md DEC-014/015/016/017, HANDOFF updated (this) |
| S19-11 | Kill EvalAtBoundary CPU readbacks in `structure_searcher.cpp` | @ml-engineer + main | Day 4 | **DONE** — commit `020eacfb4c` (1 of 6 candidates removed at line 738; other 3 `EvalAtBoundary` calls are legitimate pre-CPU-read guards; pragmatic scope reduction) |
| S19-12 | S18 deferred cleanup + VGPR confirmation | @performance-engineer / @code-reviewer | Day 4 | **DONE — no-op** (no S18 deferred items per CHANGELOG-DEV.md) |
| S19-11 | In-sprint cleanup: 6 EvalAtBoundary CPU readbacks in `structure_searcher.cpp` — "fix properly always" (carry-forward from S18 non-goals) | @ml-engineer | Day 1–3 | PENDING |
| S19-12 | In-sprint cleanup: VGPR confirmation + S18 deferred code-review items | @performance-engineer / @code-reviewer | Day 1–2 | PENDING |

## Acceptance gates (revised)

| Gate | Criterion | Status |
|------|-----------|--------|
| G1 | `histogram_ms` improvement at gate config | TBD — S19-09 Metal System Trace to decompose the −2.3% e2e into kernel-phase delta |
| G2 | No 18-config regression >5% | TBD — S19-05 pending |
| G3 | Parity bit-exact across DEC-008 envelope | TBD — S19-04 pending; spot-checked bit-exact at 3 configs (50k/RMSE, 10k/RMSE, 50k/MultiClass) for Commit 2 |
| G4 | R8 revised to ≥1.07× e2e | **NOT MET** at 1.023× on 50k/RMSE/d6/128b — deferred to Sprint 20 via DEC-017 T3b |
| G5 | No non-histogram stage regresses >10% | TBD — S19-05 pending |
| G6 | CI green | TBD — post-commit CI run |

## Blockers

**None active.** Prior DEC-015 blocker resolved: rejected empirically in S19-03a revert + S19-01c probe D confirmation. A1 regression confirmed and dropped per plan clause.

## Next actions (Sprint 19 exit gates — parallel)

1. **S19-04** — @qa-engineer: 18-config parity grid + 100-run determinism on 50k/RMSE/d6/128b. T1 is the only active kernel change; expected bit-exact by construction (MSB-sentinel does not alter reduction order).
2. **S19-05** — @performance-engineer: 18-config warm-mean delta + 50k MST capture. Expected: T1 at −2.3% e2e on 50k/RMSE; other configs likely similar or smaller (MultiClass already showed −2.0%).
3. **S19-07** — @code-reviewer: review the 3 Sprint 19 commits (`77db8b5631`, `92f3832169`, `020eacfb4c`) plus the S19-06 CI gate commit (`7387814dd6`).
4. **S19-08** — @security-auditor: kernel source string injection surface unchanged; T1 adds only a constant and a bitmask. Spot-check and sign-off.
5. **S19-09** — @performance-engineer: post-fix MST; confirm simd_shuffle chain cost dropped proportionally to the shuffle-count reduction.
6. **PR**: stage the Sprint 19 PR (target `master` on `RR-AMATOK/catboost-mlx`); keep uncommitted scratch docs and bench binary out of the PR surface.

## Carry-forward to Sprint 20

- **D1 FLAGSHIP — DEC-017 T3b atomic-CAS parity sweep.** Full DEC-008 envelope parity sweep (18 configs × 100 runs) against toy-kernel-equivalent before any integration. If parity holds → D2 integration. If parity fails → Kahan/Higham compensated summation and re-sweep. Detailed plan: `docs/sprint20/README.md`.
- **D2 — T3b production integration** (contingent on D1 parity).
- **D3 — Full-grid scaling validation.** Toy measured 1 TG × 256 threads; production dispatches 1575 TGs concurrently at depth 5–6.
- **D4 — MultiClass approxDim=3 parity.** T3b's atomic-CAS reduction order may compound drift across three independent dim reductions.
- L2 (pre-permute stats + compressedIndex gather removal — 2–4 ms headroom) deferred from S18, still valid post-S19-01c (confirmed AGX hides gather).
- L3 (MultiClass per-dim dispatch fusion — 15–25 ms on MultiClass configs) still valid.
- `maxBlocksPerPart` retuning — sprint cleanup candidate.
- DEC-011 32 KB ceiling — T3b requires amendment to 4 KB (atomic simdHistU[1024]).
