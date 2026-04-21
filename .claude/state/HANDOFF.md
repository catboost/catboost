# Handoff — CatBoost-MLX

> Last updated: 2026-04-17 by @technical-writer (Sprint 18 S18-10 docs complete, PR pending)

## Current state

- **Branch**: `mlx/sprint-18-hist-privhist-tile`
- **Last commit**: `01366aee95` (current tip — S18-10 docs commit pending)
- **Campaign**: Operation Verstappen — multi-sprint performance domination push (Sprints 16–24)

## What just happened — Sprint 18 L1a histogram accumulator re-architecture

**All gates PASS:**

| Gate | Criterion | Result |
|------|-----------|--------|
| S18-G1 | histogram_ms reduction ≥35% on gate config (≤18.7 ms) | **PASS — -66.8%** (28.75 → 9.56 ms; 9.1 ms margin) |
| S18-G2 | No config regresses >5% (18-config sweep) | **PASS — all 18 improved -56.6% to -85.5%** |
| S18-G3 | RMSE/Logloss ulp≤4, MultiClass ulp≤8; 100-run determinism | **PASS — 108/108 checkpoints ulp=0; 100/100 determinism bit-exact** |
| S18-G4 | No non-histogram stage regresses >10% on gate config | **PASS — all secondary stages improved** |
| S18-G5 | CI gate continuity | **PASS** |
| S18-07 | Code review | **PASS** |
| S18-08 | Security audit | **PASS** |

**Kernel change** (`catboost/mlx/kernels/kernel_sources.h`, commit `19fa5ce6cc`):
- `float privHist[HIST_PER_SIMD]` (4 KB/thread, 1 MB/threadgroup device-memory spill) → `threadgroup float simdHist[8][1024]` (32 KB on-chip, at Apple Silicon limit).
- Zero-init loop eliminated. Per-thread stride accumulation → cooperative 32-doc batch loop.
- D1c intra-SIMD butterfly removed (DEC-012). Cross-SIMD 8-term fold (DEC-009) unchanged.
- Barriers: 9 → 6 per dispatch. Reduction depth γ_12 → γ_7.

**The pivot (BUG-S18-001):**
Initial L1a kernel (commit `abc4c229f9`) failed all 18 parity configs by 6 orders of magnitude (4–20M ULP). Determinism PASS throughout (consistent wrong answer). Two compounding structural flaws: 1/32 doc-inclusion rate (stride/ownership mismatch) + 32× butterfly amplification (butterfly over shared slots). Root cause: D1c's reduction was ported without re-deriving the algebraic role of the butterfly under the new `simdHist` shared layout. Fixed at commit `19fa5ce6cc`. Full post-mortem: `docs/sprint18/bug_s18_001.md`.

**Perf result (18 configs, DEC-008 envelope: `approxDim ∈ {1,3}`, `N ≤ 50k`, depth 6, 50 iter):**
- Gate config: 28.75 → 9.56 ms (-66.8%). Beats nominal ablation projection by 5.94 ms.
- Range: -56.6% (50k/MultiClass/32b) to -85.5% (1k/Logloss/128b).
- The second-order win beyond spill elimination: removing the butterfly reduced dispatch barriers from 9 to 6.

**Parity result:**
- 108/108 checkpoints bit-exact across DEC-008 grid (18 configs × 6 checkpoints).
- All ULPs = 0 — cleaner than Sprint 17's 35/36 outcome.
- Higham bound tightened: γ_12 → γ_7 (~7.2e-7 → ~4.2e-7).

## Active Sprint 18 tasks

| Agent | Task | Status |
|-------|------|--------|
| @ml-product-owner | S18-00 branch cut + baseline refresh | DONE |
| @performance-engineer | S18-01 attribution | DONE — `docs/sprint18/attribution.md` |
| @research-scientist | S18-02 ablation, L1a choice | DONE — `docs/sprint18/ablation.md` |
| @ml-engineer | S18-03 kernel implementation | DONE — `19fa5ce6cc` (after BUG-S18-001 fix) |
| @qa-engineer | S18-04b parity re-run | DONE — `7ab4e8e804` |
| @performance-engineer | S18-05b perf delta + S18-09 MST | DONE — `da303866ef` |
| @mlops-engineer | S18-06 CI gate update | DONE |
| @code-reviewer | S18-07 code review | DONE — PASS |
| @security-auditor | S18-08 security pass | DONE — PASS |
| @technical-writer | S18-10 docs | DONE — this commit |

## Blockers

None.

## Next action

1. Commit S18-10 docs: `[mlx] sprint-18: S18-10 docs — results + BUG-S18-001 post-mortem + DEC-011/012`.
2. Push `mlx/sprint-18-hist-privhist-tile` to `RR-AMATOK/catboost-mlx`.
3. Open Sprint 18 PR vs `master` (RR-AMATOK fork only — never upstream per DEC-004).
4. **Sprint 19 kickoff** — @ml-product-owner to rank levers using S18-05b profiles.

## Carry-forward to Sprint 19

**Primary lever (L_writeback):** The writeback (global-atomic) phase floors N=50k configs at ~15 ms (observed in S18-05b). It is now the plurality cost at large N. Batched-atomic writeback or shared-memory prefix-scan reduction of the per-SIMD histograms before global writeback is the likely Sprint 19 L1.

**Secondary levers (deferred from Sprint 18 non-goals):**
- **L2**: Pre-permute stats + compressedIndex gather removal — 2–4 ms standalone headroom; pairs cleanly with L3's grid geometry change.
- **L3**: MultiClass per-dim dispatch fusion — 15–25 ms on MultiClass configs; zero effect on RMSE gate config. Sprint 19 headline candidate for MultiClass.
- `maxBlocksPerPart` retuning — library-path dead code for csv_train; Sprint 19 cleanup candidate.

**Constraint**: Sprint 19 results must continue to bound to DEC-008 envelope (`approxDim ∈ {1,3}`, `N ≤ 50k`, depth 6) unless the envelope is explicitly re-validated.

**Threadgroup memory note**: L1a occupies the full 32 KB Apple Silicon ceiling (DEC-011). Sprint 19 kernel changes that require additional threadgroup memory must redesign the buffer layout or tile more aggressively.
