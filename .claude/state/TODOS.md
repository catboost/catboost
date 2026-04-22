# Active Tasks — CatBoost-MLX

> Coverage: Sprints 0–15 reconstructed from git/agent-memory on 2026-04-15. Sprint 16+ is source of truth.
> Last header refresh: 2026-04-22 (post-PR #20; no active sprint).

## Current state (2026-04-22)

- **Branch**: `master` is the working base. No active sprint.
- **Tip**: `71aabaa842` (PR #20 merge — latent-bugs cleanup).
- **Production kernel**: v5 (`784f82a891`), shipped S24 D0. ULP=0 structural parity across the DEC-008 envelope.
- **R8 (honest)**: 1.01× e2e vs S16 baseline. Verstappen ≥1.5× gate remains retroactively failed from S24 D0. Do not round or inflate; the 1.90× figure is documented as superseded.
- **Carry-forward backlog**: ALL CLEARED 2026-04-22 via PR #20 (`71aabaa842`). See bottom of this file.
- **Open PRs**: none.
- **DEC-027**: deferred. Not on this backlog. Reserved for a dedicated future research sprint.

Active-sprint sections for Sprints 16 and 19 and the Operation Verstappen Campaign Scoreboard
have been removed from this header. Their closeout state is preserved in the per-sprint
sections below and in `HANDOFF.md` / `DECISIONS.md` / `CHANGELOG-DEV.md`.

---

## Sprint 21 — A1 Measurement Sprint (CLOSED — 6/6 exit gates PASS, 0× perf shipped)

**Branch**: `mlx/sprint-21-hist-tg-reduction`
**Campaign**: Operation Verstappen — battle 6 of 9 — A1 measurement sprint (pivot from TG-count reduction after D0 kill-switch fired)
**Verdict**: All 6 A1 exit gates PASS. Zero production source modified (A1-G6 discipline). Two levers retired, one promoted to viable-set.

### Closed

- [x] S21-D0 — D0 kill-switch: fixed per-TG overhead = 2.5% ± 1.3% at depth 6 (R²=0.9989). Kill-switch threshold 10% NOT met → FIRED. Variant A never activated. DEC-018 RETIRED. — @performance-engineer **DONE** (`a0c473e3b7`)
- [x] S21-D1-R3 — Host-side `eval()` sync instrumentation in `bench_boosting` (`--per-kernel-profile`). Per-dispatch stdev < 5% of mean at gate config. — @ml-engineer **DONE** (`ac378d8de6`)
- [x] S21-D1-R1 — L2 stats pre-permute direct mechanism test. Zero-gather upper bound (`stat = 1.0f`) at 1664-TG production shape: +2.61% slower (12.6 pp below 10% gate). **FALSIFIED.** DEC-019. — @ml-engineer + @performance-engineer **DONE** (`fedf9d5348`)
- [x] S21-D1-R2 — T2 sort-by-bin production-shape micro-bench. Sort+accum at 1664-TG shape: −64.8% (band 63.6–66.7%), clearing 50% gate by 28–34 pp. Parity gate B: max ULP 64, mass conservation 0 ULP. **VIABLE.** DEC-020. — @ml-engineer + @performance-engineer **DONE** (`13322feaca`)
- [x] S21-D1-R4 — Sprint 22 kickoff synthesis. Lever ranking with mechanism-direct gates; D0 kill-switch spec at ratio > 0.60; R8 honest ledger. — @technical-writer **DONE** (`a7a206b90d`)

---

## Sprint 22 — T2 Sort-by-Bin Integration (CLOSED — 4/4 gates PASS, R8 1.90×)

**Branch**: `mlx/sprint-22-t2-integration`
**Campaign**: Operation Verstappen — battle 7 of 9 — single-lever integration sprint
**Gate config**: 50k/RMSE/d6/128b
**Lever**: T2 sort-by-bin (DEC-020)
**Verdict**: **CLOSED. Verstappen ≥1.5× gate CLEARED. Cumulative R8 = 1.90×. PR #14 pending (Ramos opens).**

### Closed

- [x] S22-D0 — In-situ T2 integration probe. `hist_ms(T2) / hist_ms(T1)` = 0.328× cross-session at gate config (inside optimistic band). Kill-switch (>0.60) NOT triggered. — @ml-engineer **DONE** (`4333c82a7e`)
- [x] S22-D1 — 18-config DEC-008 parity sweep. **FAILED 18/18** at D0 T2 kernel (ULP 1,327–2,583,206). Triggered four-phase diagnostic arc (D1a→D1b→D1c). Root cause: `maxPartDocs = ceil(N/K)` uniform-partition assumption overflows under real argsort-permuted skewed splits. — @qa-engineer / @ml-engineer **DONE** (`73baadf445` contains full arc)
- [x] S22-D1a — Blit-ordering hypothesis (fill_gpu pool reuse). **REFUTED**: fill_gpu is compute, not blit; eval barriers did not fix parity. — @ml-engineer **DONE** (bundled in `73baadf445`)
- [x] S22-D1b — Depth-parity indexing hypothesis. **REFUTED**: even-depth failure pattern explained by split-distribution artifact, not indexing. — @ml-engineer **DONE** (bundled in `73baadf445`)
- [x] S22-D1c — Root-cause: `bench_boosting.cpp:526` `maxPartDocs = ceil(numDocs / numActiveParts)` overflow. Depth-1 on 50k: sizes [442, 49558] vs maxPartDocs=25000; 24558-doc overflow into neighbor slot. — @ml-engineer **DONE** (`docs/sprint22/d1c_t2_troubleshoot.md`, bundled in `73baadf445`)
- [x] S22-D2 — Option III slab-by-partOffsets fix. `sortedDocs` reorganized to per-(groupIdx, statIdx) slabs of size `numDocs` indexed by `partOffsets[partIdx]`. Buffer 5.2 MB at gate config (vs 333 MB worst-case for Option I). Overflow structurally impossible. — @ml-engineer **DONE** (`73baadf445`)
- [x] S22-D3 — Parity exit gate (independent QA). 18/18 ULP=0 bit-exact; 100/100 determinism; edge cases EC-1–EC-5 all ULP=0. Bug β (atomic-scatter float drift) does not exist — retires S21 D1-R4 Kahan concern (DEC-022). — @qa-engineer **GATE PASS** (`docs/sprint22/d3_parity_gate.md`, bundled in `73baadf445`)
- [x] S22-D4 — Perf exit gate. Ratio 0.317× cross-session (band 0.315–0.319×); S22 multiplier 1.778×; cumulative R8 = **1.07 × 1.778 = 1.90×**; Verstappen gate cleared by 40 pp. N=1000 ratio exceedances (0.651–0.694×) are structural amortization artifact, non-blocking (gate anchored at 50k). — @performance-engineer **GATE PASS** (`docs/sprint22/d4_perf_gate.md`, bundled in `73baadf445`)
- [x] S22-D5 — Code review exit gate. 0 blockers. 7 nits; NIT-6 (stray blank line) removed by security auditor; 6 nits deferred to Sprint 23. — @code-reviewer **GATE PASS** (`docs/sprint22/d5_code_review.md`, bundled in `73baadf445`)
- [x] S22-D6 — Security audit exit gate. 0 CRITICAL, 0 HIGH; bounds-proof confirms D1c overflow class structurally eliminated; max-safe-N 14.3M docs (286× headroom over 50k); zero secrets/PII. 2 MEDIUM advisory (DoS/robustness, bench harness scope). — @security-auditor **GATE PASS** (`docs/sprint22/d6_security_audit.md`, bundled in `73baadf445`)

---

## Sprint 23 — T2 Scratch→Production Promotion + NIT Catalog + Tree-Search Research (CLOSED)

**Branch**: `mlx/sprint-23-t2-promotion`
**Campaign**: Operation Verstappen — battle 8 of 9 — **CLOSED 2026-04-21**
**Gate config**: 50k/RMSE/d6/128b
**Verdict**: PASS (pre-existing-bug footnote). D0 4/4 gates (G1 with errata). R1 DEFERRED. R2 FALSIFIED. R8 1.90× unchanged. PR #15 pending.

### D0 — T2 scratch→production promotion (blocking)

- [x] S23-D0 — Promote `kernel_sources_t2_scratch.h` contents into `catboost/mlx/kernels/kernel_sources.h` under `kT2SortSource` and `kT2AccumSource` sections. Move `DispatchHistogramT2` from `bench_boosting.cpp` into `catboost/mlx/methods/histogram.cpp` with production-quality API (CB_ENSURE error handling, factored kernel registration, clean public interface). Remove `CATBOOST_MLX_HISTOGRAM_T2` compile-time flag and make T2 the default dispatch path. Per DEC-012: 4 atomic commits landed (tip `84529b47ed`). — @ml-engineer **DONE — KILL-SWITCH TRIPPED**

  **Verdict**: PASS (kill-switch tripped on pre-existing bug; records corrected; proceed to R1/R2). Parity sweep: 17/18 ULP=0; config #8 (N=10000/RMSE/128b) BIMODAL ~50/50 at (0.48231599, 0.48231912) — 105 ULP gap. Pre-existing in S22 D2/D3 tip `73baadf445`; not introduced by promotion. Gate config #14 (N=50000/RMSE/128b) 100/100 deterministic at 0.47740927. R8 1.90× record unaffected. See `docs/sprint23/d0_bimodality_verification.md` and DEC-023.

### NIT cleanup batch (from D5 code review, deferred to S23)

- [ ] S23-NIT1 — Replace inline literals (`256u`, `128u`, `0x7Fu`, `4u`) in T2 kernels with named constants from `kHistHeader` (`BLOCK_SIZE`, `BINS_PER_BYTE`, `FEATURES_PER_PACK`). File: `kernel_sources_t2_scratch.h` (or promoted `kernel_sources.h`). — @ml-engineer
- [ ] S23-NIT2 — Pull duplicate `offBase` arithmetic + `129u` magic number into a named constant (`BIN_OFFSETS_STRIDE = 129u`) in `kHistHeader` or a new `kT2Header`; add clarifying comment (`128 bins + 1 total`). — @ml-engineer
- [ ] S23-NIT3 — T2-accum empty-partition hardening: add explicit `if (partSize == 0) return;` at start of T2-accum mirroring T2-sort, eliminating reliance on float-to-uint zero-bit-aliasing from `init_value=0.0f`. — @ml-engineer
- [ ] S23-NIT4 — Add host-side CB_ENSURE that `maxBlocksPerPart == 1` when T2 is active; document the 1-block-per-partition constraint in the T2-sort kernel header comment. — @ml-engineer
- [ ] S23-NIT5 — Remove unused `numTGs` uniform from both T2-sort and T2-accum `input_names` and from host-side `numTGsArr`; remove mention from kernel header comments. — @ml-engineer
- [ ] S23-NIT7 — Harmonize feature 1-3 bin mask: change T2-accum feat 1-3 `& 0xFFu` to `& 0x7Fu` (matching T2-sort feat-0 and DEC-016 envelope), or use `& 0xFFu` for all features matching T1. Bundle with NIT1 in one pass. — @ml-engineer

### Tree-search research track (rank #2 post-T2, from d1r4_synthesis.md §3)

- [x] S23-R1 — EvalAtBoundary readback elimination (S19-11 carry-forward). **VERDICT: NOT VIABLE — architectural mismatch.** 3 live sites in `structure_searcher.cpp` (lines 290, 609, 705) are on Depthwise/Lossguide paths only. gate config (oblivious tree) never enters these paths; bench_boosting has its own inline loop that bypasses structure_searcher.cpp entirely. 0/3 sites replaced. Gate perf delta: 0 ms (sites unreachable from gate path). Parity: unchanged (17/18 ULP=0, config #8 bimodal unchanged). R8: 1.90× unchanged. Per-site: A=SKIP (depthwise restructure, no gate perf), B=SKIP (lossguide, parity harness gap), C=SKIP (scope exceeds budget, lossguide restructure). Forward: Depthwise/Lossguide benchmarks required before these sites can be targeted. See `docs/sprint23/r1_evalatboundary.md`. — @ml-engineer **DONE (no-op — kill-switch fired)**
- [x] S23-R2 — Dispatch inversion research spike. **FALSIFIED — structural algebraic blocker.** `H[f][b] = Σ_p h_p[f][b]` is not invertible; all five mask mechanisms (A–E) are algebraically or empirically rejected. Atomic contention 64× worse than DEC-023 trigger. Mechanism E is DEC-017 T3b with a different label (same +42.3% regression predicted). Day-1 kill-switch invoked; Day 2 not exercised. Do not re-enter without new mask-mechanism evidence. DEC-025 FALSIFIED. See `docs/sprint23/r2_dispatch_inversion_spike.md`. — @research-scientist **DONE (no-op — kill-switch fired)**

### Carry-forward

- [ ] S19-11 — See S23-R1 above (same task, merged).

---

---

## Sprint 24 — DEC-023 Atomic-Float Race Fix (CLOSED)

**Branch**: `mlx/sprint-24-dec023-fix`
**Campaign**: Operation Verstappen — battle 9 of 9 — **CLOSED 2026-04-21**
**Verdict**: D0 PASS on parity (DEC-023 RESOLVED). FAIL on R8 preservation (retroactive retreat 1.90× → 1.01×). Verstappen ≥1.5× gate failed. PR #16 pending.

### D0 — DEC-023 fix (DONE)

- [x] S24-D0 — v5: all-feature T1-style SIMD-shuffle accumulation. All four features (0-3) in T2-accum read from `docIndices` via T1's SIMD-shuffle + linear fold. T2-sort removed from dispatch. ULP=0 structural. All 4 acceptance criteria PASS. Commit `784f82a891`. R8 consequence: 1.90× → 1.01×. — @ml-engineer **DONE**

- [x] S24-SO-1 — Parity sweep protocol documented in `docs/sprint24/README.md §5`. Standing order effective S23 D0 forward: ≥5 runs per non-gate; 100 runs at gate. — @technical-writer **DONE**

---

## Sprint 25 — DEC-026 Cascade-Robust GAIN Research (CLOSED — FALSIFIED)

**Branch**: `mlx/sprint-25-dec026-cascade` (stacked on S24)
**Campaign**: Post-Verstappen research — R8 recovery investigation
**R8 entry position**: 1.01× (honest post-S24)
**R8 exit position**: 1.01× (unchanged — research FALSIFIED at G1)
**Verdict**: FALSIFIED at G1. Kill-switch fired cleanly on day 1. G2–G5 not attempted.
**Authority**: `DECISIONS.md DEC-026` (FALSIFIED 2026-04-21); `docs/sprint25/g1_epsilon_calibration.md` (verdict doc)

### Research track — DEC-026 (owner: @research-scientist)

- [x] S25-G1 — Epsilon calibration study. **FALSIFIED** 2026-04-21. 180-run empirical sweep (18 configs × 5 runs × 2 kernels) established ε_min = 2.200e-03 vs ε_max⁺ = 1.043e-07 → safety ratio 4.74e-05 (target ≥ 2.0). Path 5 flip gaps span the full range of legitimate top-2 separations; no ε threads the needle. Kill-switch fired. — @research-scientist **DONE (FALSIFIED)**

- [x] S25-G2 — Tiebreak implementation in scoring kernel. **CANCELLED** — blocked on G1, never attempted. — @ml-engineer **DONE (CANCELLED)**

- [x] S25-G3 — T2 Path 5 rebuild. **CANCELLED** — blocked on G2. — @ml-engineer **DONE (CANCELLED)**

- [x] S25-G4 — 18-config parity sweep + determinism. **CANCELLED** — blocked on G3. — @qa-engineer **DONE (CANCELLED)**

- [x] S25-G5 — Model-quality validation. **CANCELLED** — blocked on G4. — @qa-engineer **DONE (CANCELLED)**

### Sprint 25 merge gate

FALSIFIED. R8 stays at 1.01×. Verstappen ≥1.5× gate remains retroactively failed from S24 D0.
v5 (`catboost/mlx/kernels/kernel_sources.h` at `784f82a891`) is the final production kernel.
S25 closeout ships the G1 empirical artifacts + verdict doc as falsification evidence; no
production code changes.

**DEC-027 (deferred)**: alternative accumulation paths (e.g., XGBoost-style per-feature
deterministic radix-sum) are acknowledged as a possible future research direction but are
NOT opened at S25 close. Ramos to dedicate time later.

---

## Sprint 20–25 Backlog (one-line each, expanded per sprint)

- Sprint 20: T3b atomic-CAS — CLOSED, FALSIFIED (DEC-017 RETIRED). PR #12 merged.
- Sprint 21: A1 measurement — CLOSED, 0× perf, T2 promoted to viable-set. PR #13 merged.
- Sprint 22: T2 sort-by-bin integration — CLOSED, R8 1.90× (superseded by S24). Verstappen gate cleared at S22. PR #14 merged. Note: S22 D3 parity corrected to 17/18 (DEC-020 footnote + DEC-023).
- Sprint 23: T2 scratch→production promotion — CLOSED. 8 commits. D0 PASS (pre-existing bug). R1 DEFERRED. R2 FALSIFIED. DEC-023/024/025. PR #15 merged.
- Sprint 24: DEC-023 v5 fix — CLOSED. DEC-023 RESOLVED. R8 1.90× → 1.01× retroactive. Verstappen ≥1.5× failed. DEC-026 opened. PR #16 **merged 2026-04-22** (`1385e056ca`).
- Sprint 25: DEC-026 cascade-robust GAIN research — CLOSED, FALSIFIED at G1 (safety ratio 4.74e-05 vs 2.0 target). R8 stays at 1.01×. DEC-027 deferred for future dedicated research. PR #17 **merged 2026-04-22** (`5caa6e64cf`).
- CI unblock (PR #18): MLX 0.31+ CLI fix + stale version-pin cleanup + BUG-001 MAE sentinel narrowing. **Merged 2026-04-22** (`9b0c03fec2`). No production code changes.

---

## Carry-forward from pre-Sprint 16 — ALL CLEARED 2026-04-22 (PR #20, `71aabaa842`)

- **BUG-007** (MITIGATED 2026-04-22): Python wrapper pre-sorts group_ids (`core.py:1131-1137`); C++ `BuildDatasetFromArrays` now CB_ENSUREs sortedness. See `KNOWN_BUGS.md`.
- **bench_boosting K=10 anchor mismatch** (RESOLVED Sprint 8, TODO-022): `1.78561831` is canonical; prior `2.22267818` was a stale mismatched-params capture. See `CHANGELOG.md:27`.
- **Sibling S-1** (latent, guard hardened): `kHistOneByte` writeback race now guarded by `constexpr` + `static_assert(maxBlocksPerPart == 1, ...)` at `histogram.cpp:126`. Fix only if multi-block dispatch is ever needed.

**DEC-027 (deferred, unchanged)**: not on this backlog. Reserved for a dedicated future research sprint per S25 closeout.
