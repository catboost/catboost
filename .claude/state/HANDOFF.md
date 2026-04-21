# Handoff — CatBoost-MLX

> Last updated: 2026-04-21 (Sprint 23 CLOSED — D0 PASS/pre-existing-bug; R1 DEFERRED/harness gap; R2 FALSIFIED/structural; PR #15 pending Ramos open)

## Current state

- **Branch**: `mlx/sprint-23-t2-promotion`
- **Tip commit**: `5b9827ad93` — R2 doc + DEC-025 (Sprint 23 complete; 8 commits total)
- **Campaign**: Operation Verstappen — battle 8 of 9 CLOSED. Sprint 23 verdict: PASS with pre-existing-bug footnote.
- **Open PRs** (stacked on RR-AMATOK/catboost-mlx): #9 → #10 → #11 → #12 → #13 → #14 → **#15 pending Ramos open** (Sprint 23)
- **Known bugs**: BUG-T2-001 in `.claude/state/KNOWN_BUGS.md` — features 1-3 atomic-float race (DEC-023 OPEN, S24 D0 fix target)

## Sprint 23 — CLOSED

### Verdict: PASS (pre-existing-bug footnote). D0 4/4 gates satisfied (G1 with errata). R1 DEFERRED. R2 FALSIFIED. R8 1.90× unchanged.

**Branch tip at close**: `5b9827ad93` (8 commits total)
**Date closed**: 2026-04-21

| Track | Verdict | Commits | Key finding |
|-------|---------|---------|-------------|
| D0 — T2 scratch→production promotion | PASS (kill-switch tripped, pre-existing) | 4d1eda1f4c, 2df0bb1aed, eaf05bc21d, 84529b47ed | 17/18 ULP=0; config #8 bimodal pre-existing; gate 100/100 unaffected |
| D0 close-out + records correction | DONE | dd1c9e0a6e, be530059da | S22 D3 verdict corrected; DEC-023 opened; KNOWN_BUGS.md created |
| R1 — EvalAtBoundary elimination | DEFERRED (harness gap) | None (no-op) | 0/3 sites reachable from gate; harness must add --grow-policy flag |
| R2 — Dispatch inversion spike | FALSIFIED (structural blocker) | 441f632b10, 5b9827ad93 | H[f][b] = Σ_p h_p[f][b] not invertible; contention 64× worse than DEC-023 |

**R8 cumulative**: **1.90×** (unchanged through S23; no new perf contribution). Verstappen ≥1.5× gate cleared by 40 pp.

### D0 detail

**Parity sweep**: 17/18 ULP=0 deterministic + 1 latent bimodal at config #8 (N=10000/RMSE/128b, 105 ULP gap, ~50/50).
**Kill-switch**: TRIPPED — bimodality pre-existing in S22 D2/D3 scratch tip `73baadf445`; not introduced by promotion. Verified by @qa-engineer (`docs/sprint23/d0_bimodality_verification.md`). Promotion is innocent.
**Gate config #14** (50k/RMSE/128b): 100/100 deterministic at 0.47740927 — unaffected. R8 1.90× record clean.
**Records corrected**: S22 D3 "18/18 ULP=0" corrected to 17/18 ULP=0 + 1 latent bimodal (1-run-per-config protocol had 50% miss probability for ~50/50 race). DEC-022 scope qualifier added: "bug β does not exist" scoped to gate config only. DEC-020 footnote updated.

### R1 detail — DEFERRED

**Verdict**: NOT VIABLE at gate — architectural mismatch. 0/3 sites replaced. DEFERRED, not retired.
**Finding**: Sites A (`:290`), B (`:609`), C (`:705`) in `structure_searcher.cpp` are all on Depthwise/Lossguide paths. Gate config runs SymmetricTree (oblivious). `bench_boosting` uses its own inline oblivious-tree loop and never calls `structure_searcher.cpp`. The ~0.3 ms/iter standalone cost estimate was a theoretical projection from S16, not a measured gate-config value.
**Per-site**: A=SKIP (depthwise restructure, no gate perf), B=SKIP (lossguide, parity harness gap), C=SKIP (scope exceeds budget, fundamental lossguide restructure).
**Gate perf**: Unchanged. iter_total_ms = 19.098 ms. R8 = 1.90×.
**Forward**: Re-entry gated on `bench_boosting --grow-policy` flag addition or separate Depthwise/Lossguide benchmark harness (DEC-024).
**Doc**: `docs/sprint23/r1_evalatboundary.md`

### R2 detail — FALSIFIED

**Verdict**: FALSIFIED permanently. Structural algebraic blocker. Do not re-enter without new mask-mechanism evidence.
**Blocker**: `H[f][b] = Σ_p h_p[f][b]` is not invertible. No mask mechanism in §2 (A/B/C/D/E) can reconstruct per-partition bin sums from the single-histogram total without cost equivalent to a second full histogram pass. Atomic contention under inversion is 64× worse than the current DEC-023 trigger. Net dispatch math at gate is neutral at best (5.82 ms headroom consumed by mask cost). Mechanism E is DEC-017 T3b revisited — same contention failure mode, already falsified empirically (+42.3%).
**Doc**: `docs/sprint23/r2_dispatch_inversion_spike.md` (DEC-025)

## Sprint 22 — CLOSED

### Verdict: 4/4 exit gates PASS, R8 1.90×, Verstappen gate CLEARED

Sprint 22 integrated T2 sort-by-bin (DEC-020 VIABLE from S21). D0 in-situ probe passed kill-switch at 0.328×. D1 parity sweep failed 18/18, triggering a four-phase diagnostic arc that isolated a uniform-partition ceiling overflow bug (`bench_boosting.cpp:526 maxPartDocs = ceil(N/K)`). Option III structural fix (slab-by-partOffsets, 5.2 MB vs 333 MB worst-case for Option I) passed all four exit gates.

| Deliverable | Commit | Verdict |
|---|---|---|
| D0 in-situ T2 probe | `4333c82a7e` | PASS — ratio 0.328× optimistic |
| D1+D1a+D1b+D1c+D2 Option III fix + D3/D4/D5/D6 | `73baadf445` | 4/4 GATES PASS |

**R8 final**: cumulative = 1.07 × 1.778 = **1.90×**. Verstappen ≥1.5× gate cleared by 40 pp.

**Key side-finding**: bug β (atomic-scatter float drift, S21 D1-R4 §3) does not exist. 100/100 determinism confirmed. DEC-022 retires the Kahan concern.

**Scratch discipline maintained**: `kernel_sources.h` unmodified. T2 ships in `kernel_sources_t2_scratch.h` under `CATBOOST_MLX_HISTOGRAM_T2=1` guard. Promotion to `kernel_sources.h` is Sprint 23 D0.

---

## Sprint 21 — CLOSED

### A1 verdict: 6/6 exit gates PASS, 0× perf shipped

Sprint 21 was declared a measurement-only sprint after the D0 kill-switch fired (variant A fixed overhead = 2.5% << 10% gate). All five A1 deliverables completed:

| Deliverable | Commit | Verdict |
|---|---|---|
| D0 kill-switch | `a0c473e3b7` | FIRED — fixed overhead 2.5% ± 1.3% (R²=0.9989) |
| D1-R3 per-kernel instrumentation | `ac378d8de6` | DONE — `--per-kernel-profile` stable, stdev < 5% of mean |
| D1-R1 L2 direct mechanism test | `fedf9d5348` | FALSIFIED — zero-gather +2.61% slower (DEC-019) |
| D1-R2 T2 sort-by-bin micro-bench | `13322feaca` | VIABLE — −64.8% at 1664-TG shape (DEC-020) |
| D1-R4 synthesis + S22 plan | `a7a206b90d` | DONE — mechanism-direct gates; R8 honest ledger |

**A1-G6 discipline**: zero production source files modified on this branch. All D1-R1/R2 kernel variants were scratch/local only and have been restored.

### Two levers retired

- **DEC-018 TG-count reduction variant A — RETIRED** (never activated; D0 kill-switch fired; specification error captured — gate tested T1 amortization proxy, not T3b shape-restoration mechanism)
- **DEC-019 L2 stats pre-permute — FALSIFIED** (zero-gather upper bound +2.61% slower at 1664-TG depth-6 shape; AGX out-of-order + hardware prefetcher fully hides stats gather, consistent with S19-01c probe D)

### One lever promoted to viable-set

- **DEC-020 T2 sort-by-bin — VIABLE (rank #1 entering Sprint 22)**
  - D1-R2 at 1664-TG production shape: −64.8% histogram_ms (band 63.6–66.7%, 2σ ±2.7–4.4%)
  - Clears 50% gate by 28–34 pp
  - Gate B parity: max ULP 64, mass conservation 0 ULP across 812,800 bins
  - Ratio-transfer risk (synthetic identity-permuted harness → production argsort-permuted data) is **unproven** — Sprint 22 D0 must establish this before any integration commit

## Sprint 23 — CLOSED

### Charter: T2 scratch→production promotion + NIT cleanup + tree-search research

**Authority**: `docs/sprint22/d5_code_review.md §4` (NIT catalog, 6 items deferred); `docs/sprint21/d1r4_synthesis.md §3` (tree-search rank #2)
**Sprint scaffold**: `docs/sprint23/README.md`

### R8 honest position (post-S22)

- **Cumulative**: **1.90×** over Sprint 16-class baseline (1.07 × 1.778×)
- **Verstappen gate** (≥1.5×): **CLEARED** by 40 pp — campaign goal met
- Sprint 23 does not need to deliver additional R8 to satisfy the gate. Remaining work is production-quality promotion and the next research-track investment.

### Sprint 23 task set

| Task | Type | Owner |
|------|------|-------|
| S23-D0 — T2 scratch→production promotion | Blocking | @ml-engineer |
| S23-NIT1–NIT5,NIT7 — 6 deferred nit cleanup | Housekeeping | @ml-engineer |
| S23-R1 — EvalAtBoundary readback elimination | Compound perf | @ml-engineer |
| S23-R2 — Dispatch inversion research spike | Research | @research-scientist |

**D0 is blocking**: `kernel_sources_t2_scratch.h` contents must be promoted into `kernel_sources.h` and `DispatchHistogramT2` promoted into `catboost/mlx/methods/histogram.cpp` with production-quality API (CB_ENSURE, factored registration, clean public interface) before any championship benchmark run. The `CATBOOST_MLX_HISTOGRAM_T2=1` flag is removed; T2 becomes the default dispatch.

**NIT bundle**: address all 6 deferred nits (NIT1–NIT5, NIT7) in a single pass alongside the D0 promotion. See `docs/sprint22/d5_code_review.md §4`.

**S23-R1** (EvalAtBoundary): six `EvalAtBoundary` CPU readbacks in `structure_searcher.cpp` (`:275`, `:593`, `:653`, `:686`) — ~0.3 ms/iter standalone. Carried from S19-11. Scoping reference: `docs/sprint16/sync_inventory.md`. Bounded 0.5–1 day.

**S23-R2** (Dispatch inversion): if no concrete design surfaces within 2 days, declare unreachable for the Verstappen campaign window and defer to Sprint 24+.

### Next actions

1. **Ramos opens PR #15 for Sprint 23** (stacked on #14, branch `mlx/sprint-23-t2-promotion`).
2. **S24 D0 is the next blocking item**: DEC-023 atomic-float race fix (features 1-3 threadgroup-local reduce or int-atomic fixed-point). Kill-switch: ratio < 0.45× at gate → escalate. Acceptance: config #8 becomes 10/10 deterministic; 18/18 sweep clean; gate 100/100. See `.claude/state/DECISIONS.md DEC-023` and `.claude/state/KNOWN_BUGS.md BUG-T2-001`.
3. **S24 standing order**: parity sweep protocol ≥5 runs per non-gate config + 100 runs at gate unconditionally (carried from S23 D0 standing order).

## Standing orders (carried forward to Sprint 23)

- **No `Co-Authored-By: Claude` trailer** in any commit message — global policy.
- **RR-AMATOK fork only** — do not push or PR to `catboost/catboost` upstream.
- **DEC-012 one-structural-change-per-commit** — S23-D0 promotion will require 3–4 atomic commits.
- **Do not soften R8** — cumulative 1.90× is the honest post-S22 position; do not round to 2×.
- **Scratch discipline**: `kernel_sources.h` is the production file; modifications require a DEC-012 atomic commit with parity re-verify.

## Prior sprints — status unchanged

- **Sprint 23** — CLOSED. PR #15 pending (Ramos opens). T2 promoted to production (8 commits). R8 1.90× unchanged. D0 PASS (pre-existing bug). R1 DEFERRED. R2 FALSIFIED. DEC-023/024/025 opened. See S23 CLOSED section above.
- **Sprint 22** — CLOSED. PR #14 pending (Ramos opens). T2 SHIPPED, R8 1.90× (record stands). Verstappen gate cleared. S22 D3 verdict corrected to 17/18 (see DEC-020 footnote + DEC-023).
- **Sprint 21** — CLOSED. PR #13 pending (Ramos opens). 0× perf, A1 measurement record.
- **Sprint 20** — CLOSED. PR #12 OPEN stacked on #11. T3b DEC-017 RETIRED.
- **Sprint 19** — CLOSED. PR #11 OPEN stacked on #10. T1 DEC-016 SHIPPED (−2.3% e2e).
- **Sprint 18** — CLOSED. PR #10 OPEN stacked on #9. L1a DEC-011 SHIPPED (−66.8% histogram_ms).
- **Sprint 17** — CLOSED. PR #9 OPEN. D1c DEC-009 SHIPPED (−89–93% histogram_ms).
- **Sprints 0–16** — merged to master.
