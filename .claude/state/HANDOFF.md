# Handoff — CatBoost-MLX

> Last updated: 2026-04-20 (Sprint 21 CLOSED — 6/6 A1 exit gates PASS, 0× perf shipped; Sprint 22 kickoff)

## Current state

- **Branch**: `mlx/sprint-21-hist-tg-reduction` (Sprint 21 — state files committed, no production source modified)
- **Tip commit**: `a7a206b90d` — Sprint 21 D1-R4 synthesis + Sprint 22 kickoff plan
- **Campaign**: Operation Verstappen — battle 6 of 9 CLOSED. Battle 7 of 9 (Sprint 22) ready to cut.
- **Open PRs** (stacked on RR-AMATOK/catboost-mlx): #9 → #10 → #11 → #12 (Sprint 20 empirical record); #13 = Sprint 21 pending Ramos open

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

## Sprint 22 — KICKOFF

### Charter: T2 sort-by-bin integration (single-lever sprint)

**Authority**: `docs/sprint21/d1r4_synthesis.md` is the authoritative lever-ranking and D0 specification input.
**Sprint scaffold**: `docs/sprint22/README.md`

### Gate dependency — campaign viability

The entire Verstappen ≥1.5× campaign gate depends on Sprint 22 D0:

| S22-D0 ratio | T2 verdict | Projected cumulative e2e | Verstappen gate |
|---|---|---|---|
| ≤ 0.45 | PASS (optimistic) | ~1.96× | CLEARED +46 pp |
| ≤ 0.60 | PASS (conservative) | ~1.62–1.88× | CLEARED |
| > 0.60 | FALSIFIED | ~1.07× (current) | FAIL — pivot required |

**Kill-switch**: `hist_ms(T2) / hist_ms(T1) > 0.60` at gate config measured in-situ under real argsort-permuted training-loop partitions → T2 drops to RESEARCH; Sprint 22 pivots to tree-search restructure scoping with 0× in-sprint perf deliverable.

### R8 honest position

- Cumulative through Sprint 21 close: **~1.07× over Sprint 16-class baseline** (entirely from S17 D1c, S18 L1a, S19 T1 contributions)
- Gap to 1.5× Verstappen gate: **40% residual speedup required from Sprint 22 onward**
- Sprint 22 projected contribution (if T2 D0 PASS): +1.37× to +1.83× at gate config (midpoint 1.76×)
- Sprint 22 conservative (ratio 0.50): +1.51× — clears gate by 12 pp; stacks with S19-11 readback elimination

**No soft path exists.** L2 falsified. Variant A falsified. T3b falsified. T2 is the single mechanism-backed lever. If T2 is falsified at D0, the campaign gate is not reachable within a single additional sprint on the current kernel structure. Honest accounting — do not soften in downstream handoffs.

### Current viable-set

| Lever | Status | Sprint |
|---|---|---|
| **T2 sort-by-bin** | VIABLE rank #1 — pending S22-D0 in-situ | Sprint 22 integration |
| **S19-11 EvalAtBoundary readback** | CARRY-FORWARD — compound with T2 | Sprint 22 in-sprint (0.5–1 day) |
| **Tree-search restructure / dispatch inversion** | RESEARCH — speculative 1.5–2×, weeks of cost | Sprint 23 research spike (if T2 PASS) or Sprint 22 pivot (if T2 FAIL) |
| All other levers | FALSIFIED or SHIPPED | — |

### Next actions

1. **@ml-engineer**: S22-D0 — implement `DispatchHistogramT2` scratch variant guarded by `CATBOOST_MLX_HISTOGRAM_T2=1`; measure `ratio = hist_ms(T2) / hist_ms(T1)` at gate config via `bench_boosting --per-kernel-profile`, 3 independent runs × 49 warm iters; document in `docs/sprint22/d0_t2_production_shape.md`. A1-G6 discipline: scratch-only, no kernel_sources.h commit until D0 PASS verdict.
2. Ramos opens PR #13 for Sprint 21 (this branch).
3. If S22-D0 PASS: @qa-engineer proceeds to S22-D1 parity sweep.
4. If S22-D0 FAIL: Ramos re-decides campaign direction (tree-search restructure research spike vs gate re-scope).

## Standing orders (carried forward to Sprint 22)

- **No `Co-Authored-By: Claude` trailer** in any commit message — global policy.
- **RR-AMATOK fork only** — do not push or PR to `catboost/catboost` upstream.
- **DEC-012 one-structural-change-per-commit** — Sprint 22 D2 integration will require 4–5 atomic commits.
- **A1-G6 discipline applies to S22-D0 scratch** — no production source committed until D0 kill-switch clears.
- **Do not soften R8** — the gap-to-1.5× arithmetic in `docs/sprint21/d1r4_synthesis.md §5` is the honest view and propagates unchanged.
- **Do not skip S22-D0 kill-switch** — ratio-transfer from D1-R2 synthetic harness is unproven. Shipping T2 on in-harness evidence alone repeats the DEC-017 Sprint 20 failure mode.

## Prior sprints — status unchanged

- **Sprint 21** — CLOSED. PR #13 pending (Ramos opens). 0× perf, A1 measurement record.
- **Sprint 20** — CLOSED. PR #12 OPEN stacked on #11. T3b DEC-017 RETIRED.
- **Sprint 19** — CLOSED. PR #11 OPEN stacked on #10. T1 DEC-016 SHIPPED (−2.3% e2e).
- **Sprint 18** — CLOSED. PR #10 OPEN stacked on #9. L1a DEC-011 SHIPPED (−66.8% histogram_ms).
- **Sprint 17** — CLOSED. PR #9 OPEN. D1c DEC-009 SHIPPED (−89–93% histogram_ms).
- **Sprints 0–16** — merged to master.
