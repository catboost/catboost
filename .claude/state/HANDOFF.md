# Handoff — CatBoost-MLX

> Last updated: 2026-04-19 (Sprint 20 CLOSED via empirical falsification — T3b +42% regression vs toy −84.4%; DEC-017 RETIRED; PR #12 OPEN stacked on PR #11; Sprint 21 lever scoped to TG-count reduction)

## Current state

- **Branch**: `mlx/sprint-20-hist-atomic-cas`
- **Tip commit**: `78697fff79` — D2b design + DEC-017 retirement — ABANDON verdict
- **Prior tip (Sprint 19)**: `4113200529` — S19-13 T1 MSB-sentinel envelope guard
- **Working tree**: clean; branch pushed to origin
- **Campaign**: Operation Verstappen — multi-sprint performance domination push (Sprints 16–24), **battle 5 of 9 CLOSED AT 0× NET**
- **Open PRs** (stacked on RR-AMATOK/catboost-mlx): #9 → #10 → #11 → **#12**

## Sprint 20 — CLOSED, empirical falsification, 0× ship

**Verdict**: T3b threadgroup-atomic-CAS accumulator **FALSIFIED at D2 integration**. Toy-kernel −84.4% accumulation did not translate to production partition-fragmented dispatch; +42.3% regression at gate config (50k/RMSE/d6/128b). DEC-017 retired. No kernel change shipped. Sprint ships the empirical record and Sprint 21 redesign plan.

### Shipped commits (3, all docs/state — no kernel changes)

| Commit | Content |
|---|---|
| `9216f4941c` | D1 parity sweep — T3b 18/18 bit-exact PASS, 100/100 determinism PASS |
| `9079ad3873` | D2 falsification record — +42% regression with stage attribution |
| `78697fff79` | D2b design + DEC-017 retirement — ABANDON verdict, Sprint 21 lever scoped |

### D1 result — parity held (see `docs/sprint20/d1_parity.md`)

- 18/18 configs bit-exact vs T0 production kernel (ULP = 0 everywhere, stronger than DEC-008 requires)
- 100/100 determinism at gate config (single unique BENCH_FINAL_LOSS)
- **Critical CRITIQUE catch during D1**: T0 baseline in `microbench_algorithmic.cpp` originally omitted the DEC-009 cross-SIMD fold; corrected before parity comparison. Without that correction the T3b vs T0 ULP would have been spuriously non-zero and masked a real integration green-light.

### D2 result — STOP-BOUND FIRED (see `docs/sprint20/d2_results.md`)

Gate config (50k/RMSE/d6/128b), 50 iters, 3 independent warm runs:

| Kernel | warm_mean_ms | vs S19-tip |
|---|---|---|
| S19-tip (T1) baseline | 31.87 ms | 1.00× |
| D2 T3b | **45.3 ms** | **1.42× SLOWER** |

Stop-bound required [9.0 ms, 21.1 ms] (1.5×–3.5× improvement). D2 measured +42% **regression** — far outside. Per standing orders: STOP, do not commit. Kernel + host changes reverted; only empirical record committed.

### Stage attribution of the regression

| Stage | D2 | S19-tip | Δ |
|---|---|---|---|
| Derivatives | 0.5 ms | 0.5 ms | 0% |
| **Tree search (histogram + suffix + split)** | **41.7 ms** | **29.4 ms** | **+42%** |
| Leaf estimation | 2.5 ms | 2.5 ms | 0% |

100% of the regression lives in `tree_search_ms`. Derivatives and leaf estimation untouched.

### Root cause — dispatch-shape mismatch

T3b's accumulation speedup (−84.4%) was measured in toy-kernel isolation at **1 TG × 256 threads × 50k docs (depth-0 single partition)**, ~195 docs/thread. Production depth-6 dispatches **1638 TGs × 256 threads** (13 feature groups × 63 partitions × 2 stats), ~3 docs/thread. T3b's **fixed per-TG overhead** — zeroing 1024 `atomic_uint` slots at entry and reading them back at writeback (8 memory ops per thread) — amortizes well at 195 docs/thread (≤1% of work), but **dominates at 3 docs/thread (67% of per-TG work)**. CAS atomics also cannot pipeline like simd_shuffle chains can, so the accumulation gain itself compresses. Net: the fixed-cost structure of T3b is incompatible with the production partition count.

### Standing warning (campaign-level, locked)

**Toy-kernel ablations at single-TG root shape do not predict production partition-fragmented dispatch.** Any future lever whose benefit comes from amortization across many docs/thread must be validated against the production TG × docs-per-thread shape *before* integration commit. DEC-017 retirement post-mortem in `.claude/state/DECISIONS.md` encodes this as Sprint 21+ gating criterion.

## Sprint 20 exit gates

| Gate | Criterion | Status |
|---|---|---|
| G1 | `histogram_ms` ≤ 4 ms on gate config | **FAIL** — measured +42% regression (D2) |
| G2 | No 18-config regression > 5% | N/A — no kernel change shipped |
| G3 | Parity 108/108 bit-exact | **PASS** — 18/18 at D1, 100/100 deterministic |
| G4 | `iter_total_ms` ≤ 10.5 ms | **FAIL** — tied to G1 |
| G5 | Non-histogram stages ≤ 10% regression | **PASS** — derivatives & leaf unchanged |
| G6 | CI green | **PASS** — no kernel change; CI can't regress by construction |

**Sprint 20 exits via honest falsification, not a perf gate.** PR #12 body records the gate table unchanged.

## Model-falsification count — fifth this campaign

- DEC-013 (writeback-as-plurality) — falsified Sprint 19 Day 1
- DEC-014 original (gather sub-phase) — falsified Sprint 19 Day 2
- DEC-015 (col-major layout) — falsified Sprint 19 Day 3
- DEC-014 (A1 BATCH_DOCS=64) — falsified Sprint 19 Day 4
- **DEC-017 (T3b atomic-CAS) — falsified Sprint 20 D2 (this sprint)**

All five share the same failure mode: analytical or toy-kernel prediction does not survive production dispatch shape. The standing warning now sits in DECISIONS.md as a campaign-level gate.

## R8 status — honest accounting

| Sprint | R8 target | R8 delivered | Verdict |
|---|---|---|---|
| 19 | ≥1.07× e2e | 1.018× gate / 1.033× best | NOT MET |
| 20 | ≥2.0× e2e (projected) | 0.704× gate (i.e. +42% regression) | NOT MET — sprint abandoned pre-commit |
| **21** (forward) | **≥1.08× e2e** (Sprint 21 only) | TBD | — |

**Campaign ≥1.5× e2e target kept.** Sprint 21 lever is TG-count reduction (see §Sprint 21 below). Pipeline projection midpoint 1.27×, upper bound 1.46×. **1.5× not credibly reachable on current kernel structure and is flagged honestly** per Ramos's "do not soften" standing order. Sprint 22–24 delta vs 1.5× is a live open question for later Verstappen planning.

## Sprint 21 — scoped, not yet cut (see `docs/sprint20/d2b_design.md`)

**Lever**: TG-count reduction via partition-batching at histogram dispatch. Reduce 1638 TGs (13 groups × 63 partitions × 2 stats) to ≤ 26 TGs (13 groups × 2 stats) by aggregating all partitions into a single per-group TG that scans all docs once and accumulates into `hist[part][bin]` via global-atomic or TG-local per-partition slots. This directly inverts the D2 failure mode (small docs/thread) by restoring large docs/thread even at depth 6.

**Projected ceiling**: R8 ≥ 1.08× on gate, ≥ 1.10× on best-case (Logloss 128b). Honest upper bound from current kernel structure. Detailed section breakdown in `docs/sprint20/d2b_design.md §3–6`.

**Not started**: branch `mlx/sprint-21-hist-tg-reduction` is not yet cut. Cut point = Sprint 20 tip (`78697fff79`) once user greenlights.

## Blockers

**None active.** Sprint 20 exits cleanly. PR #12 open for review.

## Next actions (awaiting user direction)

1. Merge PR #9 → master (Sprint 17, long-standing OPEN)
2. Merge PR #10 → its base (Sprint 18)
3. Merge PR #11 → its base (Sprint 19)
4. Merge PR #12 → its base (Sprint 20) — record-only, no kernel change
5. Cut `mlx/sprint-21-hist-tg-reduction` from `78697fff79`; scaffold `docs/sprint21/README.md` mirroring d2b_design.md §3
6. Sprint 21 kickoff — @research-scientist to validate partition-batching kernel shape in toy-kernel at production docs/thread ratio (≈3 docs/thread) **before** any production integration, per Sprint 20 standing warning.

## Carry-forward to Sprint 21

- **L_TG-reduction** (DEC-018 DRAFT-S21, to be drafted): partition-batched histogram dispatch. See `docs/sprint20/d2b_design.md §3`.
- **Validation gate before D2-equivalent integration**: toy-kernel must be measured at production TG × docs/thread shape (not single-TG root shape). This is the campaign-level gate derived from DEC-017 post-mortem.
- **L2** (pre-permute stats + compressedIndex gather removal, 2–4 ms headroom) — still valid, Sprint 22 candidate.
- **L3** (MultiClass per-dim dispatch fusion, 15–25 ms on MC configs) — Sprint 23 candidate.
- **DEC-011 32 KB ceiling** — no amendment needed; T3b's 4 KB relaxation was retired with DEC-017. Ceiling stands as originally written.
- **S19-13 envelope guard** (`CB_ENSURE(maxFoldCount ≤ 127)`) — still active; ships with T1 in Sprint 19 PR.

## Prior sprints — status unchanged

- **Sprint 19** — CLOSED. PR #11 OPEN. Exit gates PASSED. T1 fuse-valid shipped (−1.76% gate / −3.23% best, bit-exact, envelope-guarded).
- **Sprint 18** — CLOSED. PR #10 OPEN. Exit gates PASSED. L1a privHist spill fix shipped (−66.8% gate, 108/108 bit-exact).
- **Sprint 17** — CLOSED. PR #9 OPEN. D1c tree reduction shipped (−89 to −93% histogram_ms, 35/36 bit-exact).
- **Sprints 0–16** — merged upstream (this fork's master).
