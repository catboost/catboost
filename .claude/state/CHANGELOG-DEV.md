# Developer Changelog — CatBoost-MLX

> Coverage: Sprints 0–15 reconstructed from git log on 2026-04-15. Sprint 16+ is source of truth.

## Sprint 22 — T2 sort-by-bin SHIPPED; Option III fix; Verstappen ≥1.5× gate CLEARED; R8 1.90× (2026-04-20, CLOSED)

**Branch**: `mlx/sprint-22-t2-integration` (cut from Sprint 21 tip `a7a206b90d`)
**Campaign**: Operation Verstappen — battle 7 of 9
**Verdict**: **CLOSED. 4/4 exit gates PASS. T2 sort-by-bin validated. Cumulative R8 = 1.90×. Verstappen ≥1.5× gate CLEARED by 40 pp.**

### Sprint arc: D0 PASS → D1 parity failure → four-phase diagnostic → Option III fix → 4/4 gates PASS

Sprint 22 began with an in-situ T2 integration probe (D0) that passed its kill-switch at 0.328× ratio — inside the optimistic band. D1 parity sweep then failed 18/18 configs (ULP 1,327–2,583,206), triggering a four-phase diagnostic arc:

- **D1a**: blit-ordering hypothesis (fill_gpu pool reuse) — REFUTED (fill_gpu is compute; eval barriers did not fix parity)
- **D1b**: depth-parity indexing hypothesis — REFUTED (even-depth pattern explained by split-distribution artifact)
- **D1c**: root cause identified — `bench_boosting.cpp:526` `maxPartDocs = ceil(numDocs / numActiveParts)` uniform-partition assumption. Under real argsort-permuted splits at depth 1 on 50k docs, partitions are [442, 49558] vs `maxPartDocs=25000`; 24558-doc overflow into the neighboring TG's `sortedDocs` slot corrupted histograms. `iters=1` always passed (depth=0 → single partition, no overflow possible).
- **D2**: Option III fix (slab-by-partOffsets). `sortedDocs` reorganized to per-(groupIdx, statIdx) slabs of size `numDocs` indexed by `partOffsets[partIdx]`. Overflow structurally impossible since `sum(partSizes) == numDocs`. Buffer 5.2 MB at gate config vs 333 MB worst-case for Option I one-line fix.

Side-finding: bug β (atomic-scatter float drift, S21 D1-R4 §3 risk) does not exist. 10/10 and 100/100 determinism confirmed post-fix. Kahan compensation concern retired (DEC-022).

### Commits landed (2 kernel/state commits)

| Commit | Content | Verdict |
|--------|---------|---------|
| `4333c82a7e` | D0 in-situ T2 probe at production shape | PASS — ratio 0.328× (optimistic band) |
| `73baadf445` | D1+D1a+D1b+D1c+D2 Option III fix + D3/D4/D5/D6 gate reports | 4/4 GATES PASS |

### Exit gates

| Gate | Criterion | Verdict |
|------|-----------|---------|
| D3 parity | 18/18 DEC-008 ULP=0; 100/100 determinism; EC-1–EC-5 all ULP=0 | **PASS** |
| D4 perf | Ratio 0.317× cross-session; cumulative R8 = 1.90×; gate cleared +40 pp | **PASS** |
| D5 code review | 0 blockers, 6 nits deferred to S23 | **PASS** |
| D6 security audit | 0 CRITICAL/HIGH; overflow class structurally eliminated; max-safe-N 14.3M | **PASS** |

### Final numbers

| Metric | Value |
|--------|-------|
| T2/T1 hist_ms ratio (gate config) | 0.317× cross-session (band 0.315–0.319×) |
| S22 e2e multiplier | 1.778× (33.958 ms → 19.098 ms iter_total) |
| Cumulative R8 post-S22 | **1.07 × 1.778 = 1.90×** |
| Verstappen gate (≥1.5×) | **CLEARED +40 pp** |
| Parity | 18/18 ULP=0; 100/100 determinism; BENCH_FINAL_LOSS T1=T2=0.47740927 |

### Decisions recorded

- **DEC-020**: status advanced from VIABLE → **SHIPPED / VALIDATED**
- **DEC-021**: Option III slab-by-partOffsets layout chosen over Option I (5.2 MB vs 333 MB; overflow structurally eliminated; 1.6 pp perf headroom vs D0)
- **DEC-022**: Kahan/compensated-summation concern RETIRED — bug β does not exist (10/10 + 100/100 determinism post-fix)

### PR #14 target

`RR-AMATOK/catboost-mlx` — stacked on PR #13 (Sprint 21). Ramos opens. Title: `[mlx] sprint-22: T2 sort-by-bin — Option III fix, 4/4 gates PASS, R8 1.90×`.

### Sprint 23 backlog (from S22 closeout)

D0 task: T2 scratch→production promotion (move `kernel_sources_t2_scratch.h` → `kernel_sources.h`, `DispatchHistogramT2` → `histogram.cpp`). 6 deferred NIT catalog items. Tree-search restructure research track (S23-R1 EvalAtBoundary readback, S23-R2 dispatch inversion spike).

---

## Sprint 21 — A1 measurement sprint; L2 FALSIFIED; T2 VIABLE; variant A RETIRED; 0× perf shipped (2026-04-20, CLOSED)

**Branch**: `mlx/sprint-21-hist-tg-reduction` (cut from Sprint 20 tip `85b6362b6e`)
**Campaign**: Operation Verstappen — battle 6 of 9
**Verdict**: **CLOSED via A1 measurement sprint.** 6/6 A1 exit gates PASS. 0× net perf delta shipped (A1-G6 discipline — no production source modified). Two levers retired; one promoted to viable-set.

### A1 pivot rationale

Sprint 21 was planned as a TG-count reduction (variant A) integration sprint. D0 kill-switch fired on day 1: fixed per-TG overhead at depth 6 = 2.5% ± 1.3% (R²=0.9989 depth regression), far below the ≥10% gate. A specification error was discovered: the D0 gate tested T1 fixed-overhead amortization as a proxy for variant A's actual mechanism (T3b shape restoration at 195 docs/thread). Ramos chose option (a): honor the kill-switch strictly. Sprint 21 retargeted to A1 — a measurement-only sprint producing production-shape evidence for two lever candidates. Generalizable lesson encoded in `feedback_ultrathink_task_planning.md`.

### Commits landed (5, all docs/instrumentation — zero kernel changes)

| Commit | Content | Verdict |
|---|---|---|
| `a0c473e3b7` | D0 kill-switch: depth-sweep regression, fixed overhead = 2.5% ± 1.3% | FIRED — variant A RETIRED (DEC-018) |
| `ac378d8de6` | D1-R3 per-kernel-profile instrumentation in `bench_boosting.cpp` | DONE — stable, stdev < 5% of mean |
| `fedf9d5348` | D1-R1 L2 direct mechanism test (`stat = 1.0f` zero-gather at 1664-TG depth-6) | FALSIFIED — +2.61% slower (DEC-019) |
| `13322feaca` | D1-R2 T2 sort-by-bin production-shape micro-bench (sort+accum, 1664-TG shape) | VIABLE — −64.8% (DEC-020) |
| `a7a206b90d` | D1-R4 synthesis + Sprint 22 kickoff plan (`docs/sprint21/d1r4_synthesis.md`) | DONE — mechanism-direct gates; R8 ledger |

### Two decisions retired

- **DEC-018 TG-count reduction variant A — RETIRED** (was DRAFT-S21, never activated). D0 kill-switch fired (2.5% << 10% gate). Specification error captured: gate tested T1 amortization proxy, not the T3b shape-restoration mechanism that was the actual savings source. `docs/sprint21/d0_attribution.md §6.2`.
- **DEC-019 L2 stats pre-permute — FALSIFIED**. Zero-gather upper bound (stat=1.0f): +2.61% slower at 1664-TG depth-6 production shape. 12.6 pp below 10% gate. AGX out-of-order execution + hardware L2 prefetcher fully hide the stats gather. Generalizes S19-01c probe D single-TG finding to multi-TG depth-6. `docs/sprint21/d1r1_l2_attribution.md`.

### One decision promoted

- **DEC-020 T2 sort-by-bin — VIABLE (pending Sprint 22 D0 in-situ)**. D1-R2 at 1664-TG production shape: −64.8% histogram_ms (band 63.6–66.7%, 2σ ±2.7–4.4%), clearing 50% gate by 28–34 pp. Gate B parity: max ULP 64, mass conservation 0 ULP across 812,800 bins. Enters Sprint 22 viable-set rank #1. Ratio-transfer risk (synthetic identity-permuted → production argsort-permuted) unproven; Sprint 22 D0 tests directly with kill-switch at ratio > 0.60. `docs/sprint21/d1r2_t2_microbench.md`.

### R8 — honest

- Sprint 21 contribution: **0× by design** (A1 measurement sprint; no perf change intended or shipped)
- Cumulative through Sprint 21: **~1.07× over Sprint 16-class baseline** (from S17/S18/S19 kernel improvements only)
- Gap to Verstappen 1.5× gate: **40% residual** — reachable iff T2 clears Sprint 22 D0 at ratio ≤ 0.60

### Sprint 21 exit gates

| Gate | Criterion | Status |
|---|---|---|
| A1-G1 | D0 kill-switch executed with production-shape evidence | PASS (`a0c473e3b7`) |
| A1-G2 | D1-R3 per-dispatch timings stable (stdev < 5% of mean) | PASS (`ac378d8de6`) |
| A1-G3 | D1-R1 binary L2 verdict at production shape | PASS — FALSIFIED (`fedf9d5348`) |
| A1-G4 | D1-R2 binary T2 verdict at production shape (sort-inclusive) | PASS — VIABLE (`13322feaca`) |
| A1-G5 | D1-R4 Sprint 22 plan has mechanism-direct gates | PASS (`a7a206b90d`) |
| A1-G6 | No kernel source committed on Sprint 21 branch | PASS (zero production source diffs) |

### PR #13 target

`RR-AMATOK/catboost-mlx` — stacked on PR #12 (Sprint 20). Ramos opens. Title: `[mlx] sprint-21: A1 measurement sprint — L2 falsified, T2 viable, variant A retired`.

---

## Sprint 20 — T3b atomic-CAS FALSIFIED at D2; DEC-017 RETIRED; 0× ship, empirical record + Sprint 21 redesign (2026-04-19, CLOSED via falsification)

**Branch**: `mlx/sprint-20-hist-atomic-cas` (cut from Sprint 19 tip `4113200529`)
**Campaign**: Operation Verstappen — battle 5 of 9 — L_accum lever (T3b variant)
**Verdict**: **FALSIFIED.** Toy-kernel −84.4% single-TG accumulation did not translate to production partition-fragmented dispatch. D2 integration measured +42.3% regression at gate config (50k/RMSE/d6/128b), far outside the stop-bound of [9.0 ms, 21.1 ms]. Kernel + host changes reverted pre-commit per standing orders. DEC-017 RETIRED. **0× net perf delta shipped this sprint.** PR #12 ships the empirical record and Sprint 21 redesign plan.

### Commits landed (3, all docs/state)

1. **`9216f4941c`** — D1 parity sweep. T3b 18/18 configs bit-exact vs T0 production kernel (ULP = 0 everywhere, stronger than DEC-008 envelope). 100-run determinism at gate config produced a single unique BENCH_FINAL_LOSS. **Critical CRITIQUE catch during implementation**: the T0 baseline in `docs/sprint19/scratch/algorithmic/microbench_algorithmic.cpp` originally omitted the DEC-009 cross-SIMD fold — without that correction the T3b vs T0 ULP would have been spuriously non-zero and masked a real D2 green-light. Harness in `docs/sprint20/scratch/microbench_parity.cpp` (905 lines); results in `docs/sprint20/d1_parity.md`.
2. **`9079ad3873`** — D2 falsification record. Three independent warm runs at gate config: D2 = 45.3 ms, S19-tip = 31.87 ms → +42.3% regression. Stage attribution via `bench_boosting --stage-profile`: derivatives 0.5 ms (0%), **tree search 41.7 ms vs 29.4 ms (+42%)**, leaf estimation 2.5 ms (0%). 100% of regression lives in the histogram kernel. Root-cause analysis in `docs/sprint20/d2_results.md`.
3. **`78697fff79`** — D2b design + DEC-017 retirement (single commit per user option A). `docs/sprint20/d2b_design.md` (229 lines, 7 sections): abandon verdict for Sprint 20, Sprint 21 lever scoping (TG-count reduction via partition-batching), R8 projection ≥1.08× gate / ≥1.10× best, Sprint 21–22–23 pipeline (midpoint 1.27×, upper bound 1.46×; 1.5× not credibly reachable and flagged honestly). DECISIONS.md DEC-017 flipped from `ACTIVE-PENDING-D3` to `RETIRED — SUPERSEDED BY EMPIRICAL FALSIFICATION` with post-mortem banner and dispatch-shape root cause math. Original DRAFT-S20 text preserved below banner per DEC-013/14/15 pattern.

### Root cause — dispatch-shape mismatch (locked as campaign-level standing warning)

Toy kernel (Sprint 19 ablation): 1 TG × 256 threads × 50k docs single partition, ≈195 docs/thread. T3b's fixed per-TG overhead (1024-slot `atomic_uint` zero-init + writeback read = 8 memory ops per thread) amortizes to ≤1% of per-TG work; accumulation gain dominates; −84.4% valid for this shape only.

Production depth-6 dispatch: 13 feature groups × 63 partitions × 2 stats = **1638 TGs**. Per TG: ~50000 / 64 partitions ≈ 781 docs → 781 / 256 ≈ **3 docs/thread**. Fixed overhead now 8 memory ops vs 12 CAS ops = **67% of per-TG work**. CAS atomics cannot pipeline like simd_shuffle chains (each CAS is a read-modify-write with conditional retry that must see the result before the next iteration). Net: the fixed-cost structure of T3b is incompatible with the production partition count.

**Standing warning (campaign-level, encoded in DECISIONS.md DEC-017 post-mortem)**: toy-kernel ablations at single-TG root shape do not predict production partition-fragmented dispatch. Any future lever whose benefit comes from amortization across many docs/thread must be validated against the production TG × docs/thread shape *before* integration commit. This is the fifth analytical/toy-kernel model falsified this campaign — the pattern is now locked and the validation gate is mandatory for Sprint 21+.

### Sprint 20 exit gates

| Gate | Criterion | Status |
|---|---|---|
| G1 | `histogram_ms` ≤ 4 ms on gate | **FAIL** (measured +42%) |
| G2 | No 18-config regression > 5% | N/A (no kernel change shipped) |
| G3 | Parity 108/108 | **PASS** (D1 18/18 + 100/100 determinism) |
| G4 | `iter_total_ms` ≤ 10.5 ms | **FAIL** (tied to G1) |
| G5 | Non-histogram stages ≤ 10% | **PASS** (derivatives & leaf unchanged) |
| G6 | CI green | **PASS** (no kernel change) |

Sprint exits via empirical falsification, not a perf gate. PR #12 body records the gate table unchanged.

### R8 status — honest

- Sprint 20 target: ≥2.0× e2e (projected from toy −84.4%).
- Sprint 20 delivered: **0.704× gate** (+42% regression) — falsified before commit.
- **Sprint 21 target reset: ≥1.08× e2e** (TG-count reduction lever, scoped in d2b_design.md §3).
- **Campaign ≥1.5× e2e kept** per user's explicit decision. Sprint 21–22–23 pipeline midpoint 1.27×, upper bound 1.46×. **1.5× not credibly reachable on current kernel structure and is flagged honestly.**

### PR #12 — opened

`https://github.com/RR-AMATOK/catboost-mlx/pull/12` — stacked on PR #11 (Sprint 19). Ships the empirical record, not performance. Merge order: #9 → #10 → #11 → #12.

---

## Sprint 19 — T1 fuse-valid (DEC-016) shipped; DEC-014/015 REJECTED empirically; S19-13 envelope guard + exit gates (2026-04-17 → 2026-04-19, EXIT-GATES PASSED)

**Branch**: `mlx/sprint-19-hist-writeback`
**Campaign**: Operation Verstappen — battle 4 of 9 — L_accum lever (pivoted from L_writeback)
**Verdict**: T1 (DEC-016) shipped at −1.76% e2e on gate config, bit-exact, deterministic, guarded. R8 ≥1.07× NOT met (1.018× actual on gate / 1.033× best) — deferred to Sprint 20 via DEC-017 (T3b atomic-CAS).

### Day 4 evening (2026-04-19) — Exit gates + S19-13 envelope guard

Five exit-gate agents launched after commit `0f992cf863`. Two completed with empirically-backed sign-offs; two returned plan-only outputs (sandbox constraints); one flagged a BLOCKER on the T1 MSB-sentinel that was then fixed in S19-13.

**S19-07 code review — BLOCKER then resolved via S19-13.** Reviewer found that `compressedIndex[...] | VALID_BIT` in `kernel_sources.h` is unsafe whenever slot-0 holds a bin value ≥ 128. The packer (`csv_train.cpp::PackFeatures`) uses 8-bit slots, so slot-0 occupies bits [24..31] — bit 31 aliases bin 128. With default `MaxBins = 255` or the `bins = 128 + NaN offset` case, the path is reachable and `p_clean = p_s & 0x7FFFFFFFu` silently rewrites bins 128..255 → 0..127. The DECISIONS.md rationale claim "Safe at ≤128 bins because packed holds four 8-bit values in bits 24–30" was off by one.

**S19-13 fix** (landed in this session, single commit):
- `catboost/mlx/methods/histogram.cpp::ComputeHistogramsImpl` — computes `maxFoldCount` during foldCountsFlatVec construction and enforces `CB_ENSURE(maxFoldCount ≤ 127u, …)` before dispatch, with diagnostic message naming DEC-016 envelope and Sprint 20 DEC-017 as the wider-envelope follow-up. Include of `<catboost/libs/helpers/exception.h>` added.
- `catboost/mlx/tests/bench_boosting.cpp::DispatchHistogram` — mirror of the host-side guard via `std::fprintf(stderr, …)` + `std::exit(1)` (CB_ENSURE header is not available in the standalone bench build path).
- `catboost/mlx/tests/bench_boosting.cpp::GenerateSyntheticDataset` — `folds = isOneHot ? (…) : cfg.NumBins − 1` for ordinals. Aligns bench's Folds with real-quantize (`csv_train.cpp::Quantize` sets `folds = numBorders` for no-NaN features). Previously bench stored `Folds = cfg.NumBins` which over-reported by 1 and caused the guard to false-trip on `--bins 128` despite actual bin values staying in [0, 126].
- `catboost/mlx/kernels/kernel_sources.h:175–182` — inline comment rewritten to state the true invariant ("Safe ONLY when every feature's fold count ≤ 127") and cross-reference the host-side guard.
- `.claude/state/DECISIONS.md::DEC-016` — rationale + scope-limit corrected, S19-07 cross-reference added.

**S19-04 parity + determinism — PASS.** 18 configs × 3 runs each on `bench_boosting_ref` (kernel `020eacfb4c` pre-T1, HEAD elsewhere) vs `bench_boosting_t1` (HEAD + S19-13). All 18 produce bit-exact `BENCH_FINAL_LOSS` across ref and t1 (ulp = 0 in all cases, DEC-008 envelope satisfied at the strictest level). 100-run determinism on 50k/RMSE/d6/128b/seed42 returns a single unique loss (0.47740927 post-S19-13) — BUG-001 structural guard holds.

**S19-05 perf delta — PASS G2.** 3-run warm-mean deltas: best −3.23% (50k/Logloss/128); gate config (50k/RMSE/128) −1.76%; worst regression +1.39% at 1k/RMSE/128 (within 3-run noise floor ±2%). No config regresses > 5%. Delivered R8 factor on gate: **1.018×**. Honest accounting preserved. Per-config JSONs written to `.cache/profiling/sprint19/after_t1/*.json` (18 files).

**S19-08 security — PASS (APPROVED).** 5-commit diff audit: no kernel-source injection surfaces, no new buffer-size surfaces, no TOCTOU from EvalAtBoundary removal (MLX host-pointer ctor copies synchronously), no subprocess/eval/pickle in `check_histogram_gate.py`, no secrets, no dependency drift. One defense-in-depth suggestion ("add bins ≤ 128 assertion") — absorbed into S19-13.

**S19-09 post-fix MST — DEFERRED.** `xcrun xctrace` remains sandbox-blocked (same condition as S18-09). Analytical stage decomposition appended to `docs/sprint19/results.md §S19-09`: first-principles probe-A projection (−19.5% e2e) vs measured (−1.76%) is an ~11× over-projection, consistent with probe-A's 86.2% being a depth-0 single-TG attribution that does not multiply cleanly across 1575 TGs × 6 depths. Pattern: fifth analytical model under-predicts the projection-to-production gap. MST capture carried to Sprint 20 under Instruments availability.

**Docs landed:** `docs/sprint19/results.md` (executive summary + per-gate detail + honest R8 accounting).

### Day 4 (2026-04-19) — Path 3 close-out: Commits 1+2 shipped, A1 empirically dropped, parallel tracks

### Day 4 (2026-04-19) — Path 3 close-out: Commits 1+2 shipped, A1 empirically dropped, parallel tracks

**Three DEC-012 kernel commits landed** on `mlx/sprint-19-hist-writeback`:

1. **`77db8b5631`** — Commit 1: extract DEC-015 side-fix. Reverted col-major layout changes in `compressed_index.h`, `kernel_sources.h`, `bench_boosting.cpp`, `csv_train.cpp`. Kept the `DispatchHistogramBatched` per-group variable correction (`featureColumnIndices`+`numGroups` replacing scalar `featureColumnIdx`) in `histogram.cpp` — a pre-existing correctness fix that would have shipped regardless.
2. **`7387814dd6`** — S19-06 CI gate widening. `benchmarks/check_histogram_gate.py` updated from `sprint17/10k` to `sprint19/baseline/50000_rmse_d6_128bins.json`. Dropped min-reduction flag; sprint-neutral messages. Dry-run triggers at +6.1% delta.
3. **`020eacfb4c`** — S19-11 scope-reduced. Removed `TMLXDevice::EvalAtBoundary(result.LeafDocIds)` at `structure_searcher.cpp:738` — a no-op flush since MLX constructor copies data into the GPU buffer synchronously. Other 3 `EvalAtBoundary` calls on that path (lines 290, 609, 705) are legitimate pre-`.data<T>()` guard-syncs, left intact. Bit-exact pre/post at 50k/RMSE/d6/128b = 0.48047778 (3 runs each).
4. **`92f3832169`** — Commit 2: DEC-016 T1 fuse-valid simd_shuffle reduction. Pack the valid flag into the MSB of `packed` at load time (`packed |= VALID_BIT` where `VALID_BIT = 0x80000000u`); derive validity from `(p_s & VALID_BIT)` inside the src broadcast loop; mask via `p_clean = p_s & 0x7FFFFFFFu` before bin extraction. Drops one `simd_shuffle` per src iteration (3 → 2). **Measurements (50k/RMSE/d6/128b, 3-run warm mean):** pre-edit 32.47 ms, post-edit 31.73 ms → **−2.3% e2e**. **Parity bit-exact at 3 configs** (50k/RMSE=0.48047778, 10k/RMSE=0.48016092, 50k/MultiClass=0.94424933). Safe at ≤128 bins (packed holds four 8-bit values in bits [0..30]; bit 31 always zero on load).

**Commit 3 (DEC-014 A1 BATCH_DOCS=64) DROPPED** per plan clause "if not reproducible, drop":
- A1 variant added to `docs/sprint19/scratch/algorithmic/microbench_algorithmic.cpp` as `kA1Source`. Toy measurement (3 runs, post-T1): A1 vs T1 mean = **−1.9%** (noise-marginal; stdev ~1%).
- Production port (lo/hi slab state in lane registers, outer stride doubled, 2-slab inner shuffle loop). Parity bit-exact (0.48047778) but **warm-mean e2e +9.4% REGRESSION** (T1-only 31.7 ms vs T1+A1 34.7 ms, 3 runs each). Register pressure from lo/hi slab state dominates the halved outer-loop saving — AGX VGPR spill hypothesis.
- A1 reverted in `kernel_sources.h`; A1 variant kept in `microbench_algorithmic.cpp` for future reference.
- Full disposition: `docs/sprint19/scratch/algorithmic/a1_empirical_drop.md`.

**Pattern note: fourth analytical model falsified this sprint.** DEC-013 writeback plurality → SUPERSEDED. DEC-014 original gather sub-phase → INVALIDATED. DEC-015 col-major layout → REJECTED (measured 0.98× vs projected 2.13×). DEC-014 (A1 BATCH_DOCS=64) → REJECTED (measured +9.4% regression vs projected −4%). Sprint 19 lesson, locked: analytical reasoning about AGX cache/register behavior is unreliable — empirical micro-bench backing is required before committing any production kernel change, and toy-kernel signal must be validated against production integration before shipping.

**R8 accounting (honest per "do not soften" standing order):**
- R8 revised mid-sprint from aggressive 1.5–1.8× e2e to **≥1.07× e2e** after S19-01 ground-truth falsified the writeback plurality model.
- Delivered: **1.023× e2e** on 50k/RMSE/d6/128b via T1 alone.
- R8 NOT met. Deferred to Sprint 20 via DEC-017 T3b atomic-CAS (toy measured −84.4% accumulation; full DEC-008 parity sweep is the Sprint 20 D1 gate).

**Documentation landed (S19-10 technical-writer pass):**
- `docs/sprint19/algorithmic_ablation.md` — T0/T1/T2/T3/T3b ablation with measured toy-kernel deltas.
- `docs/sprint20/README.md` — Sprint 20 D1–D4 plan (T3b parity sweep, production integration, full-grid scaling, MultiClass drift analysis).
- DECISIONS.md updated: DEC-014 REJECTED, DEC-015 REJECTED, DEC-016 ACTIVE, DEC-017 DRAFT-S20.
- HANDOFF.md updated with close-out status and R8 deferral.

**Exit gates PENDING (parallel tracks, unblocked):** S19-04 parity grid + 100-run determinism, S19-05 18-config perf delta + 50k MST, S19-07 code review, S19-08 security pass, S19-09 post-fix MST.

---

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
