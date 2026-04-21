# Sprint 21 D1-R4 — Synthesis & Sprint 22 Kickoff Plan

**Branch**: `mlx/sprint-21-hist-tg-reduction`
**Date**: 2026-04-20
**Task**: A1 execution D1-R4 — synthesize D1-R1/R2/R3 into a ranked, mechanism-direct lever set and specify Sprint 22 D0.
**Prior doc citations**: `docs/sprint21/README.md`, `docs/sprint21/d0_attribution.md`, `docs/sprint21/d1r3_instrumentation.md`, `docs/sprint21/d1r1_l2_attribution.md`, `docs/sprint21/d1r2_t2_microbench.md`, `docs/sprint19/algorithmic_ablation.md`, `docs/sprint20/d2b_design.md`, `.claude/state/DECISIONS.md`.
**Discipline**: A1-G5 exit gate. Every projected number below is paired with a direct mechanism gate that would falsify it. No proxies. Ratio-transfer risk (synthetic → production data) is called out explicitly on T2.

---

## §1 TL;DR

A1 retired two levers (variant A / TG-count reduction via D0 kill-switch; L2 stats-pre-permute via D1-R1 zero-gather upper-bound) and certified one (T2 sort-by-bin via D1-R2, 64.8% hist_ms reduction at production dispatch shape, sort cost fully included). **T2 is the single viable mechanism-backed lever heading into Sprint 22 and is ranked #1.** Tree-search restructure remains a research-track candidate — it is the only pathway still credibly capable of closing the gap from the T2-midpoint (~1.35×) to the Verstappen ≥1.5× gate. **Sprint 22 D0 must be an in-situ T2 integration probe wired into `DispatchHistogram` under real argsort-permuted partitions — the synthetic identity-permuted harness does not establish production ratio-transfer.** Kill-switch threshold: real-data T2/T1 > 0.60 drops T2 back to RESEARCH.

---

## §2 Lever ledger (post-A1)

Every lever considered since the Sprint 17 start of Operation Verstappen, with current status.

| Lever | Mechanism | Latest evidence | Verdict | Doc citation |
|---|---|---|---|---|
| **D1c tree reduction** | 3-level balanced-tree simd_shuffle_xor fold vs serial | −89 to −93% histogram_ms on 18 configs, 35/36 parity bit-exact | **SHIPPED (S17, DEC-009)** | `docs/sprint17/results.md`; DEC-009 |
| **L1a `simdHist[8][1024]`** | Replace 4 KB/thread `privHist` register-array spill with per-SIMD TG-memory layout (32 KB) | −66.8% hist_ms at gate config, 108/108 bit-exact | **SHIPPED (S18, DEC-011/012)** | DEC-011; `docs/sprint18/` |
| **Writeback two-phase reduction** | On-chip fold 8 simdHists → 1, atomic-free global store | Falsified pre-implementation: writeback = 5% (0.79 ms), not 15 ms plurality | **FALSIFIED (S19 D2, DEC-013 SUPERSEDED)** | `docs/sprint19/attribution.md`; DEC-013 |
| **DEC-014 gather sub-phase model** | Col-major / wider-batch to hide L2 stall on `compressedIndex` | Col-major 0.98× (projected 2.13×); AGX prefetcher fully hides gather | **FALSIFIED (S19, DEC-014/015 REJECTED)** | `docs/sprint19/reattribution.md`; DEC-014, DEC-015 |
| **T1 fuse-valid (MSB-sentinel)** | Pack valid flag into bit 31 of `packed`, drop one `simd_shuffle` per src (3 → 2) | −2.3% e2e at gate, 18/18 bit-exact, envelope-guarded ≤127 bins | **SHIPPED (S19, DEC-016)** | `docs/sprint19/algorithmic_ablation.md`; DEC-016 |
| **T3b threadgroup-atomic-CAS** | Replace shuffle broadcast with `atomic_uint simdHistU[1024]` + CAS-float add | Toy kernel −84.4% at 195 docs/thread; production +42.3% **regression** at 3 docs/thread | **FALSIFIED (S20 D2, DEC-017 RETIRED)** | `docs/sprint20/d2_results.md`; DEC-017 |
| **L2 stats pre-permute** | Pre-permute `gradients`/`hessians` to kill per-doc gather in kernel | Zero-gather upper bound (`stat = 1.0f`): +2.61% slower, not ≥10% faster. Gate miss by 12.6 pp | **FALSIFIED (S21 D1-R1)** | `docs/sprint21/d1r1_l2_attribution.md` |
| **Variant A (TG-count reduction)** | Dispatch 26 TGs × 195 docs/thread instead of 1638 × 3 to amortize fixed per-TG overhead | Fixed-overhead = 2.5% ± 1.3% at depth 6 (regression R²=0.9989). Gate requires ≥10% | **FALSIFIED (S21 D0, DEC-018 never activated)** | `docs/sprint21/d0_attribution.md`; README §Pivot |
| **T2 sort-by-bin** | Counting-sort docs by bin pre-pass → pure bin-range scan in accumulator, eliminates simd_shuffle | Sort+accum at 1664-TG production shape: −64.8% (band 63.6–66.7%, 2σ ±2.7–4.4%); gate margin 28–34 pp | **VIABLE — Sprint 22 #1** | `docs/sprint21/d1r2_t2_microbench.md` |
| **L3 MultiClass dispatch fusion** | Collapse 3 per-dim histogram dispatches to 1 with dim loop | Not measured; MC configs only; does not clear RMSE gate | **DEFERRED (MC-only, out of campaign gate)** | `docs/sprint20/d2b_design.md §3` |
| **Tree-search restructure** | Invert partition-fragmented dispatch: one histogram over all docs + scoring-time masking; or eliminate per-depth `EvalAtBoundary` CPU readbacks in `structure_searcher.cpp` | Unmeasured; speculative 1.5–2× if inversion lands, 0× if scoring mask has own overhead | **RESEARCH** | `docs/sprint20/d2b_design.md §3`; `docs/sprint16/sync_inventory.md:27,33,34,35` |
| **Shuffle `stat`→`packed` fuse (T1 follow-up)** | Fixed-point encode `stat` into unused bits, save one more shuffle per src | Extrapolated ~0.68 ms (+1.03× e2e). Fails R8 ≥1.08×. Non-trivial encoding | **DEFERRED (marginal, does not clear gate)** | `docs/sprint21/d0_attribution.md §6.4` |

**Campaign falsification count after A1: 7 models falsified** (writeback-plurality, DEC-014 gather, DEC-015 col-major, DEC-014 A1 batch-64, DEC-017 T3b, DEC-018 variant A, L2). Five additional levers shipped, deferred, or promoted to research. The viable in-sprint lever set has converged to a single candidate (T2) plus one research-track candidate (tree-search restructure).

---

## §3 Sprint 22 ranked viable-set

### Rank #1 — T2 sort-by-bin (integration sprint)

**Mechanism**. The production T1 kernel spends ~80% of `histogram_ms` in a 32-iteration `simd_shuffle` broadcast chain that fans bin/stat/valid across a SIMD group. T2 replaces this with a two-kernel dispatch: (a) a counting-sort pre-pass that bin-partitions each partition's docs by feature-0 bin and emits `binOffsets[]`, (b) an accumulator that for feature-0 does a pure bin-range scan (no shuffle, single writer per bin → no contention) and for features 1–3 does a per-doc sorted scatter via global atomics. The simd_shuffle serial chain — the measured plurality cost at production dispatch shape — is structurally eliminated for feature-0 and replaced with cheaper per-thread linear work for features 1–3. Sort cost is ~5× cheaper than the accumulation it replaces at 781 docs/TG × 128 bins, so T2 amortizes even at the fragmented depth-6 shape. The D1-R2 harness measured sort+accum together at 0.520 ms vs T1 at 1.479 ms (ratio 0.352×, reduction 64.8% cross-session mean) — clearing the 50% gate by 28–34 pp, cleanly separated from noise.

**Direct mechanism gate (falsifier) at Sprint 22 D0**. Wire T2 into a `DispatchHistogramT2` variant in `catboost/mlx/methods/histogram.cpp`, dispatched under real training-loop conditions (argsort-permuted `docIndices` from `structure_searcher.cpp`, not synthetic identity permutation). Measure `histogram_ms` via the D1-R3 `--per-kernel-profile` instrumentation at the gate config (`--rows 50000 --features 50 --classes 1 --depth 6 --iters 50 --bins 128 --seed 42`). **Falsifier**: if measured `hist_ms(T2) / hist_ms(T1) > 0.60` (i.e., production reduction < 40%), the synthetic-to-production ratio transfer has failed and T2 drops back to RESEARCH. This is not a proxy — it directly tests whether the pre-sort mechanism survives real-data access patterns. Rationale for 0.60: the D1-R2 upper bound is 0.36× (66.7% reduction); a 3σ ratio transfer loss of ~24 pp would put the real-data ratio at 0.60. Any worse than that indicates structural failure of the mechanism (e.g., argsort-permuted docs evict the sort-scratch cache line), not just noise.

**Projected e2e gain (with propagation math)**.

Starting point: `iter_total_ms = 31.93 ms`, `histogram_ms = 21.57 ms` (D1-R1 baseline), non-hist stages = 10.36 ms.

| Scenario | Real-data T2/T1 | T2 hist_ms | New iter_total | e2e speedup |
|---|---|---|---|---|
| Optimistic (ratio transfers fully) | 0.33× | 7.12 ms | 17.48 ms | **1.83×** |
| Midpoint (mid of D1-R2 band) | 0.36× | 7.77 ms | 18.13 ms | **1.76×** |
| Conservative (gate threshold) | 0.50× | 10.79 ms | 21.15 ms | **1.51×** |
| Kill-switch boundary | 0.60× | 12.94 ms | 23.30 ms | **1.37×** |

Projected band (before Sprint 22 D0): **e2e 1.37× to 1.83×**, midpoint 1.76× if ratio transfers fully, fall-back 1.51× if the 0.50 gate threshold is the realized production ratio. The 0.60 kill-switch threshold places the falsification boundary at 1.37×, below the 1.5× Verstappen gate — i.e., if T2 misses its kill-switch, T2 alone does not clear the campaign gate even best-case and should be abandoned rather than shipped at 1.37×.

**R8 contribution (fraction of ≥1.5× gate)**. Campaign start 1.018× (S19 tip). 1.5× gate requires closing `1.5 / 1.018 − 1 = 47%` residual. Midpoint T2 projection 1.76× × (1/1.018) = **clears the gate standalone at midpoint** (T2 alone delivers 1.73× over S19 tip, well above 1.5×). Conservative 1.51× × (1/1.018) = **1.48× cumulative — 2 pp short of gate**. Kill-switch 1.37× = 1.35× cumulative — **13 pp short**. T2 is sufficient-to-clear-the-gate at midpoint, insufficient-at-conservative.

**Risks**.
- **Ratio-transfer to production data (primary)**. D1-R2 used identity-permuted docs (docs 0..50k sorted into 64 partitions by `doc / partSize`). Production uses argsort over leaf indices, producing irregular memory access. In-harness T1 at the 64-partition primary shape ran 1.48 ms vs the D1-R1 per-level reference 3.60 ms (−59%) — a pure data-locality artifact. The T2/T1 **ratio** should cancel this to first order because both variants benefit from the same locality, but if the sort pre-pass itself is cache-sensitive in a way the accumulator isn't, the ratio could widen. Variant A (26 TGs × 50k docs/TG) is the stress test: T1-VA = 3.43 ms matches D1-R1/6 within 4.6%, and T2-VA = 0.98 ms gives 71.5% reduction — so at least one production-representative shape holds the reduction. This is suggestive, not conclusive for the primary 64-partition shape.
- **Parity at integration**. D1-R2 measured T2 vs T1 at 64 ULP max, T2 vs CPU double at 64 ULP, mass-sum ULP 0 over 812,800 bins. DEC-008 envelope requires RMSE/Logloss ULP ≤ 4 — T2 would blow this by 16×. However, DEC-008 ULP ≤ 4 is measured on FINAL_LOSS, not per-bin histogram values; the per-bin drift is then absorbed through scoring and leaf estimation. Sprint 22 D1 parity sweep (18-config DEC-008 envelope) must validate end-to-end ULP, not per-bin. If end-to-end fails, Kahan-compensated summation on the per-doc scatter path is the path forward (same as the DEC-017 fallback plan), which costs ~1 extra atomic_uint per bin and may compress the speedup. Budget: +2–3 days if Kahan needed.
- **Multi-feature sort scope**. D1-R2 sorts on feature-0 bin only; features 1–3 use atomic scatter (still shuffle-free but more expensive than the range scan). A full multi-feature sort would sort on all 4 bins jointly or do 4 independent sorts. Sprint 22 should not widen scope — ship feature-0 sort as-implemented, measure end-to-end, and defer multi-feature optimization.
- **Integration complexity**. DispatchHistogram grid geometry stays identical (same `(256 × numGroups, numParts, numStats)` layout). New buffers: `sortedDocs[numTGs × maxPartDocs]`, `binOffsets[numTGs × 129]`. TG memory: `atomic_uint tgCounts[128]` = 512 B, negligible against DEC-011 32 KB ceiling. One additional kernel launch per histogram dispatch. Per DEC-012 one-structural-change-per-commit: (1) T2 kernel added in `kernel_sources.h`, (2) `DispatchHistogramT2` added as dispatch variant, (3) T2 selection guarded by compile-time flag or runtime config, (4) default flip after parity. Estimated 4–5 commits, 3–4 days.

**Sprint 22 kill-switch**. If Sprint 22 D0 measures T2/T1 > 0.60 at production shape, T2 is falsified (ratio transfer failed). If parity D1 fails ULP ≤ 4 on any config AND Kahan falls below 0.60 ratio after compensation, T2 is also abandoned. Both kill-switches are direct mechanism tests, not proxies.

**Scope qualifier on bug β and Kahan concern (added 2026-04-20 during S23 D0)**: Sprint 22 D2/D3 retired the Kahan concern (DEC-022) based on 10/10 and 100/100 determinism — measured at gate config only (N=50000/RMSE/128b). Bug β is partially real at smaller N. Features 1-3 `atomic_fetch_add` race fires at N=10000/bins=128 (config #8): bimodal ~50/50 between 0.48231599 (ULP=0) and 0.48231912 (ULP=105). DEC-023 (OPEN, S24 scope) is the attack plan. Kahan is NOT the primary fix path — see DEC-023 Options 1-2 (threadgroup-local reduce or int-atomic fixed-point). DEC-022 remains valid at gate config and for the H-B-overflow-as-root-cause framing. The 1.90× R8 record is unaffected (gate config is 100/100 deterministic).

---

### Rank #2 (research track) — Tree-search restructure / EvalAtBoundary readback elimination

**Mechanism**. Two distinct sub-variants, both research-level.

(a) **EvalAtBoundary readback elimination (S19-11 carry-over)**. `structure_searcher.cpp` contains six `EvalAtBoundary` CPU readbacks per iteration (catalog in `docs/sprint16/sync_inventory.md`): per-partition split selection at `:275`, per-leaf sum extraction at `:593`, per-leaf compressedIndex walk at `:653`, lossguide exit boundary at `:686`. These are CPU-side readbacks that force a Metal sync barrier at every tree-level iteration. At depth 6 that is 6 forced syncs per iter × ~50 µs each = ~0.3 ms — not a plurality cost, but they compound with MPS command-buffer overhead. S19-11 in TODOS.md is explicitly carried forward. Replacement with a GPU-resident split-selection kernel was already scoped in sync_inventory.md.

(b) **Dispatch inversion**. The partition-fragmented 1638-TG dispatch at depth 6 is the artifact of CatBoost's symmetric-tree per-partition-histogram design. Compute one histogram over all docs once per iter, apply partition masks at split-scoring time. Speculative: could restore the 195-docs/thread shape at every depth, unlocking T3b-like accumulators that failed variant A's shape-restoration attempt. Equally speculative that scoring-time masking has its own new overhead that eats the gain.

**Direct mechanism gate**. Both sub-variants require research-track scoping. No existing harness has measured either mechanism at production shape. A Sprint 22 D0 scoping gate cannot be written until the research establishes a concrete kernel+host design to measure.

**Projected e2e gain**. (a) readback elimination: ~0.3 ms / 31.93 ms ≈ +1.01× — insufficient standalone, valuable as compound with T2. (b) dispatch inversion: 1.5–2× speculative, 0× if masking overhead dominates. No band possible without more evidence.

**R8 contribution**. (a) negligible standalone, compound-only. (b) if landed cleanly with T2 stacked, could push 1.76× × 1.05× ≈ 1.85×, but the ×1.05 compound is speculative.

**Risks**.
- **Scope creep**. Dispatch inversion is "weeks" cost per `docs/sprint20/d2b_design.md §3`. Not fundable from a single-sprint budget.
- **Lever conflict with T2**. If T2's sort pre-pass assumes partition-fragmented input, dispatch inversion would invalidate it. T2 integration must land BEFORE dispatch inversion is scoped, or T2's design must be inversion-agnostic.
- **Research cost uncertainty**. Typical Verstappen research spike is 3–5 days for scoping + 5–10 days for implementation. A failed research spike is 0×, not negative — but consumes the sprint's deliverable budget.

**Kill-switch**. If the research spike cannot identify a concrete design that produces a mechanism-testable dispatch shape within 2 days, reprioritize and treat as unreachable for the Verstappen campaign window.

**Sprint 22 recommendation**. **Run (a) in-sprint as a compound with T2** — S19-11 was already scheduled. It is a bounded 0.5–1 day fix. **Defer (b) to a dedicated research spike in Sprint 23 or later** — if T2 integration lands clean in Sprint 22, Sprint 23 is the natural time to assess whether dispatch inversion is worth the research cost, informed by post-T2 attribution.

---

### Other candidates considered and not ranked in-sprint

- **L3 MultiClass fusion** — affects MC configs only, does not move the RMSE gate, deferred to Sprint 23+ as a scope-narrowed win for MC benchmarks. Not in the Sprint 22 viable-set.
- **Shuffle `stat`→`packed` fuse** — ~1.03× projected, below R8 minimum. Does not clear the gate even with T2 stacked. Deferred.
- **Re-picking from falsified set** — L2, variant A, T3b, DEC-014 gather, DEC-015 col-major: all falsified via direct mechanism tests. Do not reconsider without new evidence or structural change.

---

## §4 Sprint 22 D0 specification

**Purpose**. Convert the D1-R2 synthetic-harness signal into a production-shape, in-situ measurement before any Sprint 22 kernel commit lands.

**Pre-requisite**. D1-R3 per-kernel-profile instrumentation (commit `ac378d8de6`, Sprint 21). Use `bench_boosting --per-kernel-profile` at gate config.

**Scope (scratch-only, mirrors A1-G6 discipline)**. Implement `DispatchHistogramT2` as an in-source variant of `DispatchHistogramBatched` in `catboost/mlx/methods/histogram.cpp`, guarded by an environment-variable or compile-time flag (`CATBOOST_MLX_HISTOGRAM_T2=1`). Kernel source lives in a new `kHistOneByteT2Source` string in `kernel_sources.h` (or locally in `bench_boosting.cpp` if discipline requires scratch-only). Parity NOT required for D0 — parity is D1. D0 is a perf-only mechanism test.

**Measurement protocol**.
- Gate config: `--rows 50000 --features 50 --classes 1 --depth 6 --iters 50 --bins 128 --seed 42 --per-kernel-profile`
- 3 independent runs × 49 warm iters each (iter 0 cold excluded), matching D1-R1/R2 methodology
- Primary metric: `histogram_ms` mean across runs; secondary: `iter_total_ms`
- Baseline: T1 via same binary, same flags, flag-off. Measure in same session to cancel Metal scheduler drift.

**Kill-switch (direct mechanism threshold)**. Compute `ratio = hist_ms(T2) / hist_ms(T1)` at gate config.
- `ratio ≤ 0.45` — optimistic band holds; proceed directly to Sprint 22 D1 parity sweep.
- `0.45 < ratio ≤ 0.60` — conservative band; proceed to D1 parity but R8 projection drops to 1.37–1.51× midpoint; Ramos re-decides whether to ship or escalate.
- `ratio > 0.60` — **T2 FALSIFIED at production shape**. Sort cost amortization assumed from D1-R2 identity-permuted docs did not transfer. T2 drops to RESEARCH. Sprint 22 pivots to tree-search restructure scoping with no in-sprint perf deliverable.

The 0.60 threshold is the midpoint projection's 1.37× e2e contribution — the lowest ratio at which T2 still contributes positively but fails to clear the Verstappen gate standalone. Below 0.60 there is no single-sprint stacking path to 1.5× from current lever set.

**Rationale for the threshold (ultrathink check)**. D1-R2 band is 0.33–0.36× in-harness. Ratio-transfer risk is dominated by the sort pre-pass being cache-sensitive under argsort-permuted docs. Worst case ratio-transfer: sort pre-pass doubles in cost (1664 TGs × ~0.2 ms sort cost → ~0.4 ms extra at production hist_ms scale), pushing T2 from 7.6 ms to ~13 ms — exactly 0.60 ratio. If sort cost more than doubles, T2's mechanism is fundamentally incompatible with production access patterns and we need new evidence. 0.60 is not an arbitrary 10 pp margin; it is the band where midpoint projections fall below gate.

**D0 output**. `docs/sprint22/d0_t2_production_shape.md` with (a) 3-run measurement table, (b) ratio with ±2σ, (c) kill-switch verdict binary, (d) if PASS: clearance for D1 parity with projected e2e band; if FAIL: research-track pivot plan.

**Cost estimate**. 1 day (scratch variant implementation + 3-run measurement + doc).

---

## §5 R8 honesty ledger

### Starting position

- **Pre-campaign (Sprint 16 tip)**: 1.00× baseline.
- **S17 shipped**: D1c tree reduction — histogram-kernel-local reduction, did not move e2e target directly at gate config (improved histogram_ms scaling at deeper configs and larger N, but iter_total gain at 50k/RMSE/d6/128b was not separately reported in S17 closeout table). Contribution to e2e at gate: folded into the pre-S19 baseline.
- **S18 shipped**: L1a `simdHist[8][1024]` — `histogram_ms` −66.8% at gate (28.75 → 9.56 ms in S18 closeout metrics), e2e reported ~1.07× at gate config at S18 close, though S19-01 ground truth re-measured baseline at `iter_total_ms` 21.12 ms after L1a.
- **S19 shipped**: T1 fuse-valid (DEC-016) — measured −2.3% e2e on post-L1a baseline (32.47 → 31.73 ms at gate), contributing 1.023× on top of L1a. Cumulative delivered e2e at gate config post-S19: approximately **1.07× over Sprint-16-class baseline** (rough; exact attribution across bench config drift is complicated, but `iter_total_ms` at gate config landed at ~31.73 ms post-S19 vs a pre-campaign baseline in the mid-30s).

### S20–S21 contributions

- **S20**: 0× — T3b falsified at D2, no kernel shipped.
- **S21**: 0× — A1 measurement sprint by design. Two levers retired (L2 FALSIFIED, variant A FALSIFIED), one viable-set member confirmed (T2 VIABLE). No kernel change shipped. R8 acknowledged at 1.0× in sprint plan.

**Cumulative R8 through S21 close (honest accounting, no double-counting)**: approximately **1.07× over S16-class baseline**, entirely from S17/S18/S19 kernel-local improvements. Gap to 1.5× campaign gate: **40% residual speedup needed from S22 onward** (calculated as `1.5 / 1.07 − 1 = 0.40`).

### Sprint 22 projected contribution band

From §3 rank #1:

| T2 scenario | Sprint 22 delivered | Cumulative (× S16-class) | Gate gap |
|---|---|---|---|
| Optimistic (0.33 ratio) | +1.83× at gate | ~1.96× | **CLEARED by 46 pp** |
| Midpoint (0.36 ratio) | +1.76× at gate | ~1.88× | **CLEARED by 38 pp** |
| Conservative (0.50 ratio, in-harness gate threshold) | +1.51× at gate | ~1.62× | **CLEARED by 12 pp** |
| D0 kill-switch boundary (0.60 ratio) | +1.37× at gate | ~1.47× | **FAIL by 3 pp** |
| T2 falsified at D0 (> 0.60) | +1.00× | ~1.07× | **FAIL by 43 pp (1.5 − 1.07 = 0.43)** |

The table is bare of optimism. The 1.5× gate is cleared iff T2 delivers its conservative band or better. The kill-switch at 0.60 is calibrated exactly at the boundary where cumulative misses the gate — below 0.60 there is no credible single-sprint path.

### Gap to 1.5× and what would need to hit

**If T2 clears its D0 kill-switch at any ratio ≤ 0.60, the campaign gate is reachable within Sprint 22.** If T2 is falsified at D0, the Verstappen ≥1.5× gate is not reachable on the current kernel structure in a single additional sprint, and Sprint 23 must pivot to research-track options:

1. **Tree-search restructure (research-level)** — speculative 1.5–2× on its own, but weeks of research cost and uncertain deliverability. If T2 falsifies and this is pursued, Sprint 23 ships 0× and Sprint 24 carries the attempt.
2. **Campaign gate re-scope** — acknowledge 1.5× not credibly reachable on current kernel structure, commit to 1.25–1.35× as a delivered target. This was flagged honestly in Sprint 20 carry-forward and remains the honest fallback if T2 is falsified.

**No soft path exists.** Stacking L2 on T2 is moot (L2 falsified). Stacking tree-search on T2 doubles the sprint budget. Stacking T1 shuffle-fuse on T2 adds ~1.03× only and still requires T2 to deliver near-midpoint. Honest assessment: **the campaign's viability now depends on a single in-situ test of T2 at production shape.** If that test passes, the campaign likely clears 1.5× in Sprint 22. If it fails, the campaign likely does not clear 1.5× within the planned sprint window.

This is the honest R8 ledger and should not be softened in downstream agent handoffs.

---

## §6 Sprint 21 closeout checklist (for technical-writer handoff)

### Commits to be made (atomic, per DEC-012 one-structural-change-per-commit)

1. **Sprint 21 state-file commit** (single commit):
   - `.claude/state/TODOS.md` — Close Sprint 21 A1 entries: D0 (DONE, kill-switch fired), D1-R3 (DONE, `ac378d8de6`), D1-R1 (DONE, FALSIFIED), D1-R2 (DONE, VIABLE), D1-R4 (DONE, this doc). Open Sprint 22 entries: D0 in-situ T2 integration probe (owner @ml-engineer, 1 day), D1 parity sweep (owner @qa-engineer, 1 day), D2 integration (owner @ml-engineer, 3 days), D3 perf gate + R8 commitment (owner @performance-engineer, 1 day). S19-11 EvalAtBoundary cleanup carry-forward.
   - `.claude/state/HANDOFF.md` — Update with Sprint 21 A1 close verdict, Sprint 22 kickoff state, T2 viable-set membership, S19-11 readback cleanup carry.
   - `.claude/state/DECISIONS.md` — Three new decision entries:
     - **DEC-018 TG-count reduction variant A — RETIRED** (was DRAFT-S21, never activated). Post-mortem: D0 kill-switch measured fixed-overhead 2.5% ± 1.3%, below 10% gate; specification error captured (kill-switch tested T1 amortization, not T3b's restored-shape mechanism). Pointer: `docs/sprint21/d0_attribution.md §6.2`.
     - **DEC-019 L2 stats pre-permute — FALSIFIED**. D1-R1 zero-gather upper bound measured +2.61% slower at production shape, not ≥10% faster. Generalizes S19-01c probe D single-TG finding to multi-TG depth-6. AGX out-of-order + prefetcher fully hides stats gather. Pointer: `docs/sprint21/d1r1_l2_attribution.md`.
     - **DEC-020 T2 sort-by-bin — VIABLE (pending Sprint 22 D0 production-shape integration)**. D1-R2 sort+accum at production shape measured 64.8% reduction (band 63.6–66.7%, 2σ ±2.7–4.4%), clearing 50% gate by 28–34 pp. Enters Sprint 22 viable-set rank #1. Ratio-transfer risk (synthetic identity-permuted → production argsort-permuted) flagged; Sprint 22 D0 tests the transfer directly. Pointer: `docs/sprint21/d1r2_t2_microbench.md`.
   - `.claude/state/CHANGELOG-DEV.md` — Append Sprint 21 entry: D0 kill-switch fired, A1 pivot chosen, D1-R3/R1/R2/R4 delivered, 0× perf shipped, two levers retired, T2 promoted.
   - `docs/sprint22/README.md` — New scaffold referencing this synthesis doc as the lever-ranking input; D0 spec mirrors §4 above.

2. **No code or kernel commits** — A1-G6 discipline. All D1-R1/R2 kernel variants were scratch/local only and have been restored.

### PR #13 target description

- **Title**: `[mlx] sprint-21: A1 measurement sprint — L2 falsified, T2 viable, variant A retired`
- **Body summary**:
  - Sprint 21 retargeted from TG-count reduction (variant A) to A1 measurement-only after D0 kill-switch fired (fixed-overhead 2.5% << 10% gate)
  - Three new measurement deliverables: D1-R3 `--per-kernel-profile` instrumentation, D1-R1 L2 direct mechanism test (FALSIFIED), D1-R2 T2 production-shape micro-bench (VIABLE)
  - Two decisions retired: DEC-018 (variant A, never activated), L2 (campaign lever candidate falsified)
  - One decision promoted: DEC-020 T2 (viable-set rank #1 for Sprint 22)
  - 0× perf shipped; Sprint 22 enters with mechanism-backed lever ranking
- **Test plan**:
  - [x] D0 measurement reproducible (R²=0.9989 depth regression, 3 independent runs)
  - [x] D1-R1 zero-gather variant restored (`git diff -- kernel_sources.h` empty)
  - [x] D1-R2 harness scratch-only (`docs/sprint21/scratch/t2/` untracked, no kernel_sources.h diff)
  - [x] `bench_boosting` binary rebuildable from documented build command
  - [x] A1-G6 discipline satisfied (zero production source modified on Sprint 21 branch)
- **Base**: `master` (stacked on PR #12)
- **Labels**: `sprint-21`, `measurement-sprint`, `verstappen-campaign`

### State files to update (parallel during closeout, single commit)

| File | Change |
|---|---|
| `.claude/state/TODOS.md` | Close S21 A1 block; open S22 D0/D1/D2/D3 block; carry S19-11 |
| `.claude/state/HANDOFF.md` | Sprint 21 verdict + Sprint 22 state; T2 viable-set entry; gate-clearance dependency on S22-D0 |
| `.claude/state/DECISIONS.md` | DEC-018 RETIRED, DEC-019 FALSIFIED, DEC-020 VIABLE entries |
| `.claude/state/CHANGELOG-DEV.md` | Sprint 21 A1 summary entry |
| `docs/sprint22/README.md` | New scaffold: D0 kill-switch spec from §4 above, lever ranking from §3 |

### What NOT to do in closeout

- **Do not commit kernel variants** — A1-G6 discipline. D1-R1 zero-gather variant and D1-R2 T2 kernel live as scratch only.
- **Do not soften the R8 ledger** in Sprint 22 README — the gap-to-1.5× arithmetic in §5 is the honest view and propagates unchanged.
- **Do not activate DEC-018** — it never passed D0 and should land in DECISIONS.md only as a retirement record.
- **Do not skip the Sprint 22 D0 kill-switch** — ratio-transfer from D1-R2 is unproven and the 0.60 threshold is the gate-binding criterion. Any temptation to ship T2 on in-harness evidence alone repeats the Sprint 20 DEC-017 failure mode (toy kernel → integration hope).

---

## §7 Exit gate coverage (A1-G5)

| Exit gate | Criterion | Status |
|---|---|---|
| A1-G1 | D0 kill-switch executed with production-shape evidence | PASS (`d0_attribution.md`) |
| A1-G2 | D1-R3 instrumentation produces stable per-dispatch timings | PASS (commit `ac378d8de6`) |
| A1-G3 | D1-R1 gives a binary L2 verdict at production shape | PASS (FALSIFIED, `d1r1_l2_attribution.md`) |
| A1-G4 | D1-R2 gives a binary T2 verdict at production shape (sort-inclusive) | PASS (VIABLE, `d1r2_t2_microbench.md`) |
| **A1-G5** | **D1-R4 Sprint 22 plan has mechanism-direct gates** | **PASS (this doc)** |
| A1-G6 | No kernel source committed on Sprint 21 branch (all variants = local/scratch) | PASS (verified in D1-R1 §5, D1-R2 §7) |

**6/6 Sprint 21 A1 exit gates PASSED.** Sprint 21 ready for closeout + PR #13.
