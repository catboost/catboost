# Sprint 21 — TG-Count Reduction via Per-Feature-Group Single-Pass Histogram

**Branch**: `mlx/sprint-21-hist-tg-reduction` (cut from Sprint 20 tip `85b6362b6e`)
**Campaign**: Operation Verstappen — battle 6 of 9
**Lever (as-planned)**: TG-count reduction, variant A (per-feature-group single-pass accumulator) — DEC-018 DRAFT-S21
**Gate config**: 50k/RMSE/d6/128b (unchanged from Sprint 19/20)
**R8 target (as-planned)**: ≥1.08× e2e at gate config

---

## STATUS BANNER (2026-04-19)

**D0 COMPLETE — KILL-SWITCH FIRED.** Measured fixed-per-TG overhead at depth 6 = **2.5% ± 1.3% of histogram_ms**, far below the ≥10% threshold. Details: `docs/sprint21/d0_attribution.md`.

**Specification error discovered during D0 (captured for campaign-level learning):** The kill-switch as written in §D0 below tested whether reducing TG count would amortize T1's fixed overhead. That premise did not apply to variant A's actual mechanism, which is **T3b-accumulator-swap at the restored 195-docs/thread shape** — TG-count reduction is the shape-restoration enabler, not the savings source. T1's fixed overhead is already low (2.5%) because Sprint 18 L1a eliminated the dominant DRAM costs. The 40% "fixed overhead" figure in `docs/sprint20/d2b_design.md §2` referred to T3b's per-thread work ratio at production shape, not T1's. See `d0_attribution.md §6.2` for the full analysis. The feedback memory at `~/.claude/projects/.../memory/feedback_ultrathink_task_planning.md` captures the generalizable lesson: write gates as direct tests of the lever's mechanism, not proxies.

**Ramos decision (2026-04-19): honor the kill-switch as written (option a).** D1 BLOCKED. Variant A will not proceed under current Sprint 21 framing. Lever re-pick pending — see §Pivot options below.

**R8 Sprint 21 target: ACKNOWLEDGED at 1.0× (no perf change to ship this sprint).** Revised to reflect kill-switch fire. Sprint 21 pivots to a data-producing scope that de-risks Sprint 22 lever selection.

---

---

## Background

Sprint 20 falsified DEC-017 (T3b threadgroup-atomic-CAS accumulator) at D2 integration: toy-kernel −84.4% single-TG accumulation measured +42.3% regression at the production gate config. Root cause is dispatch-shape mismatch — T3b's fixed per-TG overhead (1024-slot zero-init + writeback = 8 memory ops) amortizes at 195 docs/thread (toy) and dominates at 3 docs/thread (production depth 6, 1638 TGs). Parity was perfect (18/18 bit-exact, 100/100 deterministic). Full record: `docs/sprint20/d1_parity.md`, `docs/sprint20/d2_results.md`, `docs/sprint20/d2b_design.md`. DEC-017 RETIRED with post-mortem banner in `.claude/state/DECISIONS.md`.

**The standing warning locked from Sprint 20** (encoded in DEC-017 retirement, applied to Sprint 21+): toy-kernel ablations at single-TG root dispatch shape DO NOT predict production partition-fragmented dispatch. Any lever whose benefit comes from amortization across many docs/thread MUST be validated against the production TG × docs/thread shape *before* integration commit. Sprint 21 complies by making production-shape attribution the D0 gate and production-shape micro-bench the D1 gate — both *before* D2 integration touches kernel source.

Sprint 21's lever — TG-count reduction, variant A — inverts the dispatch-shape problem directly. Instead of 1638 TGs × 3 docs/thread, dispatch 26 TGs (13 feature groups × 2 stats) × 195 docs/thread. Each TG scans all N docs once and writes `histogram[part][bin]` via per-partition output slots. The per-TG doc count is restored to the toy-kernel shape, so the toy-kernel speedup on the accumulation primitive is projected to transfer **because the dispatch shape is preserved** — not assumed to transfer across shapes. This is the specific methodological fix for the DEC-017 failure mode. See `docs/sprint20/d2b_design.md §3` for the lever survey that ranked variant A above variant B (per-TG partition-batch) and above L2/L3.

---

## Pivot options (post-D0, awaiting Ramos direction)

All three honor the kill-switch fire — variant A does NOT proceed in any option. The question is what Sprint 21 ships in its place.

### A1 — Measurement sprint (recommended)

Scope retargets Sprint 21 to production-shape data that makes Sprint 22 lever selection a bet with known odds. Six analytical/toy-kernel models falsified in the current campaign; the pattern says the next perf attempt should rest on production-shape evidence, not single-TG toy.

Tasks (revised D1–D4):

- **D1-R1** L2 production-shape re-attribution. S19-01c probe D showed global-memory loads at 0% of kernel cost at single-TG shape. Measure the same at multi-TG depth 6. AGX cache behavior at small-partition (3 docs/thread) lookups may differ from single-TG shape. Kill-switch: if L2 gather fraction ≥ 10% of histogram_ms, L2 enters Sprint 22 viable set; if < 10%, L2 FALSIFIED at production too. Cost 1–2 days.
- **D1-R2** T2 sort-by-bin production-shape micro-bench. T2 measured −80.6% accumulation at single-TG toy (S19-10). Build T2 toy kernel dispatched at production shape (1638 TGs × ~3 docs/thread OR 26 TGs × ~195 docs/thread) and measure accumulation reduction. Kill-switch: if ≥ −50% accumulation at production shape, T2 enters Sprint 22 as candidate; if < −50%, T2 FALSIFIED. Cost 2–3 days.
- **D1-R3** Host-side eval() sync instrumentation in `bench_boosting`. Infrastructure for tight per-kernel-phase attribution; documented as upper-bound timing (sync removes overlap). Benefits every future sprint. Cost 0.5–1 day. Per `d0_attribution.md §8.2`.
- **D1-R4** Sprint 22 kickoff plan with data-backed lever ranking. Single doc synthesizing D1-R1/D1-R2 results, ranking L2/T2/tree-search-restructure by measured production-shape evidence. Cost 0.5 day.

**R8 Sprint 21**: 1.0× (explicit, no perf delivered). **Value delivered**: production-shape evidence for two levers + instrumentation tool.
**R8 Sprint 22**: TBD from A1 outcome; a greenlit lever (L2 or T2) carries concrete production-shape backing into integration, not toy-kernel speculation.

### A2 — T2 production-shape attempt (aggressive, higher variance)

Skip the measurement-only frame. Pick T2 as the highest-potential lever, build at production shape, integrate if micro-bench clears. Similar risk profile to Sprint 20 (toy kernel → production integration), but T2's failure mode is different from T3b's (per-partition sort cost, not per-TG fixed overhead), so the DEC-017 failure pattern does not automatically repeat. Cost 5–8 days. Upside 1.3–1.5× if successful. Falsification risk ~30–50% per campaign pattern. R8 Sprint 21: 1.3–1.5× if successful, 1.0× if falsified.

### A3 — Campaign re-scope

Formally close Sprint 21 at 0×. Sprint 22 entry point: @research-scientist designs tree-search restructure (per `docs/sprint20/d2b_design.md §3`, speculative 1.5–2×) or identifies a new lever class. Verstappen ≥1.5× target acknowledged as requiring research-level intervention, not further kernel-level levers. R8 Sprint 21: 0× by design.

**Recommendation**: **A1.** At this campaign state (six falsifications, R8 gap to close, two consecutive 0× sprints risked), production-shape evidence is the single most valuable non-perf deliverable. A2 is the same class of evidence that just failed with T3b (toy kernel → integration hope). A3 is premature — A1 produces the inputs A3 would need anyway. A1 also directly applies the new ultrathink-task-planning discipline: Sprint 22's gates will be written against measured production-shape mechanisms, not proxies.

**Ramos decision (2026-04-19): A1 CHOSEN.** Sprint 21 retargets as a measurement + instrumentation sprint. R8 acknowledged at 1.0×. See revised tasks §A1 Execution below.

---

## A1 Execution — task specs with ultrathink-written gates

Each gate is written as a **direct empirical test of the lever's mechanism**, not a proxy — per the new ultrathink-task-planning discipline. Gates that rely on proxies explicitly flag the correlation assumed and how it could break.

### D1-R3 — Host eval() sync instrumentation (run first, infrastructure)

**Lever mechanism**: n/a — this is infrastructure that makes subsequent D1-R1 and D1-R2 measurements tight. Adds `mx::eval(array)` sync points in `bench_boosting.cpp::RunIteration` to capture per-dispatch timing (histogram, suffix-sum, split-score, leaf-estimation) instead of only the 3-bucket `--stage-profile` coarse bucket.

**Success criterion** (not a kill-switch — no lever to gate): per-dispatch timing stable across 3 independent runs with stdev < 5% of mean at gate config depth 6. Sync-induced measurement distortion (no kernel overlap) documented explicitly; resulting numbers reported as UPPER BOUNDS on per-kernel cost.

**Owner**: @ml-engineer (small bench_boosting.cpp edit).
**Cost**: 0.5–1 day.
**Dependency for**: D1-R1, D1-R2 (both use this infrastructure for tighter attribution).

### D1-R1 — L2 production-shape re-attribution (direct mechanism test)

**Lever mechanism (one sentence)**: L2 pre-permutes the `stats` (gradients/hessians) arrays so the histogram kernel reads contiguous memory instead of gather-by-`partitions[doc]`.

**What a proxy gate would measure**: "stats-load fraction of histogram_ms" via sub-stage profiling. Proxy correlates with L2's savings via "if stats-load is X% of hist_ms, L2 saves up to X%". Correlation breaks if stats-load overlaps with other work (so the X% is padded with overlapped cost) — makes the proxy a loose upper bound.

**Direct mechanism test (GATE)**: build a kernel variant of the L1a accumulator with stats loads replaced by constants (`g = 1.0f, h = 1.0f` instead of `g = gradients[compressedIndex[doc]]`). Run this modified kernel at production shape (50k/RMSE/d6/128b). Measure `histogram_ms` reduction vs baseline T1.

**Kill-switch**: if modified-kernel `histogram_ms` reduction is **≥ 10%** at production shape, L2 enters Sprint 22 viable-set with measured upper-bound savings. If **< 10%**, L2 FALSIFIED at production shape (consistent with S19-01c probe D's single-TG finding generalizing).

**Methodological note**: this is the **maximum** L2 could ever save (zero gather cost). Real L2 integration costs a per-iteration O(N) permute kernel which consumes a fraction of the savings. If the upper bound is < 10%, the real lever is weaker — so the 10% gate is conservative.

**Owner**: @ml-engineer (kernel source variant) + @performance-engineer (measurement).
**Cost**: 1–2 days.
**Output**: `docs/sprint21/d1r1_l2_attribution.md`.

### D1-R2 — T2 sort-by-bin production-shape micro-bench (direct mechanism test)

**Lever mechanism (one sentence)**: T2 pre-sorts `compressedIndex` per partition by bin so the accumulation loop reads monotone-bin data, eliminating the simd_shuffle broadcast chain.

**What a proxy gate would measure**: accumulation-time alone (post-sort), as S19-10 toy-kernel reportedly did. Proxy fails to count the **per-partition sort cost**, which scales with partition count and may dominate at depth 6 (1638 partitions vs 1 in the toy).

**Direct mechanism test (GATE)**: T2 toy kernel at production dispatch shape, timing **sort + accumulation together** (total `histogram_ms`-equivalent, not accumulation alone). Dispatch shape: 1638 TGs × ~3 docs/thread (production T1 shape). Optionally also at 26 TGs × 195 docs/thread (variant A shape, in case T2 + variant A stacks as a Sprint 22 candidate).

**Kill-switch**: at 1638-TG production shape, if total T2 `histogram_ms` is **≤ 50% of T1 baseline** (i.e., −50% reduction), T2 standalone enters Sprint 22 viable-set. If **> 50%** (less improvement), T2 FALSIFIED at production shape.

**Methodological verification step**: before running, verify whether S19-10's −80.6% figure included sort cost. If not, that number overstated T2's benefit. Record the finding in the doc.

**Pre-kickoff sanity check**: S19-10 dropped T2 for "per-partition pre-pass cost substantial and unmeasured at production shape" per agent's d0 §6.3 footnote. This is exactly the gap D1-R2 closes.

**Owner**: @ml-engineer (T2 toy kernel + sort impl) + @performance-engineer (measurement).
**Cost**: 2–3 days.
**Output**: `docs/sprint21/d1r2_t2_microbench.md`.

### D1-R4 — Sprint 22 kickoff plan (synthesis)

**Purpose**: synthesize D1-R1 + D1-R2 + D1-R3 results into Sprint 22 lever ranking with ultrathink-discipline gates.

**Content requirements**:
- Lever ranking by measured production-shape evidence: L2, T2, tree-search restructure, (others as data permits)
- For each viable lever, a Sprint 22 D1-style gate written as a direct mechanism test (not a proxy)
- Explicit Sprint 22 R8 projection range based on A1's evidence
- Campaign cumulative R8 projection update through Sprint 22-23

**Owner**: @technical-writer (or main thread if @technical-writer unavailable).
**Cost**: 0.5 day.
**Output**: `docs/sprint22/README.md` scaffold.

### Execution order

Sequential (each enables the next):
1. D1-R3 (instrumentation) — enables tighter measurements for D1-R1 and D1-R2
2. D1-R1 (L2 direct test) — cheaper, binary-verdict result
3. D1-R2 (T2 direct test) — larger effort, informs Sprint 22 lever selection
4. D1-R4 (Sprint 22 planning synthesis) — after D1-R1/R2 land

Total: 4–6.5 days. **R8 Sprint 21: acknowledged 1.0× (no perf ship).** R8 Sprint 22: TBD from A1 outcome, with evidence backing.

### Sprint 21 exit gates (revised under A1)

| Gate | Criterion | Status |
|------|-----------|--------|
| A1-G1 | D0 kill-switch executed with production-shape evidence | **PASS** (d0_attribution.md) |
| A1-G2 | D1-R3 instrumentation produces stable per-dispatch timings | PENDING |
| A1-G3 | D1-R1 gives a binary L2 verdict at production shape | PENDING |
| A1-G4 | D1-R2 gives a binary T2 verdict at production shape (sort-inclusive) | PENDING |
| A1-G5 | D1-R4 Sprint 22 plan has mechanism-direct gates | PENDING |
| A1-G6 | No kernel source committed on Sprint 21 branch (all variants = local/scratch) | PENDING |

A1-G6 is a discipline gate: the measurement kernel variants built for D1-R1 and D1-R2 are **scratch/local-only**, not committed as production code. Sprint 21 ships docs + instrumentation tool (D1-R3) only.

---

---

## Tasks (as-planned, D0 executed, D1–D5 BLOCKED pending pivot decision)

### D0 — Production-shape attribution (COMPLETE; kill-switch FIRED — see `d0_attribution.md`)

*(original D0 scope below — preserved as institutional memory)*



Stage-decompose `histogram_ms` at gate config (50k/RMSE/d6/128b) into:
- Fixed per-TG overhead fraction (zero-init + writeback + dispatch)
- Per-thread accumulation work fraction
- Stats-load + compressedIndex gather fraction
- Global-atomic contention fraction (if applicable to variant A)

Output: `docs/sprint21/d0_attribution.md` with ms + % per stage, ±1 ms error bars, measured at production dispatch shape (multi-TG depth-6 dispatch — NOT single-TG root). Instrument via `bench_boosting --stage-profile` and, where needed, per-kernel-phase timing injected into `kernel_sources.h` as a debug variant (not committed — local measurement only).

**D0 kill-switch**: if fixed-overhead fraction at depth 6 is **< 10% of histogram_ms**, TG-count reduction is not the dominant lever. Sprint 21 MUST re-pick the lever before committing to R8. No D1 until this gate clears.

**D0 is measurement-only — no kernel source changes, no parity risk, no perf risk.** This is the cheap kill-switch that Sprint 19 proved the campaign needs.

### D1 — Production-shape micro-bench for variant A

Build a variant-A toy kernel (one TG per feature group × 2 stats = 26 TGs, each scanning all 50k docs with per-partition output slots) and measure `accumulation_ms` at production dispatch shape — multi-feature-group concurrent dispatch at gate config, not single-TG isolation. Harness mirrors `docs/sprint19/scratch/algorithmic/microbench_algorithmic.cpp` structure.

**Pass criterion**: measured `accumulation_ms` reduction ≥ −50% vs T1 baseline at production dispatch shape. (T3b measured −84% at *single-TG* shape — variant A is projected to preserve the per-TG docs/thread ratio, so if the accumulation primitive's speedup doesn't transfer within a 40% compression factor, the hypothesis is wrong and integration is blocked.)

**Fail path**: escalate. Do NOT integrate. Do NOT apply compensating layers without re-attribution. Sprint 19 DEC-014/A1 pattern — model falsified, sprint ends honest.

**D1 is measurement-only — toy kernel, parity checked against T0 reference only, no production kernel source touched.**

### D2 — Integration: variant A production kernel

Contingent on D1 pass. One commit per DEC-012:

1. **Kernel rewrite commit** — replace the partition-per-TG dispatch in `kernel_sources.h` with a feature-group-per-TG kernel. Each TG loops over all N docs; per-doc work reads `partitions[doc]` to route accumulation to `simdHistU[part][bin]` (per-partition output slots in TG memory). Writeback fans out across partitions at TG-end. TG memory footprint: `numParts × numBins × 4 B = 64 × 128 × 4 = 32 KB` at gate config — at the DEC-011 ceiling; must not exceed.
2. **Host dispatch commit** — update `histogram.cpp::ComputeHistogramsImpl` grid dims and per-TG metadata (partition count, first-partition index, stat index). Remove the per-partition loop at host side. `bench_boosting.cpp::DispatchHistogram` mirror.
3. **DECISIONS.md commit** — DEC-018 flipped from DRAFT to ACTIVE; DEC-011 ceiling cross-reference; S19-13 envelope guard status confirmed.

### D3 — Full DEC-008 parity sweep

Mirror Sprint 20 D1 methodology: 18 configs × 100 runs at gate config, comparing variant A vs S19-tip (T1). Bit-exact required for RMSE (ULP = 0); ULP ≤ 4 for Logloss, ULP ≤ 8 for MultiClass. 100-run determinism at gate config — single unique `BENCH_FINAL_LOSS`. Output: `docs/sprint21/d3_parity.md`.

**If parity fails on any config**: do not ship. Escalate. Kahan-compensated summation is NOT the default remediation — the per-partition output slot structure has different reduction-order properties than T3b; Kahan may be redundant or may have a different mitigation scope.

### D4 — Perf gate + R8 commitment

Mirror Sprint 20 D2 methodology: 3 independent warm runs at gate config, 50 iters, stage-attributed via `bench_boosting --stage-profile`. Pass criterion: **`iter_total_ms` ≤ 19.5 ms at gate config (≥1.08× e2e from 21.03 ms S19 baseline)**. Output: `docs/sprint21/d4_results.md`.

**If perf fails**: revert per Sprint 20 D2 precedent. Ship the empirical record only. Do not soften R8. Do not fold to a hybrid.

### D5 — Exit gates (parallel)

Launched only after D4 clears. Mirror Sprint 19 closeout:
- S21-07 code review (DEC-018 kernel correctness, DEC-011 ceiling compliance, barrier count)
- S21-08 security audit (no new injection surfaces, buffer bounds)
- S21-09 Metal System Trace (if sandbox permits) — confirm fixed-overhead fraction dropped proportionally to TG-count reduction
- S21-10 technical-writer close-out (DECISIONS.md DEC-018 ACTIVE, HANDOFF + CHANGELOG, Sprint 22 skeleton)

---

## Risks

**D0 kill-switch may fire.** If the dispatch-overhead fraction at depth 6 is smaller than the 40% estimated in `d2b_design.md §2`, the lever's upside collapses. Sprint 19 falsified four analytical models in a row; Sprint 20 falsified a fifth. Assume D0 has a real chance of firing and have an alternative lever pre-scoped: L2 re-attribution (stats pre-permute at production shape) is the honest second choice. If D0 kill-switch fires, Sprint 21 pivots to L2 D0 re-attribution, not to a compensating layer on variant A.

**Per-partition atomic contention at depth 6.** Variant A writes `histogram[part][bin]` from multiple threads across the same TG. If partition output slots live in TG memory (`simdHistU[part][bin]` = 32 KB at gate config), contention is TG-local and bounded. If they spill to global atomics (some TG memory pressure scenario), contention becomes cross-TG and will degrade fast. D1 micro-bench must measure contention rate, not just accumulation time.

**TG memory ceiling (DEC-011, 32 KB)**. Gate config sits exactly at the ceiling (64 parts × 128 bins × 4 B = 32 KB). Any increase in partition count (deeper trees) OR bin count (>128) breaks the kernel. The S19-13 envelope guard (`maxFoldCount ≤ 127`) is orthogonal — this is a partition-count ceiling. Sprint 21 must add a host-side guard: `CB_ENSURE(numParts × numBins × 4 ≤ 32 KB)` in `histogram.cpp` before dispatch. Document in DEC-018.

**Partition lookup cost in the inner loop**. Each doc's accumulation requires `partitions[doc]` gather. S19-01c probe D showed AGX hides global-memory loads behind the shuffle inner loop at single-TG shape — but that was at single-TG shape, and variant A restores that shape, so the hiding should transfer. D1 micro-bench must confirm this empirically (not assume by analogy); this is exactly the error mode DEC-017 hit.

**Load imbalance at depth 6**. Partitions at depth 6 are not uniform in size — some may hold 10 docs, some may hold 3000. A single TG scanning all 50k with per-partition output slots may idle threads during small-partition output writes. If imbalance dominates, variant A degrades. D1 must profile with realistic (non-uniform) partition distributions, not a synthetic balanced split.

**Fifth-iteration failure mode.** Five models falsified across Sprints 19–20. If Sprint 21 falsifies a sixth, the appropriate response is escalation to a research-level re-decomposition (tree-search strategy change, §3 of d2b_design.md) — NOT another compensating layer. The "fifth iteration" bar is a campaign-level signal that the lever portfolio itself needs expansion, not that the current lever needs patching.

---

## Exit gates

| Gate | Criterion |
|------|-----------|
| G1 | `histogram_ms` improvement at gate config; measured ms logged |
| G2 | No 18-config regression > 5% |
| G3 | Parity 108/108 bit-exact across DEC-008 envelope |
| G4 | `iter_total_ms` ≤ 19.5 ms at gate config (≥1.08× e2e) |
| G5 | No non-histogram stage regresses > 10% |
| G6 | CI green |

G4 is the R8 gate. **Do not soften G4 during D2/D4** — Sprint 20 precedent shows the stop-bound saves the sprint from a bad commit. If the perf gate misses, ship the empirical record and escalate.

---

## Carry-forward from Sprint 20

- **DEC-017 RETIRED** — empirically falsified. Post-mortem banner in `.claude/state/DECISIONS.md`. Standing warning on toy-kernel vs production dispatch shape locked campaign-level.
- **PR #12** — Sprint 20 stacked on #11 → #10 → #9 on `RR-AMATOK/catboost-mlx`. Ships the empirical record, not performance. Merge order: #9 → #10 → #11 → #12 → (#13 Sprint 21 when ready).
- **S19-13 envelope guard** (`CB_ENSURE(maxFoldCount ≤ 127)`) — still active; ships with T1 in PR #11. Unchanged by variant A (variant A does not pack bin values into feature slots).
- **DEC-011 32 KB ceiling** — stands. Variant A sits at the ceiling at gate config; add a partition-count host guard in D2.
- **L2, L3** — still live as Sprint 22/23 candidates per d2b_design.md §5. L2 is the pre-scoped fallback if Sprint 21 D0 kill-switch fires.
- **Verstappen ≥1.5× e2e target** — kept per standing order. Sprint 21–22–23 pipeline midpoint 1.27×, upper bound 1.46×. 1.5× not credibly reachable on current kernel structure and is flagged honestly. Sprint 24+ escalation likely required. No target re-scoping.

---

## Cumulative R8 projection (midpoint)

| Sprint | Lever | R8 | Cumulative |
|---|---|---|---|
| 19 (shipped) | T1 DEC-016 | 1.018× | 1.018× |
| 20 (abandon) | — | 1.000× | 1.018× |
| **21 (this)** | **DEC-018 TG-count reduction variant A** | **1.08×** | **1.10×** |
| 22 (proposed) | L2 re-attribution + integration | 1.05× | 1.15× |
| 23 (proposed) | Tree-search restructure OR further TG-layout | 1.10× | 1.27× |

Upper bound cumulative: 1.46×. Honest projection — no fifth analytical model reached for to close the gap to 1.5×.
