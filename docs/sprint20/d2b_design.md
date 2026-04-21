# Sprint 20 D2b — T3b Redesign: ABANDON

**Branch**: `mlx/sprint-20-hist-atomic-cas`
**Date**: 2026-04-19
**Status**: Design doc. Recommends Sprint 20 ABANDON and hands off to Sprint 21.
**References**: commit `9079ad3873` (D2 falsification record), `docs/sprint20/d2_results.md`, `docs/sprint20/README.md`, `docs/sprint19/scratch/algorithmic/microbench_algorithmic.cpp`, DEC-013/014/015/017.

---

## 1. Problem statement

The T3b threadgroup-atomic-CAS accumulator produced an **−84.4% reduction** in toy-kernel isolation (Sprint 19 algorithmic ablation) and a **+42.3% regression** when integrated into the production dispatch at the gate config 50k/RMSE/d6/128b (Sprint 20 D2). Parity was 18/18 bit-exact across the full DEC-008 envelope; the failure mode is pure performance. The kernel/host changes were reverted; the empirical record is preserved in `docs/sprint20/d2_results.md` (commit `9079ad3873`).

This doc proposes **abandoning the T3b integration path entirely** and hands off to Sprint 21 with a concrete R8 target on a different lever. No D2b kernel code is proposed. No Option C hybrid is proposed. No mid-sprint pivot to L3 is proposed.

---

## 2. Why T3b is the wrong lever

### Dispatch-shape mismatch — the empirical case

T3b's toy-kernel ablation ran:

- 1 TG × 256 threads
- All 50k docs in a single partition (root-level, no splitting)
- Each thread processes ~195 docs per dispatch
- Per thread: 195 CAS ops for 4 features = **780 CAS ops**; fixed overhead = **8 memory ops** (4 zero-init stores + 4 writeback reads)
- Work-to-overhead ratio: **780 : 8 ≈ 97×**

Production dispatch at depth 6 runs:

- 13 feature groups × 63 partitions × 2 stats = **1638 TGs** concurrently
- Per TG: ~50k / 64 partitions ≈ 781 docs across 256 threads = **~3 docs per thread**
- Per thread: 3 docs × 4 features = **12 CAS ops**; fixed overhead = **8 memory ops** (unchanged — zero-init + writeback of the 1024-slot TG-local accumulator is per-TG, not per-doc)
- Work-to-overhead ratio: **12 : 8 ≈ 1.5×**

The fixed per-TG overhead is absolute, not proportional to per-TG work. At 195 docs/thread the overhead is 1% of the per-thread cost; at 3 docs/thread it is 40%. CAS atomics compound the problem — they do not pipeline like simd_shuffle chains. Each CAS is a read-modify-write with conditional retry that must see the result before the next iteration.

**The bottleneck is the dispatch shape, not the accumulator primitive.** No redesign of the accumulator alone rescues T3b at production dispatch shape.

### Why Option C (depth-gated hybrid) does not rescue T3b

A hybrid that uses T3b at shallow depths (large partitions, ≥ ~30 docs/thread) and T1 at deep depths has been modelled as follows:

| Depth | TGs | Docs/thread | Primitive | Contribution to total histogram_ms |
|---|---|---|---|---|
| 0 | 26 | 195 | T3b wins (−84%) | ~4% of total (single root-level dispatch) |
| 1 | 52 | 97 | T3b wins (−70%) | ~8% |
| 2 | 104 | 48 | T3b marginal (~−30%) | ~14% |
| 3 | 208 | 24 | T3b break-even | ~18% |
| 4 | 416 | 12 | T3b loses (+10%) | ~22% |
| 5 | 832 | 6 | T3b loses (+25%) | ~24% |
| 6 | 1638 | 3 | T3b loses (+42%) | ~10% |

Depth-gated T3b captures depth 0–2 (~26% of histogram_ms at meaningful improvement), leaves depth 3+ on T1. Expected e2e upside: **1.04–1.08× at the gate config**.

Cost side: ~200 LOC (two kernel variants + dispatcher selector), a per-config `docs/thread` threshold constant that will drift as bin count or partition distribution changes, and a portfolio of two kernels to maintain indefinitely. Future levers (L2, L3, tree-search restructure) would have to be validated against both variants.

**Rejected.** The 1.04–1.08× ceiling does not justify the LOC and maintenance surface. Retains the dispatch-shape mismatch rather than fixing it.

---

## 3. Alternative levers survey

Each candidate is a hypothesis to be validated at production dispatch shape, not a promise.

### L2 — stats pre-permute / layout reorganization

**Hypothesis**: stats load (`gradients`/`hessians` gathered per `partitions[]`) is a hidden cost that benefits from reorganization.

**Empirical status**: **FALSIFIED (Sprint 19)**. S19-01c probe D showed global-memory loads are hidden by AGX behind the shuffle inner loop at single-TG shape. Re-integrating L2 without re-attribution would be the fifth analytical model in a row.

**To validate**: production-shape attribution — measure stats-load fraction at depth 6 (multi-TG) specifically. Likely < 5% of histogram_ms; if so, L2 cannot clear R8.

**Upside range if validated**: 1.02–1.05× e2e (speculative; no production-shape measurement).

**Sprint cost**: 2–3 days attribution + 3–5 days integration + parity sweep.

**Risk**: falsified premise. Do not commit without D0 production-shape re-attribution.

### L3 — MultiClass dispatch fusion

**Hypothesis**: MultiClass currently dispatches 3 separate histogram kernels per iter (one per approxDim). Collapsing to 1 kernel with a dim loop saves dispatch overhead.

**Empirical status**: not measured.

**To validate**: isolated MultiClass bench at production dispatch shape; measure dispatch-overhead fraction vs accumulation fraction.

**Upside range**: **2–3× speedup for MultiClass configs only**. Does not clear the RMSE/Logloss gate (R8 is measured on 50k/RMSE/d6/128b).

**Sprint cost**: 3–4 days + MultiClass-only parity sweep.

**Risk**: scope of win is narrow — MultiClass is 6 of 18 configs. Campaign gate is RMSE.

**Verdict as Sprint 20 mid-pivot**: rejected. MultiClass-only does not clear the campaign gate.

### TG-count reduction (coalesce partitions per TG)

**Hypothesis**: The dispatch-shape problem is fixable by reducing TG count. If each TG processes multiple partitions sequentially (reusing `simdHistU` with per-partition flush) OR if each TG processes a fixed doc-batch regardless of partition (with per-doc partition lookup), the per-TG doc count grows and the fixed overhead amortizes.

**Empirical status**: not measured at production dispatch shape.

**Variants**:

- **A. Per-feature-group single-pass kernel**: one TG per feature group scans all N docs, writes `histogram[part][bin]` via per-partition global atomics. Partition assignment looked up from `partitions[]`. TG count drops from 1638 to 26 (13 groups × 2 stats). Per-TG doc count becomes ~50k / 256 = 195 (matches toy-kernel shape). Projected speedup in isolation: T3b-like (−80% accumulation). Integration cost: fundamentally different grid layout, ~150 LOC, moderate kernel rewrite.

- **B. Per-TG partition-batch kernel**: each TG processes K partitions sequentially, with zero-init/writeback amortized across K partitions. TG count drops from 1638 to ~ 1638 / K. Per-TG overhead unchanged but amortized across K× more work. K=8 roughly halves fixed-overhead fraction. Integration cost: ~80 LOC, moderate.

**Upside range** (Sprint 21 projection):

- Variant A: **1.15–1.30× e2e** at gate config IF production-shape micro-bench confirms the toy-kernel speedup transfers.
- Variant B: **1.05–1.12× e2e** depending on K.

**Sprint cost**: 1–2 days D0 attribution + 2–3 days D1 production-shape micro-bench + 5–7 days D2 integration + parity sweep.

**Risk factors**:

- Per-partition global atomic contention at depth 6 (1638 partitions — if atomic contention dominates, variant A degrades).
- Partition lookup cost in the inner loop (gather from `partitions[doc]`).
- Load imbalance: partitions are not uniform in size; one TG scanning all 50k with per-partition output may idle threads when partitions are small.

**Verdict**: most credible Sprint 21 lever. Requires production-shape validation before committing R8.

### Different tree-search strategy (research-level)

**Hypothesis**: The partition-fragmented dispatch is itself the artifact of CatBoost's symmetric-tree + per-partition-histogram design. An alternative — histogram computed once per iter over all docs, with partition masks applied at split-scoring time — inverts the dispatch: more work per histogram kernel, fewer kernel dispatches.

**Empirical status**: not explored.

**Upside range**: speculative; could be 1.5–2× if the inversion lands, or 0× if scoring-time masking has its own overhead.

**Sprint cost**: weeks. Research-level.

**Verdict**: Sprint 22+ candidate. Flagged, not scoped.

---

## 4. Recommended Sprint 20 disposition: ABANDON

Sprint 20 ships as:

- Sprint 19 tip (T1, 1.018× gate / 1.033× best, R8 ≥1.07× NOT MET — already documented honestly in `docs/sprint19/results.md`)
- D1 parity sweep record (commit `9216f4941c`) — T3b 18/18 bit-exact, 100/100 deterministic
- D2 falsification record (commit `9079ad3873`) — T3b +42.3% regression at depth 6

**No D2b kernel code.** **No Option C hybrid.** **No mid-sprint pivot to L3.**

Sprint 20 net perf delivery: **0×.**

The falsification records are not dead weight — they are the evidence Sprint 21 must build on. The standing warning derived from this sprint (see §6 below) is more valuable than a rushed 1.05× hybrid would have been.

---

## 5. Sprint 21+ plan

Ramos selected "concrete R8 target upfront" — so Sprint 21 commits to a specific number on a specific lever before any integration begins. The structure:

### D0 (Sprint 21 kickoff) — production-shape attribution

Deliverable: per-stage ms breakdown at gate config 50k/RMSE/d6/128b using Metal System Trace (if sandbox-unblocked) OR analytical stage decomposition backed by production-shape micro-benches. Specifically:

- Fixed per-TG overhead fraction at depth 6 (zero-init + writeback + dispatch overhead) vs per-thread accumulation work
- Per-partition atomic contention rate (if variant A is on the table)
- Partition lookup cost in `partitions[doc]` gather pattern (if variant B is on the table)

**Kill-switch**: if D0 shows fixed-overhead fraction at depth 6 is **< 10%** of histogram_ms, then TG-count reduction is not the dominant lever and Sprint 21 must re-pick before committing to R8.

Zero parity/perf risk — D0 is measurement only, no code changes.

### D1 — lever selection with concrete R8 projection

Based on D0 attribution, select one lever. Current prior (subject to D0 override): **TG-count reduction, variant A (per-feature-group single-pass kernel)**.

**Sprint 21 R8 target: ≥1.08× e2e at gate config** (conservative midpoint; lower bound 1.05×, upper bound 1.18×).

Projection backing: variant A's dispatch shape mirrors the toy-kernel shape where T3b showed −84% (1 TG per feature group × 195 docs/thread). The toy-kernel speedup is projected to transfer this time *because the dispatch shape is preserved*, not assumed to transfer across shapes. D1 must confirm with a production-shape micro-bench (multi-feature-group dispatch at gate config) BEFORE integration begins.

Commit to R8 ≥1.08× in Sprint 21 README at D1 close. Do not soften during D2.

### D2+ — integration

Execute per DEC-012 (one structural change per commit). Estimated: kernel rewrite (1 commit) + host dispatch changes (1 commit) + DECISIONS.md amendment (1 commit) + parity/perf gate commit. Parity sweep against Sprint 19 tip must hold DEC-008 envelope. Perf must clear R8 1.08× at the gate.

### Sprint 21-22-23 cumulative pipeline

| Sprint | Lever | R8 (midpoint) | Cumulative (midpoint) |
|---|---|---|---|
| 19 (shipped) | T1 DEC-016 | 1.018× | 1.018× |
| 20 (abandon) | — | 1.000× | 1.018× |
| 21 (proposed) | TG-count reduction | 1.08× | 1.10× |
| 22 (proposed) | L2 re-attribution + integration OR L3 MultiClass fusion | 1.05× | 1.15× |
| 23 (proposed) | Tree-search restructure OR further TG-layout work | 1.10× | 1.27× |

**Midpoint cumulative by Sprint 23: 1.27×.** Upper bound cumulative: 1.46×.

**Operation Verstappen target: ≥1.5× e2e.** Per Ramos's answer 3, the target stands — no re-scoping. Therefore:

- 1.5× is **not credibly reachable by end of Sprint 23 at midpoint projections**.
- At upper-bound projections (1.46×), the campaign falls 4pp short.
- Sprint 24+ escalation is likely required: either a fifth lever or a fundamental re-decomposition (e.g. different tree-search strategy §3).

This is the honest projection. No fifth analytical model is reached for to close the gap.

---

## 6. Out of scope / explicitly rejected

Documented on the record so they are not lost to institutional memory:

- **Option A merge-partitions (from d2_results.md)** — folded into Sprint 21 TG-count reduction variant A. Same idea, different scope boundary.
- **Option C depth-gated hybrid T3b+T1** — rejected on ceiling (1.04–1.08×) vs cost (200 LOC, dispatcher selector, kernel portfolio).
- **Option D maxBlocksPerPart increase** — rejected in Sprint 19 (S19-02b variant C); with T3b falsified, premise is gone too.
- **Pivot to L3 MultiClass fusion mid-Sprint 20** — MultiClass is 6 of 18 configs; does not clear the RMSE campaign gate. Viable as Sprint 22 lever for narrow win.
- **Pivot to L2 stats pre-permute mid-Sprint 20** — premise falsified by S19-01c probe D (gather hidden by AGX). Re-integration would be fifth analytical model. Requires Sprint 21 D0 re-attribution before any commitment.
- **Kahan-compensated T3b** — D1 parity passed at 0 ulp without Kahan; Kahan added TG memory cost, not speedup. Kahan addressed parity, not the dispatch-shape problem that actually falsified T3b.

---

## 7. Honest bottom line (PR/README-ready)

Sprint 20 is abandoned. The T3b threadgroup-atomic-CAS accumulator measured −84.4% single-TG accumulation in toy-kernel isolation and +42.3% regression at the production gate config (50k/RMSE/d6/128b). Parity was 18/18 bit-exact — the failure is pure dispatch-shape mismatch: T3b's fixed per-TG overhead (1024-slot zero-init + writeback) amortizes at 195 docs/thread (toy) and dominates at 3 docs/thread (production depth 6). This is the fourth analytical model falsified in the current campaign (DEC-013 writeback, DEC-014 original gather, DEC-015 col-major, T3b-as-drop-in).

Sprint 20 ships no perf delta. Sprint 21 opens with **production-shape attribution as D0** (zero-risk measurement), then commits to **TG-count reduction (per-feature-group single-pass kernel)** at **R8 ≥1.08× e2e** with a production-shape micro-bench backing the number before integration begins.

Operation Verstappen's ≥1.5× cumulative target stands per standing order. Sprint 21-22-23 midpoint projection is 1.27× cumulative — **1.5× is not credibly reachable by Sprint 23 at midpoints**. Sprint 24+ escalation is the honest escape valve if the pipeline ships as projected. No target re-scoping.

### Standing warning (preserved in DEC-017 retirement)

Toy-kernel ablations at single-TG root dispatch shape **DO NOT predict production partition-fragmented dispatch** in this codebase. Future T* algorithmic campaigns MUST validate at production dispatch shape (multi-TG, depth-appropriate partition granularity) before committing to an R8 projection. Apply this rule to all Sprint 21+ lever evaluations.
