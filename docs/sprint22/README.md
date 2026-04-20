# Sprint 22 — T2 Sort-by-Bin Integration

**Branch**: `mlx/sprint-22-hist-t2-sort` (to be cut from Sprint 21 tip)
**Campaign**: Operation Verstappen — battle 7 of 9
**Lever**: T2 sort-by-bin (DEC-020 VIABLE, rank #1)
**Gate config**: 50k/RMSE/d6/128b (unchanged from Sprint 19–21)
**R8 target**: ≥1.51× e2e at gate config (conservative band, ratio ≤ 0.50); 1.5× Verstappen gate clears iff D0 ratio ≤ 0.60
**Authority**: `docs/sprint21/d1r4_synthesis.md` — authoritative lever-ranking, D0 specification, and R8 ledger input for this sprint

---

## §1 Background

Sprint 21 was a measurement-only sprint (A1 discipline). It produced mechanism-direct production-shape evidence for two lever candidates and retired both previous analytical models:

- **DEC-018 TG-count reduction variant A** — RETIRED (D0 kill-switch: fixed overhead 2.5% << 10% gate)
- **DEC-019 L2 stats pre-permute** — FALSIFIED (zero-gather upper bound +2.61% slower; AGX prefetcher hides gather entirely)

The single surviving viable-set member is **T2 sort-by-bin** (DEC-020), measured at −64.8% `histogram_ms` reduction at 1664-TG production dispatch shape in a scratch micro-harness (`docs/sprint21/d1r2_t2_microbench.md`). T2 enters Sprint 22 as rank #1 and the only lever with mechanism-backed production-shape evidence.

**Sprint 22 is a single-lever integration sprint.** The entire Verstappen ≥1.5× campaign gate depends on the Sprint 22 D0 in-situ ratio measurement. If T2 clears D0, the campaign gate is reachable within this sprint. If T2 is falsified at D0, the campaign gate is not reachable on the current kernel structure within a single additional sprint, and Sprint 23 must pivot to research-track options.

**Campaign falsification count entering Sprint 22**: 7 models falsified (writeback-plurality, DEC-014 gather, DEC-015 col-major, DEC-014 A1 batch-64, DEC-017 T3b, DEC-018 variant A, DEC-019 L2). Standing rule: do not re-pick from the falsified set without new structural evidence.

---

## §2 D0 Specification — In-Situ T2 Integration Probe

**Purpose**: Convert the D1-R2 synthetic-harness signal into a production-shape, in-situ measurement before any Sprint 22 kernel commit lands. The D1-R2 harness used identity-permuted docs (synthetic); production uses argsort-permuted docs from `structure_searcher.cpp`. Ratio-transfer from synthetic to production is **unproven** and must be tested before integration.

**Pre-requisite**: D1-R3 per-kernel-profile instrumentation (commit `ac378d8de6`, Sprint 21). Use `bench_boosting --per-kernel-profile` at gate config.

**Scope (scratch-only — A1-G6 discipline applies)**. Implement `DispatchHistogramT2` as an in-source variant of `DispatchHistogramBatched` in `catboost/mlx/methods/histogram.cpp`, guarded by an environment variable or compile-time flag (`CATBOOST_MLX_HISTOGRAM_T2=1`). Kernel source lives in a new `kHistOneByteT2Source` string in `kernel_sources.h` OR locally in `bench_boosting.cpp` if discipline requires scratch-only. No kernel source is committed until D0 kill-switch clears. Parity is **not required** for D0 — parity is D1.

**Measurement protocol**:
- Gate config: `--rows 50000 --features 50 --classes 1 --depth 6 --iters 50 --bins 128 --seed 42 --per-kernel-profile`
- 3 independent runs × 49 warm iterations each (iter 0 cold excluded), matching D1-R1/R2 methodology
- Primary metric: `histogram_ms` mean across runs; secondary: `iter_total_ms`
- Baseline: T1 via same binary, same flags, flag-off. Measure T1 and T2 in same session to cancel Metal scheduler drift.
- Output document: `docs/sprint22/d0_t2_production_shape.md` — 3-run measurement table, ratio with ±2σ, kill-switch verdict, and if PASS: projected e2e band; if FAIL: research-track pivot plan.

**Kill-switch (direct mechanism threshold)**. Compute `ratio = hist_ms(T2) / hist_ms(T1)` at gate config:

| ratio | T2 verdict | Action |
|---|---|---|
| ≤ 0.45 | PASS — optimistic band holds | Proceed directly to S22-D1 parity sweep |
| 0.45 < ratio ≤ 0.60 | PASS — conservative band | Proceed to D1; R8 projection drops to 1.37–1.51×; Ramos re-decides whether to ship or escalate |
| > 0.60 | **T2 FALSIFIED at production shape** | T2 drops to RESEARCH; Sprint 22 pivots to tree-search restructure scoping; 0× in-sprint perf deliverable |

**Rationale for the 0.60 threshold (from `docs/sprint21/d1r4_synthesis.md §4`)**. D1-R2 in-harness band is 0.33–0.36×. Ratio-transfer risk is dominated by the sort pre-pass being cache-sensitive under argsort-permuted docs. Worst case ratio-transfer: sort pre-pass doubles in cost (pushing T2 from ~7.6 ms to ~13 ms ≈ 0.60 ratio). If sort cost more than doubles, T2's mechanism is fundamentally incompatible with production access patterns. 0.60 is not an arbitrary margin — it is the ratio where the kill-switch boundary cumulative e2e (~1.37×) falls below the 1.5× Verstappen gate. Below 0.60 there is no single-sprint stacking path to 1.5×.

**Cost estimate**: 1 day (scratch variant implementation + 3-run measurement + doc).

---

## §3 Exit Gates

| Gate | Criterion | Owner | Blocked on |
|---|---|---|---|
| S22-D0 | `ratio = hist_ms(T2) / hist_ms(T1) ≤ 0.60` at gate config, in-situ argsort-permuted partitions | @ml-engineer | — |
| S22-D1 | 18-config parity vs DEC-008 envelope: RMSE ULP = 0; Logloss ULP ≤ 4; MultiClass ULP ≤ 8; 100-run determinism at gate config | @qa-engineer | S22-D0 PASS |
| S22-D2 | T2 production integration + default flip, per DEC-012 atomic commits (4–5 commits) | @ml-engineer | S22-D1 PASS |
| S22-D3 | 18-config perf sweep + R8 honest commitment. Documented cumulative e2e at gate config; no softening of R8 ledger | @performance-engineer | S22-D2 PASS |

**Kill-switch at D0 fires if ratio > 0.60**: Sprint 22 ends with a pivot plan to tree-search restructure. No perf deliverable. No kernel committed. Campaign gate missed; Sprint 23 must re-scope or research-track.

**Kill-switch at D1 fires if any config fails parity AND Kahan compensation falls below 0.60 ratio after compensation overhead**: T2 is also abandoned. Both kill-switches are direct mechanism tests, not proxies.

---

## §4 Lever-Set

### Rank #1 — T2 sort-by-bin (this sprint, integration)

**Mechanism**: Two-kernel dispatch replacing the 32-iteration simd_shuffle broadcast chain.
1. Counting-sort pre-pass: bin-partition each partition's docs by feature-0 bin, emit `binOffsets[]`.
2. Accumulator: feature-0 does pure bin-range scan (no shuffle, single writer per bin); features 1–3 do per-doc sorted scatter via global atomics.

simd_shuffle serial chain is structurally eliminated for feature-0. Sort cost is ~5× cheaper than the accumulation it replaces at 781 docs/TG × 128 bins.

**Integration plan (per DEC-012 one-structural-change-per-commit)**:
1. T2 kernel added to `kernel_sources.h` (or promoted from scratch on D0 PASS)
2. `DispatchHistogramT2` added as dispatch variant in `histogram.cpp`
3. T2 selection guarded by compile-time flag or runtime config
4. Host-side buffer allocation: `sortedDocs[numTGs × maxPartDocs]`, `binOffsets[numTGs × 129]`
5. Default flip after parity clears D1

TG memory: `atomic_uint tgCounts[128]` = 512 B, negligible against DEC-011 32 KB ceiling.

**New buffers**: `sortedDocs[numTGs × maxPartDocs]`, `binOffsets[numTGs × 129]`. One additional kernel launch per histogram dispatch.

### S19-11 EvalAtBoundary readback elimination (compound, in-sprint)

Six `EvalAtBoundary` CPU readbacks in `structure_searcher.cpp` (`:275`, `:593`, `:653`, `:686`) force a Metal sync barrier at each tree-level iteration. At depth 6: 6 forced syncs × ~50 µs = ~0.3 ms per iteration. Bounded 0.5–1 day fix. Scheduled as compound with T2 integration per `docs/sprint21/d1r4_synthesis.md §3 rank #2`. Carry-forward from S19-11.

### Dispatch inversion (deferred to Sprint 23 research spike)

Invert the partition-fragmented dispatch: compute one histogram over all docs, apply partition masks at split-scoring time. Speculative 1.5–2× if inversion lands, 0× if scoring mask adds its own overhead. Weeks of research cost — not fundable in a single integration sprint. Defer to Sprint 23 after T2 integration lands. If T2 is falsified at D0, Sprint 23 is the dispatch-inversion research spike by default.

---

## §5 Risks Carried Forward

### Primary — ratio transfer (synthetic → production)

D1-R2 measured T2/T1 = 0.352× in a synthetic harness with identity-permuted docs (docs 0..50k sorted into 64 partitions by `doc / partSize`). Production uses argsort over leaf indices, producing irregular memory access. The T2/T1 ratio should cancel locality artifacts to first order (both variants benefit equally from locality), but if the sort pre-pass is cache-sensitive under argsort-permuted access in a way the accumulator is not, the ratio could widen. The variant A (26 TGs × 195 docs/TG) shape in D1-R2 gave T2-VA/T1-VA = 0.285×, suggesting the mechanism holds at one production-representative shape — but this is suggestive, not conclusive for the primary 64-partition shape. Sprint 22 D0 resolves this directly.

### Parity vs DEC-008 envelope

D1-R2 per-bin max ULP = 64 — this is for raw histogram bin values, not for end-to-end training loss. DEC-008 requires final loss ULP ≤ 4 (RMSE/Logloss) / ≤ 8 (MultiClass). The per-bin drift is absorbed through scoring and leaf estimation. Sprint 22 D1 parity sweep validates end-to-end ULP across 18 configs. If any config fails, Kahan-compensated summation on the per-doc scatter path is the fallback (same as DEC-017 fallback plan) — costs ~1 extra `atomic_uint` per bin; may compress the speedup by 2–5%. Budget: +2–3 days if Kahan needed.

### Multi-feature sort scope

D1-R2 sorts on feature-0 bin only; features 1–3 use atomic scatter. Sprint 22 ships feature-0 sort as-implemented; multi-feature optimization deferred to Sprint 23+. Do not widen scope in-sprint.

### Integration complexity and DEC-011 ceiling

Grid geometry unchanged (`(256 × numGroups, numParts, numStats)`). TG memory new allocation: `atomic_uint tgCounts[128]` = 512 B against 32 KB ceiling — negligible. Sprint 22 D2 integration estimate: 4–5 DEC-012 atomic commits, 3–4 days. No ceiling risk.

---

## §6 R8 Honest Commitment

**Current position**: ~1.07× cumulative e2e over Sprint 16-class baseline (from S17/S18/S19 kernel improvements; S20 and S21 contributed 0×).

**Gap**: 40% residual speedup required from Sprint 22 onward to reach 1.5× Verstappen gate.

| T2 scenario | S22 e2e | Cumulative (× S16-class) | Verstappen gate |
|---|---|---|---|
| Optimistic (ratio 0.33) | +1.83× | ~1.96× | CLEARED +46 pp |
| Midpoint (ratio 0.36) | +1.76× | ~1.88× | CLEARED +38 pp |
| Conservative (ratio 0.50) | +1.51× | ~1.62× | CLEARED +12 pp |
| Kill-switch boundary (ratio 0.60) | +1.37× | ~1.47× | FAIL by 3 pp |
| T2 falsified at D0 (ratio > 0.60) | +1.00× | ~1.07× | FAIL by 43 pp |

**The 1.5× gate is cleared iff T2 delivers its conservative band or better.** The kill-switch at 0.60 is calibrated exactly at the boundary where the cumulative misses the gate. Below 0.60 there is no credible single-sprint stacking path.

**If T2 is falsified at D0**: the Verstappen ≥1.5× gate is not reachable on the current kernel structure within a single additional sprint. Sprint 23 options: (a) tree-search restructure research spike (speculative 1.5–2×, weeks of cost, uncertain deliverability), (b) campaign gate re-scope to 1.25–1.35× as delivered target. This is the honest fallback — do not soften.

The R8 table above must be propagated unchanged into the Sprint 22 HANDOFF.md and CHANGELOG-DEV.md at closeout. No target re-scoping mid-sprint without explicit Ramos direction.
