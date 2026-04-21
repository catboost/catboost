# Active Tasks — CatBoost-MLX

> Coverage: Sprints 0–15 reconstructed from git/agent-memory on 2026-04-15. Sprint 16+ is source of truth.

## Sprint 16 — Performance Diagnosis & First Cut

**Branch**: `mlx/sprint-16-perf-diagnosis`
**Goal**: Measure the MLX perf gap, restore state file hygiene, land the sync-storm free-move.

### Open

- [ ] S16-01 Per-stage profiling report — JSON for 9 stages × {depth 6,10} × {oblivious,depthwise,lossguide} — @performance-engineer
- [ ] S16-02 Baseline profile at `.cache/profiling/sprint16/baseline_*.json` for N∈{10k,100k,1M} × {RMSE,Logloss,MultiClass} × {32 bins,128 bins} — @performance-engineer
- [ ] S16-03 Metal System Trace `.trace` + 1-page findings note in `.cache/profiling/sprint16/` — @research-scientist
- [ ] S16-04 Sync-point inventory: every in-loop EvalNow/eval/item() with call site, frequency, cost — @performance-engineer
- [ ] S16-05 `bench_boosting` + `bench_mlx_vs_cpu.py` extended with `--stage-profile`; CPU-parity runner emits side-by-side JSON — @mlops-engineer
- [ ] S16-06 CI gate: >5% wall-clock regression on 50k regression benchmark fails PR — @mlops-engineer
- [ ] S16-07 `pointwise_target.h` sync-storm elimination: remove 18 EvalNow, add 1 EvalAtBoundary at iteration boundary; ≥10% e2e improvement on N=100k RMSE depth=6 — @ml-engineer
- [ ] S16-08 Numerical parity: RMSE bit-exact, Logloss ulp≤4, MultiClass ulp≤8 vs Sprint 15 at N∈{10k,100k,1M} — @qa-engineer
- [ ] S16-09 State files restored: 5 files in `.claude/state/` — @mlops-engineer + @technical-writer ✅ (this commit)
- [ ] S16-10 Post-fix profile: sync stage drops ≥50%, no non-sync stage regresses >10% — @performance-engineer
- [ ] S16-11 ARCHITECTURE.md sync-points updated; Sprint 16 CHANGELOG-DEV entry; Sprint 17 plan drafted from MST evidence — @technical-writer
- [ ] S16-12 Code review + security pass on S16-07 — @code-reviewer, @security-auditor

### Merge gate

All items above must be green to merge `mlx/sprint-16-perf-diagnosis` → `master`.

---

## Campaign Scoreboard — Operation Verstappen

**Sprint 24 exit targets** (all benchmarks report 32-bin and 128-bin columns):

### Primary dominance targets

- [ ] 10k regression: MLX ≤ 0.20s (1.35x faster than CPU)
- [ ] 50k regression: MLX ≤ 0.25s (1.68x faster than CPU)
- [ ] 50k binary: MLX ≤ 0.40s (1.85x faster than CPU)
- [ ] 50k multiclass: MLX ≤ 0.60s (2.2x faster than CPU)
- [ ] 500k regression: MLX ≤ 0.3x CPU
- [ ] 2M regression: MLX ≤ 0.2x CPU (5x faster)
- [ ] Beat XGBoost + LightGBM on every 50k+ benchmark
- [ ] Match or beat CatBoost CUDA A100 on Airline 10M (15s target, szilard/GBM-perf 2024)

### Secondary targets

- [ ] Histogram SIMD occupancy ≥ 70%
- [ ] Sync points per training iteration ≤ 2
- [ ] CI perf regression gate: no PR lands with >5% regression
- [ ] Correctness parity vs CPU reference: delta ≤ 1e-5

### Small-N policy

N < 5k: CPU fallback acceptable. Championship push focuses on N ≥ 10k.

---

## Sprint 19 — Accumulation Redesign (PIVOTED from Two-Phase Writeback)

**Branch**: `mlx/sprint-19-hist-writeback` (name reflects original scope — history over cosmetics)  
**Gate config**: 50k/RMSE/128b  
**Baseline** (S18 after / S19-01 ground-truth, steady-state iters 5–49): `histogram_ms` = 15.46 ms (ground-truth), `iter_total_ms` = 21.12 ms  
**Pivot**: S19-01 showed writeback = 0.79 ms (5%), accumulation = 14.30 ms (93%). R8 fired at 1.02–1.04× e2e. Ramos chose accumulation redesign. See DECISIONS.md for DEC-013 (SUPERSEDED) and DEC-014 (DRAFT).  
**Projection**: `histogram_ms` −30% min (accumulation redesign; 1.25–1.50× e2e aggressive target still live pending S19-02b)  
**Sprint length**: **Day 6** (pivot cost one day; was Day 5)

### Day 0

- [x] S19-00 — Branch cut from S18 tip (`463de74efa`); 18 baseline JSONs copied to `.cache/profiling/sprint19/baseline/`; state files scaffolded; `docs/sprint19/README.md` scaffold created — @ml-product-owner / @technical-writer **DONE**

### Day 1

- [x] ~~S19-01~~ — **COMPLETE-BUT-SUPERSEDED** — Writeback attribution (commit `d7ea14e28c`): writeback = 0.79 ms (5%), accumulation = 14.30 ms (93%). R8 fired. Evidence correct; premise (writeback as plurality) falsified. See `docs/sprint19/attribution.md` and DECISIONS.md DEC-014. — @performance-engineer
- [x] ~~S19-02~~ — **COMPLETE-BUT-SUPERSEDED** — DEC-013 draft written (two-phase writeback); DEC-013 premise invalidated by S19-01. Variant (c) 3.0 ms projection not supported by ground truth. See `docs/sprint19/ablation.md` and DECISIONS.md DEC-013 SUPERSEDED note. — @research-scientist
- [ ] S19-11 — In-sprint cleanup: 6 `EvalAtBoundary` CPU readbacks in `structure_searcher.cpp` ("fix properly always" carry-forward from S18) — @ml-engineer
- [ ] S19-12 — In-sprint cleanup: VGPR confirmation + S18 deferred code-review items — @performance-engineer / @code-reviewer

### Day 2

- [ ] S19-01b — Accumulation sub-phase attribution: re-attribute 14.30 ms accumulation across sub-phases; ±1 ms error bars; output appended to `docs/sprint19/attribution.md` — @performance-engineer **IN PROGRESS**
- [ ] S19-02b — Accumulation redesign ablation: variants A (wider batch), B (coalesced TG staging), C (per-feature specialization), D (different ownership granularity) × {bins 32,128} × {N 10k,50k}; PROPOSE → CRITIQUE → IMPLEMENT-draft → VERIFY-project → REFLECT; DEC-014 lock; output appended to `docs/sprint19/ablation.md` — @research-scientist **IN PROGRESS**

### Day 3

- [x] S19-03 (Commit 1 — DEC-015) — **BLOCKED: performance gate not met** — DEC-015 col-major compressedIndex transposed view implemented, parity 18/18 PASS (0 ULP), determinism 100/100 PASS, builds clean. Gate: `histogram_ms ~4.2 ms, e2e ~2.13×`. Measured: warm mean ref=33.7–34.2 ms, DEC-015=34.3–35.7 ms → **~0.98× e2e**. S19-01b prediction falsified. 5 modified files NOT committed. Working tree dirty. See HANDOFF.md BLOCKERS. — @ml-engineer
- [ ] S19-03 (re-spec) — **NEW: micro-benchmark re-attribution** — Build isolated Metal kernel sweeping row-major vs col-major `compressedIndex` access at N∈{10k,50k} × lineSize∈{13,25,50}. Confirm whether AGX hardware hides scatter cost or bottleneck is elsewhere. Required before next kernel intervention. — @performance-engineer
- [ ] S19-04 — Parity sweep: DEC-008 envelope (approxDim ∈ {1,3}, N ≤ 50k, all losses, 32/128 bins, 50 iter, d6); 100-run determinism on 50k/RMSE/d6/128b — @qa-engineer (HOLD: S19-03 gate must resolve first)

### Day 4+

- [ ] S19-05 — Stage-profiler delta on 18-config grid; output: `.cache/profiling/sprint19/{before,after}_*.json` + `docs/sprint19/results.md` delta table — @performance-engineer
- [ ] S19-06 — Update `benchmarks/check_histogram_gate.py` reference baseline to S18 after-JSON; verify CI gate continuity; intentional-regression dry-run — @mlops-engineer
- [ ] S19-07 — Code review: accumulation phase correctness, DEC-014 design, DEC-011 ceiling, barrier count — @code-reviewer
- [ ] S19-08 — Security pass: kernel string injection surface; no new externally-controlled buffer sizes — @security-auditor
- [ ] S19-09 — Metal System Trace re-capture on gate config; confirm accumulation improvement; output: appendix in `docs/sprint19/results.md` — @performance-engineer

### In-sprint cleanup

- [ ] S19-11 — 6 `EvalAtBoundary` CPU readbacks in `structure_searcher.cpp` (see Day 1 above)
- [ ] S19-12 — VGPR confirmation + S18 deferred code-review items (see Day 1 above)

### Docs

- [ ] S19-10 — `docs/sprint19/` full population: README (scaffold DONE; pivot update DONE), attribution, ablation, results, non_goals; `CHANGELOG-DEV.md` S19 entry; `ARCHITECTURE.md` accumulation section update; DEC-013 SUPERSEDED + DEC-014 lock (DRAFT → ACCEPTED at S19-02b close); same-PR docs standing order fulfilled — @technical-writer

### Sprint 19 merge gates

| Gate | Criterion |
|------|-----------|
| G1 | **`histogram_ms` −30% min** on 50k/RMSE/128b (accumulation = 93%; 32% accum reduction ≈ 30% histogram_ms) |
| G2 | No 18-config regression >5% |
| G3 | Parity 108/108 bit-exact across DEC-008 envelope |
| G4 | `iter_total_ms` −30% min on 50k/RMSE/128b (aggressive 1.5× target pending S19-01b/S19-02b) |
| G5 | No non-histogram stage regresses >10% |
| G6 | CI green |

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

## Sprint 23 — T2 Scratch→Production Promotion + NIT Catalog + Tree-Search Research (OPEN)

**Branch**: `mlx/sprint-23-t2-promotion` (to be cut)
**Campaign**: Operation Verstappen — battle 8 of 9
**Gate config**: 50k/RMSE/d6/128b (unchanged)
**Authority**: `docs/sprint22/d5_code_review.md §4` (NIT catalog); `docs/sprint21/d1r4_synthesis.md §3` (tree-search rank #2)

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

- [ ] S23-R1 — EvalAtBoundary readback elimination (S19-11 carry-forward). Six `EvalAtBoundary` CPU readbacks in `structure_searcher.cpp` (`:275`, `:593`, `:653`, `:686`) force Metal sync barriers at each tree-level. At depth 6: ~0.3 ms/iter standalone. Scope the GPU-resident split-selection replacement per `docs/sprint16/sync_inventory.md`. Bounded 0.5–1 day. — @ml-engineer **CARRY-FORWARD from S19-11**
- [ ] S23-R2 — Dispatch inversion research spike. Scope: can one histogram over all docs + scoring-time partition masking replace the partition-fragmented 1638-TG dispatch? If no concrete design surfaces within 2 days, declare unreachable for the Verstappen campaign window and defer. — @research-scientist **RESEARCH**

### Carry-forward

- [ ] S19-11 — See S23-R1 above (same task, merged).

---

---

## Sprint 24 — DEC-023 Atomic-Float Race Fix (OPEN)

**Branch**: TBD (cut from Sprint 23 tip)
**Campaign**: Operation Verstappen — battle 9 of 9
**Gate config**: 50k/RMSE/d6/128b (unchanged)

### D0 — DEC-023 fix (blocking)

- [ ] S24-D0 — Fix features 1-3 `atomic_fetch_add` non-determinism in T2-accum. Choose one of three fix options (DEC-023): (1) threadgroup-local reduce + single-thread commit, (2) int-atomic fixed-point accumulation, (3) Kahan/Neumaier compensated summation (standalone insufficient). Preferred: Option 1 (mirrors feat-0 design; known-clean mechanism). Budget: 1-2 days. Kill-switch: if fix degrades gate-config ratio below 0.45× (optimistic band), escalate to structural redesign. Acceptance: config #8 becomes 10/10 deterministic; 18/18 parity sweep clean; gate config remains 100/100. — @ml-engineer **OPEN**

### Standing-order update

- [ ] S24-SO-1 — Update parity sweep protocol (standing order for all future sprints): minimum 5 runs per non-gate config (catches ~50/50 bimodality at 97% probability); gate config unconditionally 100 runs. Document in `docs/sprint24/README.md §Exit Gates`. Apply retrospectively to any future sweep doc that claims determinism. — @technical-writer **OPEN**

---

## Sprint 20–24 Backlog (one-line each, expanded per sprint)

- Sprint 20: T3b atomic-CAS — CLOSED, FALSIFIED (DEC-017 RETIRED). PR #12 OPEN.
- Sprint 21: A1 measurement — CLOSED, 0× perf, T2 promoted to viable-set. PR #13 pending.
- Sprint 22: T2 sort-by-bin integration — CLOSED, R8 1.90×, Verstappen gate cleared. PR #14 pending. Note: S22 D3 parity record corrected to 17/18 (see DEC-020 footnote + DEC-023).
- Sprint 23: T2 scratch→production promotion — IN PROGRESS. D0 complete (4 commits, kill-switch tripped on pre-existing config #8 bimodal). R1/R2 pending.
- Sprint 24: DEC-023 atomic-float race fix (S24-D0) + championship benchmark

---

## Carry-forward from pre-Sprint 16

- BUG-007: nanobind path doesn't sort group_ids (silent divergence on unsorted input) — from @qa-engineer Sprint 12 review
- bench_boosting K=10 anchor mismatch: expected 2.22267818, measured 1.78561831 — flagged Sprint 7, unresolved
