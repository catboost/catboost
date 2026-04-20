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

## Sprint 22 — T2 Sort-by-Bin Integration (OPEN)

**Branch**: `mlx/sprint-22-hist-t2-sort` (to be cut)
**Campaign**: Operation Verstappen — battle 7 of 9 — single-lever integration sprint
**Gate config**: 50k/RMSE/d6/128b (unchanged)
**Lever**: T2 sort-by-bin (DEC-020 VIABLE, rank #1)
**R8 target**: ≥1.51× e2e at gate config (conservative band); 1.5× gate clears iff T2 D0 ratio ≤ 0.60
**Authority**: `docs/sprint21/d1r4_synthesis.md` §3/§4/§5/§6

### Open

- [ ] S22-D0 — In-situ T2 integration probe at production shape. Implement `DispatchHistogramT2` as a scratch variant in `catboost/mlx/methods/histogram.cpp` or locally in `bench_boosting.cpp`, guarded by env-var or compile-time flag (`CATBOOST_MLX_HISTOGRAM_T2=1`). Parity NOT required for D0 (perf-only mechanism test). Measure `histogram_ms` via `--per-kernel-profile` at gate config (3 independent runs × 49 warm iters). Compute `ratio = hist_ms(T2) / hist_ms(T1)` in same session. Kill-switch: ratio > 0.60 → T2 FALSIFIED at production shape, Sprint 22 pivots to tree-search restructure scoping. Acceptance: ratio ≤ 0.60 at gate config with ±2σ band documented. Output: `docs/sprint22/d0_t2_production_shape.md`. — @ml-engineer **OPEN** (est. 1 day; A1-G6 scratch-only discipline applies)
- [ ] S22-D1 — 18-config parity sweep against DEC-008 envelope. RMSE bit-exact (ULP = 0); Logloss ULP ≤ 4; MultiClass ULP ≤ 8. 100-run determinism at gate config. Kahan fallback budget if any config fails. Blocked on S22-D0 PASS (ratio ≤ 0.60). — @qa-engineer **BLOCKED** (est. 1 day)
- [ ] S22-D2 — T2 production integration + default-flip per DEC-012 one-structural-change-per-commit. Estimated 4–5 atomic commits: (1) T2 kernel in `kernel_sources.h`, (2) `DispatchHistogramT2` dispatch variant in `histogram.cpp`, (3) runtime/compile-time selection guard, (4) host-side guard and buffer allocation (`sortedDocs`, `binOffsets`), (5) default flip after parity clears. Blocked on S22-D1 PASS. — @ml-engineer **BLOCKED** (est. 3 days)
- [ ] S22-D3 — 18-config perf sweep at gate config + R8 honest commitment. Measure `iter_total_ms` and `histogram_ms` across 18 configs. Document cumulative e2e at gate config (no softening of R8 ledger per `docs/sprint21/d1r4_synthesis.md §5`). Blocked on S22-D2 PASS. — @performance-engineer **BLOCKED** (est. 1 day)

### Carry-forward

- [ ] S19-11 — 6 `EvalAtBoundary` CPU readbacks in `structure_searcher.cpp` (`:275`, `:593`, `:653`, `:686`). Scheduled as compound with T2 integration in Sprint 22 per `docs/sprint21/d1r4_synthesis.md §3 rank #2`. Bounded 0.5–1 day fix (~0.3 ms / 31.93 ms standalone contribution). — @ml-engineer **CARRY-FORWARD**

---

## Sprint 20–24 Backlog (one-line each, expanded per sprint)

- Sprint 20: T3b atomic-CAS — CLOSED, FALSIFIED (DEC-017 RETIRED). PR #12 OPEN.
- Sprint 21: A1 measurement — CLOSED, 0× perf, T2 promoted to viable-set. PR #13 pending.
- Sprint 22: T2 sort-by-bin integration — OPEN (single-lever sprint; campaign 1.5× gate depends on S22-D0 ratio)
- Sprint 23: Tree-search restructure / dispatch inversion research spike (if T2 clears Sprint 22; otherwise campaign re-scope)
- Sprint 24: Championship benchmark — final tuning, dominance suite vs CPU + CUDA, release polish

---

## Carry-forward from pre-Sprint 16

- BUG-007: nanobind path doesn't sort group_ids (silent divergence on unsorted input) — from @qa-engineer Sprint 12 review
- bench_boosting K=10 anchor mismatch: expected 2.22267818, measured 1.78561831 — flagged Sprint 7, unresolved
