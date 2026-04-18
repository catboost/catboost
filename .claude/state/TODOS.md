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

- [ ] S19-03 — Implement winning accumulation redesign from DEC-014 at `kernel_sources.h`; preserve DEC-011 32 KB TG memory ceiling; PROPOSE → CRITIQUE → IMPLEMENT → VERIFY → REFLECT — @ml-engineer
- [ ] S19-04 — Parity sweep: DEC-008 envelope (approxDim ∈ {1,3}, N ≤ 50k, all losses, 32/128 bins, 50 iter, d6); 100-run determinism on 50k/RMSE/d6/128b — @qa-engineer

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

## Sprint 20–24 Backlog (one-line each, expanded per sprint)

- Sprint 20: Quantization fastpath — GPU quantization on ingest, persistent device-resident datasets
- Sprint 21: Leaf + apply fusion — one command buffer for leaf estimation + tree apply
- Sprint 22: Kernel specialization — dtype/depth-specialized kernels via MLX JIT template system
- Sprint 23: Large-scale tiling — datasets exceeding Metal buffer limits, async CPU→GPU streaming
- Sprint 24: Championship benchmark — final tuning, dominance suite vs CPU + CUDA, release polish

---

## Carry-forward from pre-Sprint 16

- BUG-007: nanobind path doesn't sort group_ids (silent divergence on unsorted input) — from @qa-engineer Sprint 12 review
- bench_boosting K=10 anchor mismatch: expected 2.22267818, measured 1.78561831 — flagged Sprint 7, unresolved
