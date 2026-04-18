# Operation Verstappen — Sprint 19: Accumulation Redesign

> **Sprint 19 pivot (2026-04-17)**: The original writeback hypothesis was falsified by S19-01 ground-truth measurement. Writeback is 0.79 ms (5%) of `histogram_ms` — not the plurality. The real bottleneck is the **accumulation phase at 14.30 ms (93%)**. Sprint 19 pivots from L_writeback to L_accum. DEC-013 SUPERSEDED. DEC-014 DRAFT. See [`docs/sprint19/attribution.md`](attribution.md) for the full S19-01 data.
>
> Branch name `mlx/sprint-19-hist-writeback` reflects the original scope and is preserved for history.

## What this is

Sprint 19 is the third **structural kernel rewrite** of Operation Verstappen (battle 4 of 9). Sprint 17 eliminated the serial 255-step threadgroup reduction (-89–93% `histogram_ms`). Sprint 18 eliminated per-thread device-memory spill via `simdHist` on-chip accumulation (-56–85% `histogram_ms`). Sprint 19 originally targeted the writeback phase; S19-01 ground-truth showed writeback is 5% of the cost. **Sprint 19 now targets the accumulation phase (93% of `histogram_ms`).**

**Sprint 19 status: IN PROGRESS — Day 2 (S19-01b attribution + S19-02b ablation running in parallel).**

See [`docs/operation-verstappen.md`](../operation-verstappen.md) for the full campaign roadmap.

---

## The lever

**L_accum — accumulation redesign** (DEC-014, DRAFT — locks at S19-02b close).

S19-01 ground-truth attribution identified the accumulation phase as 14.30 ms (93%) of steady-state `histogram_ms` on the 50k/RMSE/128b gate config. The accumulation loop — where each threadgroup cooperative-batch-accumulates doc gradients into `simdHist[g][bin]` — is compute-bound at this scale. Four candidate redesign variants are under ablation in S19-01b/S19-02b:

- **(A) Wider batch accumulation**: increase the 32-doc batch size to reduce loop-body overhead per bin update.
- **(B) Coalesced threadgroup staging**: restructure staging so bin writes are coalesced across SIMD groups, reducing cross-lane latency.
- **(C) Per-feature specialization**: specialize the inner loop for feature count ranges, enabling compile-time unrolling.
- **(D) Different ownership granularity**: partition bin ownership differently (e.g., by feature rather than by bin index mod 32) to improve locality.

All 4 variants are under ablation. DEC-014 locks when S19-02b closes (end of Day 2).

Source anchor: accumulation loop in `catboost/mlx/kernels/kernel_sources.h`.  
DEC-014 rationale: `.claude/state/DECISIONS.md#DEC-014`.

---

## Sprint 19 gate config

Sprint 18 used **10k/RMSE/128b**. Sprint 19 uses **50k/RMSE/128b** — the config where the accumulation bottleneck is most visible (accumulation = 93% of `histogram_ms`).

**Baseline** (S18 after / S19-01 ground-truth, steady-state iters 5–49):

| Metric | Gate config (50k/RMSE/128b) |
|--------|----------------------------|
| `histogram_ms` (mean) | 15.46 ms |
| `histogram_ms` (median) | 15.21 ms |
| `iter_total_ms` (mean) | 21.12 ms |
| `iter_total_ms` (median) | 20.96 ms |

Source: `.cache/profiling/sprint19/baseline/50000_rmse_d6_128bins.json` (S18 after; S19-01 ground-truth confirms 15.46 ms mean).

---

## Performance projection

Revised projection after pivot. Aggressive target (1.5× e2e) still live pending S19-01b/S19-02b numbers.

| Metric | S18 after (baseline) | S19 target | Projection |
|--------|---------------------:|:----------:|:----------:|
| `histogram_ms` (gate config) | 15.46 ms | **8–11 ms** | 1.4–1.9× histogram |
| `iter_total_ms` (gate config) | 21.12 ms | **14–17 ms** | 1.25–1.50× e2e |
| Championship 50k exit (total training time) | ~0.75–0.85 s | pending S19-02b | pending |

**G1 gate (revised)**: `histogram_ms` −30% min on 50k/RMSE/128b. Accumulation = 93% of `histogram_ms` → a 32% accumulation reduction yields ~30% `histogram_ms` reduction. G4 (−30% `iter_total_ms`) aggressive 1.5× target remains live pending S19-02b.

---

## Variants under ablation (S19-02b)

| Variant | Description | Primary risk | Status |
|---------|-------------|:------------|--------|
| (A) Wider batch accumulation | Increase 32-doc batch size; reduce loop overhead per bin update | Larger batch may exceed register budget or increase divergence | Under ablation |
| (B) Coalesced TG staging | Restructure bin-write access pattern for cross-SIMD coalescing | Ownership remapping may conflict with DEC-011 stride-partition invariant | Under ablation |
| (C) Per-feature specialization | Specialize inner loop for feature count ranges; compile-time unrolling | Specialization cost grows with feature diversity; JIT compile time risk | Under ablation |
| (D) Different ownership granularity | Partition bin ownership by feature rather than bin mod 32 | May break parity guarantees from DEC-009 linear fold order | Under ablation |

Full ablation numbers and DEC-014 lock in `docs/sprint19/ablation.md` (S19-02b pending, in progress).

---

## Sub-tasks

See `.claude/state/TODOS.md` Sprint 19 section for full task list with owners and dependencies.

| ID | Summary | Owner | Status |
|----|---------|-------|--------|
| S19-00 | Branch cut, baselines, state scaffold, README scaffold | @ml-product-owner / @technical-writer | **DONE** |
| S19-01 | Writeback attribution (MST, gate config) — evidence correct, premise falsified | @performance-engineer | **COMPLETE-BUT-SUPERSEDED** |
| S19-02 | Ablation sweep (writeback variants a/b/c) — DEC-013 draft; superseded by pivot | @research-scientist | **COMPLETE-BUT-SUPERSEDED** |
| S19-01b | Accumulation sub-phase attribution (14.30 ms, ±1 ms error bars) | @performance-engineer | IN PROGRESS |
| S19-02b | Accumulation redesign ablation (variants A/B/C/D) + DEC-014 lock | @research-scientist | IN PROGRESS |
| S19-03 | Implement winning accumulation redesign from DEC-014 at `kernel_sources.h` | @ml-engineer | PENDING (Day 3) |
| S19-04 | Parity sweep (DEC-008 envelope + determinism) | @qa-engineer | PENDING |
| S19-05 | 18-config stage-profiler delta | @performance-engineer | PENDING |
| S19-06 | CI gate baseline update | @mlops-engineer | PENDING |
| S19-07 | Code review | @code-reviewer | PENDING |
| S19-08 | Security pass | @security-auditor | PENDING |
| S19-09 | MST re-capture (post-fix) | @performance-engineer | PENDING |
| S19-10 | Full docs population (this README + subdocs) | @technical-writer | IN PROGRESS |
| S19-11 | In-sprint: 6 EvalAtBoundary CPU readbacks (structure_searcher.cpp) | @ml-engineer | PENDING |
| S19-12 | In-sprint: VGPR confirmation + S18 deferred code-review items | @performance-engineer / @code-reviewer | PENDING |

---

## Same-PR docs standing order

All Sprint 19 source changes land in a single PR. That PR must include:

- `docs/sprint19/{README, attribution, ablation, results, non_goals}.md`
- `CHANGELOG-DEV.md` Sprint 19 entry (full)
- `catboost/mlx/ARCHITECTURE.md` accumulation section updated
- `.claude/state/DECISIONS.md` DEC-013 (SUPERSEDED, preserved) + DEC-014 (locked from DRAFT → ACCEPTED at S19-02b close)

No doc, no merge.

---

## Sprint 18 starting point

Sprint 18 after (full table): [`docs/sprint18/results.md`](../sprint18/results.md).  
S18 gate config anchor: **N=10k, RMSE, depth=6, 128 bins → `histogram_ms` = 9.56 ms (-66.8% vs S17)**.  
Sprint 19 gate config anchor: **N=50k, RMSE, depth=6, 128 bins → `histogram_ms` = 15.46 ms (S19-01 ground-truth, accumulation = 93%)**.
