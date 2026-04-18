# Operation Verstappen — Sprint 19: Two-Phase Histogram Writeback Reduction

## What this is

Sprint 19 is the third **structural kernel rewrite** of Operation Verstappen (battle 4 of 9). Sprint 17 eliminated the serial 255-step threadgroup reduction (-89–93% `histogram_ms`). Sprint 18 eliminated per-thread device-memory spill via `simdHist` on-chip accumulation (-56–85% `histogram_ms`). Sprint 19 targets what remains: the **writeback phase**, which floors N=50k configs at ~15 ms in steady state regardless of accumulation quality.

**Sprint 19 status: IN PROGRESS.**

See [`docs/operation-verstappen.md`](../operation-verstappen.md) for the full campaign roadmap.

---

## The lever

**L_writeback — two-phase on-chip reduction before global write** (DEC-013, DRAFT).

Under L1a (`simdHist[8][1024]`, 32 KB on-chip), the current writeback path performs one `atomic_fetch_add` per bin per SIMD group to a shared global accumulation buffer. With 8 SIMD groups and 1024 bins, this is 8,192 atomic adds per histogram dispatch — all contending against the same global buffer. At N=50k (many dispatches), this contention is the plurality cost: the S18-05b profile showed writeback at ~5 ms on the gate config and converging toward ~15 ms at N=50k/128b in steady state.

**Two-phase approach** (Ramos-approved; chosen over batched-atomic for robustness):

1. **Phase 1 (on-chip fold)**: After barrier-6 (end of accumulation), fold the 8 per-SIMD histograms in `simdHist[0..7][bin]` into `simdHist[0][bin]` using the threadgroup itself. No global memory touched. Reuses the existing 32 KB buffer — no new threadgroup memory, preserves DEC-011 ceiling.
2. **Phase 2 (single global write)**: Write `simdHist[0][bin]` to the global accumulation buffer with a single atomic-free store per bin (one threadgroup, one writer per bin at the time of global write). Eliminates all cross-threadgroup atomic contention.

**Why two-phase over batched-atomic**: Two-phase eliminates atomics entirely from the writeback path. Deterministic reduction order is parity-friendly (consistent with DEC-009's linear fold). The `simdHist` staging buffer is already on-chip post-barrier-6 — no extra allocation needed. Batched-atomic reduces contention window but does not remove the atomic floor.

Source anchor: writeback logic in `catboost/mlx/kernels/kernel_sources.h` (post-barrier-6 block).  
DEC-013 rationale: `.claude/state/DECISIONS.md#DEC-013`.

---

## Sprint 19 gate config shift

Sprint 18 used **10k/RMSE/128b** as the gate config because the accumulation lever (L1a) had full force there. Sprint 19 shifts to **50k/RMSE/128b** — the config where the writeback lever has force (writeback is a larger share of `histogram_ms` at large N).

**Baseline** (S18 after, steady-state iters 5–49):

| Metric | Gate config (50k/RMSE/128b) |
|--------|----------------------------|
| `histogram_ms` (mean) | 15.52 ms |
| `histogram_ms` (median) | 15.21 ms |
| `iter_total_ms` (mean) | 21.12 ms |
| `iter_total_ms` (median) | 20.96 ms |

Source: `.cache/profiling/sprint19/baseline/50000_rmse_d6_128bins.json` (identical to S18 after).

---

## Performance projection

Aggressive projection, Ramos-approved. Subject to R8 downgrade trigger (see below).

| Metric | S18 after (baseline) | S19 target | Projection |
|--------|---------------------:|:----------:|:----------:|
| `histogram_ms` (gate config) | 15.52 ms | **7–9 ms** | 1.7–2.2× |
| `iter_total_ms` (gate config) | 21.12 ms | **12–14 ms** | 1.5–1.8× |
| Championship 50k exit (total training time) | ~0.75–0.85 s | **0.55–0.70 s** | revised aggressive |

**R8 constraint**: If S19-01 attribution shows writeback phase is <40% of steady-state `histogram_ms` on the gate config, the 1.7–2.2× projection does not have mechanical support. Projection revises DOWN before S19-03 commits (before any kernel change ships). This is a hard gate — aggressive framing is honest only if the attribution data supports it.

---

## Variants under ablation (S19-02)

| Variant | Description | Atomics in writeback | New TG memory | Primary risk |
|---------|-------------|---------------------:|:-------------:|:------------|
| **(c) Two-phase + prefix-scan** | Phase-1 intra-TG fold via linear scan; Phase-2 atomic-free global store | 0 | 0 (reuses `simdHist[0..1023]`) | Prefix-scan over 1024 bins costs ~32 barrier-synced passes; may introduce stall if not pipelined |
| (a) Two-phase reduction | Phase-1 parallel tree-reduce over 8 simd histograms; Phase-2 atomic-free global store | 0 | 0 | Tree-reduce depth = 3 passes (fold 8→1); adds 3 barriers; net still faster if each barrier < 1.7 ms |
| (b) Batched-atomic | Group 1024 bins into M batches; atomic window per batch, serialized across batches | M × 128 | 0 | Still has atomics; gain proportional to batch granularity; weaker parity story |

**CHOSEN**: variant **(c)** — two-phase + prefix-scan. Ramos approved robustness over raw throughput; (c) is the most deterministic path (prefix-scan order is fixed, parity risk minimal). Full ablation numbers and gate-clearance analysis in `docs/sprint19/ablation.md` (pending S19-02).

---

## Sub-tasks

See `.claude/state/TODOS.md` Sprint 19 section for full task list with owners and dependencies.

| ID | Summary | Owner | Status |
|----|---------|-------|--------|
| S19-00 | Branch cut, baselines, state scaffold, README scaffold | @ml-product-owner / @technical-writer | **DONE** |
| S19-01 | Writeback attribution (MST, gate config, R8 trigger) | @performance-engineer | PENDING |
| S19-02 | Ablation sweep (a/b/c × block size × bins) | @research-scientist | PENDING |
| S19-03 | Kernel implementation (chosen variant) | @ml-engineer | PENDING |
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
- `catboost/mlx/ARCHITECTURE.md` writeback section updated
- `.claude/state/DECISIONS.md` DEC-013 (locked from DRAFT → ACCEPTED at S19-02 close)

No doc, no merge.

---

## Sprint 18 starting point

Sprint 18 after (full table): [`docs/sprint18/results.md`](../sprint18/results.md).  
S18 gate config anchor: **N=10k, RMSE, depth=6, 128 bins → `histogram_ms` = 9.56 ms (-66.8% vs S17)**.  
Sprint 19 gate config anchor: **N=50k, RMSE, depth=6, 128 bins → `histogram_ms` = 15.52 ms (S18 after steady-state, the writeback floor)**.
