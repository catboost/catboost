# Sprint 18 Design — L1: SIMD-Group-Local Shared Histogram

Owner: @research-scientist (ablation, S18-02) · @ml-engineer (implementation, S18-03)  
Branch: `mlx/sprint-18-hist-privhist-tile`

> **Status: SHIPPED — L1a fixed kernel at commit `19fa5ce6cc`. All gates PASS.**  
> This document captures the structural framing, correctness invariants, and parity rationale. The BUG-S18-001 section below documents the initial implementation failure and its fix — a teaching artifact about reduction layout portability.

---

## The lever (DEC-010)

The D1c kernel (Sprint 17) eliminated the 255-step serial reduction tail. The accumulation phase — `kernel_sources.h:131–148` — is now the plurality cost: **6.4 ms ± 1.5 ms (27%)** of the 28.75 ms post-S17 `histogram_ms` at the gate config (N=10k, RMSE, d6, 128 bins). `privHist` zero-init (lines 125–128) adds a further 4.0 ms ± 1.5 ms. Ground-truth by S18-01 attribution; see `docs/sprint18/attribution.md`.

Root cause: `float privHist[HIST_PER_SIMD]` at `kernel_sources.h:123` allocates 1024 floats × 4 B = **4 KB per thread**. At 256 threads per threadgroup that is 1 MB of thread-local state — well beyond the Apple M-series register file (~256 registers per thread). Every `privHist[f * 256 + bin] += stat` on line 145 is a device-memory read-modify-write; the zero-init loop at lines 125–128 performs 1024 spilled stores per thread before any real work starts.

**L1 replaces this with a SIMD-group-local histogram in threadgroup memory.** One histogram per SIMD group, not one per thread. At 8 SIMD groups per threadgroup (BLOCK_SIZE=256, SIMD_SIZE=32) that is 8 copies instead of 256 — a 32× reduction in histogram instances. Threadgroup memory stays on-chip; device-memory spill is eliminated.

**Stride partitioning** gives each bin a unique owner within its SIMD group. Lane `l` within SIMD group `g` owns bins `{l, l+32, l+64, …, l+992}` (32 bins per lane × 32 lanes = 1024 bins). No two lanes write the same bin within a group → zero atomics in the accumulation phase.

---

## Structural correctness guard (BUG-001)

BUG-001 (Sprint 3–7 non-determinism incident) arose from CAS-based atomics into shared histogram memory where SIMD-group scheduling was not lockstep. **L1a/b/c avoid this by construction:**

- Stride-partitioning gives each bin a **single owner** per SIMD group. There are no atomics in the accumulation phase.
- `simd_shuffle_xor` is register-only and lockstep within a SIMD group, guaranteed by MSL §6.9.
- The cross-SIMD fold is the fixed-order 8-term linear sum per DEC-009 — deterministic by construction.
- Every write has a unique owner; every cross-SIMD read occurs after a `threadgroup_barrier`.

This is a stronger correctness guarantee than the Sprint 3 design. The 100-run determinism fixture (S18-04, @qa-engineer) validates empirically what the stride-partition design guarantees structurally.

---

## Variant comparison

The three candidates span the threadgroup-memory / accumulation-passes trade-off space. L1d is the control.

| variant | per-SIMD hist width | tile count | peak threadgroup_mem (KB) | accumulation passes over docs | primary risk |
|---------|--------------------:|-----------:|-------------------------:|:----------------------------:|--------------|
| **L1a** (full 32 KB) | 1024 bins × 8 SIMD | 1 | 32 (at Apple Silicon limit) | 1 | Threadgroup-memory ceiling; no headroom for Sprint 19+ geometry changes |
| **L1b** (tiled) | 256 bins × 8 SIMD × 4 tiles | 4 | 12 | 4 | 4× doc-loop memory bandwidth cost; may be DRAM-bound at N=50k |
| **L1c** (hybrid stride) | 256 bins × 8 SIMD, 8 bins/lane | 4 | 12 | 4 | Same bandwidth risk as L1b; stride pattern is more complex |
| **L1d** (control) | — | — | 12 | 1 | Reference; should be identical to Sprint 17 after |

### L1a — full 32 KB per-SIMD histogram

`simdHist[8][1024]` (8 SIMD groups × 1024 bins × 4 B = 32 KB) in threadgroup memory, at Apple Silicon's hard ceiling. The D1c reduction phase re-uses the same buffer in place: accumulation writes into `simdHist[g][0..1023]`; a `threadgroup_barrier` separates phases; D1c folds 8 per-SIMD histograms into the final per-bin sums, writing to a 4 KB staging region at the end of the 32 KB block. Single accumulation pass over all docs. Parity story is the cleanest (no tile-boundary re-reads).

### L1b — tiled, 256-bin × 4 passes

`simdHist[8][256]` = 8 KB plus the existing `stagingHist[1024]` = 4 KB, unchanged from D1c (12 KB peak). Each of 4 tiles processes 256 bins: the inner loop runs once per tile, writing only bins in `[tile_base, tile_base+256)`. D1c reduction runs per tile. Trade-off: each document is re-visited 4× (once per tile), which multiplies DRAM bandwidth by 4 for the stats/compressedIndex gather loads.

### L1c — hybrid stride-partition, 256-float SIMD slab

Same 12 KB threadgroup memory as L1b. Each SIMD group owns a 256-float slab. Within a tile, thread `(g, l)` owns bins `{l, l+32, l+64, l+96, l+128, l+160, l+192, l+224}` — 8 bins per lane within 256-bin tiles. Accumulation runs 4 tiles; each tile writes into `simdHist[g][0..255]`. D1c reduction per tile (unchanged). Like L1b, re-reads docs 4× from global memory.

Full analytical projections (histogram_ms per variant, threadgroup-memory footprint, parity ulp, gate-clearance table) are in `docs/sprint18/ablation.md`.

---

## Numerical parity rationale

L1 changes the accumulation order. Each lane now sums its owned docs' contributions per its owned bins directly into `simdHist[g][bin]`; the D1c fold is then the existing intra-SIMD butterfly plus cross-SIMD linear sum. The total number of additions per bin is identical to Sprint 17 D1c.

Higham γ_N analysis (same methodology as DEC-008; derivation in `docs/sprint17/ablation.md` §3):

- Sprint 17 D1c reduction depth: 5 levels (intra-SIMD butterfly) + 7 levels (8-term cross-SIMD linear) = 12 effective levels → **γ_12 ≈ 7.2e-7**.
- Sprint 18 L1 reduction depth: each bin accumulates ≤ 32 docs per SIMD group × 8 SIMD groups = 256 contributions (same cardinality). Intra-SIMD butterfly now runs over per-bin 32-lane partials — same 5 levels. Cross-SIMD fold is the same 8-term linear. **Effective reduction depth is identical to D1c.**
- Expected parity: within DEC-008 bounds (RMSE ulp ≤ 4, Logloss ulp ≤ 4, MultiClass ulp ≤ 8) — no envelope adjustment needed.

The Sprint 17 0-ulp result (35/36 checkpoints bit-exact) is **not** a baseline to beat — it was lucky-within-contract per `parity_results.md`. The hard merge gate is DEC-008. The one transient 17-ulp MultiClass excursion at iter=10 (healed by iter=20) is not expected to compound under L1; reduction depth is analytically unchanged.

---

## Chosen variant: L1a

**Ramos approved L1a on Day 2 (2026-04-17). Shipped at commit `19fa5ce6cc` after BUG-S18-001 fix.**

### Rationale

S18-02 ablation (`docs/sprint18/ablation.md` §5) identifies L1a as the only variant with error-envelope gate clearance:

1. Worst-case projection (17.3 ms) clears the 18.7 ms gate with 1.4 ms margin. L1b and L1c both miss on their upper error bounds.
2. Simplest structural change (~+25 LOC vs D1c); lowest S18-07 review surface.
3. Only variant with a structural occupancy delta (1 tg/SM) — unlocks Option C (≥45% / ≤15.8 ms) if writeback-atomic contention is SM-local. S18-09 MST resolves.
4. Matches the plan's §L1a tiebreaker rationale (`/Users/ramos/.claude/plans/sprint18-hist-privhist-tile.md` §Design).
5. After BUG-S18-001 fix: intra-SIMD butterfly removed entirely, reducing effective reduction depth from γ_12 (S17) to γ_7 (S18). DEC-008 parity envelope preserved and tightened.

### Trade-off accepted

32 KB threadgroup memory at the Apple M-series hard ceiling. DEC-012 codifies this choice. Sprint 19+ threadgroup-geometry work re-negotiates if the ceiling creates scheduling pressure.

### Final kernel structure (shipped — commit `19fa5ce6cc`)

| # | `kernel_sources.h` lines | Change from D1c |
|---|---|---|
| 1 | `:123` | `float privHist[HIST_PER_SIMD]` → `threadgroup float simdHist[8][1024]` (32 KB, at Apple Silicon limit) |
| 2 | `:125–128` | Zero-init loop eliminated; threadgroup memory is implicitly initialized |
| 3 | `:131–148` | Per-thread stride loop replaced with cooperative 32-doc batch loop: `simd_shuffle` broadcast of (packed, stat) from lane `src`; only the bin-owner lane (`bin & 31 == lane_index_in_simdgroup`) writes |
| 4 | D1c intra-SIMD butterfly | **Removed entirely** — `simdHist[g][bin]` is already a full per-SIMD-group sum; no intra-SIMD reduction needed (DEC-012) |
| 5 | `:181–225` | D1c cross-SIMD 8-term linear fold (DEC-009) unchanged; reads from `simdHist` in place |
| 6 | `:229–245` | Global-atomic writeback unchanged |

**Barriers per dispatch: 6** (1 accumulation-to-reduction `threadgroup_barrier` + 5 cross-SIMD fold steps). Down from S17's 9 (5 intra-SIMD butterfly rounds + 4 cross-SIMD fold barriers had a different accounting). The intra-SIMD rounds in D1c did not require `threadgroup_barrier` (register-only), so the net reduction is the 3 explicit barriers removed by eliminating the butterfly.

---

## BUG-S18-001 — Initial implementation failure

The initial L1a kernel (commit `abc4c229f9`) failed all 18 parity configs by 6 orders of magnitude. Full post-mortem: `docs/sprint18/bug_s18_001.md`.

**Root cause in one sentence**: the D1c intra-SIMD butterfly was ported to the L1a layout without re-deriving its algebraic role, producing two compounding structural flaws: a 1/32 doc-inclusion rate (stride ownership mismatch) and a 32× amplification (butterfly over shared slots that all hold the same value).

**Why naively porting D1c's butterfly is wrong under L1a:**

```
D1c invariant (butterfly correct):
  each lane holds a DISTINCT privHist[bin] partial
  → butterfly reduces 32 distinct values → correct sum

L1a invariant (butterfly wrong):
  all 32 lanes in SIMD group g read simdHist[g][bin]
  → all 32 lanes read the SAME value
  → butterfly accumulates: value + value + ... (32 times) = value × 32  ✗

L1a fix: stride-partition gives simdHist[g][bin] a single writer
  → simdHist[g][bin] is already the full per-SIMD-group sum after accumulation
  → butterfly has no work to do and must be removed  ✓
```

The two flaws partially cancelled (1/32 × 32 ≈ 1×), so quick sanity checks on loss magnitudes appeared plausible. Only a rigorous ULP comparison revealed the true error.

**Lesson**: never port a reduction pattern across a layout change without re-deriving the algebraic correctness from the new invariant. Determinism-PASS alone is not evidence of correctness.

---

## Key source anchors

| # | File:line | Description |
|---|-----------|-------------|
| 1 | `catboost/mlx/kernels/kernel_sources.h:123` | `threadgroup float simdHist[8][1024]` — 32 KB. Replaces the old `float privHist[HIST_PER_SIMD]`. |
| 2 | `catboost/mlx/kernels/kernel_sources.h:125–128` | Zero-init loop eliminated (lines now removed). |
| 3 | `catboost/mlx/kernels/kernel_sources.h:131–148` | Cooperative 32-doc batch loop with `simd_shuffle` broadcast and stride-partition ownership check. |
| 4 | `catboost/mlx/kernels/kernel_sources.h:181–225` | Cross-SIMD 8-term linear fold (DEC-009, unchanged). Intra-SIMD butterfly removed. |
| 5 | `catboost/mlx/kernels/kernel_sources.h:229–245` | Global-atomic writeback. Unchanged. |
