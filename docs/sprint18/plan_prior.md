# Sprint 18 — Plan Prior (draft, from S17-02 ablation findings)

Owner: @research-scientist · Drafted: 2026-04-17 · Branch: `mlx/sprint-17-hist-tree-reduce`

This is a **prior** — a prioritised list of Sprint 18 levers derived from what the Sprint 17 D1c ablation implies about the remaining bottleneck structure. It is **not** a committed plan; the committed Sprint 18 plan is written after Sprint 17 merges and @performance-engineer ground-truths S17-01 against the 30% gate.

## What Sprint 17 is expected to leave on the floor

After D1c lands (projected `histogram_ms` ≈ 185 ms on 10k/RMSE/d6/128b, down from 308 ms), the reduction phase collapses from ~180 ms to ~10 ms. The remaining budget is dominated by:

| Phase | Projected post-S17 ms | Share of post-S17 kernel | Sprint 18 lever |
|-------|----------------------:|-------------------------:|-----------------|
| Private-histogram accumulation (`privHist[1024]` read-modify-write, spill to device memory) | ~130 | ~70% | **L1** |
| Global-atomic writeback | ~45 | ~24% | L2 |
| Reduction tail (D1c) | ~10 | ~5% | — (already optimal) |

Baseline diagnosis source: `docs/sprint16/mst_findings.md` §B.2 — each thread owns a 1024-float stack array = 4 KB; at 256 threads per threadgroup this is 1 MB of per-threadgroup private state, which exceeds the Apple Silicon register file and spills to device memory. Every `privHist[f * BINS_PER_BYTE + bin] += stat` becomes a device-memory round-trip.

---

## Top 3 ranked levers for Sprint 18

### L1 (HEADLINE) — SIMD-group-local shared histogram (D2 in MST findings)

**Problem.** Each thread owns 1024 bins of private histogram. These spill to device memory (`mst_findings.md` §B.2 shows 4 KB per thread × 256 threads = 1 MB per threadgroup, far exceeding the ~256-register file per thread). Every accumulation is a device-memory RMW.

**Proposal.** Collapse from 256 per-thread histograms to 8 per-SIMD-group histograms in threadgroup memory. Each SIMD group owns a 1024-float slice in threadgroup memory (8 × 1024 × 4 = 32 KB — at the Apple Silicon limit). Within a SIMD group, 32 threads accumulate into a shared 1024-slot buffer using SIMD-group-local atomics or (preferred) a deterministic stride-partitioned write pattern.

**Why this works post-D1c.** D1c uses 12 KB of threadgroup memory; L1 needs 32 KB. The plan-of-record lever on this is `hist.metal`'s original design (SIMD-slice reduction) which BUG-001 rejected. The modern rewrite is:
- Partition the 1024 bins across 32 lanes: lane `l` owns bins `l, l+32, l+64, ..., l+992` (32 bins per lane × 32 lanes = 1024 bins). Each bin is written by exactly one lane per SIMD group → **zero contention, no atomics**.
- Then D1c-style intra-SIMD shuffle reduction collapses 32 lane values per bin (already same lane → already sum).
- Then cross-SIMD reduction collapses 8 SIMD-group totals per bin.

**Expected gain.** Sprint 16 MST estimated 20–40% additional reduction on top of D1's reduction-only win (`mst_findings.md` Table D). Projected `histogram_ms`: **185 ms → 110–140 ms**, a **25–40% additional cut**. Combined with Sprint 17: `histogram_ms` drops from 308 ms baseline to 110–140 ms, a ~60% cumulative win.

**Risks.**
1. **BUG-001 regression.** Stride-partitioning eliminates contention by construction — no atomics, no CAS. This is a stronger guarantee than the SIMD-local atomic variant that broke BUG-001. Add a 100-run determinism fixture to S18 parity suite.
2. **Threadgroup memory at 32 KB is the hard Apple Silicon limit.** No headroom for other shared-memory features (e.g., staging for cross-SIMD fold). Mitigation: tile the SIMD-local histogram too (8 × 256 = 8 KB × 4 tiles, reusing the D1c pattern).
3. **Bin-partitioning reshuffle.** The per-lane bin assignment changes memory access patterns in the existing accumulator loop (line 142–148). Needs kernel refactor, not just a reduction-block swap.

**Budget estimate.** 5–7 sprint-days. Headline structural rewrite; needs own ablation (L1a: per-SIMD full-1024 layout; L1b: per-SIMD tiled 256 layout; L1c: per-SIMD with Hillis-Steele instead of stride-partition).

### L2 — Pre-permute stats and compressedIndex to remove `docIndices` gather (D3 in MST findings)

**Problem.** Line 133 `compressedIndex[docIdx * lineSize + featureColumnIdx]` and line 139 `stats[statIdx * totalNumDocs + docIdx]` both use `docIdx = docIndices[sortedPos]` — a gather-indexed load. At depth ≥ 1 the permutation is non-trivial, so these loads are scattered across device memory. Depth 5 has 32 partitions; `docIndices` is fully scrambled.

**Proposal.** Pre-dispatch kernel that writes `stats_permuted[sortedPos] = stats[docIndices[sortedPos]]` and similarly for `compressedIndex`. Histogram kernel then reads `stats_permuted[partOffset + myDocStart + d]` coalesced, eliminating the gather.

**Expected gain.** MST estimated 10–20% at depth ≥ 3, growing with depth. Sprint 16 per-depth data: depth 5 is 114 ms (37% of total hist_ms at 10k RMSE); a 15% gather-removal saving yields ~17 ms cut.

**Risks.**
1. **Extra memory cost**: `stats_permuted` = O(approxDim × numDocs) = 40 KB at 10k/approxDim=1; `compressedIndex_permuted` = O(numDocs × lineSize) = 520 KB at 10k/13cols. Trivial on M3 (64 GB unified memory); watch if 50k dataset scales linearly (~2.6 MB — still trivial).
2. **Extra dispatch latency**: one scatter kernel per depth before histogram kernel. 6 depths × 0.5 ms overhead = 3 ms amortised across 50 iterations = negligible.
3. **Correctness at depth 0**: `docIndices` is identity at depth 0, so the permutation is a no-op and the scatter is waste. Add an early-skip at depth 0.

**Budget estimate.** 2–3 days. Self-contained change; parity is trivially preserved (same data, reordered).

### L3 — Multiclass per-dim dispatch fusion (D5 in MST findings)

**Problem.** For multiclass (approxDim=3), `csv_train.cpp:3185–3204` serially invokes `DispatchHistogram()` three times per depth — once per approx dimension. Each dispatch incurs kernel-encoding overhead and serialises on its own `mx::eval`. Sprint 16 baseline shows multiclass is 2× binary at same N (596 ms vs 314 ms at 10k), consistent with the 3× dispatch fanout.

**Proposal.** Add an `approxDim` axis to the kernel's Z grid dimension. One dispatch handles all three dimensions. Requires:
- Extending `stats` access to `stats[k * totalNumDocs + docIdx]` for k ∈ [0, approxDim) — already threaded through the kernel at line 139 (via the `statIdx` z-axis).
- Changing `csv_train.cpp:3185–3204` to submit one fused dispatch instead of three.

**Expected gain.** 30–50% on **multiclass only**. Binary is unaffected. At 10k multiclass/d6/128b this means ~200 ms cut → `histogram_ms` drops from ~400 ms (post-L1 estimate) to ~230 ms.

**Risks.**
1. **Wider blast radius**: needs careful review of `DispatchHistogram()` (`csv_train.cpp:869–955`) and the downstream Phase 2 `mx::eval(toEval)` loop. Touching the outer loop is risky.
2. **Memory bandwidth**: single dispatch now loads `stats` 3× more per threadgroup. If the existing binary-regression kernel is already stats-bandwidth-bound post-L1, the multiclass fusion may not help as much as projected.
3. **Binary regression no-op**: for approxDim=1 the fusion is a rename; no functional change. Good — binary path stays untouched.

**Budget estimate.** 3–4 days. Medium-complexity refactor; parity is straightforward (same arithmetic, different grid geometry).

---

## Ranking rationale

| Rank | Lever | Scope | Gain | Risk | When to start |
|-----:|-------|-------|-----:|:----:|---------------|
| 1 | **L1** (SIMD-group-local histogram) | All configs | 25–40% post-S17 | Medium | Sprint 18 headline |
| 2 | L2 (pre-permute stats/compressedIndex) | Depth ≥ 3 | 10–20% post-L1 | Low | Sprint 18 drive-by OR Sprint 19 |
| 3 | L3 (multiclass fusion) | Multiclass only | 30–50% multiclass-only post-L1 | Medium-High | Sprint 19 |

**L1 is the headline.** It attacks the post-S17 dominant cost (privHist spill) with a pattern that also aligns with the D1c reduction design — they compose cleanly.

**L2 is a good drive-by** for Sprint 18 if L1 lands with time remaining. Pure layout change, low code risk, independent parity characteristics.

**L3 is multiclass-specific** and has a wider blast radius. Defer to Sprint 19 unless multiclass targets become time-critical for the Operation Verstappen scoreboard (szilard/GBM-perf Airline 10M is binary; A100 parity target is binary).

---

## Likely Sprint 18 gate (prior estimate)

**≥25% additional reduction in `histogram_ms`** at the S17 gate config (N=10k, RMSE, depth=6, 128 bins), relative to the post-S17 measured `histogram_ms`. Cumulative target vs Sprint 16 baseline: **≥50% end-to-end**.

This is tighter than S17's 30% because the post-S17 floor is already low (~185 ms); further cuts hit diminishing returns as accumulation costs become the asymptote.

Parity target unchanged: RMSE ulp ≤ 4, Logloss ulp ≤ 4, MultiClass ulp ≤ 8. L1's determinism argument (stride partitioning, zero atomics) keeps parity margins comfortable.

---

## Dependencies to carry from Sprint 17

1. **DEC-005 (RMSE ulp ≤ 4 relaxation)** — needs to be firm; L1 reduction error bounds assume it.
2. **DEC-006 (D1c chosen)** — threadgroup memory budget that L1 inherits: 12 KB used by D1c, 20 KB headroom for L1's per-SIMD buffers.
3. **`maxBlocksPerPart`-aware launch geometry** — L1 and L2 both respect the current `maxBlocksPerPart` clamp (1..8). Sprint 19 may revisit.

---

## What this prior is NOT

- It is **not** a committed Sprint 18 plan. Ramos approves the committed plan after Sprint 17 merges and @performance-engineer confirms the actual post-S17 cost distribution.
- It is **not** a re-scoping of Sprint 17. The D1c verdict is locked per DEC-006; any surprises go to Sprint 18, not back into 17.
- It is **not** an exhaustive lever list. D4 (redundant `mx::eval` removal) and D6 (fold range-check) from `mst_findings.md` are tiny drive-bys — include in Sprint 18 PR opportunistically.
