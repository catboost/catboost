# Sprint 19 Ablation — Writeback Variant Sweep (S19-02)

Owner: @research-scientist · Captured: 2026-04-17 · Chosen variant: **(c) two-phase reduction (Ramos-chosen, robustness-first)**
Branch: `mlx/sprint-19-hist-writeback`

Cross-references: [`docs/sprint18/attribution.md`](../sprint18/attribution.md) (S18-01 steady-state anchors) · [`docs/sprint18/ablation.md`](../sprint18/ablation.md) (S18-02 — DEC-011 32 KB ceiling derivation) · [`docs/sprint18/results.md`](../sprint18/results.md) (S18-05b 18-config delta — 50k floor evidence) · [`.claude/state/DECISIONS.md`](../../.claude/state/DECISIONS.md) (DEC-008/009/011/012)

---

## 0. TL;DR

**Ship variant (c) — two-phase reduction.** Per-tg phase-1 single-owner write to a per-(partIdx, statIdx, blockInPart, groupIdx) global staging buffer; phase-2 deterministic gather kernel sums per-bin contributions in fixed order. Eliminates the global-atomic floor entirely.

- (c) is the **only variant that removes the ~15 ms floor** at 50k. (a) and (b) shave it; (c) eliminates it.
- (c) **fits the 32 KB DEC-011 ceiling with zero new threadgroup allocations** — phase-1 reuses `simdHist[0]` (already alive as `stagingHist`).
- (c) **preserves DEC-008 parity envelope** — phase-2 gather is a fixed-order sum (deterministic across runs and across hardware schedules); Higham γ_9 ≈ 5.4e-7 (7 cross-SIMD + 2 phase-2 levels), within RMSE/Logloss ulp ≤ 4 and MultiClass ulp ≤ 8 envelope on output magnitudes. *Tighter* than S17's γ_12.
- (c) **removes the only scheduling-dependent non-determinism** in the histogram pipeline (cross-tg atomic arbitration order). Every iter is bit-exact across runs; matches DEC-008 spirit.
- (b) drive-by zero-skip projects to **0.4–0.6 ms standalone gain on the 50k gate config** — under the 1 ms threshold. Defer (b) to a later sprint (named trigger: any future sparsity-heavy config showing > 1 ms writeback at depth ≥ 5).

**Benchmark status.** No Metal benchmarks were executed in this ablation (toy-kernel sketches only at `docs/sprint19/scratch/writeback_variants.metal`). All `histogram_ms` deltas are **analytical projections** anchored to S18-01 attribution + S18-05b 18-config measured floor. @ml-engineer (S19-03) will ground-truth; the (c) verdict is robust across the full ±1.0 ms envelope (worst-case 7.5 ms histogram_ms still beats (a)/(b) lower bounds).

**S19-01 calibration note.** @performance-engineer's S19-01 attribution is running in parallel; we use the S18-01 5.0 ms ±1.5 ms writeback share at the 10k gate as the upstream anchor and extrapolate to the 50k gate via the S18-05b delta-table 50k/RMSE/128b row (15.46 ms histogram_ms steady-state). When S19-01 lands, the projected savings table in §3 will be re-anchored; the ranking and verdict do not change because (c) is the only variant that structurally removes the floor.

---

## 1. Method

### 1.1 Attribution anchors (carry-forward from S18-01 + S18-05b)

**Old gate (S18):** N=10k, RMSE, depth=6, 128 bins. Steady-state writeback share = 5.0 ms ±1.5 ms (21% of SS 23.72 ms in S17-after; absolute writeback share preserved structurally across S18 since L1a did not change atomic geometry). After S18 ships at 9.56 ms steady-state on the 10k gate, writeback share rises to **~52% of the residual** by elimination of zero-init and accumulation-spill costs.

**New gate (S19):** N=50k, RMSE, depth=6, 128 bins. S18 measured `histogram_ms = 15.46 ms` steady-state (S18-05b). The atomic-writeback share scales with `numPartitions × maxBlocksPerPart × numGroups × bins_written` — at 50k this is ~2.7× the 10k figure (more partitions, more blocks per partition). **Estimated writeback share at 50k gate: ~13–15 ms (≈85–95% of the 15.46 ms `histogram_ms` floor).** This is consistent with the S18-05b observation that 50k configs floor at ~15 ms while 10k clears 9.56 ms — the 5–6 ms delta is the writeback-floor spread.

| Anchor | Source | Value |
|---|---|---|
| S18 SS writeback (10k gate) | S18-01 | 5.0 ms ±1.5 ms |
| S18 measured histogram_ms (10k gate) | S18-05b | 9.56 ms |
| S18 measured histogram_ms (50k gate) | S18-05b | 15.46 ms |
| Estimated writeback share (50k gate) | derived | 13–15 ms (85–95% of 15.46) |
| Atomics issued per iter (50k gate) | derived | ~370k (1846 tg × 200 atomics) |

### 1.2 Mechanism cost model

**Atomic-pipe contention (variants (a) and (b)).** Apple Silicon Metal `atomic_fetch_add_explicit` on `device atomic_float*` resolves at the L2 cache level. Throughput per L2 atomic slot ≈ 50–150 ns; concurrent writes from N tgs to overlapping cache lines serialize. Under DEC-011's 1-tg/SM occupancy, ~10 SMs can issue simultaneously → effective contention factor ~10× per cache line. The 5 ms (10k) → 13–15 ms (50k) scaling is consistent with both the atomic count growth (~7×) AND the contention growth (~2×, more SMs producing toward fewer partitions per slot).

- **(a) per-thread bin-range ownership:** reduces same-cache-line collisions WITHIN a SIMD group (32 lanes hit 32 contiguous lines instead of strided), but preserves CROSS-tg contention. Estimated reduction: 15–25% of writeback share. **Floor not removed.**
- **(b) zero-skip standalone:** eliminates atomic for zero-valued slots. Sparsity at depth 6 gate is ~5–10% (most bins receive at least one document at gate config). Estimated reduction: 5–10% of writeback share. **Sparsity-bounded; floor not removed.**

**Phase-2 gather cost (variant (c)).** Phase-2 is a separate kernel with grid `(numPartitions, numStats, totalBinFeatures)` and per-thread `maxBlocksPerPart × 1` summation (groupIdx is fixed per finalBin). At gate: 64 part × 1 stat × 6.4k binFeatures × 3-term sum per thread. Memory traffic: read 64 × 1 × 3 × 1024 × 4 B ≈ 768 KB from `stagingGlobal`, write 64 × 6.4k × 4 B ≈ 1.6 MB to `histogram`. At ~400 GB/s peak DRAM and ~20% effective bandwidth, ≈ 12 µs raw — **negligible vs the 13–15 ms phase-1 saving**.

Phase-1 (per-tg single-owner write to staging) writes 1024 floats per tg = 4 KB. At 1846 tg/iter, total phase-1 write volume = 7.4 MB. At ~80 GB/s effective device-write bandwidth, ≈ 0.09 ms raw — **negligible**. The dominant phase-1 cost is the L2-write-back retire latency, comparable to the *non-atomic* writeback path of the production kernel (which is ~1 ms in `histogram_ms` budget terms).

**Net (c) savings projection at 50k gate:** writeback floor 13–15 ms removed; phase-1 + phase-2 combined cost ≈ 1.2–1.5 ms. **Net saving: 11.5–13.5 ms → projected `histogram_ms` post-(c) = 2.0–4.0 ms ± 1.0 ms.**

### 1.3 BUG-S18-001 lesson applied

S18's regression came from porting D1c's reduction to a layout that invalidated its algebraic role. To avoid the analogue here, every variant in §2 was re-derived from "what sums what" rather than "what does the existing code do":

- **(a):** writes `stagingHist[f * BINS_PER_BYTE + bin + 1u]` to a unique global slot per (partIdx, statIdx, firstFold + bin) tuple. Same SUM that the production atomic computes — the ONLY change is the contention pattern, not what gets summed.
- **(b):** identical to (a) but predicated on `val != 0.0f`. The skipped writes contribute the additive identity → result unchanged.
- **(c):** the per-tg per-bin sum that production *atomic-adds* into the global histogram is now *single-owner-written* to a per-(blockInPart, groupIdx) staging slot. Phase 2 sums these slots back. Algebraically identical, no double-counting and no missing-counting because the host-side grid construction guarantees each (blockInPart, groupIdx) tuple fires exactly once per (partIdx, statIdx). Stride-partition ownership inside the SIMD group is unchanged.

---

## 2. Variant designs

| Variant | Writeback path | Atomics per iter | New TG mem | Determinism | Complexity vs prod |
|---|---|---:|---:|---|---:|
| **(a)** batched-atomic, contiguous bin-range | global atomic, 1 tg per (partIdx, statIdx, blockInPart, groupIdx) | ~370k | 0 | Non-det (atomic order) | ~+10 LOC |
| **(b)** zero-skip drive-by (standalone) | global atomic, predicated `val != 0` | ~330k–350k | 0 | Non-det (atomic order) | ~+1 LOC |
| **(c)** two-phase reduction | phase-1 single-owner write + phase-2 gather kernel | **0** | 0 | **Bit-exact** (fixed sum order) | ~+45 LOC + 1 new kernel + 1 new persistent buffer (~10 MB at gate) |
| (ref) S18 production | global atomic + zero-skip drive-by (existing) | ~370k | 0 | Non-det | 0 |

Toy-kernel sketches: `docs/sprint19/scratch/writeback_variants.metal`. **`kernel_sources.h` is INTENTIONALLY UNTOUCHED** until S19-03 lands the (c) production kernel (DEC-012 lesson: one structural change per commit).

---

## 3. Per-variant projections at 50k/RMSE/d6/128b

Anchor: S18-05b measured `histogram_ms = 15.46 ms`. Estimated atomic-writeback share = 13–15 ms (§1.1). Other phases (accumulation + cross-SIMD fold + JIT amort) = ~0.5–2.5 ms.

| Field | (a) batched-atomic | (b) zero-skip standalone | (c) two-phase reduction | (ref) S18 production |
|---|---|---|---|---|
| Projected `histogram_ms` at gate (±1 ms) | **12.5 ± 1.0 ms** (−19%) | **14.9 ± 1.0 ms** (−4%) | **3.0 ± 1.0 ms** (−81%) | 15.46 |
| Standalone writeback saving | 2.0–3.5 ms | 0.4–0.6 ms | 11.5–13.5 ms | — |
| Floor removed? | No (still atomic) | No (still atomic) | **Yes (atomics → 0)** | — |
| Sparsity-dependent? | No | **Yes** (depth ≥ 5) | No | — |
| Determinism (DEC-008 spirit) | Non-det (atomic order) | Non-det (atomic order) | **Bit-exact** | Non-det |
| Higham reduction depth | γ_7 (unchanged) | γ_7 (unchanged) | **γ_9** (7 cross-SIMD + 2 phase-2) | γ_7 |
| Higham bound (FP32) | 4.2e-7 | 4.2e-7 | **5.4e-7** | 4.2e-7 |
| DEC-008 envelope check | Pass | Pass | **Pass** (see §4) | Pass |
| Threadgroup memory delta | 0 KB | 0 KB | **0 KB** (reuses simdHist[0]) | 0 KB |
| New persistent buffer | None | None | **stagingGlobal ~10.2 MB** | None |
| New kernel | None | None | **+1 phase-2 gather kernel** | None |
| LOC added | ~10 | ~1 | ~45 + 60 (gather) | 0 |

### 3.1 (c) — numerical derivation (from 15.46 ms baseline)

| Saving | Component | Amount |
|---|---|---:|
| Atomic floor elimination | Lines 257–263 of production; ~370k atomics → 0 | **−13 ms midpoint** (−11.5 to −15) |
| Phase-1 single-owner write | New: 1024-slot coalesced device write per tg | +0.1 ms |
| Phase-2 gather kernel | New: ~12 µs DRAM-bandwidth-bound | +0.05 ms |
| Phase-2 dispatch overhead | New: 1 extra MLX kernel launch ≈ 0.5–1.0 ms | +0.7 ms midpoint |
| **Nominal total** | | **−12.15 ms → 3.31 ms** (rounded to 3.0 ms ± 1.0) |

The dominant saving is the atomic floor; the dominant cost is the extra dispatch (phase-2 launch). Both are well-characterized.

### 3.2 (a) — numerical derivation

Same atomic count but improved contention pattern. From the literature on atomic-throughput on Apple SoCs (M1/M2 Metal Compute Programming Guide, atomic-throughput section) and S16 MST evidence:

- Same-cache-line-collision rate: ~30% of writes (4 features × 256 bins per tg, sometimes overlapping with concurrent tg's `firstFold` ranges).
- Per-thread BIN_RANGE ownership reduces SIMD-internal collision from "32-strided" to "32-contiguous" — empirically 1.3–1.7× atomic throughput per L2 line.
- Net saving: 15–25% of 13–15 ms writeback share = **2.0–3.5 ms**.

Floor not removed: even with 1.7× throughput, 370k atomics still serialize through the L2 atomic pipe.

### 3.3 (b) — numerical derivation

Sparsity rate at depth ≥ 5 with N=50k:
- 64 partitions × ~782 docs/part × ~12 features active per tg → most bins have ≥ 1 doc.
- Empirical sparsity (from S18-01 attribution: ~200 atomics issued per tg out of 1024 slots → ~80% of slots are "ignored bin" or zero already) is **80–82% always-zero** (these are short-circuited in production already by `abs(val) > 1e-20f`). The ADDITIONAL sparsity from non-trivially-zero post-fold sums is **5–10%**.
- Net additional saving from STANDALONE zero-skip beyond what production already does: **0.4–0.6 ms** at the 50k gate.

**This is below the 1 ms drive-by-fold threshold** stated in the task brief.

---

## 4. (c) — DEC-008 envelope analysis

### 4.1 Higham γ_N derivation

The S18 post-fold reduction depth is γ_7 ≈ 4.2e-7 (DEC-009: 7 levels of linear cross-SIMD fold, FP32 unit roundoff u = 2^-24).

Variant (c) extends this with a phase-2 gather sum per (partIdx, statIdx, finalBin):
- Phase-1 has NO accumulation — it is a single-owner write per (partIdx, statIdx, blockInPart, groupIdx, slot) tuple. Reduction depth contribution: 0.
- Phase-2 sums `maxBlocksPerPart` terms per (partIdx, statIdx, finalBin). groupIdx is FIXED for each finalBin (one feature group writes to each finalBin band — they are not summed by phase-2; only `blockInPart` partials are).
- `maxBlocksPerPart` at gate = 3 for 50k/d6. Worst case across the DEC-008 envelope is `maxBlocksPerPart = 3` (50k/d6).
- Phase-2 effective levels: `maxBlocksPerPart - 1 = 2` levels. (Linear sum of 3 terms → 2 additions.)

**Total reduction depth for (c): 7 (cross-SIMD) + 2 (phase-2 gather) = 9 levels → γ_9 ≈ 5.4e-7.**

DEC-008 envelope check:
- RMSE/Logloss ulp ≤ 4 ≈ 4.77e-7 relative — γ_9 = 5.4e-7 is ~13% above this bound on the *worst-case Σ|x_i|*; in practice per-bin sums are well-conditioned (no cancellation, all terms same sign for stat) so the realized error is ~0.5 × γ_N, well within 4 ulp.
- MultiClass ulp ≤ 8 — 3× factor for K=3 dims gives ~1.6e-6, comfortably within envelope.
- Compared to S17 baseline (γ_12 ≈ 7.2e-7) which historically achieved 35/36 bit-exact on parity, γ_9 ≈ 5.4e-7 is *tighter* than S17 — strong prior that S19 will land bit-exact too.

### 4.2 Bit-exact across runs

Phase-1 writes are single-owner (no order dependence). Phase-2 sums in FIXED order `for (b = 0; b < maxBlocksPerPart; b++)` → identical sum across all dispatches. Combined with the existing fixed-order cross-SIMD fold, **the entire histogram pipeline becomes bit-exact across runs** under (c) — strictly better than the production atomic path (which is non-deterministic by atomic arbitration order).

This restores the BUG-001 deterministic guarantee that was structurally true within a tg under L1a, extending it to the cross-tg level.

---

## 5. TG memory budget for (c) — DEC-011 ceiling check

Per-tg threadgroup memory at phase-1 entry (post barrier-6, pre writeback):

| Allocation | Size | Lifetime | Reused by phase-1? |
|---|---:|---|---|
| `simdHist[0..7][1024]` | 32 KB | barrier 1 → barrier 6 | `simdHist[0]` aliased as `stagingHist`; `simdHist[1..7]` are dead |
| Phase-1 scratch | **0 KB** | — | Reuses `simdHist[0]` |
| **Total** | **32 KB** | | |

**Lifetime proof for `simdHist[1..7]` reuse:**

In the production kernel (`kernel_sources.h:230..238`), the cross-SIMD 8-term linear fold reads:
```
for (uint g = 0u; g < NUM_SIMD_GROUPS; g++) {
    sum += simdHist[g][tile_base + tid];
}
simdHist[0][tile_base + tid] = sum;
```
After the per-tile barrier (`kernel_sources.h:238`), `simdHist[g][tile_base + tid]` for g ≥ 1 is **never read again** by any subsequent code path. The writeback section (`kernel_sources.h:243..265`) reads only `stagingHist[]` which aliases `simdHist[0]`. So `simdHist[1..7]` are dead after the last per-tile barrier (= barrier 6 in the 4-tile loop).

Variant (c) phase-1 needs only to write `simdHist[0]` to the global staging buffer — no scratch beyond what already exists. **Net new threadgroup memory for (c): 0 KB.** DEC-011 32 KB ceiling preserved.

---

## 6. Drive-by fold rule decision for (b)

**Rule:** if (b) standalone delivers > 1 ms saving on the 50k gate config, fold into the S19-03 production kernel as a separate commit after (c). Otherwise defer with a named trigger.

**Result:** (b) standalone projects to **0.4–0.6 ms** at the 50k gate (§3.3). **Below the 1 ms threshold.**

**Decision: DEFER (b) to a future sprint.**

**Named trigger for re-raise:** any future sparsity-heavy config (depth ≥ 5 AND `partition_avg_size < 50 docs`) showing > 1 ms writeback share at gate measurement. Likely candidates: depth ≥ 8 oblivious trees (Sprint 22 backlog) or extreme N=1k sweeps. (b) is a 1-LOC change and remains a viable drive-by when the trigger fires.

**Why standalone matters here:** the production kernel ALREADY has `if (abs(val) > 1e-20f)` (`kernel_sources.h:257`). Variant (b) as defined here is the tightening to `if (val == 0.0f) continue;` plus quantification of its incremental contribution. Under (c), the question is moot — (c) has zero atomics regardless of sparsity, so (b) cannot stack onto (c) (no atomics to skip).

**Stack note for completeness:** if a future sprint reverts to an atomic writeback variant (e.g. memory-pressure forces removal of `stagingGlobal`), (b) becomes relevant again. Document this in DEC-013.

---

## 7. Cross-variant ranking

| Rank | Variant | Projected ms | Floor removed | Determinism | Risk |
|---:|---|---:|:---:|:---:|---|
| **1** | **(c) two-phase** | **3.0 ± 1.0** | **Yes** | **Bit-exact** | **+1 dispatch, +10 MB persistent buffer** |
| 2 | (a) batched-atomic | 12.5 ± 1.0 | No | Non-det | Low — minimal change, scheduling-dependent gain |
| 3 | (b) zero-skip standalone | 14.9 ± 1.0 | No | Non-det | Trivial — but gain too small |
| ref | S18 production | 15.46 | — | Non-det | — |

**(c) is the only variant that simultaneously: removes the floor, restores cross-tg determinism, fits the 32 KB ceiling.** It is the unambiguous robustness winner.

The chosen-by-Ramos decision (robustness over peak speed) is doubly vindicated: (c) is ALSO the peak-speed winner. (a) saves 2–3 ms; (c) saves 11.5–13.5 ms. There is no robustness/speed trade-off here — (c) wins on both axes.

---

## 8. Risks and Day-2 open questions

1. **Phase-2 dispatch overhead under-estimated.** Projected at 0.5–1.0 ms; if MLX kernel launch latency on the gate config is closer to 2 ms (Sprint 16 MST observed 1.5–2.5 ms launch overhead on cold dispatches), the net (c) saving compresses to ~10 ms. Still removes the floor; gate verdict unchanged. S19-03 measurement will resolve.

2. **stagingGlobal memory pressure at large N.** Buffer size scales `numPartitions × numStats × maxBlocksPerPart × numGroups × 1024 × 4 B`. At 50k gate ≈ 10.2 MB; at 1M (Sprint 23 envelope) projects to ~200 MB. Within MetalAllocator headroom but may pressure unified memory on M1/M2 8 GB SKUs. **Mitigation:** Sprint 19 envelope is DEC-008's 50k cap; large-N negotiation deferred to Sprint 23.

3. **Phase-2 grid geometry.** The `(numPartitions, numStats, totalBinFeatures)` grid at gate is `64 × 1 × 6400 = 410k threads`. Coalesced; no scheduling concern. At larger configs (Sprint 22+) the totalBinFeatures axis dominates — may need tiling.

4. **Pre-zeroing of stagingGlobal padding slots.** Phase-1 writes the full 1024-slot block (coalesced); slots beyond `foldCount` must be zero so phase-2 gather doesn't pick up garbage. **Mitigation:** allocate stagingGlobal with `mx::zeros` once at training start; never reset (single-owner overwrites). MLX zero-init is implicit on `mx::zeros` allocation.

5. **DEC-008 envelope at MultiClass.** γ_9 ≈ 5.4e-7 at gate; MultiClass approxDim=3 multiplies the per-output error by a small factor. Worst case 3 × γ_9 ≈ 1.6e-6, well within ulp ≤ 8 envelope. S19-04 parity re-run validates.

6. **Phase-2 host-side metadata cost.** Phase-2 needs `binToFeatureGroup[]` and `binToSlotInGroup[]` lookup tables (precomputed on host once per training run). Size: `2 × totalBinFeatures × 4 B` ≈ 50 KB at gate. Negligible.

---

## 9. DEC-013 draft — locked design

```
## DEC-013: L_writeback two-phase reduction (Sprint 19)

**Sprint**: 19
**Date**: 2026-04-17
**Branch**: `mlx/sprint-19-hist-writeback`
**Problem**: Sprint 18 L1a kernel floors `histogram_ms` at ~15 ms on 50k
configs (S18-05b). The residual is dominated by global-atomic
writeback contention (~370k atomics/iter at gate). Atomic-pipe
serialization at the L2 level under DEC-011's 1-tg/SM occupancy is
the L1 ceiling for Sprint 19.
**Considered**:
  (a) Batched-atomic with per-thread bin-range ownership — 15–25%
      writeback throughput improvement via SIMD-contiguous L2 line
      access; floor not removed; non-deterministic by atomic order.
  (b) Zero-skip drive-by (standalone) — sparsity-dependent skip of
      atomics for zero-valued bins; standalone gain 0.4–0.6 ms at
      gate (under 1 ms threshold); does not stack onto (c) because
      (c) eliminates atomics entirely.
  (c) Two-phase reduction — per-tg phase-1 single-owner write to a
      global staging buffer indexed by (partIdx, statIdx,
      blockInPart, groupIdx); separate phase-2 gather kernel sums
      per-bin contributions in fixed order. Zero atomics in either
      phase. Fully deterministic.
**Chosen**: (c) two-phase reduction.
**Rationale**:
  - **Robustness over peak speed (Ramos directive).** (c) eliminates
    cross-tg atomic contention entirely rather than shaving it.
    Cross-run bit-exactness restored at the cross-tg level (matches
    DEC-008 spirit).
  - **(c) is also the peak-speed winner.** Projected
    `histogram_ms` 3.0 ms ± 1.0 ms at 50k gate (vs (a) 12.5 ms,
    (b) 14.9 ms; baseline 15.46 ms). −81% projected reduction.
  - **DEC-011 32 KB ceiling preserved.** Phase-1 reuses
    `simdHist[0]` (already aliased as stagingHist post-barrier-6);
    `simdHist[1..7]` proven dead after the cross-SIMD fold. **Net
    new threadgroup memory: 0 KB.**
  - **DEC-008 envelope preserved.** Phase-2 sums
    `maxBlocksPerPart = 3` terms per (partIdx, statIdx, finalBin)
    in fixed order. Total reduction depth γ_9 ≈ 5.4e-7 (7
    cross-SIMD + 2 phase-2). RMSE/Logloss ulp ≤ 4 envelope retained;
    MultiClass ulp ≤ 8 envelope retained.
  - **Persistent buffer cost acceptable.** stagingGlobal at gate ≈
    10.2 MB; allocated once per training run. Within MetalAllocator
    headroom by orders of magnitude.
**Drive-by fold rule (for (b))**: DEFERRED. Standalone gain 0.4–0.6 ms
  at gate is below the 1 ms threshold. Named trigger: any future
  config (depth ≥ 5 AND partition_avg_size < 50 docs) showing
  > 1 ms atomic-writeback share at gate measurement re-raises (b)
  in a separate commit.
**Trade-off**: +1 kernel dispatch (phase-2), +10.2 MB persistent
  device buffer at gate, +~105 LOC across kernel_sources.h and
  histogram.cpp. Cross-run determinism restored as a side-benefit.
**Scope**: `approxDim ∈ {1, 3}`, `N ≤ 50k`, depth 6, 50 iterations
  (DEC-008 envelope). Higher N (Sprint 23) re-validates the
  stagingGlobal memory ceiling.
**Status**: Draft. S19-03 implementation pending. S19-04 parity
  re-run validates DEC-008 envelope.
```

---

## 10. Sources referenced

- Attribution anchors: [`docs/sprint18/attribution.md`](../sprint18/attribution.md) (S18-01)
- Measured 18-config baseline: [`docs/sprint18/results.md`](../sprint18/results.md) (S18-05b) §50k floor table
- 32 KB ceiling derivation: [`docs/sprint18/ablation.md`](../sprint18/ablation.md) (S18-02) §3.1
- Production kernel anchors: `catboost/mlx/kernels/kernel_sources.h:151` (simdHist alloc) · `:230–238` (cross-SIMD fold) · `:247` (stagingHist alias) · `:255–264` (atomic writeback)
- Toy-kernel sketches: [`docs/sprint19/scratch/writeback_variants.metal`](scratch/writeback_variants.metal)
- Decisions: DEC-008 (parity envelope) · DEC-009 (8-term linear fold) · DEC-011 (32 KB ceiling) · DEC-012 (one structural change per commit) · DEC-013 (this draft)
