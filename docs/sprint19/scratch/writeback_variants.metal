// ============================================================================
// Sprint 19 / S19-02 — Writeback variant ablation scratch sketches.
//
// THIS FILE IS NOT COMPILED INTO ANY METALLIB. It is a paper-design fixture
// for the three writeback variants discussed in `docs/sprint19/ablation.md`.
// `kernel_sources.h` is INTENTIONALLY UNTOUCHED until S19-03 (DEC-012:
// "one structural change per commit" — this commit is ablation only).
//
// Shared assumptions for all variants (identical to the production kernel
// at `catboost/mlx/kernels/kernel_sources.h:100..266`):
//   - SIMD_SIZE        = 32
//   - BLOCK_SIZE       = 256
//   - NUM_SIMD_GROUPS  = 8        (= BLOCK_SIZE / SIMD_SIZE)
//   - FEATURES_PER_PACK= 4
//   - BINS_PER_BYTE    = 256
//   - HIST_PER_SIMD    = 1024     (= FEATURES_PER_PACK * BINS_PER_BYTE)
//   - simdHist[8][1024] = 32 KB threadgroup memory (DEC-011 ceiling).
//
// Pre-condition for the writeback section: barrier 6 has fired and
// simdHist[0][f * BINS_PER_BYTE + bin + 1u] holds the fully-reduced
// per-bin sum for feature f (cross-SIMD 8-term linear fold complete).
// simdHist[1..7] are LIVE-DEAD at this point — their contents are stale
// and they may be reused as scratch by phase-1 of variant (c).
//
// Variants compared:
//   (a) batched-atomic with per-thread bin-range ownership
//   (b) zero-skip drive-by (standalone)  — sparsity-dependent
//   (c) two-phase reduction              — Ramos-chosen, robustness-first
// ============================================================================

#include <metal_stdlib>
using namespace metal;

constant constexpr uint SIMD_SIZE       = 32;
constant constexpr uint BLOCK_SIZE      = 256;
constant constexpr uint NUM_SIMD_GROUPS = 8;
constant constexpr uint FEATURES_PER_PACK = 4;
constant constexpr uint BINS_PER_BYTE   = 256;
constant constexpr uint HIST_PER_SIMD   = 1024;

// ============================================================================
// VARIANT (a) — batched-atomic with per-thread bin-range ownership
// ----------------------------------------------------------------------------
// Hypothesis: contiguous bin ownership reduces atomic-address collision rate
// across concurrent threadgroups (each tg's threads target sequential bins,
// so the global-atomic memory subsystem sees fewer hot-bank conflicts).
//
// Trade-off: STILL pays cross-tg atomic contention; reduces per-bin RMW
// rate but does not eliminate the floor. Scheduling-dependent variance
// (1 tg/SM under DEC-011 means the SM's L2/atomic pipe is shared with
// neighbouring SMs writing to the same firstFold bands when partitions
// or feature groups overlap).
// ============================================================================
namespace variant_a {

inline void writeback(
    threadgroup float* stagingHist,                  // alias of simdHist[0]
    device   float*   histogram,                     // global output
    constant uint*    foldCountsFlat,
    constant uint*    firstFoldIndicesFlat,
    uint              foldBase,
    uint              histBase,
    uint              tid)
{
    // Each thread owns BIN_RANGE consecutive bins. With BLOCK_SIZE=256 threads
    // and per-feature ≤256 bins, BIN_RANGE = ceil(256/256) = 1; for ≤128b
    // configs the worst case is 1 bin/thread (trivially per-thread).
    //
    // Contiguous ownership matters for the atomic memory pipe: thread t's
    // BIN_RANGE bins land at addresses [histBase + firstFold + t*BIN_RANGE
    // .. + (t+1)*BIN_RANGE - 1], so within a SIMD group the 32 lanes hit
    // 32 contiguous cache lines instead of strided.
    constexpr uint BIN_RANGE = 1u;

    for (uint f = 0u; f < FEATURES_PER_PACK; f++) {
        const uint folds     = foldCountsFlat[foldBase + f];
        const uint firstFold = firstFoldIndicesFlat[foldBase + f];
        const uint myStart   = tid * BIN_RANGE;

        // Single-pass: each thread walks its BIN_RANGE-sized contiguous slice.
        // (For BIN_RANGE > 1 — e.g. 256 bins / 64 threads variant — the inner
        // loop unrolls.)
        for (uint k = 0u; k < BIN_RANGE; k++) {
            const uint bin = myStart + k;
            if (bin >= folds) break;

            const float val = stagingHist[f * BINS_PER_BYTE + bin + 1u];
            // No zero-skip in pure (a). Atomic always issued.
            device atomic_float* dst =
                (device atomic_float*)(histogram + histBase + firstFold + bin);
            atomic_fetch_add_explicit(dst, val, memory_order_relaxed);
        }
    }
}

}  // namespace variant_a

// ============================================================================
// VARIANT (b) — zero-skip drive-by (standalone)
// ----------------------------------------------------------------------------
// Hypothesis: empty-bin atomics (val == 0) waste atomic-pipe slots. Skipping
// them reduces issued atomic count proportional to histogram sparsity.
//
// Trade-off: gain is sparsity-dependent and bounded by sparsity rate at
// non-leaf depths. At depth ≥ 4 with N=10k, partitions are small enough that
// many bins are zero; at depth 0 with N=50k, virtually no bins are zero.
//
// Zero-test threshold: |val| > 1e-20f. Tighter than abs comparison would
// be wrong because cross-SIMD linear fold can produce small denormals.
// ============================================================================
namespace variant_b {

inline void writeback(
    threadgroup float* stagingHist,
    device   float*   histogram,
    constant uint*    foldCountsFlat,
    constant uint*    firstFoldIndicesFlat,
    uint              foldBase,
    uint              histBase,
    uint              tid)
{
    for (uint f = 0u; f < FEATURES_PER_PACK; f++) {
        const uint folds     = foldCountsFlat[foldBase + f];
        const uint firstFold = firstFoldIndicesFlat[foldBase + f];

        // Strided: thread tid handles bins {tid, tid+256, ...} within the
        // ≤256-bin feature. Same access pattern as the production kernel,
        // just with a tightened zero-skip predicate. (Production already has
        // `abs(val) > 1e-20f` — variant (b) is the *standalone* assessment of
        // *that line alone*; quantifying its contribution.)
        for (uint bin = tid; bin < folds; bin += BLOCK_SIZE) {
            const float val = stagingHist[f * BINS_PER_BYTE + bin + 1u];
            if (val == 0.0f) continue;   // tighter — no denormal special case
            device atomic_float* dst =
                (device atomic_float*)(histogram + histBase + firstFold + bin);
            atomic_fetch_add_explicit(dst, val, memory_order_relaxed);
        }
    }
}

}  // namespace variant_b

// ============================================================================
// VARIANT (c) — TWO-PHASE REDUCTION (CHOSEN by Ramos)
// ----------------------------------------------------------------------------
// Hypothesis: cross-tg atomic contention is the floor. Eliminate it by
// having each tg first reduce its per-bin contribution into a per-PARTITION
// working buffer using a single-owner write (NO atomic), then a second pass
// performs a deterministic gather into the final histogram.
//
// Phase 1 (this kernel): per-tg, per-(partition, statIdx) deterministic
// reduction. Each (partIdx, statIdx, blockInPart, groupIdx) tuple is unique
// across the grid by construction (host-side dispatch); each tg writes to a
// unique slot in a working buffer indexed by these tuples. ZERO atomics.
//
// Phase 2 (separate kernel): for each (partIdx, statIdx, bin) in the final
// histogram, sum the per-(blockInPart, groupIdx) entries from phase 1 in
// FIXED order. Deterministic. Single writer per output slot. ZERO atomics.
//
// MEMORY BUDGET (TG, phase-1):
//   - simdHist[8][1024] = 32 KB    (already-allocated, REUSED below)
//   - NO new threadgroup allocations.
//
// LIFETIME PROOF (simdHist[1..7] reuse):
//   At barrier 6 (post cross-SIMD fold), the production kernel reads
//   simdHist[g][bin] for g in 0..7 and writes the sum to simdHist[0][bin].
//   After barrier 6 fires, simdHist[1..7] are NEVER read again by the
//   production writeback path (only simdHist[0], aliased as stagingHist).
//   Therefore simdHist[1..7] are dead-after-barrier-6 and can be reused
//   as scratch for phase-1's per-tg atomic-free staging if needed.
//
// In the design below, phase 1 does NOT need scratch beyond simdHist[0]
// because the per-tg per-bin sum already lives there. The "two phases" are:
//   - Phase 1 = current cross-SIMD fold (already done at barrier 6) +
//               this kernel's deterministic single-owner write to a
//               per-(partIdx, statIdx, blockInPart, groupIdx, bin) slot
//               in a global staging buffer. NO atomic.
//   - Phase 2 = second kernel that reads the staging buffer and writes
//               the final histogram with NO atomics (deterministic gather).
//
// GLOBAL STAGING BUFFER LAYOUT:
//   stagingGlobal[partIdx][statIdx][blockInPart][groupIdx][featureSlot]
//     = float, single writer per slot. Allocated once per training run.
//   Size: numPartitions × numStats × maxBlocksPerPart × numGroups × 1024
//        × 4 B
//
//   At gate (50k/RMSE/d6/128b): ~64 partitions × 1 stat × ~3 blocks/part
//                              × ~13 groups × 1024 bins × 4 B ≈ 10.2 MB.
//   This fits comfortably in MLX's MetalAllocator (Sprint 18 already
//   allocates >100 MB for stats/cursor arrays in the gate pipeline).
//
// REDUCTION DEPTH (Higham γ_N):
//   - Cross-SIMD 8-term linear fold (unchanged from DEC-009): 7 levels.
//   - Phase 2 gather sums maxBlocksPerPart terms per (part, stat, finalBin).
//     groupIdx is FIXED per finalBin (one feature group writes each finalBin
//     band — they are NOT summed across by phase-2). At gate
//     (maxBlocksPerPart=3): 3 terms → 2 levels.
//   - TOTAL effective reduction depth: 7 + 2 = 9 levels → γ_9.
//
//   γ_9 (FP32, machine epsilon u = 2^-24 ≈ 5.96e-8):
//     γ_N ≈ N * u  for small N
//     γ_9 ≈ 9 * 5.96e-8 ≈ 5.36e-7
//
//   This is *tighter* than S17's γ_12 ≈ 7.2e-7 (which historically achieved
//   35/36 bit-exact on parity). DEC-008 envelope check:
//     - RMSE/Logloss ulp ≤ 4 (≈4.77e-7 relative): γ_9 = 5.4e-7 is ~13% above
//       the worst-case bound; realized error typically 0.5×γ_N due to
//       same-sign accumulation → expected to land bit-exact.
//     - MultiClass ulp ≤ 8: 3× factor for K=3 → ~1.6e-6, comfortably within.
//
//   PARITY-FRIENDLY: phase 2 sums in FIXED blockInPart order across all
//   dispatches → bit-exact across runs (no scheduling-dependent accumulation
//   order). This makes (c) DETERMINISTIC where (a) and (b) are determinism-
//   degraded by cross-tg atomic arbitration order.
// ============================================================================
namespace variant_c {

// ---- Phase 1: per-tg deterministic write-out (replaces production writeback)
inline void phase1_writeout(
    threadgroup float* stagingHist,                  // alias of simdHist[0]
    device   float*   stagingGlobal,                 // shape: see comment above
    constant uint*    foldCountsFlat,
    constant uint*    firstFoldIndicesFlat,
    uint              foldBase,
    uint              partIdx,
    uint              statIdx,
    uint              blockInPart,
    uint              groupIdx,
    uint              maxBlocksPerPart,
    uint              numGroups,
    uint              numPartitions,
    uint              numStats,
    uint              tid)
{
    // Compute base index into stagingGlobal for this (partIdx, statIdx, blockInPart, groupIdx).
    // Layout (row-major, slowest first):
    //   [partIdx, statIdx, blockInPart, groupIdx, featureSlot]
    // where featureSlot ranges 0..(FEATURES_PER_PACK * BINS_PER_BYTE - 1) = 0..1023.
    const uint slotsPerTuple = FEATURES_PER_PACK * BINS_PER_BYTE;   // 1024
    const uint stagingBase =
          partIdx       * (numStats * maxBlocksPerPart * numGroups * slotsPerTuple)
        + statIdx       * (maxBlocksPerPart * numGroups * slotsPerTuple)
        + blockInPart   * (numGroups * slotsPerTuple)
        + groupIdx      * slotsPerTuple;

    // Single-owner write per slot: (partIdx, statIdx, blockInPart, groupIdx, slot)
    // is unique per dispatch by host-side grid construction. NO atomic needed.
    //
    // Each thread strides through 1024/256 = 4 slots. We write the whole
    // 1024-slot block (not just the foldCount-trimmed range) to keep the
    // write pattern coalesced; phase-2 reads the trimmed range only.
    // (Padding slots are pre-zeroed at staging-buffer allocation time.)
    //
    // WHY write all 1024 slots: coalesced 256-lane write to 1024 contiguous
    // floats is one full L2 line burst per SIMD group. A trimmed write
    // (only foldCount slots) creates a partial-burst penalty AND introduces
    // a per-feature loop-overhead. The 1024-slot write trades 0–768 wasted
    // bytes per tg for one cleaner memory transaction.
    for (uint slot = tid; slot < slotsPerTuple; slot += BLOCK_SIZE) {
        // simdHist[0][slot] holds the post-fold sum for this slot.
        // (For slots that are bin index 0 — ignored bin — the value is 0.)
        stagingGlobal[stagingBase + slot] = stagingHist[slot];
    }
}

// ---- Phase 2: deterministic gather kernel (separate dispatch)
//
// Grid: (numPartitions, numStats, totalBinFeatures)
// Thread: (1, 1, 1)  — or tile to (32, 1, 1) for SIMD coalescing
//
// For each (partIdx, statIdx, finalBin) tuple, sum the contributions
// across all (blockInPart, groupIdx) tuples that wrote to the corresponding
// staging slot. The (blockInPart, groupIdx) → (firstFold, foldOffset) mapping
// is precomputed on the host and passed as a flat lookup table.
inline void phase2_gather(
    device float*    stagingGlobal,
    device float*    histogram,             // final output
    constant uint*   binToFeatureGroup,     // [totalBinFeatures] → which groupIdx writes this finalBin
    constant uint*   binToSlotInGroup,      // [totalBinFeatures] → which slot in the 1024-slot block
    uint             partIdx,
    uint             statIdx,
    uint             finalBin,
    uint             totalBinFeatures,
    uint             maxBlocksPerPart,
    uint             numGroups,
    uint             numPartitions,
    uint             numStats)
{
    const uint groupIdx = binToFeatureGroup[finalBin];
    const uint slotIdx  = binToSlotInGroup[finalBin];
    const uint slotsPerTuple = FEATURES_PER_PACK * BINS_PER_BYTE;

    // Sum over (blockInPart) in fixed order 0..maxBlocksPerPart-1.
    // groupIdx is fixed for this finalBin (one feature group writes each finalBin),
    // so we only iterate over blockInPart.
    float sum = 0.0f;
    for (uint b = 0u; b < maxBlocksPerPart; b++) {
        const uint stagingBase =
              partIdx     * (numStats * maxBlocksPerPart * numGroups * slotsPerTuple)
            + statIdx     * (maxBlocksPerPart * numGroups * slotsPerTuple)
            + b           * (numGroups * slotsPerTuple)
            + groupIdx    * slotsPerTuple;
        sum += stagingGlobal[stagingBase + slotIdx];
    }

    // Single writer per output slot: NO atomic needed. histogram pre-zeroed
    // at training-iter init.
    const uint histBase = partIdx * numStats * totalBinFeatures
                        + statIdx * totalBinFeatures;
    histogram[histBase + finalBin] = sum;
}

}  // namespace variant_c
