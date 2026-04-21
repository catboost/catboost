// ============================================================================
// Sprint 19 / S19-02b — Accumulation variant ablation scratch sketches.
//
// THIS FILE IS NOT COMPILED INTO ANY METALLIB. Paper-design fixture for the
// four accumulation redesign variants discussed in
// `docs/sprint19/ablation_accumulation.md`. `kernel_sources.h` is INTENTIONALLY
// UNTOUCHED until S19-03 re-spec lands the chosen variant
// (DEC-012: one structural change per commit).
//
// Pivot context (S19-02 → S19-02b):
//   S19-01 attribution (@performance-engineer, 2026-04-17) measured
//   accumulation = 14.30 ms (93% of 15.43 ms histogram_ms) at 50k/RMSE/d6/128b
//   gate, with writeback only 0.79 ms (5%). The S19-02 writeback ablation
//   (DEC-013) becomes a 5%-lever — Ramos pivoted to accumulation redesign.
//   This file (S19-02b) ablates the four accumulation variants.
//
// Shared assumptions (identical to production at kernel_sources.h:165–209):
//   - SIMD_SIZE        = 32
//   - BLOCK_SIZE       = 256
//   - NUM_SIMD_GROUPS  = 8        (= BLOCK_SIZE / SIMD_SIZE)
//   - FEATURES_PER_PACK= 4
//   - BINS_PER_BYTE    = 256
//   - HIST_PER_SIMD    = 1024     (= FEATURES_PER_PACK * BINS_PER_BYTE)
//   - simdHist[8][1024] = 32 KB threadgroup memory (DEC-011 ceiling).
//
// Pre-condition (entry to accumulation block): zero-init complete (barrier 1
// fired in production). All variants below leave the post-accumulation
// invariant unchanged: simdHist[g][bin] holds the per-SIMD-group per-bin sum,
// ready for the cross-SIMD 8-term linear fold (DEC-009, unchanged).
//
// Cost-model anchors (from S19-01 attribution + kernel structure):
//   - Per TG: 9.08 µs accumulation total (14.30 ms / 1575 TGs).
//   - Per outer-batch step (256 docs): ~3.0 µs (≈ 3 steps per TG at gate).
//   - Per outer step: 8 SIMD groups × 32 doc-loads = 256 docs processed.
//     Per SIMD group inner loop: 32 iters × 3 simd_shuffle × (1 valid-test
//     uniform branch + 4 feature unpack/check/conditional-add).
//
// CRITICAL READING of kernel_sources.h:175 (re-checked 2026-04-17):
//   `batch_start = simd_id * SIMD_SIZE` then `+= NUM_SIMD_GROUPS * SIMD_SIZE`.
//   ⇒ SIMD groups process DISJOINT doc ranges, NOT the same 32 docs.
//   The task brief's variant (B) framing of "8×-redundancy of 8 SIMD groups
//   each hitting the same global cache line" is INCORRECT for this code path.
//   docIndices/stats global loads are cache-amortized only across feature-
//   group dispatches (different groupIdx re-read same docIndices), not
//   across intra-TG SIMD groups. Variant (B) below is re-derived from
//   the correct mechanism: decouple load cadence from shuffle cadence.
//
// Variants ablated:
//   (A) Wider batch — 64 or 128 docs per SIMD group per outer batch
//   (B) TG-memory staging of docs (re-derived: decouples load ≠ dedup)
//   (C) Per-feature kernel specialization — 4 dispatches × 1 feature
//   (D) Different ownership granularity — 16-lane or 8-lane stride-partition
// ============================================================================

#include <metal_stdlib>
using namespace metal;

constant constexpr uint SIMD_SIZE          = 32;
constant constexpr uint BLOCK_SIZE         = 256;
constant constexpr uint NUM_SIMD_GROUPS    = 8;
constant constexpr uint FEATURES_PER_PACK  = 4;
constant constexpr uint BINS_PER_BYTE      = 256;
constant constexpr uint HIST_PER_SIMD      = 1024;

// ============================================================================
// VARIANT (A) — Wider batch size
// ----------------------------------------------------------------------------
// Hypothesis: doubling the inner shuffle loop from 32 → 64 iters amortizes
// the per-iter `valid_s` uniform branch + per-feature `bin & 31 == lane`
// predicate setup over more docs, reducing loop-overhead fraction.
//
// Two sub-variants ablated:
//   (A1) BATCH_DOCS=64, single-load — each lane holds 2 packed/stat values
//        in registers across the outer batch boundary; inner loop 64 iters
//        with src in {0..63} but simd_shuffle only addresses {0..31}, so
//        we shuffle from the "second slab" via a separate shuffle src in
//        {0..31} from the second-doc register.
//   (A2) BATCH_DOCS=64, double-shuffle — two passes of 32-shuffle each;
//        NO per-lane register doubling (each lane still holds 1 doc per
//        pass); amortized loop-counter init is the only saving.
//
// Detailed (A1) analysis below — the meaningful variant.
// ============================================================================
namespace variant_a1_batch64_double_register {

inline void accumulate(
    threadgroup float simdHist[NUM_SIMD_GROUPS][HIST_PER_SIMD],
    device const uint*  docIndices,
    device const uint*  compressedIndex,
    device const float* stats,
    constant uint*      foldCountsFlat,
    uint                partOffset,
    uint                myDocStart,
    uint                myDocCount,
    uint                lineSize,
    uint                featureColumnIdx,
    uint                statIdx,
    uint                totalNumDocs,
    uint                foldBase,
    uint                lane,
    uint                simd_id)
{
    constexpr uint BATCH_DOCS = 64;   // 2× production
    constexpr uint STRIDE     = NUM_SIMD_GROUPS * BATCH_DOCS;   // 8 * 64 = 512

    // Each lane holds 2 docs in registers across the inner shuffle loop.
    // Register pressure: was {packed, stat, valid} = 3 regs × 4 B = 12 B/lane.
    // Now: {packed_lo, packed_hi, stat_lo, stat_hi, valid_lo, valid_hi}
    //      = 6 regs × 4 B = 24 B/lane. +12 B/lane × 256 threads = +3 KB
    //      register pressure per TG. Apple Silicon AGX VGPR file: ~256 VGPR
    //      per thread typical; this ~6-VGPR delta is well-bounded.
    for (uint batch_start = simd_id * BATCH_DOCS;
         batch_start < myDocCount;
         batch_start += STRIDE) {

        const uint d_lo = batch_start + lane;
        const uint d_hi = batch_start + SIMD_SIZE + lane;
        const bool valid_lo = (d_lo < myDocCount);
        const bool valid_hi = (d_hi < myDocCount);

        uint  packed_lo = 0u, packed_hi = 0u;
        float stat_lo   = 0.0f, stat_hi = 0.0f;

        if (valid_lo) {
            const uint sortedPos = partOffset + myDocStart + d_lo;
            const uint docIdx    = docIndices[sortedPos];
            packed_lo = compressedIndex[docIdx * lineSize + featureColumnIdx];
            stat_lo   = stats[statIdx * totalNumDocs + docIdx];
        }
        if (valid_hi) {
            const uint sortedPos = partOffset + myDocStart + d_hi;
            const uint docIdx    = docIndices[sortedPos];
            packed_hi = compressedIndex[docIdx * lineSize + featureColumnIdx];
            stat_hi   = stats[statIdx * totalNumDocs + docIdx];
        }

        // Inner: shuffle each of the 64 docs from its owning lane.
        // Lanes 0..31 hold "lo slab", lane src in {0..31} broadcasts.
        // Then "hi slab", lane src in {0..31} broadcasts the _hi register.
        for (uint slab = 0u; slab < 2u; ++slab) {
            for (uint src = 0u; src < SIMD_SIZE; ++src) {
                const uint  p_s     = (slab == 0u)
                    ? simd_shuffle(packed_lo, src) : simd_shuffle(packed_hi, src);
                const float s_s     = (slab == 0u)
                    ? simd_shuffle(stat_lo,   src) : simd_shuffle(stat_hi,   src);
                const bool  valid_s = (slab == 0u)
                    ? simd_shuffle(valid_lo,  src) : simd_shuffle(valid_hi,  src);
                if (!valid_s) continue;

                for (uint f = 0u; f < FEATURES_PER_PACK; ++f) {
                    const uint bin = (p_s >> (24u - 8u * f)) & 0xFFu;
                    if (bin < foldCountsFlat[foldBase + f] + 1u &&
                        (bin & (SIMD_SIZE - 1u)) == lane) {
                        simdHist[simd_id][f * BINS_PER_BYTE + bin] += s_s;
                    }
                }
            }
        }
    }
}

}  // namespace variant_a1_batch64_double_register

// ============================================================================
// VARIANT (A2) — BATCH_DOCS=128, quad-pass shuffle (no register inflation)
// ----------------------------------------------------------------------------
// Same as A1 but stride 4× wider — each lane holds 4 docs. Register pressure
// rises to ~12 VGPR delta/lane. No win on amortized loop overhead beyond A1
// because inner shuffle ops still dominate; halves the outer loop iter count
// at gate (3 → 1). At gate the outer loop already runs ≤3 iters per TG, so
// halving has < 1 µs/TG benefit. Documented for completeness but A1 is the
// representative variant for projection. Implementation analogous to A1
// with `slab in {0,1,2,3}` and 4 register pairs per lane.
// ============================================================================

// ============================================================================
// VARIANT (B) — TG-memory staging of docs (CORRECTED MECHANISM)
// ----------------------------------------------------------------------------
// Hypothesis (CORRECTED — task brief mechanism is wrong; see file header):
//   The current path has each SIMD group ISSUE its own 3 global loads inside
//   the outer batch loop. These loads have ~200-cycle device-memory latency.
//   The 32-iter shuffle loop that follows depends only on register state,
//   so the load-latency cannot overlap with shuffle work *within a SIMD group*.
//
//   Staging proposal: have ALL 256 threads cooperatively load the next 32-doc
//   batch into TG memory (tileCompressed[32], tileStats[32]) using coalesced
//   256-thread strided reads, then have each SIMD group's 32 lanes read
//   tile[lane] for the shuffle (or skip the shuffle entirely — read tile[src]
//   directly with broadcast-via-TG-memory). This:
//     (i) issues 32 loads in parallel from 256 threads → coalesces into 1
//         L2 burst per stream instead of 8 partial bursts (one per SIMD group).
//     (ii) hides device-load latency behind one tile-load barrier instead of
//          letting it stall the inner shuffle loop's first 32 iters.
//     (iii) eliminates simd_shuffle calls — replaced by tile[src] reads
//           (TG-memory access ~10 cycles, simd_shuffle ~5 cycles intra-SIMD;
//           comparable, but tile reads parallelize across all 256 lanes
//           reading the same tile slot).
//
// CRITICAL TG-MEMORY BUDGET PROBLEM (highlighted in task brief):
//   tileCompressed[32 × lineSize] uint32 + tileStats[32] float per SIMD group.
//   At lineSize ≤ 4 (gate: 100 features / 4 features/pack = 25 columns; each
//   compressedIndex row is `lineSize` uint32s, where lineSize is the number
//   of uint32 packed-feature columns per doc = 25 at gate):
//     tile per SIMD: 32 × 25 × 4 + 32 × 4 = 3200 + 128 = 3328 B
//     × 8 SIMD groups = 26,624 B = 26 KB
//     + simdHist[8][1024] = 32 KB
//     = 58 KB → EXCEEDS 32 KB DEC-011 CEILING.
//
// Two sub-options for (B):
//   (B1) Time-share simdHist vs tile: phase-separated. Stage tile, accumulate
//        into a temp register pattern, then write to simdHist after the
//        accumulation phase ends. PROBLEM: stride-partition ownership means
//        each lane owns 32 disjoint bins; writes to simdHist must happen
//        DURING the shuffle inner loop, not deferred. Time-sharing requires
//        re-architecting to phase-separated accumulation+writeback per
//        sub-tile. Complex; unclear gain.
//   (B2) Shared-tile (single tile across all SIMD groups, NOT per-SIMD):
//        ONE tile[32 × lineSize] = 3200 B + tileStats[32] = 128 B = 3328 B.
//        Total TG: 32 KB simdHist + 3.3 KB tile = 35.3 KB → STILL exceeds.
//        Reduce simdHist tile? Would require re-tiling DEC-011 layout
//        (bigger problem). Or reduce HIST_PER_SIMD to 768 — invalid because
//        FEATURES_PER_PACK × BINS_PER_BYTE is fixed by data layout.
//
//   (B3) Reduced-tile (16 docs per tile instead of 32): halves tile to 1664 B.
//        Total: 32 + 1.6 = 33.6 KB → STILL exceeds.
//
// CONCLUSION: Variant (B) requires DEC-011 ceiling renegotiation regardless
// of sub-option chosen. SURFACED AS EXPLICIT BLOCKER in §3 of the ablation
// doc. Keep (B) sketched here for completeness — kernel below uses (B2)
// shared-tile shape and assumes a hypothetical 35 KB ceiling. Implementation
// would need to re-tile the cross-SIMD fold to a smaller simdHist (e.g.
// simdHist[4][1024] = 16 KB; 4-term cross-SIMD fold; γ_3 vs γ_7).
//
// THIS VARIANT IS BLOCKED ON CEILING RENEGOTIATION AT DEC-011 — see §3.
// ============================================================================
namespace variant_b2_shared_tile {

inline void accumulate(
    threadgroup float simdHist[NUM_SIMD_GROUPS][HIST_PER_SIMD],
    threadgroup uint  tileCompressed[32 * 25],   // 3.2 KB at gate lineSize=25
    threadgroup float tileStats[32],             // 128 B
    device const uint*  docIndices,
    device const uint*  compressedIndex,
    device const float* stats,
    constant uint*      foldCountsFlat,
    uint                partOffset,
    uint                myDocStart,
    uint                myDocCount,
    uint                lineSize,
    uint                featureColumnIdx,
    uint                statIdx,
    uint                totalNumDocs,
    uint                foldBase,
    uint                tid,
    uint                lane,
    uint                simd_id)
{
    // Outer batch: 32 docs at a time, ALL 8 SIMD groups process the SAME 32
    // docs (tile is shared). Stride += 32 docs per outer iter.
    for (uint batch_start = 0u;
         batch_start < myDocCount;
         batch_start += 32u) {

        // Cooperative tile load: 256 threads, 32 docs × (1 stat + lineSize
        // uint32s). At lineSize=25 → 32 × 26 = 832 ops; 256 threads handle
        // ⌈832/256⌉ = 4 ops each. Coalesced device reads.
        threadgroup_barrier(mem_flags::mem_threadgroup);   // pre-tile fence

        for (uint k = tid; k < 32u; k += BLOCK_SIZE) {
            const uint d = batch_start + k;
            if (d < myDocCount) {
                const uint sortedPos = partOffset + myDocStart + d;
                const uint docIdx    = docIndices[sortedPos];
                tileStats[k] = stats[statIdx * totalNumDocs + docIdx];
                for (uint c = 0u; c < lineSize; ++c) {
                    tileCompressed[k * lineSize + c] = compressedIndex[docIdx * lineSize + c];
                }
            } else {
                tileStats[k] = 0.0f;
                for (uint c = 0u; c < lineSize; ++c) {
                    tileCompressed[k * lineSize + c] = 0u;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);   // tile loaded

        // Each SIMD group's 32 lanes process the same 32 docs from the tile.
        // Equivalent to the inner shuffle loop, but reads from TG memory
        // instead of register-shuffle.
        for (uint src = 0u; src < 32u; ++src) {
            const uint d = batch_start + src;
            if (d >= myDocCount) break;   // uniform across SIMD group

            const uint  p_s = tileCompressed[src * lineSize + featureColumnIdx];
            const float s_s = tileStats[src];

            for (uint f = 0u; f < FEATURES_PER_PACK; ++f) {
                const uint bin = (p_s >> (24u - 8u * f)) & 0xFFu;
                if (bin < foldCountsFlat[foldBase + f] + 1u &&
                    (bin & (SIMD_SIZE - 1u)) == lane) {
                    simdHist[simd_id][f * BINS_PER_BYTE + bin] += s_s;
                }
            }
        }
    }
    // NOTE: barriers added per outer batch (≈ 25 batches at gate / 1 SIMD-
    // group worth) = ~50 extra barriers/TG vs production's 1. Each barrier
    // ≈ 50 ns on AGX → +2.5 µs/TG additional barrier cost. Offsets ~25%
    // of projected load-latency saving. See §3 for net projection.
}

}  // namespace variant_b2_shared_tile

// ============================================================================
// VARIANT (C) — Per-feature kernel specialization
// ----------------------------------------------------------------------------
// Hypothesis: split the kernel into 4 separate dispatches, one per feature
// slot in the FEATURES_PER_PACK group. Each dispatch processes ONE feature
// (8-bit slice from the 32-bit packed value) per doc. This:
//   - Eliminates the inner `for (uint f = 0; f < 4; ++f)` loop (4 iterations
//     unrolled into 4 separate dispatches).
//   - Replaces the bin-extract shift `(p_s >> (24 - 8*f)) & 0xFF` with a
//     compile-time constant (no per-iter shift overhead).
//   - Halves the simdHist size: each dispatch only needs 256 bins per SIMD
//     group, not 1024. simdHist[8][256] = 8 KB instead of 32 KB.
//   - Frees TG memory: 32 - 8 = 24 KB headroom. Could compose with a
//     reduced (B) tile (variant B+C composability — see §5).
//
// COST OF MORE DISPATCHES:
//   - 4× MLX kernel dispatches per feature group (× 25 groups = 100 dispatches/iter
//     instead of 25). MLX dispatch overhead: ~20–50 µs/dispatch (Sprint 16 MST).
//   - At 25 → 100 dispatches: +75 dispatches × ~30 µs = +2.25 ms/iter
//     OVERHEAD. This eats most of the saving from removing the 4-feature loop.
//
// Per-dispatch saving: removing the 4-iter inner loop saves ~5–10% of the
// shuffle inner loop's per-iter cost (the conditional add is the dominant
// op, not the shift/loop). Net per-iter saving ~0.5 µs/TG.
// Across 1575 TGs: ~0.8 ms saving from kernel work.
// But +2.25 ms dispatch overhead ⇒ NET LOSS of ~1.5 ms.
//
// VERDICT: (C) is a structural net-loss at gate config because dispatch
// overhead dominates the per-iter saving. Could be revisited at much larger
// N where per-TG work dominates dispatch overhead, but DEC-008 envelope
// caps at N=50k. KILLED for Sprint 19.
//
// NO TOY KERNEL — analytically dominated. Documented for completeness; the
// dispatch-cost arithmetic is sufficient to eliminate (C) without coding it.
// ============================================================================
namespace variant_c_per_feature {
// (No implementation — analytically dominated by dispatch overhead.)
}  // namespace variant_c_per_feature

// ============================================================================
// VARIANT (D) — Different ownership granularity (16-lane stride-partition)
// ----------------------------------------------------------------------------
// Hypothesis: under current 32-lane ownership, lane l owns bins {l, l+32,
// l+64, ..., l+992}. Each shuffled doc triggers a per-feature predicate:
//   - foldCounts check: 1 ALU + 1 branch
//   - lane-owner check: (bin & 31) == lane: 1 ALU + 1 branch
//   - simdHist write: 1 TG-memory RMW
//
// The lane-owner predicate filters 31 of 32 lanes (only 1 lane writes per
// (doc, feature) pair). 31/32 of the lanes do "wasted" predicate work and
// no useful TG write. Switching to 16-lane ownership: 2 lanes own each bin,
// each writing half the docs that hit that bin (separated by parity of
// the doc-iteration order). Saves: only 14 of 16 lanes do nothing → 87.5%
// wasted vs current 96.9% wasted.
//
// BUT this requires a 1-level intra-SIMD reduction: 2 lanes contribute to
// the same bin in different shuffle iterations. To avoid double-counting,
// each pair of lanes must agree on which one writes — one uses
// `simd_shuffle_xor(val, 16)` to sum the partials before writing.
//
// CRITICAL ALGEBRAIC RISK (BUG-S18-001 lesson):
//   Stride-partition's zero-atomic property comes from the structural
//   guarantee that exactly ONE lane writes each bin slot per (doc, feature).
//   Reducing to 16-lane ownership breaks this: 2 lanes both want to write
//   bin b; they must coordinate via either:
//     (i)  an explicit XOR shuffle reduction before write (1 extra
//          shuffle + 1 conditional → adds ALU/cycle overhead per write).
//     (ii) lock-step accumulator via simd_shuffle reduce (changes the
//          algorithm semantics — no longer single-owner).
//     (iii) atomics on simdHist (REGRESSION — re-introduces BUG-001).
//
// Option (i) is the only safe path. The XOR reduction IS the intra-SIMD
// butterfly that DEC-012 explicitly removed under L1a, because under the
// shared-memory layout the butterfly is structurally redundant (BUG-S18-001).
//
// Re-introducing it under 16-lane ownership requires:
//   1. Each pair of owner-lanes (l, l+16) must compute partial sums for
//      bins they jointly own.
//   2. simd_shuffle_xor(partial, 16) brings the partner's value to both
//      lanes; one of the pair (e.g. lane in [0,15]) writes the sum.
//   3. The "pair" in (1) requires per-doc partial accumulation in a
//      register before the shuffle — but partials accumulate across
//      multiple docs in the inner loop, requiring per-bin per-doc state
//      that doesn't fit in registers (lane holds 32 bins / 16-lane =
//      2 bins worth of state across the inner shuffle loop).
//   4. To make this work you'd hold 2 partial-sum registers per lane,
//      flush them per-doc into the simdHist via XOR-reduce-then-write.
//
// The structural hazard: this is the pattern that broke BUG-S18-001
// (intra-SIMD butterfly + shared-memory write = 32× amplification).
// The mathematical structure here differs (partials are per-doc not
// per-bin shared), but the BUG-S18-001 lesson is "don't combine
// layout + algorithm changes without re-deriving the algebraic role".
// Re-derivation feasible but requires careful proof.
//
// Higham γ_N impact: each 16-lane bin now sums 2 docs' contributions via
// the XOR shuffle BEFORE the simdHist write. This is +1 reduction level
// per bin per doc. Across the full kernel: cross-SIMD fold depth γ_7 →
// γ_8 (1 extra level per intra-SIMD reduction). Realized error grows
// from 4.2e-7 to 4.8e-7, still within DEC-008 envelope.
//
// 8-lane ownership: 4 lanes own each bin. Requires 2 XOR-reduce levels
// (γ_7 → γ_9). Wasted-lane fraction drops to 7/8 = 87.5% → wait, that's
// worse arithmetic, let me recompute: 8-lane stride means 4 lanes own each
// bin. Of 32 lanes, 4 contribute to a given bin (12.5% useful), so 87.5%
// wasted — same as 16-lane. But XOR depth doubles. NET: 8-lane is strictly
// worse than 16-lane.
//
// 16-lane is the sweet spot. Sketched below.
// ============================================================================
namespace variant_d_16lane_ownership {

inline void accumulate(
    threadgroup float simdHist[NUM_SIMD_GROUPS][HIST_PER_SIMD],
    device const uint*  docIndices,
    device const uint*  compressedIndex,
    device const float* stats,
    constant uint*      foldCountsFlat,
    uint                partOffset,
    uint                myDocStart,
    uint                myDocCount,
    uint                lineSize,
    uint                featureColumnIdx,
    uint                statIdx,
    uint                totalNumDocs,
    uint                foldBase,
    uint                lane,
    uint                simd_id)
{
    // 16-lane ownership: lane l owns bins where (bin & 15) == (lane & 15).
    // Pairs (l, l+16) jointly own each bin; one writes after XOR-reduce.
    const uint owner_mask = 15u;   // bin & 15 == lane & 15
    const uint pair_lane  = lane ^ 16u;   // partner in the pair

    for (uint batch_start = simd_id * SIMD_SIZE;
         batch_start < myDocCount;
         batch_start += NUM_SIMD_GROUPS * SIMD_SIZE) {

        const uint d     = batch_start + lane;
        const bool valid = (d < myDocCount);

        uint  packed = 0u;
        float stat   = 0.0f;
        if (valid) {
            const uint sortedPos = partOffset + myDocStart + d;
            const uint docIdx    = docIndices[sortedPos];
            packed = compressedIndex[docIdx * lineSize + featureColumnIdx];
            stat   = stats[statIdx * totalNumDocs + docIdx];
        }

        for (uint src = 0u; src < SIMD_SIZE; ++src) {
            const uint  p_s     = simd_shuffle(packed, src);
            const float s_s     = simd_shuffle(stat,   src);
            const bool  valid_s = simd_shuffle(valid,  src);
            if (!valid_s) continue;

            for (uint f = 0u; f < FEATURES_PER_PACK; ++f) {
                const uint bin = (p_s >> (24u - 8u * f)) & 0xFFu;
                const bool in_range = (bin < foldCountsFlat[foldBase + f] + 1u);
                const bool im_owner = ((bin & owner_mask) == (lane & owner_mask));

                // Per-pair partial: each lane's "candidate value" is s_s if
                // it's an owner of bin and in-range, else 0.
                const float my_val   = (in_range && im_owner) ? s_s : 0.0f;
                // XOR reduce across lane pair (l, l+16): both lanes get the
                // sum. This is 1 simd_shuffle_xor at distance 16.
                const float pair_sum = my_val + simd_shuffle_xor(my_val, 16u);

                // Lower-half lane (l < 16) of the pair writes to simdHist.
                // Upper-half lane skips. Eliminates double-write hazard.
                if (in_range && im_owner && lane < 16u) {
                    simdHist[simd_id][f * BINS_PER_BYTE + bin] += pair_sum;
                }
            }
        }
    }
    // NOTE: extra simd_shuffle_xor per (doc, feature, src-shuffle) iteration:
    //   32 × 4 = 128 extra XOR shuffles per outer batch step.
    // simd_shuffle_xor cost on AGX: ~5 cycles, but the inner loop is bound
    // by the simdHist write throughput, not by ALU. Net effect is small
    // (re-quantified in §3): +0.05 µs/outer-step, but the XOR enables a
    // lane occupancy improvement that's hard to bound analytically.
    //
    // The IMPORTANT question: does halving the wasted-lane fraction speed
    // up the inner loop? The answer depends on whether the inner loop is
    // bottlenecked by simdHist write throughput or by ALU/branch latency.
    // S19-01b sub-phase data (pending) needed to settle this.
}

}  // namespace variant_d_16lane_ownership
