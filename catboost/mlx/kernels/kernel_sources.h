#pragma once

// Metal kernel body strings for CatBoost-MLX.
// These are passed to mx::fast::metal_kernel() as the `source` (body) and `header` parameters.
// The MLX API auto-generates the Metal function signature from input_names/output_names.
//
// IMPORTANT: Variable names in the source MUST exactly match the names in input_names/output_names
// passed to metal_kernel(). MLX maps them to [[buffer(N)]] in order of declaration.
//
// Scalar inputs (0-dim mx::array) become `const constant T& name` in the generated signature.
// Array inputs become `const device T* name` (or `const constant T*` if small).
// Array outputs become `device T*` (or `device atomic<T>*` if atomic_outputs=true).
// Metal thread attributes are auto-detected from the source string.

#include <string>

namespace NCatboostMlx {
namespace KernelSources {

// ============================================================================
// Shared header for histogram kernels
// ============================================================================

static const std::string kHistHeader = R"metal(
#include <metal_stdlib>
using namespace metal;

constant constexpr uint SIMD_SIZE = 32;
constant constexpr uint FEATURES_PER_PACK = 4;
constant constexpr uint BINS_PER_BYTE = 256;
constant constexpr uint BLOCK_SIZE = 256;
constant constexpr uint NUM_SIMD_GROUPS = BLOCK_SIZE / SIMD_SIZE;
constant constexpr uint HIST_PER_SIMD = FEATURES_PER_PACK * BINS_PER_BYTE;
constant constexpr uint TOTAL_HIST_SIZE = NUM_SIMD_GROUPS * HIST_PER_SIMD;
)metal";

// ============================================================================
// Histogram kernel for one-byte features (4 features packed per uint32)
// Batched: processes all feature groups in a single dispatch.
//
// Input names (in order):
//   compressedIndex, stats, docIndices, partOffsets, partSizes,
//   featureColumnIndices, lineSize, maxBlocksPerPart, numGroups,
//   foldCountsFlat, firstFoldIndicesFlat,
//   totalBinFeatures, numStats, totalNumDocs
//
// Output names: histogram
//
// Grid:   (256 * maxBlocksPerPart * numGroups, numPartitions, numStats)
// Thread: (256, 1, 1)
//
// Each threadgroup processes ONE feature group (4 packed features).
// groupIdx and blockInPart are extracted from the X grid position.
// Groups write to non-overlapping firstFoldIndices offsets, so no
// cross-group atomics are needed.
//
// BUG-001 FIX (deterministic accumulation):
//   Root cause: the original implementation used CAS-based float atomic adds into a
//   shared threadgroup histogram. All BLOCK_SIZE (256) threads raced on the same
//   HIST_PER_SIMD (1024) slots. SIMD groups within a threadgroup do NOT execute in
//   lockstep with each other on Apple Silicon, and even within a SIMD group the
//   hardware's CAS arbitration order for simultaneous accesses to the same address
//   is not architecturally guaranteed. This produced non-deterministic histogram
//   values across dispatches.
//
//   Fix (L1a, Sprint 18): replace per-thread private 4 KB arrays (privHist[1024],
//   spilled to device memory) with a per-SIMD-group shared histogram in threadgroup
//   memory (simdHist[8][1024] = 32 KB total, fully on-chip). Stride-partition
//   ownership: lane l of SIMD group g owns bins {l, l+32, l+64, ..., l+992}.
//   Each bin has exactly one writer per SIMD group → zero atomics in accumulation,
//   BUG-001 structurally prevented by construction (no CAS, no races).
//
//   Threadgroup memory (L1a):
//     simdHist[NUM_SIMD_GROUPS][HIST_PER_SIMD] = 8 × 1024 × 4 B = 32 KB (at limit)
//   This replaces the Sprint 17 layout:
//     simdHist[8][256] = 8 KB  +  stagingHist[1024] = 4 KB  = 12 KB
//   The Sprint 18 buffer is 2.67× larger but eliminates the per-thread device-
//   memory spill (256 threads × 4 KB = 1 MB per threadgroup spill eliminated).
//
//   Reduction phase (Sprint 18, BUG-S18-001 fix): single 8-term cross-SIMD
//   linear fold (DEC-009, fixed g=0..7 order, 7 addition levels → γ_7 ≈ 4.2e-7
//   FP32). The D1c intra-SIMD simd_shuffle_xor butterfly was REMOVED when the
//   layout changed — stride-partition accumulation already produces the full
//   per-SIMD-group per-bin sum in simdHist[g][bin], so the intra-SIMD butterfly
//   was redundant AND incorrect (all 32 lanes would read the same address and
//   amplify by 32×; see BUG-S18-001 root-cause comment below). Output target:
//   simdHist[0][bin] acts as stagingHist, reusing the first 1024 slots of the
//   32 KB buffer. Peak threadgroup memory: 32 KB during accumulation and
//   reduction — at the Apple Silicon threadgroup limit, any bump to
//   NUM_SIMD_GROUPS or HIST_PER_SIMD requires re-tiling (see host-side
//   static_assert in kernel_sources.cpp).
//
//   Performance: eliminates (a) 1 MB/tg device-memory spill traffic during
//   accumulation (RMW now goes to on-chip threadgroup SRAM), (b) 256 × 1024-entry
//   zero-init loops per threadgroup (threadgroup memory init is single-owner
//   strided, replacing the per-thread broadcast). Expected savings: 6.4–8.9 ms/iter
//   (27–38% of histogram_ms) per S18-01 attribution.
// ============================================================================

static const std::string kHistOneByteSource = R"metal(
    // Map grid to work — extract groupIdx and blockInPart from X dimension
    const uint tgX       = threadgroup_position_in_grid.x;
    const uint partIdx   = threadgroup_position_in_grid.y;
    const uint statIdx   = threadgroup_position_in_grid.z;
    const uint blockInPart = tgX % maxBlocksPerPart;
    const uint groupIdx    = tgX / maxBlocksPerPart;

    // Bounds check for feature groups
    if (groupIdx >= numGroups) return;

    // Load partition bounds
    const uint partOffset = partOffsets[partIdx];
    const uint partSize   = partSizes[partIdx];

    if (partSize == 0) return;

    // Check if this block is active
    const uint docsPerBlock = (partSize + maxBlocksPerPart - 1) / maxBlocksPerPart;
    const uint myDocStart = blockInPart * docsPerBlock;
    if (myDocStart >= partSize) return;
    const uint myDocEnd = min(myDocStart + docsPerBlock, partSize);
    const uint myDocCount = myDocEnd - myDocStart;

    // Which compressed column to read for this group
    const uint featureColumnIdx = featureColumnIndices[groupIdx];

    // Per-group fold metadata (4 entries per group)
    const uint foldBase = groupIdx * FEATURES_PER_PACK;

    // L1a (BUG-S18-001 fix): Per-SIMD-group shared histogram in threadgroup memory.
    //
    // Layout: simdHist[NUM_SIMD_GROUPS][HIST_PER_SIMD]
    //         = simdHist[8][1024] = 32 KB (exactly at Apple Silicon threadgroup limit)
    //
    // Stride-partition ownership (BUG-001 structural guard):
    //   lane l of SIMD group g owns bins {l, l+32, l+64, ..., l+992}
    //   (32 bins per lane × 32 lanes = 1024 bins per SIMD group)
    //   Each bin has exactly one writer per SIMD group → no atomics, no contention.
    //
    // After accumulation the cross-SIMD fold (DEC-009) reads simdHist[g][bin]
    // and writes the final sum into simdHist[0][bin], which serves as stagingHist.
    //
    // BUG-S18-001 root cause (L1a broken): The original accumulation loop assigned
    // one doc per thread (d = tid, stride BLOCK_SIZE), meaning all 32 lanes in a
    // SIMD group processed 32 DIFFERENT docs with 32 different bins. The predicate
    // (bin & 31) == lane fired with probability 1/32 → only 1/32 of docs contributed.
    // Separately, the intra-SIMD D1c butterfly then multiplied each already-full
    // simdHist[g][bin] slot by 32 (all lanes read the same shared address, butterfly
    // summed it 32 times). Net effect: 1/32 × 32 = correct magnitude but wrong bin
    // geometry → 6-orders-of-magnitude parity failure. Both flaws removed here.
    threadgroup float simdHist[NUM_SIMD_GROUPS][HIST_PER_SIMD]; // 32 KB

    const uint tid     = thread_index_in_threadgroup;
    const uint lane    = tid & (SIMD_SIZE - 1u);   // 0..31 within SIMD group
    const uint simd_id = tid >> 5u;                // 0..7 SIMD group index

    // Zero-init: each lane zeros its owned stride for its SIMD group.
    // Lane l zeros simdHist[simd_id][l], simdHist[simd_id][l+32], ...
    // Total writes per thread: 32 (1024 bins / 32 lanes). Barrier 1.
    for (uint b = lane; b < HIST_PER_SIMD; b += SIMD_SIZE) {
        simdHist[simd_id][b] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup); // barrier 1: zero-init complete

    // Cooperative 32-doc batch accumulation — no atomics, no contention.
    //
    // Each SIMD group strides through its own disjoint batch window:
    //   batch_start = simd_id * SIMD_SIZE, then += NUM_SIMD_GROUPS * SIMD_SIZE
    // Within each batch of 32 docs, lane `src` loaded packed/stat; all 32 lanes
    // in the group receive them via simd_shuffle, and only the bin-owner lane
    // (bin & 31 == lane) writes. Every doc contributes exactly once.
    //
    // No intra-SIMD butterfly needed here: simdHist[g][bin] accumulates the full
    // per-SIMD-group per-bin sum directly — each bin slot has one writer per group.
    // DEC-016 T1 fuse-valid: pack the valid flag into the MSB of `packed` so
    // the per-src loop needs only 2 simd_shuffles instead of 3. Each feature
    // slot is 8 bits wide (packer: csv_train.cpp PackFeatures), so slot-0 uses
    // bits [24..31] — bit 31 aliases bin value 128. Safe ONLY when every
    // feature's fold count ≤ 127 (equivalently, all bin values ≤ 127).
    // Host-side CB_ENSURE in DispatchHistogramBatched enforces this; callers
    // beyond the envelope are rejected loudly. See S19-07 code review.
    const uint VALID_BIT = 0x80000000u;

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
            packed = compressedIndex[docIdx * lineSize + featureColumnIdx] | VALID_BIT;
            stat   = stats[statIdx * totalNumDocs + docIdx];
        }

        // Broadcast each of the 32 docs in this batch to all 32 lanes.
        // simd_shuffle(x, src): all lanes receive lane src's value of x.
        for (uint src = 0u; src < SIMD_SIZE; ++src) {
            const uint  p_s = simd_shuffle(packed, src);
            const float s_s = simd_shuffle(stat,   src);
            if ((p_s & VALID_BIT) == 0u) continue;   // uniform branch across the SIMD group
            const uint p_clean = p_s & 0x7FFFFFFFu;

            // Per-feature accumulation: only the bin-owner lane writes.
            // (bin & 31) == lane ensures each slot has exactly one writer → no RMW races.
            for (uint f = 0u; f < FEATURES_PER_PACK; ++f) {
                const uint bin = (p_clean >> (24u - 8u * f)) & 0xFFu;
                if (bin < foldCountsFlat[foldBase + f] + 1u &&
                    (bin & (SIMD_SIZE - 1u)) == lane) {
                    simdHist[simd_id][f * BINS_PER_BYTE + bin] += s_s;
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup); // barrier 2: accumulation complete

    // Reduction (DEC-009, cross-SIMD linear fold only — intra-SIMD butterfly removed).
    //
    // After correct stride-partition accumulation, simdHist[g][bin] already holds
    // the full per-SIMD-group per-bin sum. No per-lane partials exist to fold via
    // butterfly. Only the 8-term cross-SIMD linear fold (DEC-009) is needed.
    //
    // Barrier count: 1 (zero-init) + 1 (accumulation) + 4 (one barrier per tile × 4)
    //   = 6 total. Down from broken-L1a's 10, down from Sprint 17's 9.
    //
    // Reduction depth: 7-term linear fold → γ_7 ≈ 4.2e-7 FP32. Tighter than S17
    //   (which had 5-xor + 7-linear = 12 levels → γ_12 ≈ 7.2e-7). DEC-008 compliant.

    for (uint tile = 0u; tile < FEATURES_PER_PACK; tile++) {   // 4 tiles × 256 bins = 1024 bins
        const uint tile_base = tile * BINS_PER_BYTE;

        // --- Cross-SIMD fold (DEC-009): 256 threads each accumulate 8 SIMD-group values ---
        // Thread tid handles bin tid (256 threads map exactly to 256 bins per tile).
        // The 8-term sum runs in fixed simd_id order 0..7 → deterministic result.
        // Final sum written to simdHist[0][tile_base + tid] — stagingHist alias below.
        if (tid < BINS_PER_BYTE) {
            float sum = 0.0f;
            for (uint g = 0u; g < NUM_SIMD_GROUPS; g++) {
                sum += simdHist[g][tile_base + tid];
            }
            simdHist[0][tile_base + tid] = sum;   // simdHist[0] acts as stagingHist
        }
        threadgroup_barrier(mem_flags::mem_threadgroup); // 1 barrier per tile (4 total)
    }
    // Total barriers in histogram kernel body:
    //   barrier 1 (zero-init) + barrier 2 (accumulation) + 4 (cross-SIMD, 1/tile × 4) = 6

    // Writeback: read fully-reduced histogram from simdHist[0] (our in-place stagingHist).
    // Contract: stagingHist[f * BINS_PER_BYTE + bin + 1u] — preserved exactly.
    // simdHist[0][f * BINS_PER_BYTE + bin + 1u] holds the same value stagingHist
    // held in Sprint 17: the all-thread sum for feature f, bin+1 (1-indexed CatBoost bins).
    threadgroup float* stagingHist = simdHist[0];   // alias, no copy

    const uint histBase = partIdx * numStats * totalBinFeatures + statIdx * totalBinFeatures;

    for (uint f = 0u; f < FEATURES_PER_PACK; f++) {
        const uint folds = foldCountsFlat[foldBase + f];
        const uint firstFold = firstFoldIndicesFlat[foldBase + f];

        for (uint bin = tid; bin < folds; bin += BLOCK_SIZE) {
            const float val = stagingHist[f * BINS_PER_BYTE + bin + 1u];
            if (abs(val) > 1e-20f) {
                // Always use atomics: multiple blocks per partition OR multiple
                // groups share the same output buffer (different offsets, but
                // the buffer was initialized to zero so atomics are safe).
                device atomic_float* dst = (device atomic_float*)(histogram + histBase + firstFold + bin);
                atomic_fetch_add_explicit(dst, val, memory_order_relaxed);
            }
        }
    }
)metal";

// ============================================================================
// Shared header for scoring kernels
// ============================================================================

static const std::string kScoreHeader = R"metal(
#include <metal_stdlib>
using namespace metal;

constant constexpr uint SCORE_BLOCK_SIZE = 256;
)metal";

// ============================================================================
// Suffix-sum transform kernel — deterministic threadgroup scan (BUG-001 fix)
//
// Converts raw per-bin histogram counts into suffix sums so that ordinal
// split scoring becomes O(1) per bin instead of O(bins).
//
// For ordinal feature with bins h[0..F-1]:
//   h'[b] = sum(h[b..F-1])  (reverse inclusive prefix scan)
//   h'[folds-1] is intentionally left unwritten (written as 0 by init_value);
//   this matches the serial implementation and prevents the scorer from
//   selecting an all-right split that has no left-side documents.
//
// OneHot features are skipped (their bins are independent categories).
//
// Grid:   (numFeatures, numPartitions_times_approxDim, numStats)
// Thread: (256, 1, 1)  — one threadgroup per (feature, partition, stat) triple
//                        256 threads >= 255 bins max, one thread per bin
//
// BUG-001 FIX — Root cause of non-determinism in the previous implementation:
//   The previous code used simd_prefix_inclusive_sum + simd_broadcast for the
//   multi-pass (bins > 32) path.  Empirical testing (10 runs on fixed inputs)
//   showed alternating values at the first written bin slot, proving that
//   simd_broadcast reads from an architecturally-undefined lane state across
//   separate Metal command-buffer submissions.  The simd_broadcast spec says the
//   source lane must be active (convergent) — this is not guaranteed when the
//   active-lane mask changes between the conditional read and the broadcast call.
//
// Fix — explicit Hillis-Steele inclusive scan in threadgroup shared memory:
//   1. Each thread t loads h[folds-1-t] into scanBuf[t] (reversed order so
//      a left-to-right inclusive prefix sum computes right-to-left suffix sums).
//      Threads with t >= folds load 0.0f.
//   2. Hillis-Steele up-sweep: log2(256)=8 rounds.  Round r adds scanBuf[t]
//      to scanBuf[t - 2^r] for t >= 2^r.  A threadgroup_barrier separates each
//      round — the addition order is fixed by the algorithm, not by hardware
//      scheduling.  The result is a deterministic inclusive prefix sum.
//   3. Write-back: thread t (t >= 1, t < folds) writes scanBuf[t] to
//      histogram_out at bin (folds-1-t).  Thread 0 / bin (folds-1) is skipped
//      per CatBoost serial semantics.
//
// Memory: threadgroup float scanBuf[256] = 1 KB per threadgroup.  Well within
//   the 32 KB threadgroup memory limit on all Apple Silicon GPUs.
//
// Performance: 8 barrier rounds vs the old ceil(bins/32) chunk iterations plus
//   simd intrinsic calls.  For bins=96 the old path had 3 chunk passes; the
//   new path always does exactly 8 passes.  At bins=32 the new path also does 8
//   passes instead of 1, but suffix-sum is not the hot path — it is dominated
//   by histogram build and split scoring.  Cold-start improvement (344→109 ms)
//   from TODO-008 is preserved because the kernel-compile cache hit is unchanged.
// ============================================================================

static const std::string kSuffixSumSource = R"metal(
    // Each threadgroup handles one (feature, partition, stat) triple.
    // Thread index is the bin index (reversed: thread 0 = bin folds-1).
    const uint t       = thread_index_in_threadgroup;   // 0..255
    const uint featIdx = threadgroup_position_in_grid.x;
    const uint partIdx = threadgroup_position_in_grid.y;
    const uint statIdx = threadgroup_position_in_grid.z;

    if (featIdx >= numFeatures) return;

    // Skip one-hot features — their histogram entries are direct lookups.
    if (featureIsOneHot[featIdx] != 0u) return;

    const uint folds = featureFolds[featIdx];
    if (folds <= 1u) return;

    const uint firstFold = featureFirstFold[featIdx];
    const uint base = partIdx * numStats * totalBinFeatures + statIdx * totalBinFeatures;

    // Step 1: Load bins into shared buffer in reversed order.
    // Thread t maps to bin (folds-1-t). Threads t >= folds load 0.
    // After the scan, scanBuf[t] = h[folds-1] + h[folds-2] + ... + h[folds-1-t]
    //                             = suffix sum h'[folds-1-t].
    threadgroup float scanBuf[256];
    scanBuf[t] = (t < folds) ? histogram[base + firstFold + (folds - 1u - t)] : 0.0f;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: Hillis-Steele inclusive prefix scan (log2(256) = 8 rounds).
    // Round r: each thread t adds the value from thread (t - 2^r) if t >= 2^r.
    // The barrier between rounds guarantees every thread sees the previous
    // round's writes before reading — the addition order is fixed and identical
    // across all dispatches.
    for (uint stride = 1u; stride < 256u; stride <<= 1u) {
        float addend = (t >= stride) ? scanBuf[t - stride] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        scanBuf[t] += addend;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Step 3: Write results back.
    // Thread t covers bin (folds-1-t).  Thread 0 (= bin folds-1) is skipped:
    // leaving bin folds-1 as 0 (init_value) matches the serial reference which
    // does not write the rightmost bin (no documents go to the left of it).
    if (t >= 1u && t < folds) {
        histogram_out[base + firstFold + (folds - 1u - t)] = scanBuf[t];
    }
)metal";

// ============================================================================
// Split scoring + threadgroup reduction kernel
//
// Each thread evaluates one bin-feature candidate, summing gain across all
// partitions and approxDim dimensions. Threadgroup-level argmax reduction
// produces one best split per block. CPU does final reduction over blocks.
//
// Grid:   (SCORE_BLOCK_SIZE * numBlocks, 1, 1)
// Thread: (SCORE_BLOCK_SIZE, 1, 1)
// ============================================================================

static const std::string kScoreSplitsSource = R"metal(
    const uint globalIdx = threadgroup_position_in_grid.x * SCORE_BLOCK_SIZE
                         + thread_index_in_threadgroup;

    threadgroup float  sharedGain[SCORE_BLOCK_SIZE];
    threadgroup uint   sharedFeat[SCORE_BLOCK_SIZE];
    threadgroup uint   sharedBin[SCORE_BLOCK_SIZE];

    float myGain = -INFINITY;
    uint myFeatIdx = 0xFFFFFFFF;
    uint myBinIdx = 0;

    if (globalIdx < totalBinFeatures) {
        // Find which feature this bin-feature belongs to
        uint featIdx = 0;
        uint binInFeature = globalIdx;
        for (uint f = 0; f < numFeatures; f++) {
            uint folds = featureFolds[f];
            if (binInFeature < folds) {
                featIdx = f;
                break;
            }
            binInFeature -= folds;
        }

        const uint firstFold = featureFirstFold[featIdx];

        // Sum gain across all partitions and all approx dimensions
        float totalGain = 0.0f;

        for (uint k = 0; k < approxDim; k++) {
            const uint dimHistBase = k * numPartitions * numStats * totalBinFeatures;
            const uint dimStatsBase = k * numPartitions;

            for (uint p = 0; p < numPartitions; p++) {
                const float totalSum = partTotalSum[dimStatsBase + p];
                const float totalWeight = partTotalWeight[dimStatsBase + p];

                const uint histBase = dimHistBase + p * numStats * totalBinFeatures;

                // After suffix-sum transform, this gives right-side sum directly
                float sumRight = histogram[histBase + firstFold + binInFeature];
                float weightRight = 0.0f;
                if (numStats > 1u) {
                    weightRight = histogram[histBase + totalBinFeatures + firstFold + binInFeature];
                }

                float sumLeft = totalSum - sumRight;
                float weightLeft = totalWeight - weightRight;

                if (weightLeft < 1e-15f || weightRight < 1e-15f) continue;

                totalGain += (sumLeft * sumLeft) / (weightLeft + l2RegLambda)
                           + (sumRight * sumRight) / (weightRight + l2RegLambda)
                           - (totalSum * totalSum) / (totalWeight + l2RegLambda);
            }
        }

        myGain = totalGain;
        myFeatIdx = featIdx;
        myBinIdx = binInFeature;
    }

    // Threadgroup argmax reduction
    sharedGain[thread_index_in_threadgroup] = myGain;
    sharedFeat[thread_index_in_threadgroup] = myFeatIdx;
    sharedBin[thread_index_in_threadgroup] = myBinIdx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = SCORE_BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (thread_index_in_threadgroup < stride) {
            uint other = thread_index_in_threadgroup + stride;
            if (sharedGain[other] > sharedGain[thread_index_in_threadgroup]) {
                sharedGain[thread_index_in_threadgroup] = sharedGain[other];
                sharedFeat[thread_index_in_threadgroup] = sharedFeat[other];
                sharedBin[thread_index_in_threadgroup] = sharedBin[other];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Thread 0 writes per-block result
    if (thread_index_in_threadgroup == 0) {
        const uint blockIdx = threadgroup_position_in_grid.x;
        bestScores[blockIdx] = sharedGain[0];
        bestFeatureIds[blockIdx] = sharedFeat[0];
        bestBinIds[blockIdx] = sharedBin[0];
    }
)metal";

// ============================================================================
// Score splits kernel with precomputed bin-to-feature lookup table (OPT-2)
//
// Identical algorithm to kScoreSplitsSource but replaces the serial
// feature-search loop with a single indexed load from binToFeature[].
//
// Inputs differ from kScoreSplitsSource by the extra `binToFeature` buffer
// inserted after featureIsOneHot.
//
// Grid:   (SCORE_BLOCK_SIZE * numBlocks, 1, 1)
// Thread: (SCORE_BLOCK_SIZE, 1, 1)
// ============================================================================

static const std::string kScoreSplitsLookupSource = R"metal(
    const uint globalIdx = threadgroup_position_in_grid.x * SCORE_BLOCK_SIZE
                         + thread_index_in_threadgroup;

    threadgroup float  sharedGain[SCORE_BLOCK_SIZE];
    threadgroup uint   sharedFeat[SCORE_BLOCK_SIZE];
    threadgroup uint   sharedBin[SCORE_BLOCK_SIZE];

    float myGain = -INFINITY;
    uint myFeatIdx = 0xFFFFFFFF;
    uint myBinIdx = 0;

    if (globalIdx < totalBinFeatures) {
        // O(1) feature lookup — replaces serial loop over features
        const uint featIdx = binToFeature[globalIdx];
        const uint firstFold = featureFirstFold[featIdx];
        // binInFeature = globalIdx - firstFold (relative bin index within feature)
        const uint binInFeature = globalIdx - firstFold;

        // Sum gain across all partitions and all approx dimensions
        float totalGain = 0.0f;

        for (uint k = 0; k < approxDim; k++) {
            const uint dimHistBase  = k * numPartitions * numStats * totalBinFeatures;
            const uint dimStatsBase = k * numPartitions;

            for (uint p = 0; p < numPartitions; p++) {
                const float totalSum    = partTotalSum[dimStatsBase + p];
                const float totalWeight = partTotalWeight[dimStatsBase + p];

                const uint histBase = dimHistBase + p * numStats * totalBinFeatures;

                // After suffix-sum transform, gives right-side sum directly
                float sumRight    = histogram[histBase + firstFold + binInFeature];
                float weightRight = 0.0f;
                if (numStats > 1u) {
                    weightRight = histogram[histBase + totalBinFeatures + firstFold + binInFeature];
                }

                float sumLeft    = totalSum    - sumRight;
                float weightLeft = totalWeight - weightRight;

                if (weightLeft < 1e-15f || weightRight < 1e-15f) continue;

                totalGain += (sumLeft * sumLeft)   / (weightLeft  + l2RegLambda)
                           + (sumRight * sumRight)  / (weightRight + l2RegLambda)
                           - (totalSum * totalSum)  / (totalWeight + l2RegLambda);
            }
        }

        myGain    = totalGain;
        myFeatIdx = featIdx;
        myBinIdx  = binInFeature;
    }

    // Threadgroup argmax reduction
    sharedGain[thread_index_in_threadgroup] = myGain;
    sharedFeat[thread_index_in_threadgroup] = myFeatIdx;
    sharedBin[thread_index_in_threadgroup]  = myBinIdx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = SCORE_BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (thread_index_in_threadgroup < stride) {
            uint other = thread_index_in_threadgroup + stride;
            if (sharedGain[other] > sharedGain[thread_index_in_threadgroup]) {
                sharedGain[thread_index_in_threadgroup] = sharedGain[other];
                sharedFeat[thread_index_in_threadgroup] = sharedFeat[other];
                sharedBin[thread_index_in_threadgroup]  = sharedBin[other];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Thread 0 writes per-block result
    if (thread_index_in_threadgroup == 0) {
        const uint blockIdx = threadgroup_position_in_grid.x;
        bestScores[blockIdx]     = sharedGain[0];
        bestFeatureIds[blockIdx] = sharedFeat[0];
        bestBinIds[blockIdx]     = sharedBin[0];
    }
)metal";

// ============================================================================
// Leaf accumulation kernel header and source
//
// Accumulates per-leaf gradient/hessian sums from all documents.
//
// BUG-001 FIX: Complete redesign for deterministic accumulation.
//
//   Previous design: one thread per document, multiple threadgroups.
//   Multiple threadgroups raced on the same leaf slot via atomic_fetch_add
//   (cross-threadgroup non-determinism) and within each threadgroup threads
//   raced via CAS on shared memory (intra-threadgroup non-determinism).
//
//   New design: one threadgroup total (strided document loop), per-thread
//   private accumulators, fixed-order sequential reduction.
//
//   - Strided loop: each thread i processes docs i, i+LEAF_BLOCK_SIZE, ...
//     Each thread's document subset is fixed and deterministic.
//   - Per-thread private array (LEAF_PRIV_SIZE = MAX_APPROX_DIM*MAX_LEAVES*2
//     = 1280 floats = 5 KB) — zero contention during accumulation.
//   - Sequential reduction (LEAF_BLOCK_SIZE passes, one per thread) with
//     threadgroup_barrier between passes — fixed addition order.
//   - Single threadgroup → no cross-threadgroup global atomics at all.
//     Final write to global output is non-atomic (exactly one write per slot).
//
// Grid:   (LEAF_BLOCK_SIZE, 1, 1)  — always exactly ONE threadgroup
// Thread: (LEAF_BLOCK_SIZE, 1, 1)
//
// NOTE: Callers must use grid = (LEAF_BLOCK_SIZE, 1, 1) NOT the old
//       (LEAF_BLOCK_SIZE * numBlocks, 1, 1) multi-threadgroup dispatch.
//       The kernel now iterates internally over all numDocs.
// ============================================================================

static const std::string kLeafAccumHeader = R"metal(
#include <metal_stdlib>
using namespace metal;

constant constexpr uint LEAF_BLOCK_SIZE = 256;
constant constexpr uint MAX_LEAVES = 64;
constant constexpr uint MAX_APPROX_DIM = 10;
// Per-thread private storage: MAX_APPROX_DIM * MAX_LEAVES * 2 = 1280 floats = 5 KB
constant constexpr uint LEAF_PRIV_SIZE = MAX_APPROX_DIM * MAX_LEAVES * 2;
// Chunk size for multi-pass accumulation (depth > 6): always MAX_LEAVES
constant constexpr uint LEAF_CHUNK_SIZE = MAX_LEAVES;
)metal";

static const std::string kLeafAccumSource = R"metal(
    // BUG-001 FIX: Per-thread private accumulator — zero contention.
    // Each thread processes docs [thread_idx, thread_idx+LEAF_BLOCK_SIZE, ...].
    // All writes go to thread-private stack memory: no atomics, no races.
    float privSums[LEAF_PRIV_SIZE];

    const uint totalEntries = approxDim * numLeaves * 2u;

    // Zero per-thread private sums
    for (uint i = 0u; i < totalEntries; i++) {
        privSums[i] = 0.0f;
    }

    // Strided document loop: each thread covers a deterministic non-overlapping
    // subset of documents — no contention with other threads.
    for (uint d = thread_index_in_threadgroup; d < numDocs; d += LEAF_BLOCK_SIZE) {
        const uint leaf = partitions[d];
        if (leaf < numLeaves) {
            for (uint k = 0u; k < approxDim; k++) {
                const float grad = gradients[k * numDocs + d];
                const float hess = hessians[k * numDocs + d];
                privSums[k * numLeaves * 2u + leaf * 2u]       += grad;
                privSums[k * numLeaves * 2u + leaf * 2u + 1u]  += hess;
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Fixed-order sequential reduction via shared staging array.
    // Thread 0 initialises staging; threads 1..LEAF_BLOCK_SIZE-1 add in order.
    // Addition order is fixed across all dispatches → deterministic result.
    threadgroup float stagingSums[LEAF_PRIV_SIZE];

    if (thread_index_in_threadgroup == 0u) {
        for (uint i = 0u; i < totalEntries; i++) {
            stagingSums[i] = privSums[i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint t = 1u; t < LEAF_BLOCK_SIZE; t++) {
        if (thread_index_in_threadgroup == t) {
            for (uint i = 0u; i < totalEntries; i++) {
                stagingSums[i] += privSums[i];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Single-threadgroup design: no other threadgroup writes to the same slots.
    // Non-atomic global write is correct and deterministic.
    for (uint i = thread_index_in_threadgroup; i < totalEntries; i += LEAF_BLOCK_SIZE) {
        const float val = stagingSums[i];
        const uint k       = i / (numLeaves * 2u);
        const uint rem     = i % (numLeaves * 2u);
        const uint leaf    = rem / 2u;
        const uint is_hess = rem % 2u;

        if (is_hess == 0u) {
            gradSums[k * numLeaves + leaf] = val;
        } else {
            hessSums[k * numLeaves + leaf] = val;
        }
    }
)metal";

// ============================================================================
// Chunked leaf accumulation kernel — multi-pass variant for depth > 6
//
// Identical algorithm to kLeafAccumSource but processes only the leaf slice
// [chunkBase, chunkBase + chunkSize) per dispatch.  The caller issues
// ceil(numLeaves / LEAF_CHUNK_SIZE) dispatches, each with a different chunkBase,
// and concatenates the per-chunk grad/hess sums into the final output arrays.
//
// This avoids allocating a per-thread private array proportional to numLeaves:
// LEAF_PRIV_SIZE is fixed at MAX_APPROX_DIM * LEAF_CHUNK_SIZE * 2 = 1280 floats
// (5 KB) regardless of total numLeaves — identical to the single-pass kernel.
//
// Extra inputs vs kLeafAccumSource:
//   chunkBase  — uint32: first leaf index handled by this pass
//   chunkSize  — uint32: number of leaves in this pass (<= LEAF_CHUNK_SIZE)
//
// Output shapes: [approxDim * chunkSize] each (caller slices into full output)
//
// Grid/Thread: same as kLeafAccumSource — (256, 1, 1) single threadgroup.
// ============================================================================

static const std::string kLeafAccumChunkedSource = R"metal(
    // BUG-001 FIX pattern: per-thread private accumulator, fixed-order reduction.
    // This pass handles leaves [chunkBase, chunkBase + chunkSize).
    // Private array is always LEAF_PRIV_SIZE regardless of total numLeaves.
    float privSums[LEAF_PRIV_SIZE];

    const uint totalEntries = approxDim * chunkSize * 2u;

    // Zero per-thread private sums
    for (uint i = 0u; i < totalEntries; i++) {
        privSums[i] = 0.0f;
    }

    // Strided document loop: only accumulate docs whose leaf falls in this chunk.
    for (uint d = thread_index_in_threadgroup; d < numDocs; d += LEAF_BLOCK_SIZE) {
        const uint leaf = partitions[d];
        if (leaf >= chunkBase && leaf < chunkBase + chunkSize) {
            const uint localLeaf = leaf - chunkBase;  // index within this chunk
            for (uint k = 0u; k < approxDim; k++) {
                const float grad = gradients[k * numDocs + d];
                const float hess = hessians[k * numDocs + d];
                privSums[k * chunkSize * 2u + localLeaf * 2u]      += grad;
                privSums[k * chunkSize * 2u + localLeaf * 2u + 1u] += hess;
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Fixed-order sequential reduction (identical to kLeafAccumSource).
    threadgroup float stagingSums[LEAF_PRIV_SIZE];

    if (thread_index_in_threadgroup == 0u) {
        for (uint i = 0u; i < totalEntries; i++) {
            stagingSums[i] = privSums[i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint t = 1u; t < LEAF_BLOCK_SIZE; t++) {
        if (thread_index_in_threadgroup == t) {
            for (uint i = 0u; i < totalEntries; i++) {
                stagingSums[i] += privSums[i];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write chunk results to output arrays (caller offsets into full-size output).
    // Output layout: [approxDim * chunkSize] — grad and hess in separate buffers.
    for (uint i = thread_index_in_threadgroup; i < totalEntries; i += LEAF_BLOCK_SIZE) {
        const float val    = stagingSums[i];
        const uint k       = i / (chunkSize * 2u);
        const uint rem     = i % (chunkSize * 2u);
        const uint localLeaf = rem / 2u;
        const uint is_hess = rem % 2u;

        if (is_hess == 0u) {
            gradSums[k * chunkSize + localLeaf] = val;
        } else {
            hessSums[k * chunkSize + localLeaf] = val;
        }
    }
)metal";

// ============================================================================
// Tree apply kernel — port of ApplyObliviousTree CPU loop to a single
// Metal dispatch.
//
// Replaces O(depth) separate MLX op dispatches (slice/shift/mask/compare/or
// per depth level) with one kernel invocation that processes all depth levels
// per thread in a tight inner loop.
//
// Input names (in order):
//   compressedData  — [numDocs * lineSize] uint32: packed feature columns
//   splitColIdx     — [depth] uint32: which ui32 column per split level
//   splitShift      — [depth] uint32: right-shift count per level
//   splitMask       — [depth] uint32: bit mask after shift per level
//   splitThreshold  — [depth] uint32: bin threshold per level
//   splitIsOneHot   — [depth] uint32: 1 if OneHot (equality), 0 if ordinal (>)
//   leafValues      — [numLeaves * approxDim] float32: leaf values (lr already baked in)
//   cursorIn        — [approxDim * numDocs] float32: existing cursor (read-only)
//   numDocs         — scalar uint32
//   depth           — scalar uint32: number of split levels (== log2(numLeaves))
//   lineSize        — scalar uint32: number of uint32 columns per document
//   approxDim       — scalar uint32: number of prediction dimensions (1 for binary/regression)
//
// Output names:
//   cursorOut       — [approxDim * numDocs] float32: updated cursor (cursor + leaf delta)
//   partitionsOut   — [numDocs] uint32: leaf index per document (eliminates O(depth) recompute)
//
// NOTE: cursorOut is the mutable output. Each thread reads cursorIn for its own
//   doc slot, adds the leaf value, and writes cursorOut — no atomics needed.
//   Binary/regression: cursorOut[d] = cursorIn[d] + leafValues[leafIdx]
//   Multiclass:        cursorOut[k * numDocs + d] = cursorIn[k * numDocs + d]
//                                                  + leafValues[leafIdx * approxDim + k]
//
// Grid:   (numDocs, 1, 1) rounded up to threadgroup boundary
// Thread: (TREE_APPLY_BLOCK_SIZE, 1, 1) = (256, 1, 1)
//
// Design rationale:
//   - No shared memory: each thread is fully independent (reads own doc, writes own slots).
//   - No atomics: each output slot is owned by exactly one thread.
//   - depth loop is fully unrolled at runtime (depth <= 6 in practice); all reads
//     are coalesced (consecutive docs access consecutive compressedData rows).
//   - Learning rate is already baked into leafValues by ComputeLeafValues; no extra
//     scalar multiply needed.
//   - Handles depth=0 correctly: leafIdx stays 0, all docs go to leaf 0.
// ============================================================================

static const std::string kTreeApplyHeader = R"metal(
#include <metal_stdlib>
using namespace metal;

constant constexpr uint TREE_APPLY_BLOCK_SIZE = 256;
)metal";

static const std::string kTreeApplySource = R"metal(
    // One thread per document.
    const uint globalDocIdx = threadgroup_position_in_grid.x * TREE_APPLY_BLOCK_SIZE
                            + thread_index_in_threadgroup;

    if (globalDocIdx >= numDocs) return;

    // Compute leaf index by applying all split levels.
    // For each level d:
    //   featureVal = (compressedData[docIdx * lineSize + col] >> shift) & mask
    //   goRight    = (isOneHot) ? (featureVal == threshold) : (featureVal > threshold)
    //   leafIdx   |= goRight << d
    uint leafIdx = 0u;
    const uint docBase = globalDocIdx * lineSize;

    for (uint d = 0u; d < depth; d++) {
        const uint col       = splitColIdx[d];
        const uint shift     = splitShift[d];
        const uint mask      = splitMask[d];
        const uint threshold = splitThreshold[d];
        const uint isOneHot  = splitIsOneHot[d];

        const uint packed     = compressedData[docBase + col];
        const uint featureVal = (packed >> shift) & mask;

        uint goRight;
        if (isOneHot != 0u) {
            goRight = (featureVal == threshold) ? 1u : 0u;
        } else {
            goRight = (featureVal > threshold) ? 1u : 0u;
        }
        leafIdx |= (goRight << d);
    }

    // Write updated cursor for all approxDim dimensions.
    // cursorIn/cursorOut layout: [approxDim, numDocs] row-major = k * numDocs + doc
    // leafValues layout:         [numLeaves, approxDim] row-major = leafIdx * approxDim + k
    // Each thread owns slots (k * numDocs + globalDocIdx) for all k — no contention.
    for (uint k = 0u; k < approxDim; k++) {
        const uint slot = k * numDocs + globalDocIdx;
        cursorOut[slot] = cursorIn[slot] + leafValues[leafIdx * approxDim + k];
    }

    // Write partition (leaf) assignment directly from the computed leafIdx.
    // Eliminates the redundant O(depth) MLX bitwise-op recompute in tree_applier.cpp.
    // depth=0 edge case: leafIdx=0 for all docs (loop above never executes), correct.
    partitionsOut[globalDocIdx] = leafIdx;
)metal";

// ============================================================================
// Depthwise tree apply kernel.
//
// For depthwise (non-symmetric) trees, each internal node at depth d has its
// own (feature, bin, isOneHot) split — unlike oblivious trees where all leaves
// at depth d share one split.
//
// Node indexing: nodes are numbered 0..numNodes-1 in BFS order.
//   depth 0: node 0            (the root)
//   depth 1: nodes 1, 2
//   depth 2: nodes 3, 4, 5, 6
//   depth d: nodes [2^d - 1 .. 2^(d+1) - 2]    (numNodes = 2^maxDepth - 1 total)
//
// Tree traversal per document:
//   nodeIdx = 0   (root)
//   for d in [0, depth):
//     extract featureVal from nodeIdx's split descriptor
//     goRight = (isOneHot ? featureVal==thresh : featureVal>thresh)
//     nodeIdx = 2*nodeIdx + 1 + goRight    (left child = 2n+1, right child = 2n+2)
//   leafIdx = nodeIdx - (numNodes)         (leaves are at nodeIdx in [numNodes..numNodes+numLeaves-1])
//
// Input names (in order):
//   compressedData  — [numDocs * lineSize] uint32
//   nodeColIdx      — [numNodes] uint32: feature column per internal node
//   nodeShift       — [numNodes] uint32: right-shift per node
//   nodeMask        — [numNodes] uint32: bit mask per node
//   nodeThreshold   — [numNodes] uint32: bin threshold per node
//   nodeIsOneHot    — [numNodes] uint32: 1=OneHot, 0=ordinal
//   leafValues      — [numLeaves * approxDim] float32
//   cursorIn        — [approxDim * numDocs] float32
//   numDocs         — scalar uint32
//   depth           — scalar uint32 (actual depth of tree, not maxDepth)
//   lineSize        — scalar uint32
//   approxDim       — scalar uint32
//
// Output names:
//   cursorOut       — [approxDim * numDocs] float32
//   partitionsOut   — [numDocs] uint32 (leaf index per doc)
//
// Grid:   (numDocs, 1, 1) rounded up to threadgroup boundary
// Thread: (TREE_APPLY_BLOCK_SIZE, 1, 1) = (256, 1, 1)
// ============================================================================

static const std::string kTreeApplyDepthwiseSource = R"metal(
    // One thread per document.
    const uint globalDocIdx = threadgroup_position_in_grid.x * TREE_APPLY_BLOCK_SIZE
                            + thread_index_in_threadgroup;

    if (globalDocIdx >= numDocs) return;

    const uint docBase = globalDocIdx * lineSize;

    // Traverse the binary tree from root (nodeIdx=0) down `depth` levels.
    // BFS layout: left child of node n = 2n+1, right child = 2n+2.
    uint nodeIdx = 0u;
    for (uint d = 0u; d < depth; d++) {
        const uint col       = nodeColIdx[nodeIdx];
        const uint shift     = nodeShift[nodeIdx];
        const uint mask      = nodeMask[nodeIdx];
        const uint threshold = nodeThreshold[nodeIdx];
        const uint isOneHot  = nodeIsOneHot[nodeIdx];

        const uint packed     = compressedData[docBase + col];
        const uint featureVal = (packed >> shift) & mask;

        uint goRight;
        if (isOneHot != 0u) {
            goRight = (featureVal == threshold) ? 1u : 0u;
        } else {
            goRight = (featureVal > threshold) ? 1u : 0u;
        }
        nodeIdx = 2u * nodeIdx + 1u + goRight;
    }

    // numNodes = 2^depth - 1; leaves start at nodeIdx == numNodes.
    // leafIdx = nodeIdx - numNodes.
    const uint numNodes = (1u << depth) - 1u;
    const uint leafIdx  = nodeIdx - numNodes;

    // Update cursor: add leaf value for each approxDim slot.
    for (uint k = 0u; k < approxDim; k++) {
        const uint slot = k * numDocs + globalDocIdx;
        cursorOut[slot] = cursorIn[slot] + leafValues[leafIdx * approxDim + k];
    }

    // Write partition (leaf) assignment.
    partitionsOut[globalDocIdx] = leafIdx;
)metal";

}  // namespace KernelSources
}  // namespace NCatboostMlx
