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
//   Fix: replace all shared-memory atomic accumulation with per-thread private
//   histograms (thread-local stack arrays). Each thread accumulates only its own
//   documents (loop stride BLOCK_SIZE, starting at thread_index_in_threadgroup)
//   with no contention — zero atomics during accumulation. After accumulation, a
//   fixed-order sequential reduction across threads (t=1, 2, ..., BLOCK_SIZE-1)
//   folds all per-thread contributions into thread-0's histogram using threadgroup
//   shared memory as a staging area. The reduction order is fixed, making the
//   final histogram bit-for-bit identical across all dispatches.
//
//   Memory: per-thread stack arrays are thread-local (spill to device memory if
//   register pressure is high, but stay off the shared threadgroup address space).
//   Threadgroup shared memory is used only during the reduction phase (4 KB for
//   the HIST_PER_SIMD float staging buffer).
//
//   Performance: private histogram accumulation eliminates all stall cycles from
//   CAS retries. The sequential reduction adds O(BLOCK_SIZE) passes over HIST_PER_SIMD
//   entries, but each pass is a simple load+add+store without contention.
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

    // BUG-001 FIX: Per-thread private histograms — no shared-memory atomics.
    // Each thread accumulates its own document subset into a private stack array.
    // Thread d processes documents d, d+BLOCK_SIZE, d+2*BLOCK_SIZE, ...
    // This is a fixed, deterministic subset for each thread index, giving
    // identical results for identical inputs across all dispatches.
    //
    // Layout: privHist[FEATURES_PER_PACK * BINS_PER_BYTE] = [4][256] = 1024 floats
    // Size per thread: 4096 bytes (thread-local; spills to device memory if needed).
    float privHist[HIST_PER_SIMD];

    // Zero per-thread private histogram
    for (uint i = 0u; i < HIST_PER_SIMD; i++) {
        privHist[i] = 0.0f;
    }

    // Accumulate documents into private histogram — no atomics, no contention
    for (uint d = thread_index_in_threadgroup; d < myDocCount; d += BLOCK_SIZE) {
        const uint sortedPos = partOffset + myDocStart + d;
        const uint docIdx = docIndices[sortedPos];

        // Load packed features (4 one-byte features in one uint32)
        const uint packed = compressedIndex[docIdx * lineSize + featureColumnIdx];

        // Load the statistic for this document
        const float stat = stats[statIdx * totalNumDocs + docIdx];

        // Accumulate into per-thread private histogram (no contention)
        for (uint f = 0u; f < FEATURES_PER_PACK; f++) {
            const uint bin = (packed >> (24u - 8u * f)) & 0xFFu;
            if (bin < foldCountsFlat[foldBase + f] + 1u) {
                privHist[f * BINS_PER_BYTE + bin] += stat;
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // D1c: SIMD-shuffle intra-SIMD reduction + fixed-order cross-SIMD fold.
    //
    // Replaces the serial 255-step reduction (one barrier per thread) with:
    //   - 5 simd_shuffle_xor rounds per bin (zero threadgroup barriers, register-only)
    //   - 2 threadgroup barriers per tile (SIMD-leader write, then cross-SIMD read)
    //   - 4 tiles cover all HIST_PER_SIMD = 1024 bins → 8 barriers total (down from 255)
    //
    // Threadgroup memory:
    //   simdHist[NUM_SIMD_GROUPS][BINS_PER_BYTE] = 8 × 256 × 4 B = 8 KB
    //   stagingHist[HIST_PER_SIMD]               = 1024 × 4 B    = 4 KB
    //   Peak                                                       = 12 KB (37% of 32 KB limit)
    //
    // Correctness: for each bin b, all 32 threads in a SIMD group load privHist[b]
    // (same bin index, different threads' private histograms), then butterfly-sum
    // across the 32 lanes. After 5 XOR rounds all lanes hold the same value: the
    // per-SIMD-group sum for bin b. Lane 0 stores to simdHist[simd_id][b].
    // The cross-SIMD fold (8-term sequential, fixed simd_id order 0..7) produces
    // the final all-thread sum in stagingHist — deterministic because the simd_id
    // order is compile-time fixed.
    //
    // Determinism: simd_shuffle_xor is bit-deterministic within a SIMD group
    // (Apple Metal Shading Language Spec §6.9 — lanes execute in lockstep).
    // The 8-term cross-SIMD sum runs in fixed simd_id order → also bit-deterministic.
    // The reduction order differs from the serial fold, so absolute values will
    // differ by a few ULP (within the DEC-005 ≤ 4 ULP RMSE parity gate).
    //
    // Zero-skip fast-path (abs(val) > 1e-20f): kept in the global-atomic writeback
    // (lines 192–198 below) where it guards device-memory traffic. Fusing it into
    // the cross-SIMD fold is an open micro-optimisation deferred to Sprint 18.
    threadgroup float simdHist[NUM_SIMD_GROUPS][BINS_PER_BYTE]; // 8 KB
    threadgroup float stagingHist[HIST_PER_SIMD];               // 4 KB (downstream interface unchanged)

    const uint tid     = thread_index_in_threadgroup;
    const uint lane    = tid & (SIMD_SIZE - 1u);   // 0..31 within SIMD group
    const uint simd_id = tid >> 5u;                // 0..7 SIMD group index

    for (uint tile = 0u; tile < FEATURES_PER_PACK; tile++) {   // 4 tiles × 256 bins = 1024 bins
        const uint tile_base = tile * BINS_PER_BYTE;

        // --- Intra-SIMD reduction: 5 shuffle rounds, zero threadgroup barriers ---
        // Every thread loads privHist[tile_base + bin] for the same bin index
        // across all 32 lanes in the SIMD group, then butterfly-sums across lanes.
        // Each bin in [0, BINS_PER_BYTE) is handled by exactly one iteration of
        // this loop; the 32 threads in the SIMD group cover 32 different document
        // subsets but the same bin.
        for (uint bin = 0u; bin < BINS_PER_BYTE; bin++) {
            float val = privHist[tile_base + bin];

            // Butterfly XOR reduction: after 5 rounds all 32 lanes hold
            // the same value — the sum of this SIMD group's 32 threads for bin.
            val += simd_shuffle_xor(val, 16u);
            val += simd_shuffle_xor(val,  8u);
            val += simd_shuffle_xor(val,  4u);
            val += simd_shuffle_xor(val,  2u);
            val += simd_shuffle_xor(val,  1u);

            if (lane == 0u) {
                simdHist[simd_id][bin] = val;   // SIMD-leader stores per-group sum
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup); // barrier 1 of 2 per tile

        // --- Cross-SIMD fold: 256 threads each accumulate 8 SIMD-group values ---
        // Thread tid handles bin tid (256 threads map exactly to 256 bins per tile).
        // The 8-term sum runs in fixed simd_id order 0..7 → deterministic result.
        if (tid < BINS_PER_BYTE) {
            float sum = 0.0f;
            for (uint g = 0u; g < NUM_SIMD_GROUPS; g++) {
                sum += simdHist[g][tid];
            }
            stagingHist[tile_base + tid] = sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup); // barrier 2 of 2 per tile
    }
    // Total barriers: 2 per tile × 4 tiles = 8 (down from 255 in the serial fold).

    // Write results from stagingHist (fully reduced into thread 0's pass) to global
    const uint histBase = partIdx * numStats * totalBinFeatures + statIdx * totalBinFeatures;

    for (uint f = 0u; f < FEATURES_PER_PACK; f++) {
        const uint folds = foldCountsFlat[foldBase + f];
        const uint firstFold = firstFoldIndicesFlat[foldBase + f];

        for (uint bin = thread_index_in_threadgroup; bin < folds; bin += BLOCK_SIZE) {
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
