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

    // Threadgroup histogram using atomic_uint with CAS-based float add.
    // Layout: [FEATURES_PER_PACK][BINS_PER_BYTE]
    threadgroup atomic_uint sharedHist[HIST_PER_SIMD];

    // Zero the histogram
    for (uint i = thread_index_in_threadgroup; i < HIST_PER_SIMD; i += BLOCK_SIZE) {
        atomic_store_explicit(&sharedHist[i], as_type<uint>(0.0f), memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Process documents — all threads accumulate via CAS-based float atomic add
    for (uint d = thread_index_in_threadgroup; d < myDocCount; d += BLOCK_SIZE) {
        const uint sortedPos = partOffset + myDocStart + d;
        const uint docIdx = docIndices[sortedPos];

        // Load packed features (4 one-byte features in one uint32)
        const uint packed = compressedIndex[docIdx * lineSize + featureColumnIdx];

        // Load the statistic for this document
        const float stat = stats[statIdx * totalNumDocs + docIdx];

        // Accumulate into histogram for each of the 4 features
        for (uint f = 0; f < FEATURES_PER_PACK; f++) {
            const uint bin = (packed >> (24 - 8 * f)) & 0xFF;
            if (bin < foldCountsFlat[foldBase + f] + 1) {
                const uint histIdx = f * BINS_PER_BYTE + bin;
                uint old_val = atomic_load_explicit(&sharedHist[histIdx], memory_order_relaxed);
                uint new_val;
                do {
                    new_val = as_type<uint>(as_type<float>(old_val) + stat);
                } while (!atomic_compare_exchange_weak_explicit(
                    &sharedHist[histIdx], &old_val, new_val,
                    memory_order_relaxed, memory_order_relaxed));
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write results to global histogram buffer
    const uint histBase = partIdx * numStats * totalBinFeatures + statIdx * totalBinFeatures;

    for (uint f = 0; f < FEATURES_PER_PACK; f++) {
        const uint folds = foldCountsFlat[foldBase + f];
        const uint firstFold = firstFoldIndicesFlat[foldBase + f];

        for (uint bin = thread_index_in_threadgroup; bin < folds; bin += BLOCK_SIZE) {
            const float val = as_type<float>(atomic_load_explicit(&sharedHist[f * BINS_PER_BYTE + bin + 1], memory_order_relaxed));
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
// Suffix-sum transform kernel — parallel SIMD-group version (TODO-008)
//
// Converts raw per-bin histogram counts into suffix sums (in-place) so that
// ordinal split scoring becomes O(1) per bin instead of O(bins).
//
// For ordinal feature with bins h[0..F-1]:
//   h'[b] = sum(h[b..F-1])  (reverse inclusive scan)
//   h'[folds-1] is intentionally left unwritten (set to 0 by init_value);
//   this matches the serial implementation and prevents the scorer from
//   selecting an all-right split that has no left-side documents.
//
// OneHot features are skipped (their bins are independent categories).
//
// Grid:   (numFeatures, numPartitions_times_approxDim, numStats)
// Thread: (32, 1, 1)  — one SIMD group per (feature, part, stat) triple
//
// Parallelism over folds using simd_prefix_inclusive_sum:
//   - Each threadgroup (32 lanes) handles one (feature, partition, stat).
//   - For folds <= 32: each active lane covers one bin. Load the bins in
//     reverse order (lane 0 = h[folds-1], lane 1 = h[folds-2], ...).
//     simd_prefix_inclusive_sum gives lane i the sum h[folds-1..folds-1-i].
//     Write back to bins folds-2 downto 0 (skip folds-1 per serial semantics).
//   - For folds > 32: multiple right-to-left passes of 32 bins each.
//     Carry the total from each chunk into the next chunk leftward.
//     Within each chunk the simd scan handles the intra-chunk reduction.
//
// Note on numerical precision:
//   The simd_prefix_inclusive_sum uses a hardware binary-tree reduction
//   which produces a different floating-point addition order than the serial
//   scalar loop. Results are mathematically equivalent but may differ in the
//   last ULP. The downstream scorer (FindBestSplitGPU) is tolerant of these
//   differences; final loss matches the serial version to float32 precision.
// ============================================================================

static const std::string kSuffixSumSource = R"metal(
    const uint lane    = thread_index_in_threadgroup;   // 0..31
    const uint featIdx = threadgroup_position_in_grid.x;
    const uint partIdx = threadgroup_position_in_grid.y;
    const uint statIdx = threadgroup_position_in_grid.z;

    if (featIdx >= numFeatures) return;

    // Skip one-hot features — their histogram entries are direct lookups
    if (featureIsOneHot[featIdx] != 0u) return;

    const uint folds = featureFolds[featIdx];
    if (folds <= 1u) return;

    const uint firstFold = featureFirstFold[featIdx];
    const uint base = partIdx * numStats * totalBinFeatures + statIdx * totalBinFeatures;

    // ---- Single-pass case: folds <= 32 ----
    // Lane i covers bin (folds-1-i). Load in reverse so prefix = suffix.
    if (folds <= 32u) {
        // Load this lane's bin (reversed).
        // Lanes beyond [0, folds-1] load 0 so they don't contribute.
        float val = (lane < folds) ? histogram[base + firstFold + (folds - 1u - lane)] : 0.0f;

        // simd_prefix_inclusive_sum(val) at lane i = sum(val[0..i])
        // Since val[0] = h[folds-1], val[1] = h[folds-2], ...,
        // result at lane i = h[folds-1] + h[folds-2] + ... + h[folds-1-i]
        //                   = h'[folds-1-i]  (suffix sum at bin folds-1-i)
        float suffixVal = simd_prefix_inclusive_sum(val);

        // Write: skip lane 0 (bin folds-1) per serial semantics.
        // Lane i >= 1 writes to bin (folds-1-i).
        if (lane >= 1u && lane < folds) {
            histogram_out[base + firstFold + (folds - 1u - lane)] = suffixVal;
        }
        return;
    }

    // ---- Multi-pass case: folds > 32 ----
    // Process 32 bins per pass, right-to-left.
    // carry = running total from all chunks to the right of the current chunk.
    float carry = 0.0f;

    // Number of complete 32-wide chunks plus possible partial first chunk
    // (we iterate right-to-left so the last chunk in the buffer is first).
    // total bins = folds; chunk i (0=rightmost) covers bins [folds-32*(i+1) .. folds-32*i - 1]
    // For a partial leftmost chunk, only the rightmost part of the lane range is active.

    const uint numChunks = (folds + 31u) / 32u;  // ceil(folds/32)

    for (uint chunk = 0u; chunk < numChunks; chunk++) {
        // Chunk 0 is the rightmost 32 bins: [folds-32, folds-1]
        // Chunk k covers bins starting at: binStart = (chunk < numChunks-1) ?
        //   folds - 32*(chunk+1) : 0
        uint chunkEnd;    // exclusive, one past last bin of this chunk
        uint chunkStart;  // inclusive first bin of this chunk

        chunkEnd   = folds - 32u * chunk;               // e.g. chunk 0 → folds
        chunkStart = (chunkEnd > 32u) ? (chunkEnd - 32u) : 0u;
        uint chunkSize = chunkEnd - chunkStart;           // ≤ 32

        // Lane i covers bin chunkStart + (chunkSize - 1 - lane) within the chunk
        // (reversed so prefix sum = suffix sum within chunk)
        uint binInChunk = (lane < chunkSize) ? (chunkSize - 1u - lane) : 0u;
        uint globalBin  = chunkStart + binInChunk;

        float val = (lane < chunkSize) ? histogram[base + firstFold + globalBin] : 0.0f;

        // Intra-chunk suffix sum via prefix on reversed data
        float suffixVal = simd_prefix_inclusive_sum(val) + carry;

        // carry for the next (leftward) chunk = total sum of this chunk
        // = suffixVal at lane (chunkSize-1), which is the leftmost active lane
        carry = simd_broadcast(suffixVal, chunkSize - 1u);

        // Write results:
        //   For the rightmost chunk (chunk==0): skip the last bin (bin folds-1)
        //     which corresponds to lane 0. For all other chunks: write all lanes.
        bool skipLastBin = (chunk == 0u) && (lane == 0u);
        if (!skipLastBin && lane < chunkSize) {
            histogram_out[base + firstFold + globalBin] = suffixVal;
        }
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
// Accumulates per-leaf gradient/hessian sums from all documents using
// threadgroup-local CAS-based atomic float add, then atomic writeback
// to global output.
//
// Grid:   (LEAF_BLOCK_SIZE * ceil(numDocs / LEAF_BLOCK_SIZE), 1, 1)
// Thread: (LEAF_BLOCK_SIZE, 1, 1)
// ============================================================================

static const std::string kLeafAccumHeader = R"metal(
#include <metal_stdlib>
using namespace metal;

constant constexpr uint LEAF_BLOCK_SIZE = 256;
constant constexpr uint MAX_LEAVES = 64;
constant constexpr uint MAX_APPROX_DIM = 10;
)metal";

static const std::string kLeafAccumSource = R"metal(
    // Shared accumulators: [approxDim][numLeaves][2] (grad, hess)
    // Max: 10 * 64 * 2 = 1280 entries = 5 KB
    threadgroup atomic_uint sharedSums[MAX_APPROX_DIM * MAX_LEAVES * 2];

    const uint totalShared = approxDim * numLeaves * 2;
    for (uint i = thread_index_in_threadgroup; i < totalShared; i += LEAF_BLOCK_SIZE) {
        atomic_store_explicit(&sharedSums[i], as_type<uint>(0.0f), memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint globalIdx = threadgroup_position_in_grid.x * LEAF_BLOCK_SIZE
                         + thread_index_in_threadgroup;

    if (globalIdx < numDocs) {
        const uint docIdx = globalIdx;
        const uint leaf = partitions[docIdx];

        if (leaf < numLeaves) {
            for (uint k = 0; k < approxDim; k++) {
                const float grad = gradients[k * numDocs + docIdx];
                const float hess = hessians[k * numDocs + docIdx];

                const uint gradIdx = k * numLeaves * 2 + leaf * 2;
                const uint hessIdx = gradIdx + 1;

                // CAS-based float atomic add for gradient
                uint old_val = atomic_load_explicit(&sharedSums[gradIdx], memory_order_relaxed);
                uint new_val;
                do {
                    new_val = as_type<uint>(as_type<float>(old_val) + grad);
                } while (!atomic_compare_exchange_weak_explicit(
                    &sharedSums[gradIdx], &old_val, new_val,
                    memory_order_relaxed, memory_order_relaxed));

                // CAS-based float atomic add for hessian
                old_val = atomic_load_explicit(&sharedSums[hessIdx], memory_order_relaxed);
                do {
                    new_val = as_type<uint>(as_type<float>(old_val) + hess);
                } while (!atomic_compare_exchange_weak_explicit(
                    &sharedSums[hessIdx], &old_val, new_val,
                    memory_order_relaxed, memory_order_relaxed));
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write threadgroup partial sums to global output using atomics
    for (uint i = thread_index_in_threadgroup; i < totalShared; i += LEAF_BLOCK_SIZE) {
        float val = as_type<float>(atomic_load_explicit(&sharedSums[i], memory_order_relaxed));
        if (abs(val) > 1e-20f) {
            uint remainder = i % (numLeaves * 2);
            uint k = i / (numLeaves * 2);
            uint leaf = remainder / 2;
            uint is_hess = remainder % 2;

            if (is_hess == 0) {
                device atomic_float* dst = (device atomic_float*)(gradSums + k * numLeaves + leaf);
                atomic_fetch_add_explicit(dst, val, memory_order_relaxed);
            } else {
                device atomic_float* dst = (device atomic_float*)(hessSums + k * numLeaves + leaf);
                atomic_fetch_add_explicit(dst, val, memory_order_relaxed);
            }
        }
    }
)metal";

}  // namespace KernelSources
}  // namespace NCatboostMlx
