// hist.metal — Histogram computation kernel for CatBoost-MLX.
// Translates catboost/cuda/methods/greedy_subsets_searcher/kernel/hist_one_byte.cu to Metal.
//
// This kernel computes per-feature, per-bin gradient/weight histograms from:
//   - Compressed feature index (uint32, 4 one-byte features packed per word)
//   - Per-document statistics (gradients, weights)
//   - Document partition assignments (leaf membership)
//
// Algorithm:
//   1. Each threadgroup processes a stripe of documents for one partition
//   2. Accumulate bin statistics in threadgroup memory (no atomics within group)
//   3. SIMD-synchronized writes avoid bank conflicts
//   4. Reduce across SIMD groups within the threadgroup
//   5. Write results to global histogram buffer (atomic if multiple blocks)

#include <metal_stdlib>
using namespace metal;

// Constants
constant constexpr uint SIMD_SIZE = 32;
constant constexpr uint FEATURES_PER_PACK = 4;  // 4 one-byte features per uint32
constant constexpr uint BINS_PER_BYTE = 256;

// --------------------------------------------------------------------------
// One-byte feature histogram kernel
//
// Grid:   (docBlocks * featureGroupBlocks, numPartitions, numStats)
// Thread: (BLOCK_SIZE, 1, 1) where BLOCK_SIZE = 256
//
// Each threadgroup:
//   - Processes up to FEATURES_PER_PACK features (one uint32 column)
//   - Iterates over a stripe of documents in the partition
//   - Accumulates into threadgroup histogram of size [4 features * 256 bins]
// --------------------------------------------------------------------------

kernel void histogram_one_byte_features(
    // Compressed feature index: [numDocs, numUi32PerDoc]
    const device uint*       compressedIndex  [[buffer(0)]],
    // Per-document statistic (gradient or weight): [numDocs]
    const device float*      stats            [[buffer(1)]],
    // Partition table: [numPartitions] with {offset, size}
    const device uint*       partOffsets      [[buffer(2)]],
    const device uint*       partSizes        [[buffer(3)]],
    // Feature column index (which ui32 column this group reads)
    constant uint&           featureColumnIdx [[buffer(4)]],
    // Line size = numUi32PerDoc (stride between documents)
    constant uint&           lineSize         [[buffer(5)]],
    // Number of docs blocks per partition
    constant uint&           maxBlocksPerPart [[buffer(6)]],
    // Per-feature fold counts: [4] for the 4 features in this pack
    const device uint*       foldCounts       [[buffer(7)]],
    // Per-feature first fold index: [4]
    const device uint*       firstFoldIndices [[buffer(8)]],
    // Total bin-features in histogram
    constant uint&           totalBinFeatures [[buffer(9)]],
    // Number of statistics (1 for gradient-only, 2 for gradient+weight)
    constant uint&           numStats         [[buffer(10)]],
    // Total number of documents (stride for stats array)
    constant uint&           totalNumDocs     [[buffer(11)]],
    // Output histogram: [numPartitions, numStats, totalBinFeatures]
    device float*            histogram        [[buffer(12)]],

    // Metal thread identifiers
    uint3 threadgroup_position_in_grid   [[threadgroup_position_in_grid]],
    uint3 threadgroups_per_grid          [[threadgroups_per_grid]],
    uint  thread_index_in_threadgroup    [[thread_index_in_threadgroup]],
    uint  simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
    uint  thread_index_in_simdgroup      [[thread_index_in_simdgroup]]
) {
    // Map grid to work
    const uint partIdx   = threadgroup_position_in_grid.y;
    const uint statIdx   = threadgroup_position_in_grid.z;
    const uint blockInPart = threadgroup_position_in_grid.x;

    // Load partition bounds
    const uint partOffset = partOffsets[partIdx];
    const uint partSize   = partSizes[partIdx];

    if (partSize == 0) return;

    // Check if this block is active (may have more blocks than docs)
    constexpr uint BLOCK_SIZE = 256;
    const uint docsPerBlock = (partSize + maxBlocksPerPart - 1) / maxBlocksPerPart;
    const uint myDocStart = blockInPart * docsPerBlock;
    if (myDocStart >= partSize) return;
    const uint myDocEnd = min(myDocStart + docsPerBlock, partSize);
    const uint myDocCount = myDocEnd - myDocStart;

    // Threadgroup histogram: [SIMD_GROUPS][FEATURES_PER_PACK][BINS_PER_BYTE]
    // Each SIMD group gets its own histogram slice to avoid conflicts.
    constexpr uint NUM_SIMD_GROUPS = BLOCK_SIZE / SIMD_SIZE;  // 8
    constexpr uint HIST_PER_SIMD = FEATURES_PER_PACK * BINS_PER_BYTE;  // 4 * 256 = 1024
    constexpr uint TOTAL_HIST_SIZE = NUM_SIMD_GROUPS * HIST_PER_SIMD;  // 8 * 1024 = 8192

    threadgroup float sharedHist[TOTAL_HIST_SIZE];

    // Zero the histogram
    for (uint i = thread_index_in_threadgroup; i < TOTAL_HIST_SIZE; i += BLOCK_SIZE) {
        sharedHist[i] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each SIMD group gets its own histogram slice
    const uint simdHistBase = simdgroup_index_in_threadgroup * HIST_PER_SIMD;

    // Process documents
    for (uint d = thread_index_in_threadgroup; d < myDocCount; d += BLOCK_SIZE) {
        const uint docIdx = partOffset + myDocStart + d;

        // Load packed features (4 one-byte features in one uint32)
        const uint packed = compressedIndex[docIdx * lineSize + featureColumnIdx];

        // Load the statistic for this document
        // Stats layout: [numStats, totalNumDocs] — stride is totalNumDocs, not partition size
        const float stat = stats[statIdx * totalNumDocs + docIdx];

        // Accumulate into per-SIMD histogram for each of the 4 features
        for (uint f = 0; f < FEATURES_PER_PACK; f++) {
            const uint bin = (packed >> (24 - 8 * f)) & 0xFF;

            // Only accumulate if bin is within valid range
            if (bin < foldCounts[f] + 1) {
                const uint histIdx = simdHistBase + f * BINS_PER_BYTE + bin;
                sharedHist[histIdx] += stat;
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce across SIMD groups: sum all SIMD group histograms into group 0
    // Each thread reduces one histogram entry across all SIMD groups
    for (uint i = thread_index_in_threadgroup; i < HIST_PER_SIMD; i += BLOCK_SIZE) {
        float sum = 0.0f;
        for (uint s = 0; s < NUM_SIMD_GROUPS; s++) {
            sum += sharedHist[s * HIST_PER_SIMD + i];
        }
        sharedHist[i] = sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write results to global histogram buffer
    // Layout: histogram[partIdx * numStats * totalBinFeatures + statIdx * totalBinFeatures + binFeatureIdx]
    const uint histBase = partIdx * numStats * totalBinFeatures + statIdx * totalBinFeatures;

    for (uint f = 0; f < FEATURES_PER_PACK; f++) {
        const uint folds = foldCounts[f];
        const uint firstFold = firstFoldIndices[f];

        for (uint bin = thread_index_in_threadgroup; bin < folds; bin += BLOCK_SIZE) {
            const float val = sharedHist[f * BINS_PER_BYTE + bin + 1];  // bin 0 is "less than first border"
            if (abs(val) > 1e-20f) {
                // Use atomic add when multiple blocks contribute to same partition
                if (maxBlocksPerPart > 1) {
                    // Metal 3.0+ supports atomic float add in device memory
                    device atomic_float* dst = (device atomic_float*)(histogram + histBase + firstFold + bin);
                    atomic_fetch_add_explicit(dst, val, memory_order_relaxed);
                } else {
                    histogram[histBase + firstFold + bin] = val;
                }
            }
        }
    }
}

// --------------------------------------------------------------------------
// Half-byte feature histogram kernel (for features with <= 16 bins)
// Packs 8 features per uint32 (4 bits each)
// --------------------------------------------------------------------------

kernel void histogram_half_byte_features(
    const device uint*       compressedIndex  [[buffer(0)]],
    const device float*      stats            [[buffer(1)]],
    const device uint*       partOffsets      [[buffer(2)]],
    const device uint*       partSizes        [[buffer(3)]],
    constant uint&           featureColumnIdx [[buffer(4)]],
    constant uint&           lineSize         [[buffer(5)]],
    constant uint&           maxBlocksPerPart [[buffer(6)]],
    const device uint*       foldCounts       [[buffer(7)]],
    const device uint*       firstFoldIndices [[buffer(8)]],
    constant uint&           totalBinFeatures [[buffer(9)]],
    constant uint&           numStats         [[buffer(10)]],
    constant uint&           totalNumDocs     [[buffer(11)]],
    device float*            histogram        [[buffer(12)]],

    uint3 threadgroup_position_in_grid   [[threadgroup_position_in_grid]],
    uint3 threadgroups_per_grid          [[threadgroups_per_grid]],
    uint  thread_index_in_threadgroup    [[thread_index_in_threadgroup]],
    uint  simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint BLOCK_SIZE = 256;
    constexpr uint FEATURES_PER_PACK_HALF = 8;  // 8 half-byte features per uint32
    constexpr uint BINS_PER_NIBBLE = 16;
    constexpr uint NUM_SIMD_GROUPS = BLOCK_SIZE / SIMD_SIZE;
    constexpr uint HIST_PER_SIMD = FEATURES_PER_PACK_HALF * BINS_PER_NIBBLE;  // 128
    constexpr uint TOTAL_HIST_SIZE = NUM_SIMD_GROUPS * HIST_PER_SIMD;

    const uint partIdx   = threadgroup_position_in_grid.y;
    const uint statIdx   = threadgroup_position_in_grid.z;
    const uint blockInPart = threadgroup_position_in_grid.x;

    const uint partOffset = partOffsets[partIdx];
    const uint partSize   = partSizes[partIdx];
    if (partSize == 0) return;

    const uint docsPerBlock = (partSize + maxBlocksPerPart - 1) / maxBlocksPerPart;
    const uint myDocStart = blockInPart * docsPerBlock;
    if (myDocStart >= partSize) return;
    const uint myDocEnd = min(myDocStart + docsPerBlock, partSize);
    const uint myDocCount = myDocEnd - myDocStart;

    threadgroup float sharedHist[TOTAL_HIST_SIZE];

    for (uint i = thread_index_in_threadgroup; i < TOTAL_HIST_SIZE; i += BLOCK_SIZE) {
        sharedHist[i] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint simdHistBase = simdgroup_index_in_threadgroup * HIST_PER_SIMD;

    for (uint d = thread_index_in_threadgroup; d < myDocCount; d += BLOCK_SIZE) {
        const uint docIdx = partOffset + myDocStart + d;
        const uint packed = compressedIndex[docIdx * lineSize + featureColumnIdx];
        const float stat = stats[statIdx * totalNumDocs + docIdx];

        for (uint f = 0; f < FEATURES_PER_PACK_HALF; f++) {
            const uint bin = (packed >> (28 - 4 * f)) & 0xF;
            if (bin < foldCounts[f] + 1) {
                sharedHist[simdHistBase + f * BINS_PER_NIBBLE + bin] += stat;
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce across SIMD groups
    for (uint i = thread_index_in_threadgroup; i < HIST_PER_SIMD; i += BLOCK_SIZE) {
        float sum = 0.0f;
        for (uint s = 0; s < NUM_SIMD_GROUPS; s++) {
            sum += sharedHist[s * HIST_PER_SIMD + i];
        }
        sharedHist[i] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write to global
    const uint histBase = partIdx * numStats * totalBinFeatures + statIdx * totalBinFeatures;
    for (uint f = 0; f < FEATURES_PER_PACK_HALF; f++) {
        const uint folds = foldCounts[f];
        const uint firstFold = firstFoldIndices[f];
        for (uint bin = thread_index_in_threadgroup; bin < folds; bin += BLOCK_SIZE) {
            const float val = sharedHist[f * BINS_PER_NIBBLE + bin + 1];
            if (abs(val) > 1e-20f) {
                if (maxBlocksPerPart > 1) {
                    device atomic_float* dst = (device atomic_float*)(histogram + histBase + firstFold + bin);
                    atomic_fetch_add_explicit(dst, val, memory_order_relaxed);
                } else {
                    histogram[histBase + firstFold + bin] = val;
                }
            }
        }
    }
}

// --------------------------------------------------------------------------
// Binary feature histogram kernel (for features with 2 values)
// 32 binary features per uint32 (1 bit each)
// --------------------------------------------------------------------------

kernel void histogram_binary_features(
    const device uint*       compressedIndex  [[buffer(0)]],
    const device float*      stats            [[buffer(1)]],
    const device uint*       partOffsets      [[buffer(2)]],
    const device uint*       partSizes        [[buffer(3)]],
    constant uint&           featureColumnIdx [[buffer(4)]],
    constant uint&           lineSize         [[buffer(5)]],
    constant uint&           maxBlocksPerPart [[buffer(6)]],
    constant uint&           numBinaryFeatures [[buffer(7)]],
    const device uint*       firstFoldIndices [[buffer(8)]],
    constant uint&           totalBinFeatures [[buffer(9)]],
    constant uint&           numStats         [[buffer(10)]],
    constant uint&           totalNumDocs     [[buffer(11)]],
    device float*            histogram        [[buffer(12)]],

    uint3 threadgroup_position_in_grid   [[threadgroup_position_in_grid]],
    uint  thread_index_in_threadgroup    [[thread_index_in_threadgroup]],
    uint  simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint BLOCK_SIZE = 256;
    constexpr uint NUM_SIMD_GROUPS = BLOCK_SIZE / SIMD_SIZE;

    const uint partIdx   = threadgroup_position_in_grid.y;
    const uint statIdx   = threadgroup_position_in_grid.z;
    const uint blockInPart = threadgroup_position_in_grid.x;

    const uint partOffset = partOffsets[partIdx];
    const uint partSize   = partSizes[partIdx];
    if (partSize == 0) return;

    const uint docsPerBlock = (partSize + maxBlocksPerPart - 1) / maxBlocksPerPart;
    const uint myDocStart = blockInPart * docsPerBlock;
    if (myDocStart >= partSize) return;
    const uint myDocEnd = min(myDocStart + docsPerBlock, partSize);
    const uint myDocCount = myDocEnd - myDocStart;

    // For binary features: histogram is just [SIMD_GROUPS][32] (one sum per bit)
    constexpr uint MAX_BINARY = 32;
    constexpr uint TOTAL_HIST_SIZE = NUM_SIMD_GROUPS * MAX_BINARY;

    threadgroup float sharedHist[TOTAL_HIST_SIZE];

    for (uint i = thread_index_in_threadgroup; i < TOTAL_HIST_SIZE; i += BLOCK_SIZE) {
        sharedHist[i] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint simdHistBase = simdgroup_index_in_threadgroup * MAX_BINARY;

    for (uint d = thread_index_in_threadgroup; d < myDocCount; d += BLOCK_SIZE) {
        const uint docIdx = partOffset + myDocStart + d;
        const uint packed = compressedIndex[docIdx * lineSize + featureColumnIdx];
        const float stat = stats[statIdx * totalNumDocs + docIdx];

        for (uint f = 0; f < numBinaryFeatures; f++) {
            const uint bit = (packed >> (31 - f)) & 1;
            if (bit) {
                sharedHist[simdHistBase + f] += stat;
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce
    for (uint i = thread_index_in_threadgroup; i < MAX_BINARY; i += BLOCK_SIZE) {
        float sum = 0.0f;
        for (uint s = 0; s < NUM_SIMD_GROUPS; s++) {
            sum += sharedHist[s * MAX_BINARY + i];
        }
        sharedHist[i] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write to global
    const uint histBase = partIdx * numStats * totalBinFeatures + statIdx * totalBinFeatures;
    for (uint f = thread_index_in_threadgroup; f < numBinaryFeatures; f += BLOCK_SIZE) {
        const float val = sharedHist[f];
        if (abs(val) > 1e-20f) {
            const uint firstFold = firstFoldIndices[f];
            if (maxBlocksPerPart > 1) {
                device atomic_float* dst = (device atomic_float*)(histogram + histBase + firstFold);
                atomic_fetch_add_explicit(dst, val, memory_order_relaxed);
            } else {
                histogram[histBase + firstFold] = val;
            }
        }
    }
}
