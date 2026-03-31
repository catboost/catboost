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
//
// Input names (in order):
//   compressedIndex, stats, docIndices, partOffsets, partSizes,
//   featureColumnIdx, lineSize, maxBlocksPerPart,
//   foldCounts, firstFoldIndices,
//   totalBinFeatures, numStats, totalNumDocs
//
// Output names: histogram
//
// Grid:   (maxBlocksPerPart, numPartitions, numStats)
// Thread: (256, 1, 1)
// ============================================================================

static const std::string kHistOneByteSource = R"metal(
    // Map grid to work
    const uint partIdx   = threadgroup_position_in_grid.y;
    const uint statIdx   = threadgroup_position_in_grid.z;
    const uint blockInPart = threadgroup_position_in_grid.x;

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

    // Threadgroup histogram using atomic_uint with CAS-based float add.
    // Metal supports atomic_uint in threadgroup but NOT atomic_float.
    // We use as_type<uint/float> bit-casting with compare_exchange for safe float accumulation.
    // Layout: [FEATURES_PER_PACK][BINS_PER_BYTE]
    threadgroup atomic_uint sharedHist[HIST_PER_SIMD];

    // Zero the histogram
    for (uint i = thread_index_in_threadgroup; i < HIST_PER_SIMD; i += BLOCK_SIZE) {
        atomic_store_explicit(&sharedHist[i], as_type<uint>(0.0f), memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Process documents — all threads accumulate via CAS-based float atomic add
    for (uint d = thread_index_in_threadgroup; d < myDocCount; d += BLOCK_SIZE) {
        // Use doc-index indirection for partition-sorted access
        const uint sortedPos = partOffset + myDocStart + d;
        const uint docIdx = docIndices[sortedPos];

        // Load packed features (4 one-byte features in one uint32)
        const uint packed = compressedIndex[docIdx * lineSize + featureColumnIdx];

        // Load the statistic for this document
        const float stat = stats[statIdx * totalNumDocs + docIdx];

        // Accumulate into histogram for each of the 4 features
        for (uint f = 0; f < FEATURES_PER_PACK; f++) {
            const uint bin = (packed >> (24 - 8 * f)) & 0xFF;
            if (bin < foldCounts[f] + 1) {
                const uint histIdx = f * BINS_PER_BYTE + bin;
                // CAS-based float atomic add on threadgroup atomic_uint
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
        const uint folds = foldCounts[f];
        const uint firstFold = firstFoldIndices[f];

        for (uint bin = thread_index_in_threadgroup; bin < folds; bin += BLOCK_SIZE) {
            const float val = as_type<float>(atomic_load_explicit(&sharedHist[f * BINS_PER_BYTE + bin + 1], memory_order_relaxed));
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
)metal";

}  // namespace KernelSources
}  // namespace NCatboostMlx
