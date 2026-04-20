#pragma once
// kernel_sources_t2_scratch.h — Sprint 22 D0 scratch: T2 sort-by-bin kernel sources
//
// SCRATCH-ONLY — NOT for production use.  This header exists solely to support the
// Sprint 22 D0 in-situ T2 integration probe in bench_boosting.cpp.
//
// Include guard: only included when bench_boosting.cpp is compiled with
//   -DCATBOOST_MLX_HISTOGRAM_T2=1
//
// Kernel sources are taken verbatim from the Sprint 21 D1-R2 micro-bench harness
// (docs/sprint21/scratch/t2/microbench_t2.cpp lines 286–459), which passed all
// sanity gates.  No changes to catboost/mlx/kernels/kernel_sources.h.
//
// Kill-switch: if S22-D0 ratio > 0.60, this file is deleted with no impact on
// production.  If D0 PASS, T2 kernel sources graduate to kernel_sources.h in D1.
//
// A1-G6 compliance: this file does NOT modify any existing production source.
// It is an additive scratch artifact, analogous to docs/sprint21/scratch/t2/.

#include <string>

namespace NCatboostMlx {
namespace KernelSources {

// ============================================================================
// T2-sort kernel source
//
// Counting sort of each partition's docs by feature-0 (top-byte) bin.
// Produces sortedDocs[] and binOffsets[] for T2-accum.
//
// Input names (must match exactly in metal_kernel() call):
//   compressedIndex, docIndices, partOffsets, partSizes,
//   featureColumnIndices, lineSize, maxBlocksPerPart, numGroups,
//   numPartitions, numStats, numTGs, maxPartDocs, totalNumDocs
//
// Output names: sortedDocs, binOffsets
// atomic_outputs = false (each TG writes to its own disjoint slot)
//
// Grid: (256 * maxBlocksPerPart * numGroups, numPartitions, numStats)
// Thread: (256, 1, 1)
// ============================================================================
static const std::string kT2SortSource = R"metal(
    // Same grid decomposition as kHistOneByteSource
    const uint tgX          = threadgroup_position_in_grid.x;
    const uint partIdx      = threadgroup_position_in_grid.y;
    const uint statIdx      = threadgroup_position_in_grid.z;
    const uint blockInPart  = tgX % maxBlocksPerPart;
    const uint groupIdx     = tgX / maxBlocksPerPart;

    if (groupIdx >= numGroups) return;
    if (blockInPart != 0) return;  // only one block per partition (maxBlocksPerPart=1)

    const uint featureColumnIdx = featureColumnIndices[groupIdx];
    const uint partOffset       = partOffsets[partIdx];
    const uint partSize         = partSizes[partIdx];
    const uint tid              = thread_index_in_threadgroup;

    if (partSize == 0) return;

    // Counting sort: step 1 — count docs per feature-0 bin
    // feature-0 bin = top byte of packed uint32 = bits [24..31]
    // Cap at 127 (folds <= 127 per DEC-016 T1 envelope; bin 0 = missing)
    threadgroup atomic_uint tgCounts[128];
    for (uint b = tid; b < 128u; b += 256u)
        atomic_store_explicit(&tgCounts[b], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < partSize; i += 256u) {
        const uint docIdx = docIndices[partOffset + i];
        const uint packed = compressedIndex[docIdx * lineSize + featureColumnIdx];
        const uint bin    = (packed >> 24u) & 0x7Fu;  // bits [24..30], 7-bit value
        atomic_fetch_add_explicit(&tgCounts[bin], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: exclusive prefix scan → tgOffsets (thread 0 only; partSize ≤ maxPartDocs)
    threadgroup uint tgOffsets[129];
    if (tid == 0) {
        uint acc = 0;
        for (uint b = 0; b < 128u; ++b) {
            tgOffsets[b] = acc;
            acc += atomic_load_explicit(&tgCounts[b], memory_order_relaxed);
        }
        tgOffsets[128] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3: scatter docs into per-slot sortedDocs[] using per-bin atomic cursors
    threadgroup atomic_uint tgCursors[128];
    for (uint b = tid; b < 128u; b += 256u)
        atomic_store_explicit(&tgCursors[b], tgOffsets[b], memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Per-(groupIdx, partIdx, statIdx) slot in sortedDocs[]
    const uint slotBase = (groupIdx * numPartitions * numStats + partIdx * numStats + statIdx)
                        * maxPartDocs;

    for (uint i = tid; i < partSize; i += 256u) {
        const uint docIdx = docIndices[partOffset + i];
        const uint packed = compressedIndex[docIdx * lineSize + featureColumnIdx];
        const uint bin    = (packed >> 24u) & 0x7Fu;
        const uint pos    = atomic_fetch_add_explicit(&tgCursors[bin], 1u, memory_order_relaxed);
        sortedDocs[slotBase + pos] = docIdx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write binOffsets for this TG
    const uint offBase = (groupIdx * numPartitions * numStats + partIdx * numStats + statIdx)
                       * 129u;
    for (uint b = tid; b <= 128u; b += 256u)
        binOffsets[offBase + b] = tgOffsets[b];
)metal";

// ============================================================================
// T2-accum kernel source
//
// Reads sorted docs from T2-sort, accumulates histogram using the same output
// layout as kHistOneByteSource (T1).
//
// Feature 0: pure bin-range scan (T2's core benefit — no simd_shuffle)
// Features 1–3: per-doc stride over sorted docs, atomic_fetch_add per bin
//
// BIN CONVENTION matches T1 exactly:
//   T1 output[histBase + firstFold + k] = sum for docs with raw bin = k + 1.
//   T2 writes raw bin b (b >= 1) to histogram[histBase + firstFold + b - 1].
//
// atomic_outputs = true: histogram output is device atomic<float>*
//   required for atomic_fetch_add_explicit casts.
//
// Input names:
//   sortedDocs, binOffsets, compressedIndex, stats,
//   featureColumnIndices, foldCountsFlat, firstFoldIndicesFlat,
//   lineSize, maxBlocksPerPart, numGroups,
//   numPartitions, numStats, numTGs, maxPartDocs, totalBinFeatures, totalNumDocs
//
// Output names: histogram
// Grid/Thread: same as T2-sort
// ============================================================================
static const std::string kT2AccumSource = R"metal(
    const uint tgX          = threadgroup_position_in_grid.x;
    const uint partIdx      = threadgroup_position_in_grid.y;
    const uint statIdx      = threadgroup_position_in_grid.z;
    const uint blockInPart  = tgX % maxBlocksPerPart;
    const uint groupIdx     = tgX / maxBlocksPerPart;

    if (groupIdx >= numGroups) return;
    if (blockInPart != 0) return;

    const uint featureColumnIdx = featureColumnIndices[groupIdx];
    const uint foldBase         = groupIdx * 4u;
    const uint tid              = thread_index_in_threadgroup;

    // Locate this TG's sorted docs and bin offsets
    const uint slotBase = (groupIdx * numPartitions * numStats + partIdx * numStats + statIdx)
                        * maxPartDocs;
    const uint offBase  = (groupIdx * numPartitions * numStats + partIdx * numStats + statIdx)
                        * 129u;

    // Output histogram base (same layout as kHistOneByteSource)
    const uint histBase = partIdx * numStats * totalBinFeatures
                        + statIdx * totalBinFeatures;

    const uint totalDocsInPart = binOffsets[offBase + 128u];

    // All 4 features: accumulate over sorted doc list using atomic_fetch_add.
    //
    // BIN CONVENTION — must match kHistOneByteSource (T1) exactly:
    //   T1 output[firstFold + k] = sum of stats for docs with RAW BIN = k + 1.
    //   T2 writes docs with raw bin b (b >= 1) to histogram[firstFold + b - 1].
    for (uint f = 0u; f < 4u; ++f) {
        const uint foldCount  = foldCountsFlat[foldBase + f];
        const uint firstFold  = firstFoldIndicesFlat[foldBase + f];
        if (foldCount == 0u) continue;  // padding slot in last group

        if (f == 0u) {
            // Feature 0: bin-range scan using sorted order — no simd_shuffle.
            // Sort key = raw bin (0..127), indexed by tgOffsets.
            // Skip bin 0 (missing). For bin b = 1..foldCount:
            //   sum docs in range [binOffsets[offBase + b], binOffsets[offBase + b + 1])
            //   write to histogram[histBase + firstFold + b - 1]
            for (uint b = tid + 1u; b <= foldCount; b += 256u) {
                const uint start = binOffsets[offBase + b];
                const uint end   = binOffsets[offBase + b + 1u];
                float sum = 0.0f;
                for (uint i = start; i < end; ++i) {
                    const uint docIdx = sortedDocs[slotBase + i];
                    sum += stats[statIdx * totalNumDocs + docIdx];
                }
                // Single-writer per bin (b assigned to this thread via stride)
                device atomic_float* dst = (device atomic_float*)(
                    histogram + histBase + firstFold + b - 1u);
                atomic_fetch_add_explicit(dst, sum, memory_order_relaxed);
            }
        } else {
            // Features 1–3: stride over sorted docs (sorted by feature-0 bin).
            // Per-doc: get feature-f raw bin, skip bin 0 (missing), write to firstFold + b - 1.
            for (uint i = tid; i < totalDocsInPart; i += 256u) {
                const uint docIdx = sortedDocs[slotBase + i];
                const uint packed = compressedIndex[docIdx * lineSize + featureColumnIdx];
                const float s     = stats[statIdx * totalNumDocs + docIdx];
                const uint b      = (packed >> (24u - 8u * f)) & 0xFFu;
                if (b >= 1u && b <= foldCount) {
                    device atomic_float* dst = (device atomic_float*)(
                        histogram + histBase + firstFold + b - 1u);
                    atomic_fetch_add_explicit(dst, s, memory_order_relaxed);
                }
            }
        }
    }
)metal";

}  // namespace KernelSources
}  // namespace NCatboostMlx
