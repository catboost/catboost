#pragma once

// ============================================================================
// G1 gain-dump + Path 5 reconstruction kernels — Sprint 25 DEC-026 G1 ONLY
//
// This header is NOT part of production. It defines two kernel sources used
// only by `benchmarks/sprint25/g1/g1_gain_dump.cpp`:
//
//   1. `kScoreSplitsDumpSource`
//      - Identical per-candidate GAIN computation to kScoreSplitsLookupSource
//        (catboost/mlx/kernels/kernel_sources.h).
//      - SKIPS the threadgroup argmax reduction and writes a dense per-bin
//        GAIN tensor: gainOut[globalIdx] = totalGain, featOut[globalIdx] =
//        featIdx, binOut[globalIdx] = binInFeature.
//      - Output length = totalBinFeatures (one entry per candidate at the
//        current partition set).  The kernel is invoked once per depth level;
//        each dispatch produces the full candidate-GAIN landscape for a
//        single (iter, depth-level) inside RunIteration.
//
//   2. `kT2AccumPath5Source`
//      - Pre-v5 reconstruction: feature-0 uses the bin-range scan over
//        sortedDocs (the topology that produces Value B at config #8),
//        features 1-3 use int-atomic fixed-point accumulation (SCALE=2^30)
//        to make features 1-3 deterministic.
//      - Reads from sortedDocs + binOffsets produced by a deterministic
//        T2-sort (pre-v5 serial-scatter; see `kT2SortPath5Source` below).
//      - Histogram output is the same atomic_float* layout as production so
//        the rest of the graph (suffix-sum, scoring) is unchanged.
//
//   3. `kT2SortPath5Source`
//      - Verbatim copy of the pre-v5 deterministic serial-scatter T2-sort
//        kernel from master `9f3b99c7d2` (kernel_sources.h lines 1008-1108
//        at that commit).  Included here so the G1 binary does not depend
//        on whichever version of kT2SortSource lives in production
//        kernel_sources.h at the S25 tip.
//
// Constants (T2_BIN_CAP, BIN_OFFSETS_STRIDE, FEATURES_PER_PACK, etc.) are
// defined in KernelSources::kHistHeader (catboost/mlx/kernels/kernel_sources.h)
// and shared with the production kernels — we reuse that header verbatim.
// ============================================================================

#include <string>

namespace G1Kernels {

// ============================================================================
// kScoreSplitsDumpSource — per-candidate GAIN dump
//
// Signature inputs (must match metal_kernel input_names order):
//   histogram, partTotalSum, partTotalWeight,
//   featureFirstFold, featureFolds, featureIsOneHot, binToFeature,
//   numFeatures, totalBinFeatures, numStats, l2RegLambda,
//   numPartitions, approxDim
//
// Outputs:
//   gainOut[totalBinFeatures] : float32 — per-candidate GAIN
//   featOut[totalBinFeatures] : uint32  — featureIdx for this candidate
//   binOut [totalBinFeatures] : uint32  — binInFeature for this candidate
//
// Notes on semantic equivalence with kScoreSplitsLookupSource:
//   - Identical per-candidate GAIN arithmetic (same suffix-sum-transformed
//     histogram input, same (sumLeft^2/(wLeft+l2) + sumRight^2/(wRight+l2)
//     - totalSum^2/(totalWeight+l2)) aggregation across partitions × approxDim).
//   - Invalid candidates (weightLeft < 1e-15 or weightRight < 1e-15 in every
//     partition × k) yield totalGain = 0.0f here; the production kernel
//     initialises myGain = -INFINITY and only writes bestScores when some
//     partition contributes a finite term.  The Tail-1/Tail-2 analysis
//     depends only on GAINs that the production kernel would have considered
//     (i.e. GAIN > -INFINITY in production = GAIN != 0.0f here or featIdx
//     valid), so the -INFINITY sentinel is collapsed to 0.0f for a
//     well-defined float tensor.
//   - featIdx/binInFeature follow the same binToFeature lookup and
//     `binInFeature = globalIdx - firstFold` formula as the production
//     lookup kernel.
//
// Grid:   (SCORE_BLOCK_SIZE * numBlocks, 1, 1)   with numBlocks = ceil(totalBinFeatures/256)
// Thread: (SCORE_BLOCK_SIZE, 1, 1)
// ============================================================================
static const std::string kScoreSplitsDumpSource = R"metal(
    const uint globalIdx = threadgroup_position_in_grid.x * SCORE_BLOCK_SIZE
                         + thread_index_in_threadgroup;

    if (globalIdx >= totalBinFeatures) return;

    // O(1) feature lookup — matches kScoreSplitsLookupSource.
    const uint featIdx      = binToFeature[globalIdx];
    const uint firstFold    = featureFirstFold[featIdx];
    const uint binInFeature = globalIdx - firstFold;

    float totalGain = 0.0f;
    bool  anyValid  = false;

    for (uint k = 0; k < approxDim; k++) {
        const uint dimHistBase  = k * numPartitions * numStats * totalBinFeatures;
        const uint dimStatsBase = k * numPartitions;

        for (uint p = 0; p < numPartitions; p++) {
            const float totalSum    = partTotalSum[dimStatsBase + p];
            const float totalWeight = partTotalWeight[dimStatsBase + p];

            const uint histBase = dimHistBase + p * numStats * totalBinFeatures;

            float sumRight    = histogram[histBase + firstFold + binInFeature];
            float weightRight = 0.0f;
            if (numStats > 1u) {
                weightRight = histogram[histBase + totalBinFeatures + firstFold + binInFeature];
            }

            float sumLeft    = totalSum    - sumRight;
            float weightLeft = totalWeight - weightRight;

            if (weightLeft < 1e-15f || weightRight < 1e-15f) continue;

            totalGain += (sumLeft  * sumLeft)  / (weightLeft  + l2RegLambda)
                       + (sumRight * sumRight) / (weightRight + l2RegLambda)
                       - (totalSum * totalSum) / (totalWeight + l2RegLambda);
            anyValid = true;
        }
    }

    // Collapse -INFINITY sentinel to 0.0 so downstream host-side decode is well-defined.
    // The flag `anyValid` is not written; analysis compares GAINs directly, and
    // an all-invalid candidate (no partition contributes) is effectively GAIN = 0,
    // which cannot win against any positive-GAIN candidate in the argmax.
    gainOut[globalIdx] = anyValid ? totalGain : 0.0f;
    featOut[globalIdx] = featIdx;
    binOut [globalIdx] = binInFeature;
)metal";

// ============================================================================
// kT2SortPath5Source — deterministic serial-scatter T2-sort (pre-v5)
//
// VERBATIM reconstruction of the kT2SortSource from master `9f3b99c7d2`
// (kernel_sources.h lines 1008-1108 at that commit), with the DEC-023 v2
// serial-scatter fix applied.  This is the "deterministic T2-sort" used by
// Path 5 at S24 D0 before v5 removed the bin-range scan path entirely.
//
// The only input difference from production kT2SortSource is the output
// name — the caller registers this kernel separately from production to
// avoid MLX kernel-cache collisions.
// ============================================================================
static const std::string kT2SortPath5Source = R"metal(
    const uint tgX          = threadgroup_position_in_grid.x;
    const uint partIdx      = threadgroup_position_in_grid.y;
    const uint statIdx      = threadgroup_position_in_grid.z;
    const uint blockInPart  = tgX % maxBlocksPerPart;
    const uint groupIdx     = tgX / maxBlocksPerPart;

    if (groupIdx >= numGroups) return;
    if (blockInPart != 0) return;

    const uint featureColumnIdx = featureColumnIndices[groupIdx];
    const uint partOffset       = partOffsets[partIdx];
    const uint partSize         = partSizes[partIdx];
    const uint tid              = thread_index_in_threadgroup;

    if (partSize == 0) return;

    // Step 1 — count docs per feature-0 bin.
    threadgroup atomic_uint tgCounts[T2_BIN_CAP];
    for (uint b = tid; b < T2_BIN_CAP; b += BLOCK_SIZE)
        atomic_store_explicit(&tgCounts[b], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < partSize; i += BLOCK_SIZE) {
        const uint docIdx = docIndices[partOffset + i];
        const uint packed = compressedIndex[docIdx * lineSize + featureColumnIdx];
        const uint bin    = (packed >> 24u) & 0x7Fu;
        atomic_fetch_add_explicit(&tgCounts[bin], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2 — exclusive prefix scan → tgOffsets (thread 0).
    threadgroup uint tgOffsets[BIN_OFFSETS_STRIDE];
    if (tid == 0) {
        uint acc = 0;
        for (uint b = 0; b < T2_BIN_CAP; ++b) {
            tgOffsets[b] = acc;
            acc += atomic_load_explicit(&tgCounts[b], memory_order_relaxed);
        }
        tgOffsets[T2_BIN_CAP] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3 — deterministic serial scatter (DEC-023 v2 fix).  Thread 0 walks
    // the input in order i = 0..partSize-1; within-bin doc order in sortedDocs[]
    // matches the input partition order, run-to-run.
    threadgroup atomic_uint tgCursors[T2_BIN_CAP];
    for (uint b = tid; b < T2_BIN_CAP; b += BLOCK_SIZE)
        atomic_store_explicit(&tgCursors[b], tgOffsets[b], memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint slotBase = (groupIdx * numStats + statIdx) * totalNumDocs + partOffsets[partIdx];

    if (tid == 0) {
        for (uint i = 0; i < partSize; ++i) {
            const uint docIdx = docIndices[partOffset + i];
            const uint packed = compressedIndex[docIdx * lineSize + featureColumnIdx];
            const uint bin    = (packed >> 24u) & 0x7Fu;
            const uint pos    = atomic_fetch_add_explicit(&tgCursors[bin], 1u,
                                                          memory_order_relaxed);
            sortedDocs[slotBase + pos] = docIdx;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint offBase = (groupIdx * numPartitions * numStats + partIdx * numStats + statIdx)
                       * BIN_OFFSETS_STRIDE;
    for (uint b = tid; b <= T2_BIN_CAP; b += BLOCK_SIZE)
        binOffsets[offBase + b] = tgOffsets[b];
)metal";

// ============================================================================
// kT2AccumPath5Source — Path 5 deterministic accumulation (Value-B producing)
//
// Feature 0: bin-range scan over sortedDocs (pre-v5 topology, deterministic
//            because the sort above is serial-scatter and the inner loop is
//            sequential; single writer per bin via TG-stride ownership).
//
// Features 1-3: stride over sortedDocs with int-atomic fixed-point
//            accumulation.  Each per-doc contribution `s = stats[...]` is
//            converted to an int32 via `round(s * SCALE)` where SCALE = 2^30,
//            atomically added to a parallel uint32 histogram layout, and the
//            result converted back to float at writeback.
//
// Why int fixed-point is deterministic:
//   - Integer addition is associative and commutative; any thread-schedule
//     order yields the same 32-bit sum.
//   - At SCALE = 2^30 the LSB is 2^-30 ≈ 9.3e-10, well below the 1-2 ULP/bin
//     drift signal we are measuring (FP32 gap at 0.48... is ~1.44e-8/ULP).
//   - Range: |s| in bench_boosting is bounded by |residual|·|hess|, well
//     within 2^32/SCALE ≈ 4.0 before overflow.  Partition sums ≤ partSize
//     ·max|s|; at config #8 partSize ≤ 156 and max|s| ≈ 1.0 at iter 0.
//     Fixed-point headroom: 2^31/SCALE = 2.0; sums of residuals across a
//     partition stay well within that bound for the 50-iter training
//     trajectory (residuals shrink after iter 1).
//
// Output histogram is float32 (atomic_float* per production layout).  The
// int-atomic scratch is threadgroup-local; results are converted and emitted
// via atomic_fetch_add on the shared float histogram at writeback.
//
// BIN CONVENTION (unchanged from kT2AccumSource):
//   T1 output[firstFold + k] = sum for docs with raw bin = k + 1.
//   Path 5 writes docs with raw bin b (b >= 1) to histogram[firstFold + b - 1].
//
// Input names (ORDER MATTERS — matches GetT2AccumPath5Kernel() registration):
//   sortedDocs, binOffsets, compressedIndex, stats,
//   featureColumnIndices, foldCountsFlat, firstFoldIndicesFlat,
//   partOffsets,
//   lineSize, maxBlocksPerPart, numGroups,
//   numPartitions, numStats,
//   totalBinFeatures, totalNumDocs
//
// Output: histogram (atomic_float*)
// Grid/Thread: same as kT2SortPath5Source.
// ============================================================================
static const std::string kT2AccumPath5Source = R"metal(
    const uint tgX          = threadgroup_position_in_grid.x;
    const uint partIdx      = threadgroup_position_in_grid.y;
    const uint statIdx      = threadgroup_position_in_grid.z;
    const uint blockInPart  = tgX % maxBlocksPerPart;
    const uint groupIdx     = tgX / maxBlocksPerPart;

    if (groupIdx >= numGroups) return;
    if (blockInPart != 0) return;

    const uint featureColumnIdx = featureColumnIndices[groupIdx];
    const uint foldBase         = groupIdx * FEATURES_PER_PACK;
    const uint tid              = thread_index_in_threadgroup;

    const uint slotBase = (groupIdx * numStats + statIdx) * totalNumDocs + partOffsets[partIdx];
    const uint offBase  = (groupIdx * numPartitions * numStats + partIdx * numStats + statIdx)
                        * BIN_OFFSETS_STRIDE;
    const uint histBase = partIdx * numStats * totalBinFeatures
                        + statIdx * totalBinFeatures;

    const uint totalDocsInPart = binOffsets[offBase + T2_BIN_CAP];
    if (totalDocsInPart == 0) return;

    // Int fixed-point scale: 2^30.  LSB = 2^-30 ≈ 9.3e-10 (below FP32 ULP
    // at the relevant loss magnitudes).  Signed values are mapped via
    // bias-addition then atomic-sub at readback — the natural way is
    // to use a signed int32 with two's complement wraparound, which Metal
    // supports via atomic_int.  We use `atomic_int` (signed) for the scratch.
    constexpr float SCALE = 1073741824.0f;            // 2^30
    constexpr float INV_SCALE = 1.0f / SCALE;

    // Per-feature threadgroup int scratch for features 1-3.  3 features × 128 bins × 4 B = 1.5 KB.
    threadgroup atomic_int simdHistInt[3][T2_BIN_CAP];
    for (uint f = 0u; f < 3u; ++f) {
        for (uint b = tid; b < T2_BIN_CAP; b += BLOCK_SIZE)
            atomic_store_explicit(&simdHistInt[f][b], 0, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint f = 0u; f < FEATURES_PER_PACK; ++f) {
        const uint foldCount  = foldCountsFlat[foldBase + f];
        const uint firstFold  = firstFoldIndicesFlat[foldBase + f];
        if (foldCount == 0u) continue;

        if (f == 0u) {
            // Feature 0: bin-range scan over sortedDocs (Value-B topology).
            // Same code as pre-v5 kT2AccumSource feature-0 path.
            for (uint b = tid + 1u; b <= foldCount; b += BLOCK_SIZE) {
                const uint start = binOffsets[offBase + b];
                const uint end   = binOffsets[offBase + b + 1u];
                float sum = 0.0f;
                for (uint i = start; i < end; ++i) {
                    const uint docIdx = sortedDocs[slotBase + i];
                    sum += stats[statIdx * totalNumDocs + docIdx];
                }
                device atomic_float* dst = (device atomic_float*)(
                    histogram + histBase + firstFold + b - 1u);
                atomic_fetch_add_explicit(dst, sum, memory_order_relaxed);
            }
        } else {
            // Features 1-3: stride over sortedDocs; int-atomic fixed-point.
            // fi = f - 1 indexes into simdHistInt[0..2].
            const uint fi = f - 1u;
            for (uint i = tid; i < totalDocsInPart; i += BLOCK_SIZE) {
                const uint docIdx = sortedDocs[slotBase + i];
                const uint packed = compressedIndex[docIdx * lineSize + featureColumnIdx];
                const float s     = stats[statIdx * totalNumDocs + docIdx];
                const uint b      = (packed >> (24u - 8u * f)) & 0x7Fu;
                if (b >= 1u && b <= foldCount) {
                    // Round-to-nearest to int32.  rint() is deterministic FP→int.
                    const int delta = int(rint(s * SCALE));
                    atomic_fetch_add_explicit(&simdHistInt[fi][b], delta,
                                              memory_order_relaxed);
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Writeback for features 1-3: convert int → float, atomic_add into global histogram.
    for (uint f = 1u; f < FEATURES_PER_PACK; ++f) {
        const uint foldCount = foldCountsFlat[foldBase + f];
        const uint firstFold = firstFoldIndicesFlat[foldBase + f];
        if (foldCount == 0u) continue;
        const uint fi = f - 1u;
        for (uint b = tid + 1u; b <= foldCount; b += BLOCK_SIZE) {
            const int ival = atomic_load_explicit(&simdHistInt[fi][b],
                                                  memory_order_relaxed);
            if (ival != 0) {
                const float val = float(ival) * INV_SCALE;
                device atomic_float* dst = (device atomic_float*)(
                    histogram + histBase + firstFold + b - 1u);
                atomic_fetch_add_explicit(dst, val, memory_order_relaxed);
            }
        }
    }
)metal";

}  // namespace G1Kernels
