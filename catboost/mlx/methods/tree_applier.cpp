#include "tree_applier.h"

#include <catboost/mlx/kernels/kernel_sources.h>
#include <catboost/libs/logging/logging.h>

#include <mlx/mlx.h>
#include <mlx/fast.h>

namespace NCatboostMlx {

    void ApplyObliviousTree(
        TMLXDataSet& dataset,
        const TVector<TObliviousSplitLevel>& splits,
        const mx::array& leafValues,
        ui32 approxDimension
    ) {
        const ui32 numDocs  = dataset.GetNumDocs();
        const ui32 depth    = static_cast<ui32>(splits.size());
        const auto& compressedData = dataset.GetCompressedIndex().GetCompressedData();
        const ui32 lineSize = dataset.GetCompressedIndex().GetNumUi32PerDoc();

        // Pack per-level split descriptors into flat uint32 arrays.
        // Five parallel arrays of length `depth` — one entry per split level.
        // For depth=0: produce a single-element placeholder so MLX sees a
        // non-empty buffer; the Metal kernel checks (d < depth) so the split
        // loop body never executes, and all docs land on leaf 0.
        const ui32 bufDepth = (depth == 0) ? 1u : depth;
        TVector<uint32_t> colIdxVec(bufDepth, 0u), shiftVec(bufDepth, 0u),
                          maskVec(bufDepth, 0u),    threshVec(bufDepth, 0u),
                          isOneHotVec(bufDepth, 0u);
        for (ui32 d = 0; d < depth; ++d) {
            colIdxVec[d]   = splits[d].FeatureColumnIdx;
            shiftVec[d]    = splits[d].Shift;
            maskVec[d]     = splits[d].Mask;
            threshVec[d]   = splits[d].BinThreshold;
            isOneHotVec[d] = splits[d].IsOneHot ? 1u : 0u;
        }

        auto splitColIdxArr = mx::array(reinterpret_cast<const int32_t*>(colIdxVec.data()),
                                        {static_cast<int>(bufDepth)}, mx::uint32);
        auto splitShiftArr  = mx::array(reinterpret_cast<const int32_t*>(shiftVec.data()),
                                        {static_cast<int>(bufDepth)}, mx::uint32);
        auto splitMaskArr   = mx::array(reinterpret_cast<const int32_t*>(maskVec.data()),
                                        {static_cast<int>(bufDepth)}, mx::uint32);
        auto splitThreshArr = mx::array(reinterpret_cast<const int32_t*>(threshVec.data()),
                                        {static_cast<int>(bufDepth)}, mx::uint32);
        auto splitOneHotArr = mx::array(reinterpret_cast<const int32_t*>(isOneHotVec.data()),
                                        {static_cast<int>(bufDepth)}, mx::uint32);

        auto numDocsArr   = mx::array(static_cast<uint32_t>(numDocs),          mx::uint32);
        auto depthArr     = mx::array(static_cast<uint32_t>(depth),             mx::uint32);
        auto lineSizeArr  = mx::array(static_cast<uint32_t>(lineSize),          mx::uint32);
        auto approxDimArr = mx::array(static_cast<uint32_t>(approxDimension),   mx::uint32);

        // Flatten compressedData to 1D [numDocs * lineSize].
        auto flatData = mx::reshape(compressedData,
            {static_cast<int>(numDocs * lineSize)});

        // Flatten leafValues to [numLeaves * approxDimension] row-major.
        // approxDimension == 1: [numLeaves] — reshape is a no-op.
        // approxDimension >  1: [numLeaves, K] — flatten to [numLeaves * K].
        const ui32 numLeaves    = 1u << depth;
        auto flatLeafValues     = mx::reshape(leafValues,
            {static_cast<int>(numLeaves * approxDimension)});

        // Flatten the existing cursor to [approxDimension * numDocs] for the kernel.
        // This is the read-only cursorIn — the kernel outputs cursorOut = cursorIn + delta.
        auto& cursor     = dataset.GetCursor();
        auto cursorShape = cursor.shape();
        auto flatCursor  = mx::reshape(cursor,
            {static_cast<int>(approxDimension * numDocs)});

        // ---- Metal kernel dispatch ----
        // One thread per document; no shared memory; no atomics.
        // Thread d computes leafIdx from all split levels then writes:
        //   cursorOut[k * numDocs + d] = cursorIn[k * numDocs + d]
        //                               + leafValues[leafIdx * approxDim + k]
        // for all k in [0, approxDimension).
        auto kernel = mx::fast::metal_kernel(
            "apply_oblivious_tree",
            /*input_names=*/{
                "compressedData", "splitColIdx", "splitShift", "splitMask",
                "splitThreshold", "splitIsOneHot", "leafValues", "cursorIn",
                "numDocs", "depth", "lineSize", "approxDim"
            },
            /*output_names=*/{"cursorOut", "partitionsOut"},
            /*source=*/KernelSources::kTreeApplySource,
            /*header=*/KernelSources::kTreeApplyHeader,
            /*ensure_row_contiguous=*/true,
            /*atomic_outputs=*/false
        );

        const int blockSize = 256;
        const int numBlocks = (static_cast<int>(numDocs) + blockSize - 1) / blockSize;
        auto grid = std::make_tuple(numBlocks * blockSize, 1, 1);
        auto tg   = std::make_tuple(blockSize, 1, 1);

        auto results = kernel(
            /*inputs=*/{
                flatData, splitColIdxArr, splitShiftArr, splitMaskArr,
                splitThreshArr, splitOneHotArr, flatLeafValues, flatCursor,
                numDocsArr, depthArr, lineSizeArr, approxDimArr
            },
            /*output_shapes=*/{{static_cast<int>(approxDimension * numDocs)}, {static_cast<int>(numDocs)}},
            /*output_dtypes=*/{mx::float32, mx::uint32},
            grid, tg,
            /*template_args=*/{},
            /*init_value=*/std::nullopt,
            /*verbose=*/false,
            /*stream=*/mx::Device::gpu
        );

        // Restore the original cursor shape and write back to the dataset.
        cursor = mx::reshape(results[0], cursorShape);

        // Partition assignments come directly from the kernel's second output —
        // leafIdx was already computed per-thread; no O(depth) MLX recompute needed.
        dataset.GetPartitions() = results[1];

        // EvalAtBoundary: materialise cursor and partitions written by this kernel
        // before any downstream code reads them (e.g. ComputeDerivatives next iter).
        TMLXDevice::EvalAtBoundary({cursor, dataset.GetPartitions()});

        CATBOOST_DEBUG_LOG << "CatBoost-MLX: Applied tree depth=" << depth
            << " to " << numDocs << " documents (Metal kernel)" << Endl;
    }

    void ApplyDepthwiseTree(
        TMLXDataSet& dataset,
        const TVector<TObliviousSplitLevel>& nodeSplits,
        const mx::array& leafValues,
        ui32 depth,
        ui32 approxDimension
    ) {
        const ui32 numDocs   = dataset.GetNumDocs();
        const ui32 numNodes  = (depth == 0) ? 0u : ((1u << depth) - 1u);
        const auto& compressedData = dataset.GetCompressedIndex().GetCompressedData();
        const ui32 lineSize  = dataset.GetCompressedIndex().GetNumUi32PerDoc();

        Y_VERIFY(nodeSplits.size() == numNodes,
            "ApplyDepthwiseTree: nodeSplits.size() != 2^depth - 1");

        // For depth=0: produce a 1-element placeholder so the Metal kernel sees a
        // non-empty buffer; the traversal loop never executes (depth==0).
        const ui32 bufNodes = (numNodes == 0) ? 1u : numNodes;
        TVector<uint32_t> colIdxVec(bufNodes, 0u), shiftVec(bufNodes, 0u),
                          maskVec(bufNodes, 0u),    threshVec(bufNodes, 0u),
                          isOneHotVec(bufNodes, 0u);
        for (ui32 n = 0; n < numNodes; ++n) {
            colIdxVec[n]   = nodeSplits[n].FeatureColumnIdx;
            shiftVec[n]    = nodeSplits[n].Shift;
            maskVec[n]     = nodeSplits[n].Mask;
            threshVec[n]   = nodeSplits[n].BinThreshold;
            isOneHotVec[n] = nodeSplits[n].IsOneHot ? 1u : 0u;
        }

        auto nodeColIdxArr  = mx::array(reinterpret_cast<const int32_t*>(colIdxVec.data()),
                                        {static_cast<int>(bufNodes)}, mx::uint32);
        auto nodeShiftArr   = mx::array(reinterpret_cast<const int32_t*>(shiftVec.data()),
                                        {static_cast<int>(bufNodes)}, mx::uint32);
        auto nodeMaskArr    = mx::array(reinterpret_cast<const int32_t*>(maskVec.data()),
                                        {static_cast<int>(bufNodes)}, mx::uint32);
        auto nodeThreshArr  = mx::array(reinterpret_cast<const int32_t*>(threshVec.data()),
                                        {static_cast<int>(bufNodes)}, mx::uint32);
        auto nodeOneHotArr  = mx::array(reinterpret_cast<const int32_t*>(isOneHotVec.data()),
                                        {static_cast<int>(bufNodes)}, mx::uint32);

        auto numDocsArr   = mx::array(static_cast<uint32_t>(numDocs),        mx::uint32);
        auto depthArr     = mx::array(static_cast<uint32_t>(depth),           mx::uint32);
        auto lineSizeArr  = mx::array(static_cast<uint32_t>(lineSize),        mx::uint32);
        auto approxDimArr = mx::array(static_cast<uint32_t>(approxDimension), mx::uint32);

        auto flatData = mx::reshape(compressedData,
            {static_cast<int>(numDocs * lineSize)});

        const ui32 numLeaves = 1u << depth;
        auto flatLeafValues  = mx::reshape(leafValues,
            {static_cast<int>(numLeaves * approxDimension)});

        auto& cursor     = dataset.GetCursor();
        auto cursorShape = cursor.shape();
        auto flatCursor  = mx::reshape(cursor,
            {static_cast<int>(approxDimension * numDocs)});

        auto kernel = mx::fast::metal_kernel(
            "apply_depthwise_tree",
            /*input_names=*/{
                "compressedData", "nodeColIdx", "nodeShift", "nodeMask",
                "nodeThreshold", "nodeIsOneHot", "leafValues", "cursorIn",
                "numDocs", "depth", "lineSize", "approxDim"
            },
            /*output_names=*/{"cursorOut", "partitionsOut"},
            /*source=*/KernelSources::kTreeApplyDepthwiseSource,
            /*header=*/KernelSources::kTreeApplyHeader,
            /*ensure_row_contiguous=*/true,
            /*atomic_outputs=*/false
        );

        const int blockSize = 256;
        const int numBlocks = (static_cast<int>(numDocs) + blockSize - 1) / blockSize;
        auto grid = std::make_tuple(numBlocks * blockSize, 1, 1);
        auto tg   = std::make_tuple(blockSize, 1, 1);

        auto results = kernel(
            /*inputs=*/{
                flatData, nodeColIdxArr, nodeShiftArr, nodeMaskArr,
                nodeThreshArr, nodeOneHotArr, flatLeafValues, flatCursor,
                numDocsArr, depthArr, lineSizeArr, approxDimArr
            },
            /*output_shapes=*/{{static_cast<int>(approxDimension * numDocs)}, {static_cast<int>(numDocs)}},
            /*output_dtypes=*/{mx::float32, mx::uint32},
            grid, tg,
            /*template_args=*/{},
            /*init_value=*/std::nullopt,
            /*verbose=*/false,
            /*stream=*/mx::Device::gpu
        );

        cursor = mx::reshape(results[0], cursorShape);
        dataset.GetPartitions() = results[1];

        // EvalAtBoundary: materialise cursor and partitions written by this kernel
        // before any downstream code reads them (e.g. ComputeDerivatives next iter).
        TMLXDevice::EvalAtBoundary({cursor, dataset.GetPartitions()});

        CATBOOST_DEBUG_LOG << "CatBoost-MLX: Applied depthwise tree depth=" << depth
            << " to " << numDocs << " documents (Metal kernel)" << Endl;
    }

    // -------------------------------------------------------------------------
    // Lossguide tree apply + inference helpers
    // -------------------------------------------------------------------------

    mx::array ComputeLeafIndicesLossguide(
        const mx::array& compressedData,
        const std::unordered_map<ui32, TObliviousSplitLevel>& nodeSplitMap,
        const TVector<ui32>& leafBfsIds,
        ui32 numDocs,
        ui32 numLeaves
    ) {
        if (numLeaves <= 1u || nodeSplitMap.empty()) {
            // Degenerate tree: all docs in leaf 0.
            return mx::zeros({static_cast<int>(numDocs)}, mx::uint32);
        }

        // Build an inverse set: BFS node indices that are leaf nodes.
        std::unordered_map<ui32, ui32> bfsToLeafId;  // bfsIdx → dense leaf id
        bfsToLeafId.reserve(leafBfsIds.size());
        for (ui32 k = 0; k < static_cast<ui32>(leafBfsIds.size()); ++k) {
            bfsToLeafId[leafBfsIds[k]] = k;
        }

        // CPU traversal: for each doc, walk the BFS tree from root until we reach a leaf.
        auto flatData = mx::reshape(compressedData, {static_cast<int>(numDocs), -1});
        // EvalAtBoundary required — raw CPU pointer read follows immediately.
        TMLXDevice::EvalAtBoundary(flatData);
        const uint32_t* dataPtr = flatData.data<uint32_t>();
        const ui32 lineSize = static_cast<ui32>(flatData.shape(1));

        std::vector<uint32_t> leafVec(numDocs);
        for (ui32 d = 0; d < numDocs; ++d) {
            ui32 nodeIdx = 0u;
            // Traverse until nodeIdx is a leaf (found in bfsToLeafId, not in nodeSplitMap).
            while (nodeSplitMap.count(nodeIdx) > 0) {
                const auto& ns = nodeSplitMap.at(nodeIdx);
                uint32_t packed = dataPtr[d * lineSize + ns.FeatureColumnIdx];
                uint32_t fv = (packed >> ns.Shift) & ns.Mask;
                uint32_t goRight = ns.IsOneHot
                    ? (fv == ns.BinThreshold ? 1u : 0u)
                    : (fv >  ns.BinThreshold ? 1u : 0u);
                nodeIdx = 2u * nodeIdx + 1u + goRight;
            }
            Y_VERIFY(bfsToLeafId.count(nodeIdx) > 0,
                "ComputeLeafIndicesLossguide: BFS traversal ended at unknown node");
            leafVec[d] = bfsToLeafId.at(nodeIdx);
        }

        return mx::array(
            reinterpret_cast<const int32_t*>(leafVec.data()),
            {static_cast<int>(numDocs)}, mx::uint32
        );
    }

    void ApplyLossguideTree(
        TMLXDataSet& dataset,
        const std::unordered_map<ui32, TObliviousSplitLevel>& /*nodeSplitMap*/,
        const TVector<ui32>& /*leafBfsIds*/,
        const mx::array& leafDocIds,
        const mx::array& leafValues,
        ui32 numLeaves,
        ui32 approxDimension
    ) {
        const ui32 numDocs = dataset.GetNumDocs();

        // leafDocIds is the per-document dense leaf assignment computed during search.
        // For training data this comes directly from TLossguideTreeStructure::LeafDocIds.
        // nodeSplitMap and leafBfsIds are not needed here — they are used by
        // ComputeLeafIndicesLossguide for validation/inference data inference.

        auto& cursor     = dataset.GetCursor();
        auto cursorShape = cursor.shape();

        if (approxDimension == 1) {
            // Scalar path: gather leafValues[leafDocIds[d]] for each doc.
            auto docLeafVals = mx::take(leafValues, mx::astype(leafDocIds, mx::int32), 0);
            auto flatCursor  = mx::reshape(cursor, {static_cast<int>(numDocs)});
            cursor = mx::reshape(mx::add(flatCursor, docLeafVals), cursorShape);
        } else {
            // Multi-class path: leafValues is [numLeaves, K].
            // docLeafVals = leafValues[leafDocIds[d]] → [numDocs, K]
            auto docLeafVals = mx::take(leafValues, mx::astype(leafDocIds, mx::int32), 0);
            // Transpose to [K, numDocs] and add to cursor [K, numDocs].
            auto transposed = mx::transpose(docLeafVals);  // [K, numDocs]
            cursor = mx::reshape(mx::add(
                mx::reshape(cursor, {static_cast<int>(approxDimension), static_cast<int>(numDocs)}),
                transposed),
                cursorShape
            );
        }

        // Update partitions: for lossguide trees the partition is just the dense leaf id.
        dataset.GetPartitions() = leafDocIds;

        // EvalAtBoundary: materialise cursor and partitions written by this kernel
        // before any downstream code reads them (e.g. ComputeDerivatives next iter).
        TMLXDevice::EvalAtBoundary({cursor, dataset.GetPartitions()});

        CATBOOST_DEBUG_LOG << "CatBoost-MLX: Applied lossguide tree with "
            << numLeaves << " leaves to " << numDocs << " documents" << Endl;
    }

}  // namespace NCatboostMlx
