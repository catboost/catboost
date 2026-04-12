#include "structure_searcher.h"
#include <catboost/mlx/methods/score_calcer.h>
#include <catboost/mlx/methods/leaves/leaf_estimator.h>
#include <catboost/libs/logging/logging.h>

namespace NCatboostMlx {

    TPartitionLayout ComputePartitionLayout(
        const mx::array& partitions, ui32 numDocs, ui32 numPartitions
    ) {
        // GPU-resident bucket sort — zero CPU-GPU syncs.
        //
        // Step 1: stable sort doc indices by their partition assignment.
        //   MLX argsort on uint32 uses a stable merge sort.
        //   Docs within the same partition appear in ascending docIdx order,
        //   matching the original CPU scatter-sort behaviour.
        auto docIndices = mx::astype(
            mx::argsort(partitions, /*axis=*/0), mx::uint32
        );

        // Step 2: count docs per partition via scatter-add of ones.
        //   Use int32 accumulator — exact for all values up to 2^31 (~2.1B docs),
        //   removing the previous float32 ceiling of 2^24 (~16M docs).
        auto onesI = mx::ones({static_cast<int>(numDocs)}, mx::int32);
        auto partSizesI = mx::scatter_add_axis(
            mx::zeros({static_cast<int>(numPartitions)}, mx::int32),
            partitions, onesI, 0
        );

        // Step 3: exclusive prefix sum for partition start offsets.
        auto partOffsetsI = mx::cumsum(partSizesI, /*axis=*/0, /*reverse=*/false, /*inclusive=*/false);

        TPartitionLayout layout;
        layout.DocIndices  = docIndices;
        layout.PartSizes   = mx::astype(partSizesI, mx::uint32);
        layout.PartOffsets = mx::astype(partOffsetsI, mx::uint32);

        // No EvalNow here — arrays are consumed lazily by the histogram kernel
        // in the same MLX graph, avoiding an unnecessary CPU-GPU sync point.
        return layout;
    }

    TObliviousTreeStructure SearchTreeStructure(
        TMLXDataSet& dataset,
        ui32 maxDepth,
        float l2RegLambda,
        ui32 approxDimension
    ) {
        TObliviousTreeStructure result;
        result.Splits.reserve(maxDepth);
        result.SplitProperties.reserve(maxDepth);

        const auto& features = dataset.GetCompressedIndex().GetFeatures();
        const ui32 numDocs = dataset.GetNumDocs();

        for (ui32 depth = 0; depth < maxDepth; ++depth) {
            const ui32 numPartitions = 1u << depth;  // 2^depth leaves at current level

            CATBOOST_DEBUG_LOG << "CatBoost-MLX: Searching depth " << depth
                << " (" << numPartitions << " partitions)" << Endl;

            // Compute partition layout from current assignments
            auto layout = ComputePartitionLayout(
                dataset.GetPartitions(), numDocs, numPartitions);

            // Step 1-2: Compute histograms per dimension; compute partition stats
            // once for all dims in a single GPU dispatch (OPT-1: eliminates
            // approxDim CPU-GPU round trips per depth level).
            TVector<THistogramResult> perDimHistograms;
            perDimHistograms.reserve(approxDimension);

            for (ui32 k = 0; k < approxDimension; ++k) {
                // Slice gradients and hessians for dimension k (histogram only)
                mx::array dimGrads, dimHess;
                if (approxDimension == 1) {
                    dimGrads = mx::reshape(dataset.GetGradients(), {static_cast<int>(numDocs)});
                    dimHess  = mx::reshape(dataset.GetHessians(),  {static_cast<int>(numDocs)});
                } else {
                    dimGrads = mx::slice(dataset.GetGradients(),
                        {static_cast<int>(k), 0}, {static_cast<int>(k + 1), static_cast<int>(numDocs)});
                    dimGrads = mx::reshape(dimGrads, {static_cast<int>(numDocs)});
                    dimHess  = mx::slice(dataset.GetHessians(),
                        {static_cast<int>(k), 0}, {static_cast<int>(k + 1), static_cast<int>(numDocs)});
                    dimHess  = mx::reshape(dimHess, {static_cast<int>(numDocs)});
                }

                auto histResult = ComputeHistograms(
                    dataset, dimGrads, dimHess,
                    layout.DocIndices, layout.PartOffsets, layout.PartSizes,
                    numPartitions
                );
                perDimHistograms.push_back(std::move(histResult));
            }

            // Single GPU dispatch for all-dimension partition stats (OPT-1).
            // ComputeLeafSumsGPU handles approxDim internally via the [approxDim,numDocs]
            // layout — no CPU readback needed here; GPU arrays passed directly to
            // FindBestSplitGPU overload that accepts mx::array partition totals.
            mx::array allGradSums, allHessSums;
            ComputeLeafSumsGPU(
                dataset.GetGradients(), dataset.GetHessians(),
                dataset.GetPartitions(),
                numDocs, numPartitions, approxDimension,
                allGradSums, allHessSums
            );
            // allGradSums / allHessSums: [approxDim * numPartitions] float32 — stays on GPU

            // Step 3: Find best split on GPU using pre-computed GPU partition sums
            // and binToFeature lookup table (OPT-1 + OPT-2 combined path).
            auto bestSplit = FindBestSplitGPU(
                perDimHistograms,
                allGradSums, allHessSums,
                features,
                l2RegLambda,
                numPartitions,
                approxDimension
            );

            if (!bestSplit.Defined()) {
                CATBOOST_INFO_LOG << "CatBoost-MLX: No valid split found at depth " << depth
                    << ", stopping tree growth" << Endl;
                break;
            }

            CATBOOST_DEBUG_LOG << "CatBoost-MLX: Best split at depth " << depth
                << ": feature=" << bestSplit.FeatureId << " bin=" << bestSplit.BinId
                << " gain=" << bestSplit.Gain << Endl;

            // Build the split level descriptor
            const auto& feat = features[bestSplit.FeatureId];
            TObliviousSplitLevel split;
            split.FeatureColumnIdx = static_cast<ui32>(feat.Offset);
            split.Shift = feat.Shift;
            split.Mask = feat.Mask >> feat.Shift;
            split.BinThreshold = bestSplit.BinId;
            split.IsOneHot = feat.OneHotFeature;

            result.Splits.push_back(split);
            result.SplitProperties.push_back(bestSplit);

            // Apply this split level to update partition assignments for next depth.
            // leafIdx |= (featureValue > threshold) << depth
            {
                auto& partitions = dataset.GetPartitions();
                const auto& compressedData = dataset.GetCompressedIndex().GetCompressedData();

                auto column = mx::slice(compressedData,
                    {0, static_cast<int>(split.FeatureColumnIdx)},
                    {static_cast<int>(numDocs), static_cast<int>(split.FeatureColumnIdx + 1)}
                );
                column = mx::reshape(column, {static_cast<int>(numDocs)});

                auto featureValues = mx::bitwise_and(
                    mx::right_shift(column, mx::array(static_cast<int>(split.Shift))),
                    mx::array(static_cast<int>(split.Mask))
                );

                // Compare: OneHot uses equality, ordinal uses greater-than
                mx::array goRight;
                if (feat.OneHotFeature) {
                    goRight = mx::equal(featureValues,
                        mx::array(static_cast<int>(split.BinThreshold)));
                } else {
                    goRight = mx::greater(featureValues,
                        mx::array(static_cast<int>(split.BinThreshold)));
                }
                auto bits = mx::astype(goRight, mx::uint32);
                bits = mx::left_shift(bits, mx::array(static_cast<int>(depth)));

                partitions = mx::bitwise_or(partitions, bits);
                TMLXDevice::EvalNow(partitions);
            }
        }

        return result;
    }

    TDepthwiseTreeStructure SearchDepthwiseTreeStructure(
        TMLXDataSet& dataset,
        ui32 maxDepth,
        float l2RegLambda,
        ui32 approxDimension
    ) {
        TDepthwiseTreeStructure result;
        const auto& features = dataset.GetCompressedIndex().GetFeatures();
        const ui32 numDocs = dataset.GetNumDocs();

        // NodeSplits will hold one entry per BFS node: total = 2^maxDepth - 1 if the
        // tree grows to full depth, but we fill incrementally and stop early if no
        // valid split is found for any partition.
        result.NodeSplits.reserve((1u << maxDepth) - 1u);

        for (ui32 depth = 0; depth < maxDepth; ++depth) {
            const ui32 numPartitions = 1u << depth;  // 2^depth leaves at current level

            CATBOOST_DEBUG_LOG << "CatBoost-MLX Depthwise: Searching depth " << depth
                << " (" << numPartitions << " partitions)" << Endl;

            // Compute partition layout from current assignments
            auto layout = ComputePartitionLayout(
                dataset.GetPartitions(), numDocs, numPartitions);

            // Compute histograms per approxDim dimension
            TVector<THistogramResult> perDimHistograms;
            perDimHistograms.reserve(approxDimension);

            for (ui32 k = 0; k < approxDimension; ++k) {
                mx::array dimGrads, dimHess;
                if (approxDimension == 1) {
                    dimGrads = mx::reshape(dataset.GetGradients(), {static_cast<int>(numDocs)});
                    dimHess  = mx::reshape(dataset.GetHessians(),  {static_cast<int>(numDocs)});
                } else {
                    dimGrads = mx::slice(dataset.GetGradients(),
                        {static_cast<int>(k), 0}, {static_cast<int>(k + 1), static_cast<int>(numDocs)});
                    dimGrads = mx::reshape(dimGrads, {static_cast<int>(numDocs)});
                    dimHess  = mx::slice(dataset.GetHessians(),
                        {static_cast<int>(k), 0}, {static_cast<int>(k + 1), static_cast<int>(numDocs)});
                    dimHess  = mx::reshape(dimHess, {static_cast<int>(numDocs)});
                }

                auto histResult = ComputeHistograms(
                    dataset, dimGrads, dimHess,
                    layout.DocIndices, layout.PartOffsets, layout.PartSizes,
                    numPartitions
                );
                perDimHistograms.push_back(std::move(histResult));
            }

            // Compute GPU partition sums for all partitions.
            mx::array allGradSums, allHessSums;
            ComputeLeafSumsGPU(
                dataset.GetGradients(), dataset.GetHessians(),
                dataset.GetPartitions(),
                numDocs, numPartitions, approxDimension,
                allGradSums, allHessSums
            );

            // --- Depthwise: find the best split per partition ---
            // Each partition (leaf at current depth) gets its own best split.
            // Partitions are ordered 0..numPartitions-1 in BFS leaf order, which
            // matches the bit-encoding used in UpdatePartitions (leafIdx bit-field).
            //
            // We assign splits in BFS node order so NodeSplits[nodeAtDepthD + p]
            // is the split for the p-th leaf at depth D.
            //
            // Strategy: run FindBestSplitGPU once per partition, slicing the
            // histogram and partition sums arrays to isolate that partition.

            // Materialise partition sums once before the per-partition loop.
            TMLXDevice::EvalNow({allGradSums, allHessSums});
            const float* gsPtr = allGradSums.data<float>();
            const float* hsPtr = allHessSums.data<float>();

            bool anyValidSplit = false;
            for (ui32 p = 0; p < numPartitions; ++p) {
                // Slice histogram for this single partition: [numStats * totalBinFeatures] float32.
                // fullHist.Histograms layout: [numPartitions * numStats * totalBins] float32.
                TVector<THistogramResult> singlePartHist;
                singlePartHist.reserve(approxDimension);
                for (ui32 k = 0; k < approxDimension; ++k) {
                    const auto& fullHist = perDimHistograms[k];
                    const ui32 totalBins = fullHist.TotalBinFeatures;
                    const ui32 numStats  = fullHist.NumStats;
                    int startRow = static_cast<int>(p * numStats * totalBins);
                    int endRow   = static_cast<int>((p + 1) * numStats * totalBins);
                    auto sliced = mx::slice(
                        mx::reshape(fullHist.Histograms,
                            {static_cast<int>(numPartitions * numStats * totalBins)}),
                        {startRow}, {endRow}
                    );
                    THistogramResult singleHist;
                    singleHist.Histograms        = mx::reshape(sliced,
                        {static_cast<int>(numStats * totalBins)});
                    singleHist.TotalBinFeatures  = totalBins;
                    singleHist.NumStats          = numStats;
                    singlePartHist.push_back(std::move(singleHist));
                }

                // Slice partition sums for this partition.
                // allGradSums / allHessSums layout: [approxDim * numPartitions] float32.
                // For each dim k, partition p is at k * numPartitions + p.
                TVector<float> partGradVec(approxDimension), partHessVec(approxDimension);
                for (ui32 k = 0; k < approxDimension; ++k) {
                    partGradVec[k] = gsPtr[k * numPartitions + p];
                    partHessVec[k] = hsPtr[k * numPartitions + p];
                }
                auto partGradArr = mx::array(partGradVec.data(),
                    {static_cast<int>(approxDimension)}, mx::float32);
                auto partHessArr = mx::array(partHessVec.data(),
                    {static_cast<int>(approxDimension)}, mx::float32);

                auto bestSplit = FindBestSplitGPU(
                    singlePartHist,
                    partGradArr, partHessArr,
                    features,
                    l2RegLambda,
                    /*numPartitions=*/1u,
                    approxDimension
                );

                // Build the split descriptor for this node (BFS position = numPartitions-1+p).
                TObliviousSplitLevel nodeSplit;
                if (bestSplit.Defined()) {
                    anyValidSplit = true;
                    const auto& feat = features[bestSplit.FeatureId];
                    nodeSplit.FeatureColumnIdx = static_cast<ui32>(feat.Offset);
                    nodeSplit.Shift            = feat.Shift;
                    nodeSplit.Mask             = feat.Mask >> feat.Shift;
                    nodeSplit.BinThreshold     = bestSplit.BinId;
                    nodeSplit.IsOneHot         = feat.OneHotFeature;

                    CATBOOST_DEBUG_LOG << "CatBoost-MLX Depthwise: depth=" << depth
                        << " part=" << p << ": feature=" << bestSplit.FeatureId
                        << " bin=" << bestSplit.BinId << " gain=" << bestSplit.Gain << Endl;
                } else {
                    // No valid split for this partition — reuse a no-op descriptor
                    // (threshold=0, mask=0 → all docs go left; leaf value unchanged).
                    nodeSplit.FeatureColumnIdx = 0u;
                    nodeSplit.Shift            = 0u;
                    nodeSplit.Mask             = 0u;
                    nodeSplit.BinThreshold     = 0u;
                    nodeSplit.IsOneHot         = false;
                }
                result.NodeSplits.push_back(nodeSplit);
            }

            if (!anyValidSplit) {
                CATBOOST_INFO_LOG << "CatBoost-MLX Depthwise: No valid splits at depth "
                    << depth << ", stopping" << Endl;
                break;
            }
            result.Depth = depth + 1;

            // Update partition assignments for the next depth level.
            // Depthwise update: for each partition p, apply its own split to the docs in that partition.
            // Because partitions are bit-encoded (leafIdx = bitmask), we need to set bit `depth`
            // for docs in partition p that go right.  The existing partition value already encodes
            // which leaf the doc belongs to via its lower `depth` bits.
            //
            // Vectorised MLX approach (no per-partition CPU loop):
            //   1. Build a per-doc "threshold" array by gathering nodeSplits[partition] for each doc.
            //   2. Extract the feature value per doc using the corresponding node's (col, shift, mask).
            //   3. Evaluate the go-right condition per doc.
            //   4. OR the result bit into partitions.
            //
            // For simplicity (and correctness), we recompute per-partition on CPU by updating
            // each partition's docs in sequence using MLX masked scatter.
            {
                const auto& compressedData = dataset.GetCompressedIndex().GetCompressedData();
                auto& partitions = dataset.GetPartitions();
                TMLXDevice::EvalNow(partitions);

                // Accumulate the bit update across all partitions in one MLX expression:
                // goRightBits = sum over p of: ((featureVals[p] > thresh[p]) AND (part == p)) << depth
                //
                // We compute this as a single vectorised expression over all docs:
                //   For each partition p, create a mask (part == p), extract featureVal with p's
                //   split, compare, AND with mask, shift. Sum across partitions → OR into partitions.
                mx::array updateBits = mx::zeros({static_cast<int>(numDocs)}, mx::uint32);

                for (ui32 p = 0; p < numPartitions; ++p) {
                    const auto& nodeSplit = result.NodeSplits[
                        result.NodeSplits.size() - numPartitions + p];
                    if (nodeSplit.Mask == 0u) continue;  // no-op split

                    // Docs in this partition
                    auto inPartition = mx::astype(
                        mx::equal(partitions, mx::array(static_cast<uint32_t>(p), mx::uint32)),
                        mx::uint32
                    );

                    // Extract feature values for this partition's split
                    auto column = mx::slice(compressedData,
                        {0, static_cast<int>(nodeSplit.FeatureColumnIdx)},
                        {static_cast<int>(numDocs), static_cast<int>(nodeSplit.FeatureColumnIdx + 1)}
                    );
                    column = mx::reshape(column, {static_cast<int>(numDocs)});

                    auto featureValues = mx::bitwise_and(
                        mx::right_shift(column, mx::array(static_cast<int>(nodeSplit.Shift))),
                        mx::array(static_cast<int>(nodeSplit.Mask))
                    );

                    mx::array goRight;
                    if (nodeSplit.IsOneHot) {
                        goRight = mx::equal(featureValues,
                            mx::array(static_cast<int>(nodeSplit.BinThreshold)));
                    } else {
                        goRight = mx::greater(featureValues,
                            mx::array(static_cast<int>(nodeSplit.BinThreshold)));
                    }

                    // Apply split only within this partition
                    auto bits = mx::multiply(
                        mx::astype(goRight, mx::uint32),
                        inPartition
                    );
                    updateBits = mx::add(updateBits, bits);
                }

                // Shift accumulated go-right bits to depth position and OR into partitions
                auto shiftedBits = mx::left_shift(updateBits,
                    mx::array(static_cast<int>(depth)));
                partitions = mx::bitwise_or(partitions, shiftedBits);
                TMLXDevice::EvalNow(partitions);
            }
        }

        return result;
    }

}  // namespace NCatboostMlx
