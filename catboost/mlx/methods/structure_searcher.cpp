#include "structure_searcher.h"
#include <catboost/mlx/methods/score_calcer.h>
#include <catboost/mlx/methods/leaves/leaf_estimator.h>
#include <catboost/libs/logging/logging.h>

#include <queue>

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

    TLossguideTreeStructure SearchLossguideTreeStructure(
        TMLXDataSet& dataset,
        ui32 maxLeaves,
        float l2RegLambda,
        ui32 approxDimension,
        ui32 maxDepth
    ) {
        const auto& features = dataset.GetCompressedIndex().GetFeatures();
        const ui32 numDocs = dataset.GetNumDocs();

        // The lossguide tree is represented in BFS node order.
        // We allocate NodeSplits on demand as we expand leaves.
        // BFS index 0 = root.  Children of node n: left = 2n+1, right = 2n+2.
        //
        // Internal state:
        //   leafDocIds[doc]   — dense leaf index (0..numLeaves-1) per document.
        //                       Updated when a leaf is split.
        //   leafBfsId[leaf]   — BFS node index for dense leaf k.
        //   leafDepth[leaf]   — depth of dense leaf k (root = depth 0).
        //
        // Priority queue: (gain, leafId) — max-heap.
        TLossguideTreeStructure result;
        result.NumLeaves = 1;

        // Initialize all docs in leaf 0 (root).
        std::vector<uint32_t> leafDocVec(numDocs, 0u);

        // BFS metadata per dense leaf
        std::vector<ui32> leafBfsId = {0u};    // leaf 0 is at BFS node 0 (root)
        std::vector<ui32> leafDepth = {0u};    // root is at depth 0

        // NodeSplitMap: sparse map from BFS node index → split descriptor.
        // Using an unordered_map avoids O(2^depth) allocation for unbalanced trees.
        // For max_leaves=31 with default max_depth=6, we have at most 30 entries.
        std::unordered_map<ui32, TObliviousSplitLevel> nodeSplitMap;

        // Priority queue: (gain, leafId)
        // We store candidate splits lazily — compute on first need.
        // At each step we evaluate all current leaves (or just re-evaluate
        // the two new children after a split).
        //
        // Strategy: maintain a pool of evaluated leaf splits.
        //   splitPool[leafId] = best split for that leaf (or invalid if none).
        //   gainPool[leafId]  = gain of that split.
        // The priority queue contains (gain, leafId) pairs.
        struct TLeafSplitEntry {
            float               Gain;
            ui32                LeafId;
            TBestSplitProperties Split;
            bool operator<(const TLeafSplitEntry& o) const { return Gain < o.Gain; }
        };
        std::priority_queue<TLeafSplitEntry> pq;

        // Helper: compute the best split for a single leaf.
        // Returns false if no valid split exists.
        auto evalLeaf = [&](ui32 leafId) -> bool {
            ui32 bfsNode = leafBfsId[leafId];

            // Build a temporary MLX array for this leaf's doc membership.
            // leafDocVec[doc] == leafId for docs in this leaf.
            auto leafDocArr = mx::array(
                reinterpret_cast<const int32_t*>(leafDocVec.data()),
                {static_cast<int>(numDocs)}, mx::uint32
            );

            // Compute partition layout treating this leaf's docs as partition 0.
            // For histogram computation we need:
            //   - docIndices sorted by partition (one partition = docs in this leaf first)
            //   - partOffsets[0] = 0
            //   - partSizes[0]   = number of docs in this leaf
            //
            // Use a temporary single-partition layout.
            // partitions for this sub-problem: all docs with value 0 if in leaf, else
            // irrelevant (we only pass leafDocs to the histogram kernel via DocIndices).
            //
            // Simplest approach: create a 0/1 partition array where 0 = "in this leaf".
            // Then ComputePartitionLayout with 1 real partition + a "trash" partition.
            // Actually we have 2 partitions: 0=inLeaf, 1=notInLeaf, but we only care
            // about partition 0's histograms.

            // leafPartitions[d] = 0 if leafDocVec[d]==leafId, else 1.
            auto leafPart = mx::astype(
                mx::not_equal(leafDocArr,
                    mx::array(static_cast<uint32_t>(leafId), mx::uint32)),
                mx::uint32
            );  // 0 for in-leaf, 1 for out-of-leaf

            auto layout = ComputePartitionLayout(leafPart, numDocs, 2u);

            // Compute histograms for 2 partitions; slice partition 0.
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

                auto fullHist = ComputeHistograms(
                    dataset, dimGrads, dimHess,
                    layout.DocIndices, layout.PartOffsets, layout.PartSizes,
                    2u  // 2 partitions
                );

                // Slice partition 0 (in-leaf docs) from the full histogram.
                const ui32 totalBins = fullHist.TotalBinFeatures;
                const ui32 numStats  = fullHist.NumStats;
                auto flatHist = mx::reshape(fullHist.Histograms,
                    {static_cast<int>(2u * numStats * totalBins)});
                auto sliced = mx::slice(flatHist,
                    {0}, {static_cast<int>(numStats * totalBins)});
                THistogramResult singleHist;
                singleHist.Histograms       = mx::reshape(sliced,
                    {static_cast<int>(numStats * totalBins)});
                singleHist.TotalBinFeatures = totalBins;
                singleHist.NumStats         = numStats;
                perDimHistograms.push_back(std::move(singleHist));
            }

            // Compute partition sums for this leaf.
            mx::array allGradSums, allHessSums;
            ComputeLeafSumsGPU(
                dataset.GetGradients(), dataset.GetHessians(),
                leafPart,
                numDocs, 2u, approxDimension,
                allGradSums, allHessSums
            );
            TMLXDevice::EvalNow({allGradSums, allHessSums});

            const float* gsPtr = allGradSums.data<float>();
            const float* hsPtr = allHessSums.data<float>();

            // Slice partition 0 sums: layout is [approxDim * 2 partitions].
            TVector<float> partGradVec(approxDimension), partHessVec(approxDimension);
            for (ui32 k = 0; k < approxDimension; ++k) {
                partGradVec[k] = gsPtr[k * 2 + 0];  // partition 0
                partHessVec[k] = hsPtr[k * 2 + 0];
            }
            auto partGradArr = mx::array(partGradVec.data(),
                {static_cast<int>(approxDimension)}, mx::float32);
            auto partHessArr = mx::array(partHessVec.data(),
                {static_cast<int>(approxDimension)}, mx::float32);

            auto bestSplit = FindBestSplitGPU(
                perDimHistograms,
                partGradArr, partHessArr,
                features,
                l2RegLambda,
                /*numPartitions=*/1u,
                approxDimension
            );

            if (!bestSplit.Defined()) {
                return false;
            }

            // Optionally enforce max depth
            if (maxDepth > 0 && leafDepth[leafId] >= maxDepth) {
                return false;
            }

            CATBOOST_DEBUG_LOG << "CatBoost-MLX Lossguide: leaf=" << leafId
                << " bfs=" << bfsNode << " depth=" << leafDepth[leafId]
                << " feature=" << bestSplit.FeatureId
                << " bin=" << bestSplit.BinId
                << " gain=" << bestSplit.Gain << Endl;

            pq.push({bestSplit.Gain, leafId, bestSplit});
            return true;
        };

        // Evaluate root leaf to bootstrap the priority queue.
        evalLeaf(0u);

        while (result.NumLeaves < maxLeaves && !pq.empty()) {
            // Pop the best candidate
            auto top = pq.top();
            pq.pop();

            const ui32 leafId  = top.LeafId;
            const float gain   = top.Gain;
            const auto& split  = top.Split;

            // Sanity: this leaf must still be a leaf (not previously split).
            // Since we only add each leaf once to the queue, this should always hold.
            Y_VERIFY(leafId < leafBfsId.size(),
                "Lossguide: leafId out of range");

            const ui32 bfsNode  = leafBfsId[leafId];
            const ui32 myDepth  = leafDepth[leafId];
            const ui32 leftBfs  = 2u * bfsNode + 1u;
            const ui32 rightBfs = 2u * bfsNode + 2u;

            CATBOOST_DEBUG_LOG << "CatBoost-MLX Lossguide: splitting leaf=" << leafId
                << " bfs=" << bfsNode << " gain=" << gain
                << " → left_bfs=" << leftBfs << " right_bfs=" << rightBfs << Endl;

            // Record the split for this BFS node in the sparse map.
            const auto& feat = features[split.FeatureId];
            TObliviousSplitLevel nodeSplit;
            nodeSplit.FeatureColumnIdx = static_cast<ui32>(feat.Offset);
            nodeSplit.Shift            = feat.Shift;
            nodeSplit.Mask             = feat.Mask >> feat.Shift;
            nodeSplit.BinThreshold     = split.BinId;
            nodeSplit.IsOneHot         = feat.OneHotFeature;
            nodeSplitMap[bfsNode]      = nodeSplit;

            // Assign dense IDs for the two children.
            // Left child takes the current leafId slot (left reuses leafId).
            // Right child gets a new slot at the end.
            const ui32 leftLeafId  = leafId;
            const ui32 rightLeafId = static_cast<ui32>(leafBfsId.size());

            leafBfsId.push_back(rightBfs);
            leafBfsId[leftLeafId] = leftBfs;
            leafDepth.push_back(myDepth + 1u);
            leafDepth[leftLeafId] = myDepth + 1u;
            result.NumLeaves++;

            // Update leafDocVec: docs in leafId that go right now get rightLeafId.
            // Extract feature values for docs in this leaf.
            const auto& compressedData = dataset.GetCompressedIndex().GetCompressedData();
            TMLXDevice::EvalNow(compressedData);
            const uint32_t* dataPtr = compressedData.data<uint32_t>();
            const ui32 lineSize = dataset.GetCompressedIndex().GetNumUi32PerDoc();

            for (ui32 d = 0; d < numDocs; ++d) {
                if (leafDocVec[d] != leafId) continue;
                uint32_t packed = dataPtr[d * lineSize + nodeSplit.FeatureColumnIdx];
                uint32_t fv = (packed >> nodeSplit.Shift) & nodeSplit.Mask;
                uint32_t goRight = nodeSplit.IsOneHot
                    ? (fv == nodeSplit.BinThreshold ? 1u : 0u)
                    : (fv >  nodeSplit.BinThreshold ? 1u : 0u);
                if (goRight) {
                    leafDocVec[d] = rightLeafId;
                }
                // else: stays at leftLeafId (same as leafId)
            }

            // Evaluate the two new children and push to the queue if they have valid splits.
            evalLeaf(leftLeafId);
            evalLeaf(rightLeafId);
        }

        // Store final results
        result.NodeSplitMap = std::move(nodeSplitMap);
        result.LeafBfsIds.assign(leafBfsId.begin(), leafBfsId.end());

        // Build the LeafDocIds MLX array from leafDocVec
        result.LeafDocIds = mx::array(
            reinterpret_cast<const int32_t*>(leafDocVec.data()),
            {static_cast<int>(numDocs)}, mx::uint32
        );
        TMLXDevice::EvalNow(result.LeafDocIds);

        CATBOOST_INFO_LOG << "CatBoost-MLX Lossguide: tree built with "
            << result.NumLeaves << " leaves, "
            << result.NodeSplitMap.size() << " internal nodes" << Endl;

        return result;
    }

}  // namespace NCatboostMlx
