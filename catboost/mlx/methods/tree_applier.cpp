#include "tree_applier.h"

#include <catboost/libs/logging/logging.h>

namespace NCatboostMlx {

    void ApplyObliviousTree(
        TMLXDataSet& dataset,
        const TVector<TObliviousSplitLevel>& splits,
        const mx::array& leafValues,
        ui32 approxDimension
    ) {
        const ui32 numDocs = dataset.GetNumDocs();
        const ui32 depth = splits.size();
        const auto& compressedData = dataset.GetCompressedIndex().GetCompressedData();
        const ui32 lineSize = dataset.GetCompressedIndex().GetNumUi32PerDoc();

        // TODO(Phase 7): Replace with Metal kernel dispatch (apply_oblivious_tree in leaves.metal)
        // For now, use MLX ops as reference implementation.
        //
        // Compute leaf index for each document using MLX array operations:
        // leafIdx = 0
        // for each level:
        //   featureValues = (compressedData[:, col] >> shift) & mask
        //   leafIdx |= (featureValues > threshold) << level

        auto leafIndices = mx::zeros({static_cast<int>(numDocs)}, mx::uint32);

        for (ui32 level = 0; level < depth; ++level) {
            const auto& split = splits[level];

            // Extract the feature column
            auto column = mx::slice(compressedData,
                {0, static_cast<int>(split.FeatureColumnIdx)},
                {static_cast<int>(numDocs), static_cast<int>(split.FeatureColumnIdx + 1)}
            );
            column = mx::reshape(column, {static_cast<int>(numDocs)});

            // Extract feature value: (column >> shift) & mask
            auto featureValues = mx::bitwise_and(
                mx::right_shift(column, mx::array(static_cast<int>(split.Shift))),
                mx::array(static_cast<int>(split.Mask))
            );

            // Compare to threshold: OneHot uses equality, ordinal uses greater-than
            mx::array goRight;
            if (split.IsOneHot) {
                goRight = mx::equal(featureValues,
                    mx::array(static_cast<int>(split.BinThreshold)));
            } else {
                goRight = mx::greater(featureValues,
                    mx::array(static_cast<int>(split.BinThreshold)));
            }

            // Cast bool to uint32 and shift to correct bit position
            auto bits = mx::astype(goRight, mx::uint32);
            bits = mx::left_shift(bits, mx::array(static_cast<int>(level)));

            // OR into leaf index
            leafIndices = mx::bitwise_or(leafIndices, bits);
        }

        // Gather leaf values using leaf indices
        auto docLeafValues = mx::take(leafValues, leafIndices, 0);

        // Update cursor: cursor += leafValues[leafIdx]
        auto& cursor = dataset.GetCursor();
        if (approxDimension > 1) {
            // leafValues: [numLeaves, K], docLeafValues: [numDocs, K]
            // Transpose to [K, numDocs] to match cursor shape [K, numDocs]
            docLeafValues = mx::transpose(docLeafValues);
            cursor = mx::add(cursor, docLeafValues);
        } else {
            // Cursor may be [1, numDocs] or [numDocs] — flatten, add, then restore shape.
            auto cursorShape = cursor.shape();
            cursor = mx::add(
                mx::reshape(cursor, {static_cast<int>(numDocs)}),
                docLeafValues
            );
            cursor = mx::reshape(cursor, cursorShape);
        }

        // Update partition assignments
        dataset.GetPartitions() = leafIndices;

        TMLXDevice::EvalNow({cursor, dataset.GetPartitions()});

        CATBOOST_DEBUG_LOG << "CatBoost-MLX: Applied tree depth=" << depth
            << " to " << numDocs << " documents" << Endl;
    }

}  // namespace NCatboostMlx
