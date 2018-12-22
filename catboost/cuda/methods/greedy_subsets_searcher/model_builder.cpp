#include "model_builder.h"

#include <catboost/libs/helpers/exception.h>

#include <util/stream/labeled.h>
#include <util/generic/array_ref.h>

using NCatboostCuda::TLeafPath;

static void ValidateParameters(
    const TConstArrayRef<TLeafPath> leaves,
    const TConstArrayRef<double> leafWeights,
    const TConstArrayRef<TVector<float>> leafValues) {

    CB_ENSURE(!leaves.empty(), "Error: empty tree");

    const auto depth = leaves.front().Splits.size();
    const auto expectedLeavesCount = size_t(1) << depth;
    CB_ENSURE(leaves.size() == expectedLeavesCount, LabeledOutput(leaves.size(), expectedLeavesCount));

    CB_ENSURE(leaves.size() == leafValues.size(), LabeledOutput(leaves.size(), leafValues.size()));
    CB_ENSURE(leaves.size() == leafWeights.size(), LabeledOutput(leaves.size(), leafWeights.size()));

    for (size_t i = 1; i < leaves.size(); ++i) {
        CB_ENSURE(leaves[i].Splits == leaves.front().Splits, LabeledOutput(i));
    }

    for (size_t i = 1; i < leafValues.size(); ++i) {
        CB_ENSURE(leafValues[i].size() == leafValues.front().size(), LabeledOutput(i));
    }
}

namespace NCatboostCuda {

    template <>
    TObliviousTreeModel BuildTreeLikeModel<TObliviousTreeModel>(const TVector<TLeafPath>& leaves,
                                                                const TVector<double>& leafWeights,
                                                                const TVector<TVector<float>>& leafValues) {

        ValidateParameters(leaves, leafWeights, leafValues);

        const auto depth = leaves.front().Splits.size();
        const auto leavesCount = size_t(1) << depth;
        const auto outputDimention = leafValues.front().size();

        TVector<ui32> binIds(leavesCount);

        ui32 checkSum = 0;
        for (size_t i = 0; i < leavesCount; ++i) {
            ui32 bin = 0;
            for (size_t level = 0; level < depth; ++level) {
                const auto direction = leaves[i].Directions[level];
                bin |= ((direction == ESplitValue::Zero) ? 0 : 1) << level;
            }

            Y_VERIFY(bin < leavesCount);
            binIds[i] = bin;
            checkSum += bin;
        }

        const auto expectedCheckSum = leavesCount * (leavesCount - 1) / 2;
        CB_ENSURE(checkSum == expectedCheckSum, LabeledOutput(checkSum, expectedCheckSum, leavesCount));

        TVector<double> resultWeights(leavesCount);
        TVector<float> resultValues(outputDimention * leavesCount);

        for (size_t i = 0; i < leavesCount; ++i) {
            ui32 bin = binIds[i];
            resultWeights[bin] = leafWeights[i];
            for (size_t dim  = 0; dim < outputDimention; ++dim) {
                resultValues[bin * outputDimention + dim] = leafValues[i][dim];
            }
        }

        TObliviousTreeStructure structure;
        structure.Splits = leaves[0].Splits;

        return TObliviousTreeModel(
            std::move(structure),
            std::move(resultValues),
            std::move(resultWeights),
            outputDimention);
    }
}
