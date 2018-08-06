#include "model_builder.h"

namespace NCatboostCuda {

    template<>
    TObliviousTreeModel BuildTreeLikeModel<TObliviousTreeModel>(const TVector<TLeafPath>& leaves,
                                                                const TVector<double>& leafWeights,
                                                                const TVector<TVector<float>>& leafValues) {
        CB_ENSURE(leaves.size(), "Error: empty tree");
        CB_ENSURE(leaves.size() == leafValues.size());
        CB_ENSURE(leaves.size() == leafWeights.size());

        const ui32 outputDim = leafValues[0].size();
        for (auto& leaf : leaves) {
            CB_ENSURE(leaf.Splits == leaves[0].Splits);
        }


        TObliviousTreeStructure structure;
        structure.Splits = leaves[0].Splits;
        const ui32 leavesCount = 1 << structure.GetDepth();
        CB_ENSURE(leaves.size() == leavesCount);

        TVector<ui32> binIds(leaves.size());

        ui32 checkSum = 0;
        for (ui32 i = 0; i < leaves.size(); ++i) {
            ui32 bin = 0;
            for (ui32 level = 0; level < structure.Splits.size(); ++level) {
                ESplitValue direction = leaves[i].Directions[level];
                bin |= ((direction == ESplitValue::Zero) ? 0 : 1) << level;
            }
            Y_VERIFY(bin < leavesCount);
            binIds[i] = bin;
            checkSum += bin;
        }
        CB_ENSURE(checkSum == leavesCount * (leavesCount - 1) / 2, checkSum << " " << leavesCount);

        CB_ENSURE(leafWeights.size() == leavesCount);
        for (auto& values : leafValues) {
            CB_ENSURE(values.size() == outputDim);
        }

        TVector<double> resultWeights(leavesCount);
        TVector<float> resultValues(leafValues.size() * leavesCount);

        for (ui32 i = 0; i < leavesCount; ++i) {
            ui32 bin = binIds[i];
            resultWeights[bin] = leafWeights[i];
            for (ui32 dim  = 0; dim < outputDim; ++dim) {
                resultValues[bin * outputDim + dim] = leafValues[i][dim];
            }
        }
        return TObliviousTreeModel(std::move(structure),
                                   resultValues,
                                   resultWeights,
                                   outputDim
                                   );

    }
}
