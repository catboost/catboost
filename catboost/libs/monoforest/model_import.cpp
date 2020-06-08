#include "model_import.h"


namespace NMonoForest {
    THolder<IModelImporter<TObliviousTree>> MakeCatBoostImporter(const TFullModel& model) {
        TCatBoostGrid grid(model);
        auto scaleAndBias = model.GetScaleAndBias();

        const auto& trees = *model.ModelTrees.Get();
        CB_ENSURE(trees.IsOblivious());

        const auto& treeSizes = trees.GetTreeSizes();
        const auto& treeStartOffsets = trees.GetTreeStartOffsets();
        const auto& leafValues = trees.GetLeafValues();
        const auto& leafWeights = trees.GetLeafWeights();
        const auto& leafCounts = trees.GetTreeLeafCounts();
        const auto& splits = trees.GetTreeSplits();

        TVector<TBinarySplit> binSplits;
        binSplits.reserve(trees.GetBinFeatures().size());
        for (const auto& split : trees.GetBinFeatures()) {
            binSplits.emplace_back(grid.ToBinarySplit(split));
        }


        int leafValuesOffset = 0;
        int leafWeightsOffset = 0;
        TAdditiveModel<TObliviousTree> additiveModel;
        for (auto tree : xrange(trees.GetTreeCount())) {
            TVector<TBinarySplit> treeSplits;
            for (auto idx : xrange(treeStartOffsets[tree], treeStartOffsets[tree] + treeSizes[tree])) {
                treeSplits.push_back(binSplits[splits[idx]]);
            }
            TVector<double> treeLeafValues(
                leafValues.begin() + leafValuesOffset,
                leafValues.begin() + leafValuesOffset + leafCounts[tree] * trees.GetDimensionsCount());

            double  bias = tree == 0 ? scaleAndBias.Bias : 0;
            for (auto& leafVal : treeLeafValues) {
                leafVal = scaleAndBias.Scale * (bias + leafVal);
            }
            leafValuesOffset += leafCounts[tree] * trees.GetDimensionsCount();
            TVector<double> treeLeafWeights(
                leafWeights.begin() + leafWeightsOffset,
                leafWeights.begin() + leafWeightsOffset + leafCounts[tree]);
            leafWeightsOffset += leafCounts[tree];
            additiveModel.AddWeakModel(TObliviousTree({treeSplits},
                                              std::move(treeLeafValues), std::move(treeLeafWeights),
                                              trees.GetDimensionsCount()));
        }
        return MakeHolder<TCatBoostObliviousModelImporter>(std::move(grid), std::move(additiveModel));
    }
}
