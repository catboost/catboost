#include "model_build_helper.h"

TObliviousTreeBuilder::TObliviousTreeBuilder(const TVector<TFloatFeature>& allFloatFeatures, const TVector<TCatFeature>& allCategoricalFeatures, int approxDimension)
    : ApproxDimension(approxDimension)
    , FloatFeatures(allFloatFeatures)
    , CatFeatures(allCategoricalFeatures)
{
    for (int i = 0; i < FloatFeatures.ysize(); ++i) {
        CB_ENSURE(FloatFeatures[i].FeatureIndex == i && FloatFeatures[i].FlatFeatureIndex != -1);
    }
    for (int i = 0; i < CatFeatures.ysize(); ++i) {
        CB_ENSURE(CatFeatures[i].FeatureIndex == i && CatFeatures[i].FlatFeatureIndex != -1);
    }
}

void TObliviousTreeBuilder::AddTree(const TVector<TModelSplit>& modelSplits,
                                    const TVector<TVector<double>>& treeLeafValues,
                                    const TVector<double>& treeLeafWeights
) {
    auto& leafValues = LeafValues.emplace_back();
    CB_ENSURE(ApproxDimension == treeLeafValues.ysize());
    auto leafCount = treeLeafValues.at(0).size();
    leafValues.resize(ApproxDimension * leafCount);
    for (size_t dimension = 0; dimension < treeLeafValues.size(); ++dimension) {
        for (size_t leafId = 0; leafId < leafCount; ++leafId) {
            leafValues[leafId * ApproxDimension + dimension] = treeLeafValues[dimension][leafId];
        }
    }
    if (!treeLeafWeights.empty()) {
        LeafWeights.push_back(treeLeafWeights);
    }
    Trees.emplace_back(modelSplits);
}

TObliviousTrees TObliviousTreeBuilder::Build() {
    TSet<TModelSplit> modelSplitSet;
    for (const auto& tree : Trees) {
        for (const auto& split : tree) {
            modelSplitSet.insert(split);
            if (split.Type == ESplitType::OnlineCtr) {
                auto& proj = split.OnlineCtr.Ctr.Base.Projection;
                for (const auto& binF : proj.BinFeatures) {
                    modelSplitSet.insert(TModelSplit(binF));
                }
                for (const auto& oheFeature : proj.OneHotFeatures) {
                    modelSplitSet.insert(TModelSplit(oheFeature));
                }
            }
        }
    }
    // indexing binary tree splits
    THashMap<TModelSplit, int> binFeatureIndexes;
    for (const auto& split : modelSplitSet) {
        const int binFeatureIdx = binFeatureIndexes.ysize();
        Y_ASSERT(!binFeatureIndexes.has(split));
        binFeatureIndexes[split] = binFeatureIdx;
    }
    Y_ASSERT(modelSplitSet.size() == binFeatureIndexes.size());
    // filling binary tree splits
    TObliviousTrees result;
    result.ApproxDimension = ApproxDimension;
    result.LeafValues = LeafValues;
    result.LeafWeights = LeafWeights;
    for (const auto& treeStruct : Trees) {
        for (const auto& split : treeStruct) {
            result.TreeSplits.push_back(binFeatureIndexes.at(split));
        }
        if (result.TreeStartOffsets.empty()) {
            result.TreeStartOffsets.push_back(0);
        } else {
            result.TreeStartOffsets.push_back(result.TreeStartOffsets.back() + result.TreeSizes.back());
        }
        result.TreeSizes.push_back(treeStruct.ysize());
    }
    for (const auto& split : modelSplitSet) {
        if (split.Type == ESplitType::FloatFeature) {
            if (result.FloatFeatures.empty() || result.FloatFeatures.back().FeatureIndex != split.FloatFeature.FloatFeature) {
                auto& ref = result.FloatFeatures.emplace_back();
                ref = FloatFeatures[split.FloatFeature.FloatFeature];
                ref.Borders.clear();
            }
            result.FloatFeatures.back().Borders.push_back(split.FloatFeature.Split);
        } else if (split.Type == ESplitType::OneHotFeature) {
            if (result.OneHotFeatures.empty() || result.OneHotFeatures.back().CatFeatureIndex != split.OneHotFeature.CatFeatureIdx) {
                auto& ref = result.OneHotFeatures.emplace_back();
                ref.CatFeatureIndex = split.OneHotFeature.CatFeatureIdx;
            }
            result.OneHotFeatures.back().Values.push_back(split.OneHotFeature.Value);
        } else {
            if (result.CtrFeatures.empty() || result.CtrFeatures.back().Ctr != split.OnlineCtr.Ctr) {
                result.CtrFeatures.emplace_back();
                result.CtrFeatures.back().Ctr = split.OnlineCtr.Ctr;
            }
            result.CtrFeatures.back().Borders.push_back(split.OnlineCtr.Border);
        }
    }
    result.CatFeatures = CatFeatures;

    result.UpdateMetadata();
    return result;
}
