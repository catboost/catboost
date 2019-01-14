#include "model_build_helper.h"

#include <catboost/libs/helpers/exception.h>

#include <util/generic/algorithm.h>
#include <util/generic/hash.h>
#include <util/generic/hash_set.h>
#include <util/generic/set.h>
#include <util/generic/xrange.h>
#include <util/generic/ylimits.h>


TObliviousTreeBuilder::TObliviousTreeBuilder(const TVector<TFloatFeature>& allFloatFeatures, const TVector<TCatFeature>& allCategoricalFeatures, int approxDimension)
    : ApproxDimension(approxDimension)
    , FloatFeatures(allFloatFeatures)
    , CatFeatures(allCategoricalFeatures)
{
    if (!FloatFeatures.empty()) {
        CB_ENSURE(IsSorted(FloatFeatures.begin(), FloatFeatures.end(),
            [](const TFloatFeature &f1, const TFloatFeature &f2) {
                return f1.FeatureId < f2.FeatureId && f1.FlatFeatureIndex < f2.FlatFeatureIndex;
            }),
            "Float features should be sorted"
        );
        FloatFeaturesInternalIndexesMap.resize((size_t)FloatFeatures.back().FeatureIndex + 1, Max<size_t>());
        for (auto i : xrange(FloatFeatures.size())) {
            FloatFeaturesInternalIndexesMap.at((size_t)FloatFeatures[i].FeatureIndex) = i;
        }
    }
    if (!CatFeatures.empty()) {
        CB_ENSURE(IsSorted(CatFeatures.begin(), CatFeatures.end(),
            [] (const TCatFeature& f1, const TCatFeature& f2) {
                return f1.FeatureId < f2.FeatureId && f1.FlatFeatureIndex < f2.FlatFeatureIndex;
            }),
            "Cat features should be sorted"
        );
        CatFeaturesInternalIndexesMap.resize((size_t)CatFeatures.back().FeatureIndex + 1, Max<size_t>());
        for (auto i : xrange(CatFeatures.size())) {
            CatFeaturesInternalIndexesMap.at((size_t)CatFeatures[i].FeatureIndex) = i;
        }
    }
}

void TObliviousTreeBuilder::AddTree(const TVector<TModelSplit>& modelSplits,
                                    const TVector<TVector<double>>& treeLeafValues,
                                    TConstArrayRef<double> treeLeafWeights
) {
    CB_ENSURE(ApproxDimension == treeLeafValues.ysize());
    auto leafCount = treeLeafValues.at(0).size();

    TVector<double> leafValues(ApproxDimension * leafCount);

    for (size_t dimension = 0; dimension < treeLeafValues.size(); ++dimension) {
        CB_ENSURE(treeLeafValues[dimension].size() == (1u << modelSplits.size()));
        for (size_t leafId = 0; leafId < leafCount; ++leafId) {
            leafValues[leafId * ApproxDimension + dimension] = treeLeafValues[dimension][leafId];
        }
    }
    AddTree(modelSplits, leafValues, treeLeafWeights);
}

void TObliviousTreeBuilder::AddTree(const TVector<TModelSplit>& modelSplits,
                                    TConstArrayRef<double> treeLeafValues,
                                    TConstArrayRef<double> treeLeafWeights
) {
    CB_ENSURE((1u << modelSplits.size()) * ApproxDimension == treeLeafValues.size());
    LeafValues.insert(LeafValues.end(), treeLeafValues.begin(), treeLeafValues.end());
    if (!treeLeafWeights.empty()) {
        LeafWeights.push_back(TVector<double>(treeLeafWeights.begin(), treeLeafWeights.end()));
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
        Y_ASSERT(!binFeatureIndexes.contains(split));
        binFeatureIndexes[split] = binFeatureIdx;
    }
    Y_ASSERT(modelSplitSet.size() == binFeatureIndexes.size());
    // filling binary tree splits
    TObliviousTrees result;
    result.ApproxDimension = ApproxDimension;
    result.LeafValues = LeafValues;
    result.LeafWeights = LeafWeights;
    result.CatFeatures = CatFeatures;
    result.FloatFeatures = FloatFeatures;
    for (auto& feature : result.FloatFeatures) {
        feature.Borders.clear();
    }
    for (auto& feature : result.CatFeatures) {
        feature.UsedInModel = false;
    }
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
    THashSet<int> usedCatFeatureIndexes;
    for (const auto& split : modelSplitSet) {
        if (split.Type == ESplitType::FloatFeature) {
            const size_t internalFloatIndex = FloatFeaturesInternalIndexesMap.at((size_t)split.FloatFeature.FloatFeature);
            result.FloatFeatures.at(internalFloatIndex).Borders.push_back(split.FloatFeature.Split);
        } else if (split.Type == ESplitType::OneHotFeature) {
            usedCatFeatureIndexes.insert(split.OneHotFeature.CatFeatureIdx);
            if (result.OneHotFeatures.empty() || result.OneHotFeatures.back().CatFeatureIndex != split.OneHotFeature.CatFeatureIdx) {
                auto& ref = result.OneHotFeatures.emplace_back();
                ref.CatFeatureIndex = split.OneHotFeature.CatFeatureIdx;
            }
            result.OneHotFeatures.back().Values.push_back(split.OneHotFeature.Value);
        } else {
            const auto& projection = split.OnlineCtr.Ctr.Base.Projection;
            usedCatFeatureIndexes.insert(projection.CatFeatures.begin(), projection.CatFeatures.end());
            if (result.CtrFeatures.empty() || result.CtrFeatures.back().Ctr != split.OnlineCtr.Ctr) {
                result.CtrFeatures.emplace_back();
                result.CtrFeatures.back().Ctr = split.OnlineCtr.Ctr;
            }
            result.CtrFeatures.back().Borders.push_back(split.OnlineCtr.Border);
        }
    }
    for (auto usedCatFeatureIdx : usedCatFeatureIndexes) {
        result.CatFeatures[CatFeaturesInternalIndexesMap.at(usedCatFeatureIdx)].UsedInModel = true;
    }
    result.UpdateMetadata();
    return result;
}
