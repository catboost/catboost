#pragma once

#include <catboost/libs/model/model.h>

#include <catboost/libs/helpers/exception.h>

#include <util/generic/set.h>

class TObliviousTreeBuilder {
public:
    TObliviousTreeBuilder(TVector<TFloatFeature>& allFloatFeatures, TVector<TCatFeature>& allCategoricalFeatures)
        : FloatFeatures(allFloatFeatures)
        , CatFeatures(allCategoricalFeatures)
    {
        for (int i = 0; i < FloatFeatures.ysize(); ++i) {
            CB_ENSURE(FloatFeatures[i].FeatureIndex == i && FloatFeatures[i].FlatFeatureIndex != -1);
        }
        for (int i = 0; i < CatFeatures.ysize(); ++i) {
            CB_ENSURE(CatFeatures[i].FeatureIndex == i && CatFeatures[i].FlatFeatureIndex != -1);
        }
    }
    void AddTree(TVector<TModelSplit>& modelSplits, TVector<TVector<double>>& treeLeafValues) {
        auto& leafValues = LeafValues.emplace_back();
        if (ApproxDimension == 0) {
            ApproxDimension = treeLeafValues.ysize();
        } else {
            CB_ENSURE(ApproxDimension == treeLeafValues.ysize());
        }
        auto leafCount = treeLeafValues.at(0).size();
        leafValues.resize(ApproxDimension * leafCount);
        for (size_t dimension = 0; dimension < treeLeafValues.size(); ++dimension) {
            for (size_t leafId = 0; leafId < leafCount; ++leafId) {
                leafValues[leafId * ApproxDimension + dimension] = treeLeafValues[dimension][leafId];
            }
        }
        Trees.emplace_back(modelSplits);
    }

    TObliviousTrees Build() {
        yset<TModelSplit> ModelSplitSet;
        for (const auto& tree : Trees) {
            for (const auto& split : tree) {
                ModelSplitSet.insert(split);
                if (split.Type == ESplitType::OnlineCtr) {
                    auto& proj = split.OnlineCtr.Ctr.Base.Projection;
                    for (const auto& binF : proj.BinFeatures) {
                        ModelSplitSet.insert(TModelSplit(binF));
                    }
                    for (const auto& oheFeature : proj.OneHotFeatures) {
                        ModelSplitSet.insert(TModelSplit(oheFeature));
                    }
                }
            }
        }
        TObliviousTrees result;
        result.ApproxDimension = ApproxDimension;
        result.LeafValues = LeafValues;
        // indexing binary tree splits
        THashMap<TModelSplit, int> binFeatureIndexes;
        TVector<TModelSplit> usedSplits(ModelSplitSet.begin(), ModelSplitSet.end());
        for (int i = 0; i < usedSplits.ysize(); ++i) {
            int binFeatureIdx = binFeatureIndexes.ysize();
            Y_ASSERT(!binFeatureIndexes.has(usedSplits[i]));
            binFeatureIndexes[usedSplits[i]] = binFeatureIdx;
        }
        Y_ASSERT(usedSplits.size() == binFeatureIndexes.size());
        // filling binary tree splits
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
        for (const auto& split : usedSplits) {
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
private:
    int ApproxDimension = 0;

    TVector<TVector<TModelSplit>> Trees;
    TVector<TVector<double>> LeafValues;
    TVector<TFloatFeature> FloatFeatures;
    TVector<TCatFeature> CatFeatures;
};
