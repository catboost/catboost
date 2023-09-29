#include "carry.h"

#include <catboost/libs/model/model_build_helper.h>
#include <catboost/libs/model/static_ctr_provider.h>


namespace {
    struct TCompareByName {
        inline bool operator()(const TFloatFeature& feature, const TString& name) {
            return feature.FeatureId == name;
        }
    };
    struct TCompareByFlatIndex {
        inline bool operator()(const TFloatFeature& feature, const int index) {
            return feature.Position.FlatIndex == index;
        }
    };
    struct TCompareByIndex {
        inline bool operator()(const TFloatFeature& feature, const int index) {
            return feature.Position.Index == index;
        }
    };

    template <typename Cmp, typename V>
    TVector<TFeaturePosition> GetPositionBy(const TFullModel& model, const TVector<V>& factors) {
        TVector<TFeaturePosition> factorIds;
        const auto& floatFeatures = model.ModelTrees->GetFloatFeatures();
        for (const auto& factor : factors) {
            const auto it = std::find_if(
                floatFeatures.begin(),
                floatFeatures.end(),
                [&factor](const auto& node) { return Cmp()(node, factor); });

            CB_ENSURE(it != floatFeatures.end(), "feature not found");
            factorIds.emplace_back(it->Position);
        }
        return factorIds;
    }

    ui64 RemoveMask(ui64 path, ui64 mask) {
        // reduce path in oblivious tree by mask
        ui64 result = 0;
        size_t depth = 0;
        while (path) {
            if ((mask & 1) == 0) {
                result |= (path & 1) << depth++;
            }
            mask /= 2;
            path /= 2;
        }
        return result;
    }

    bool CheckCtrIndependice(const TFullModel* model, const TVector<TFeaturePosition>& factorIds) {
        const auto ctrProvider = dynamic_cast<const TStaticCtrProvider*>(model->CtrProvider.Get());

        // no ctr at all
        if (ctrProvider == nullptr) {
            return true;
        }

        // check if each of factorId in factorIds not precent in each bin feature of ctrs
        for (const auto& [ctr, table] : ctrProvider->CtrData.LearnCtrs) {
            for (const auto& floatSplit : ctr.Projection.BinFeatures) {
                const auto it = std::find_if(
                    factorIds.begin(),
                    factorIds.end(),
                    [&floatSplit](const auto& factor) { return floatSplit.FloatFeature == factor.Index; });
                if (it == factorIds.end()) {
                    return false;
                }
            }
        }
        return true;
    }

    void EnsureCarryConstraints(const TFullModel& model, const TVector<TFeaturePosition>& factorIds) {
        CB_ENSURE(model.GetDimensionsCount() == 1, "Only one-dimensional models are supported");
        CB_ENSURE(model.IsOblivious(), "Only oblivious trees are supported");
        CB_ENSURE(CheckCtrIndependice(&model, factorIds), "Carry for float features used in ctrs not implemented");
    }

    void ClearFloatFeatures(
        TVector<TFloatFeature>& floatFeatures,
        const TVector<TFeaturePosition>& factorToRemove
    ) {
        EraseIf(floatFeatures, [&factorToRemove](const auto& factor) {
            return FindIf(factorToRemove, [&factor](const auto& toRemove) {
                       return factor.Position.Index == toRemove.Index;
                   }) != factorToRemove.end();
        });

        for (auto& feature : floatFeatures) {
            const auto shift = CountIf(factorToRemove, [&feature](const auto& toRemove) {
                return feature.Position.FlatIndex > toRemove.FlatIndex;
            });
            feature.Position.Index -= shift;
            feature.Position.FlatIndex -= shift;
        }
    }

    void ClearCatFeatures(
        TVector<TCatFeature>& catFeatures,
        const TVector<TFeaturePosition>& factorToRemove
    ) {
        for (auto& feature : catFeatures) {
            const auto shift = CountIf(factorToRemove, [&feature](const auto& toRemove) {
                return feature.Position.FlatIndex > toRemove.FlatIndex;
            });
            feature.Position.FlatIndex -= shift;
        }
    }

    void FixSplitFeatureIndexes(
        TModelSplit& split,
        const TVector<TFeaturePosition>& factorToRemove
    ) {
        switch (split.Type) {
            case ESplitType::FloatFeature: {
                auto& featureId = split.FloatFeature.FloatFeature;
                featureId -= CountIf(factorToRemove, [&featureId](const auto& node) { return node.Index < featureId; });
                break;
            }
            case ESplitType::OnlineCtr:
                for (auto& floatSplit : split.OnlineCtr.Ctr.Base.Projection.BinFeatures) {
                    auto& featureId = floatSplit.FloatFeature;
                    featureId -= CountIf(factorToRemove, [&featureId](const auto& node) { return node.Index < featureId; });
                }
                break;
            case ESplitType::OneHotFeature:
            case ESplitType::EstimatedFeature:
                break;
            default:
                CB_ENSURE(false, "Unexpected split type");
        }
    }

    TIntrusivePtr<ICtrProvider> RebuildCtrProvider(const TFullModel& model, const TVector<TFeaturePosition>& factorIds) {
        const auto ctrProvider = dynamic_cast<const TStaticCtrProvider*>(model.CtrProvider.Get());
        if (ctrProvider == nullptr) {
            return nullptr;
        }

        TCtrData ctrData;
        for (const auto& [orignCtr, valueTable] : ctrProvider->CtrData.LearnCtrs) {
            TModelCtrBase ctr = orignCtr;
            for (auto& floatSplit : ctr.Projection.BinFeatures) {
                auto& featureId = floatSplit.FloatFeature;
                featureId -= CountIf(factorIds, [&featureId](const auto& node) { return node.Index < featureId; });
            }
            ctrData.LearnCtrs.emplace(ctr, valueTable);
        }
        return MakeIntrusive<TStaticCtrProvider>(ctrData);
    }
}

TFullModel CarryModel(const TFullModel& model, const TVector<TFeaturePosition>& factorIds, const TVector<TVector<double>>& factorValues) {
    EnsureCarryConstraints(model, factorIds);

    const auto trees = model.ModelTrees;
    const auto biasFromModel = model.GetScaleAndBias().GetBiasRef();

    TVector<double> bias(factorValues.front().size(), biasFromModel.empty() ? 0 : biasFromModel.front());
    TVector<TFloatFeature> floatFeatures(trees->GetFloatFeatures().begin(), trees->GetFloatFeatures().end());
    TVector<TCatFeature> catFeatures(trees->GetCatFeatures().begin(), trees->GetCatFeatures().end());

    // update factors FlatIndex with respect to carried factors
    ClearFloatFeatures(floatFeatures, factorIds);
    ClearCatFeatures(catFeatures, factorIds);

    const auto& data = trees->GetModelTreeData();
    const auto& binFeatures = trees->GetBinFeatures();
    const auto& applyData = trees->GetApplyData();
    const auto& leafOffsets = applyData->TreeFirstLeafOffsets;

    TObliviousTreeBuilder builder(floatFeatures, catFeatures, {}, {}, factorValues.front().size());
    for (size_t treeIdx = 0; treeIdx < data->GetTreeSizes().size(); ++treeIdx) {
        ui64 mask = 0;
        TVector<ui64> paths(factorValues.front().size(), 0);

        TVector<TModelSplit> modelSplits;
        int splitIdxBegin = data->GetTreeStartOffsets()[treeIdx];
        for (int depth = 0; depth < data->GetTreeSizes()[treeIdx]; ++depth) {
            auto split = binFeatures[data->GetTreeSplits()[splitIdxBegin + depth]];
            if (split.Type == ESplitType::FloatFeature) {
                const auto it = FindIf(factorIds, [&split](const auto& factor) { return split.FloatFeature.FloatFeature == factor.Index; });
                if (it != factorIds.end()) {
                    mask |= (1ull << depth);
                    const auto& values = factorValues[it - factorIds.begin()];
                    for (size_t i = 0; i < values.size(); ++i) {
                        if (split.FloatFeature.Split < values[i]) {
                            paths[i] |= (1ull << depth);
                        }
                    }
                }
            }

            if ((mask >> depth) == 0) {
                FixSplitFeatureIndexes(split, factorIds);
                modelSplits.emplace_back(std::move(split));
            }
        }

        TConstArrayRef<double> leafValuesRef(
            data->GetLeafValues().begin() + leafOffsets[treeIdx],
            data->GetLeafValues().begin() + leafOffsets[treeIdx] + trees->GetDimensionsCount() * (1ull << data->GetTreeSizes()[treeIdx]));

        if (!modelSplits.empty()) {
            TVector<double> leafValues(paths.size() * (1ull << modelSplits.size()), 0);
            for (size_t idx = 0; idx < leafValuesRef.size(); ++idx) {
                for (size_t i = 0; i < paths.size(); ++i) {
                    if ((idx & mask) == paths[i]) {
                        leafValues[RemoveMask(idx, mask) * paths.size() + i] = leafValuesRef[idx];
                    }
                }
            }
            builder.AddTree(modelSplits, leafValues, TConstArrayRef<double>());
        } else {
            for (size_t i = 0; i < paths.size(); ++i) {
                bias[i] += leafValuesRef[paths[i]];
            }
        }
    }

    TFullModel result;
    builder.Build(result.ModelTrees.GetMutable());
    result.SetScaleAndBias(TScaleAndBias(model.GetScaleAndBias().Scale, bias));
    result.CtrProvider = RebuildCtrProvider(model, factorIds);
    result.ModelInfo["model_guid"] = CreateGuidAsString();
    result.UpdateDynamicData();
    return result;
}

TFullModel UpliftModel(const TFullModel& model, const TVector<TFeaturePosition>& factorIds, const TVector<double>& baseValues, const TVector<double>& nextValues) {
    EnsureCarryConstraints(model, factorIds);
    CB_ENSURE(nextValues.size() == baseValues.size(), "Base and next values for factors must have same size");
    CB_ENSURE(factorIds.size() == baseValues.size(), "Factor names and factor values must have same size");

    const auto trees = model.ModelTrees;
    TVector<TFloatFeature> floatFeatures(trees->GetFloatFeatures().begin(), trees->GetFloatFeatures().end());
    TVector<TCatFeature> catFeatures(trees->GetCatFeatures().begin(), trees->GetCatFeatures().end());
    ClearFloatFeatures(floatFeatures, factorIds);
    ClearCatFeatures(catFeatures, factorIds);

    const auto biasFromModel = model.GetScaleAndBias().GetBiasRef();
    const auto scale = model.GetScaleAndBias().Scale;
    double bias = 0;

    const auto& data = trees->GetModelTreeData();
    const auto& binFeatures = trees->GetBinFeatures();
    const auto& applyData = trees->GetApplyData();
    const auto& leafOffsets = applyData->TreeFirstLeafOffsets;

    TObliviousTreeBuilder builder(floatFeatures, catFeatures, {}, {}, 1);
    for (size_t treeIdx = 0; treeIdx < data->GetTreeSizes().size(); ++treeIdx) {
        ui64 mask = 0;
        ui64 basePath = 0;
        ui64 nextPath = 0;
        TVector<TModelSplit> modelSplits;
        int splitIdxBegin = data->GetTreeStartOffsets()[treeIdx];
        for (int depth = 0; depth < data->GetTreeSizes()[treeIdx]; ++depth) {
            auto split = binFeatures[data->GetTreeSplits()[splitIdxBegin + depth]];
            if (split.Type == ESplitType::FloatFeature) {
                const auto it = FindIf(factorIds, [&split](const auto& factor) { return split.FloatFeature.FloatFeature == factor.Index; });
                if (it != factorIds.end()) {
                    const auto base = split.FloatFeature.Split < baseValues[it - factorIds.begin()];
                    const auto next = split.FloatFeature.Split < nextValues[it - factorIds.begin()];
                    mask |= (1ull << depth);

                    if (base) {
                        basePath |= (1ull << depth);
                    }

                    if (next) {
                        nextPath |= (1ull << depth);
                    }
                }
            }
            if ((mask >> depth) == 0) {
                FixSplitFeatureIndexes(split, factorIds);
                modelSplits.emplace_back(std::move(split));
            }
        }

        if (mask) {
            TConstArrayRef<double> leafValuesRef(
                data->GetLeafValues().begin() + leafOffsets[treeIdx],
                data->GetLeafValues().begin() + leafOffsets[treeIdx] + trees->GetDimensionsCount() * (1ull << data->GetTreeSizes()[treeIdx]));
            if (!modelSplits.empty()) {
                TVector<double> leafValues(1ull << modelSplits.size(), 0.0);
                for (size_t idx = 0; idx < leafValuesRef.size(); ++idx) {
                    leafValues[RemoveMask(idx, mask)] =
                        leafValuesRef[(idx & (~mask)) | (nextPath & mask)] -
                        leafValuesRef[(idx & (~mask)) | (basePath & mask)];
                }
                builder.AddTree(modelSplits, leafValues, TConstArrayRef<double>());
            } else {
                bias += leafValuesRef[nextPath] - leafValuesRef[basePath];
            }
        }
    }

    TFullModel result;
    builder.Build(result.ModelTrees.GetMutable());
    result.SetScaleAndBias(TScaleAndBias(scale, {bias}));
    result.CtrProvider = RebuildCtrProvider(model, factorIds);
    result.ModelInfo["model_guid"] = CreateGuidAsString();
    result.UpdateDynamicData();
    return result;
}

TFullModel CarryModelByFeatureIndex(const TFullModel& model, const TVector<int>& factorFeatureIndexes, const TVector<TVector<double>>& factorsValues) {
    return CarryModel(model, GetPositionBy<TCompareByIndex>(model, factorFeatureIndexes), factorsValues);
}

TFullModel CarryModelByFlatIndex(const TFullModel& model, const TVector<int>& factorFlatIndexes, const TVector<TVector<double>>& factorsValues) {
    return CarryModel(model, GetPositionBy<TCompareByFlatIndex>(model, factorFlatIndexes), factorsValues);
}

TFullModel CarryModelByName(const TFullModel& model, const TVector<TString>& factorNames, const TVector<TVector<double>>& factorsValues) {
    return CarryModel(model, GetPositionBy<TCompareByName>(model, factorNames), factorsValues);
}

TFullModel UpliftModelByFeatureIndex(const TFullModel& model, const TVector<int>& factorFeatureIndexes, const TVector<double>& baseValues, const TVector<double>& nextValues) {
    return UpliftModel(model, GetPositionBy<TCompareByIndex>(model, factorFeatureIndexes), baseValues, nextValues);
}

TFullModel UpliftModelByFlatIndex(const TFullModel& model, const TVector<int>& factorFlatIndexes, const TVector<double>& baseValues, const TVector<double>& nextValues) {
    return UpliftModel(model, GetPositionBy<TCompareByFlatIndex>(model, factorFlatIndexes), baseValues, nextValues);
}

TFullModel UpliftModelByName(const TFullModel& model, const TVector<TString>& factorNames, const TVector<double>& baseValues, const TVector<double>& nextValues) {
    return UpliftModel(model, GetPositionBy<TCompareByName>(model, factorNames), baseValues, nextValues);
}
