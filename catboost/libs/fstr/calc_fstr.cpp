#include "calc_fstr.h"

#include "compare_documents.h"
#include "feature_str.h"
#include "sage_values.h"
#include "shap_values.h"
#include "shap_interaction_values.h"
#include "util.h"

#include <catboost/libs/data/features_layout_helpers.h>
#include <catboost/libs/data/model_dataset_compatibility.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/query_info_helper.h>
#include <catboost/libs/logging/logging.h>
#include <catboost/libs/logging/profile_info.h>
#include <catboost/private/libs/options/enum_helpers.h>
#include <catboost/private/libs/options/restrictions.h>

#include <util/generic/algorithm.h>
#include <util/generic/cast.h>
#include <util/generic/hash.h>
#include <util/generic/xrange.h>
#include <util/string/builder.h>
#include <util/string/cast.h>
#include <util/system/compiler.h>
#include <util/system/info.h>

#include <cmath>
#include <functional>


using namespace NCB;


static TVector<TMxTree> BuildTrees(
    const THashMap<TFeature, int, TFeatureHash>& featureToIdx,
    const TFullModel& model)
{
    CB_ENSURE_INTERNAL(model.IsOblivious(), "BuildTrees are supported only for symmetric trees");

    TVector<TMxTree> trees(model.GetTreeCount());
    const auto binFeatures = model.ModelTrees->GetBinFeatures();
    for (int treeIdx = 0; treeIdx < trees.ysize(); ++treeIdx) {
        auto& tree = trees[treeIdx];
        const int leafCount = (1uLL << model.ModelTrees->GetModelTreeData()->GetTreeSizes()[treeIdx]);

        tree.Leaves.resize(leafCount);
        for (int leafIdx = 0; leafIdx < leafCount; ++leafIdx) {
            tree.Leaves[leafIdx].Vals.resize(model.ModelTrees->GetDimensionsCount());
        }
        auto firstTreeLeafPtr = model.ModelTrees->GetFirstLeafPtrForTree(treeIdx);
        for (int leafIdx = 0; leafIdx < leafCount; ++leafIdx) {
            for (int dim = 0; dim < (int)model.ModelTrees->GetDimensionsCount(); ++dim) {
                tree.Leaves[leafIdx].Vals[dim] = firstTreeLeafPtr[leafIdx
                    * model.ModelTrees->GetDimensionsCount() + dim];
            }
        }
        auto treeSplitsStart = model.ModelTrees->GetModelTreeData()->GetTreeStartOffsets()[treeIdx];
        auto treeSplitsStop = treeSplitsStart + model.ModelTrees->GetModelTreeData()->GetTreeSizes()[treeIdx];
        for (auto splitIdx = treeSplitsStart; splitIdx < treeSplitsStop; ++splitIdx) {
            auto feature = GetFeature(
                model,
                binFeatures[model.ModelTrees->GetModelTreeData()->GetTreeSplits()[splitIdx]]
            );
            tree.SrcFeatures.push_back(featureToIdx.at(feature));
        }
    }
    return trees;
}

static THashMap<TFeature, int, TFeatureHash> GetFeatureToIdxMap(
    const TFullModel& model,
    TVector<TFeature>* features)
{
    THashMap<TFeature, int, TFeatureHash> featureToIdx;
    const auto& modelBinFeatures = model.ModelTrees->GetBinFeatures();
    int binFeaturesNum = modelBinFeatures.size();

    for (auto binSplit : model.ModelTrees->GetModelTreeData()->GetTreeSplits()) {
        if (binSplit >= binFeaturesNum) {
            continue;
        }
        TFeature feature = GetFeature(model, modelBinFeatures[binSplit]);
        if (featureToIdx.contains(feature)) {
            continue;
        }
        int featureIdx = featureToIdx.ysize();
        featureToIdx[feature] = featureIdx;
        features->push_back(feature);
    }
    return featureToIdx;
}

TVector<std::pair<double, TFeature>> CalcFeatureEffectAverageChange(
    const TFullModel& model,
    TConstArrayRef<double> weights)
{
    if (model.GetTreeCount() == 0) {
        return TVector<std::pair<double, TFeature>>();
    }
    TVector<double> effect;
    TVector<TFeature> features;

    THashMap<TFeature, int, TFeatureHash> featureToIdx = GetFeatureToIdxMap(model, &features);
    if (model.IsOblivious()) {
        TVector<TMxTree> trees = BuildTrees(featureToIdx, model);

        TVector<TConstArrayRef<double>> mxTreeWeightsPresentation;
        auto applyData = model.ModelTrees->GetApplyData();
        auto leafOffsetPtr = applyData->TreeFirstLeafOffsets.data();
        const auto leafSizes = model.ModelTrees->GetModelTreeData()->GetTreeSizes();
        const int approxDimension = model.ModelTrees->GetDimensionsCount();
        for (size_t treeIdx = 0; treeIdx < model.GetTreeCount(); ++treeIdx) {
            mxTreeWeightsPresentation.push_back(
                TConstArrayRef<double>(
                    weights.data() + leafOffsetPtr[treeIdx] / approxDimension,
                    (1ull << leafSizes[treeIdx])
                )
            );
        }
        effect = CalcEffect(
            trees,
            mxTreeWeightsPresentation
        );
    } else {
        effect = CalcEffectForNonObliviousModel(
            model,
            featureToIdx,
            weights
        );
    }

    TVector<std::pair<double, int>> effectWithFeature;
    for (int i = 0; i < effect.ysize(); ++i) {
        effectWithFeature.emplace_back(effect[i], i);
    }
    StableSort(effectWithFeature.begin(), effectWithFeature.end(), std::greater<std::pair<double, int>>());

    TVector<std::pair<double, TFeature>> result;
    for (int i = 0; i < effectWithFeature.ysize(); ++i) {
        result.emplace_back(effectWithFeature[i].first, features[effectWithFeature[i].second]);
    }
    return result;
}

static TVector<std::pair<double, TFeature>> CalcFeatureEffectAverageChange(
    const TFullModel& model,
    const TDataProviderPtr dataset,
    NPar::ILocalExecutor* localExecutor)
{
    TVector<double> leavesStatistics;
    TConstArrayRef<double> weights;
    if (dataset) {
        CB_ENSURE(dataset->GetObjectCount() != 0, "no docs in pool");
        CB_ENSURE(dataset->MetaInfo.GetFeatureCount() > 0, "no features in pool");
        CATBOOST_INFO_LOG << "Used dataset leave statistics for fstr calculation" << Endl;

        leavesStatistics = CollectLeavesStatistics(*dataset, model, localExecutor);
        weights = leavesStatistics;
    } else {
        CB_ENSURE(
            !model.ModelTrees->GetModelTreeData()->GetLeafWeights().empty(),
            "CalcFeatureEffect requires either non-empty LeafWeights in model or provided dataset"
        );
        weights = model.ModelTrees->GetModelTreeData()->GetLeafWeights();
    }
    return CalcFeatureEffectAverageChange(model, weights);
}

TVector<std::pair<double, TFeature>> CalcFeatureEffect(
    const TFullModel& model,
    const TDataProviderPtr dataset,
    EFstrType type,
    NPar::ILocalExecutor* localExecutor,
    ECalcTypeShapValues calcType)
{
    type = AdjustFeatureImportanceType(type, model.GetLossFunctionName());
    if (type != EFstrType::PredictionValuesChange) {
        CB_ENSURE_SCALE_IDENTITY(model.GetScaleAndBias(), "feature effect");
    }
    if (type == EFstrType::LossFunctionChange) {
        CB_ENSURE(
            dataset,
            "Dataset is not provided for " << EFstrType::LossFunctionChange << ", choose "
                << EFstrType::PredictionValuesChange << " fstr type explicitly or provide dataset.");
        return CalcFeatureEffectLossChange(model, dataset, localExecutor, calcType);
    } else {
        CB_ENSURE_INTERNAL(
            type == EFstrType::PredictionValuesChange || type == EFstrType::InternalFeatureImportance,
            "Inappropriate fstr type " << type);
        return CalcFeatureEffectAverageChange(model, dataset, localExecutor);
    }
}

TVector<TFeatureEffect> CalcRegularFeatureEffect(
    const TVector<std::pair<double, TFeature>>& internalEffect,
    const TFullModel& model)
{
    int catFeaturesCount = model.GetNumCatFeatures();
    int floatFeaturesCount = model.GetNumFloatFeatures();
    int textFeaturesCount = model.GetNumTextFeatures();
    int embeddingFeaturesCount = model.GetNumEmbeddingFeatures();
    TVector<double> catFeatureEffect(catFeaturesCount);
    TVector<double> floatFeatureEffect(floatFeaturesCount);
    TVector<double> textFeatureEffect(textFeaturesCount);
    TVector<double> embeddingFeatureEffect(embeddingFeaturesCount);

    for (const auto& effectWithSplit : internalEffect) {
        TFeature feature = effectWithSplit.second;
        switch (feature.Type) {
            case ESplitType::FloatFeature:
                floatFeatureEffect[feature.FeatureIdx] += effectWithSplit.first;
                break;
            case ESplitType::OneHotFeature:
                catFeatureEffect[feature.FeatureIdx] += effectWithSplit.first;
                break;
            case ESplitType::OnlineCtr: {
                auto& proj = feature.Ctr.Base.Projection;
                int featuresInSplit = proj.BinFeatures.ysize() + proj.CatFeatures.ysize()
                    + proj.OneHotFeatures.ysize();
                double addEffect = effectWithSplit.first / featuresInSplit;
                for (const auto& binFeature : proj.BinFeatures) {
                    floatFeatureEffect[binFeature.FloatFeature] += addEffect;
                }
                for (auto catIndex : proj.CatFeatures) {
                    catFeatureEffect[catIndex] += addEffect;
                }
                for (auto oneHotFeature : proj.OneHotFeatures) {
                    catFeatureEffect[oneHotFeature.CatFeatureIdx] += addEffect;
                }
                break;
            }
            case ESplitType::EstimatedFeature: {
                if (feature.EstimatedFeature.SourceFeatureType == EEstimatedSourceFeatureType::Text) {
                    textFeatureEffect[feature.EstimatedFeature.SourceFeatureId] += effectWithSplit.first;
                } else {
                    CB_ENSURE(
                        feature.EstimatedFeature.SourceFeatureType == EEstimatedSourceFeatureType::Embedding
                    );
                    embeddingFeatureEffect[feature.EstimatedFeature.SourceFeatureId] += effectWithSplit.first;
                }
                break;
            }
        }
    }

    TVector<TFeatureEffect> regularFeatureEffect;
    for (int i = 0; i < catFeatureEffect.ysize(); ++i) {
        regularFeatureEffect.push_back(
            TFeatureEffect(catFeatureEffect[i], EFeatureType::Categorical, i)
        );
    }
    for (int i = 0; i < floatFeatureEffect.ysize(); ++i) {
        regularFeatureEffect.push_back(
            TFeatureEffect(floatFeatureEffect[i], EFeatureType::Float, i)
        );
    }
    for (int i = 0; i < textFeatureEffect.ysize(); ++i) {
        regularFeatureEffect.push_back(
            TFeatureEffect(textFeatureEffect[i], EFeatureType::Text, i)
        );
    }
    for (int i = 0; i < embeddingFeatureEffect.ysize(); ++i) {
        regularFeatureEffect.push_back(
            TFeatureEffect(embeddingFeatureEffect[i], EFeatureType::Embedding, i)
        );
    }

    Sort(
        regularFeatureEffect.rbegin(),
        regularFeatureEffect.rend(),
        [](const TFeatureEffect& left, const TFeatureEffect& right) {
            return left.Score < right.Score ||
                (left.Score == right.Score && left.Feature.Index > right.Feature.Index);
        }
    );
    return regularFeatureEffect;
}


TVector<double> GetFeatureEffectForLinearIndices(
    const TVector<std::pair<double, TFeature>>& featureEffect,
    const TFullModel& model)
{
    TVector<TFeatureEffect> regularEffect = CalcRegularFeatureEffect(featureEffect, model);

    const NCB::TFeaturesLayout layout = MakeFeaturesLayout(model);

    TVector<double> effect(layout.GetExternalFeatureCount());
    for (const auto& featureEffect : regularEffect) {
        int externalFeatureIdx = layout.GetExternalFeatureIdx(
            featureEffect.Feature.Index,
            featureEffect.Feature.Type
        );
        effect[externalFeatureIdx] = featureEffect.Score;
    }

    return effect;
}

TVector<double> CalcRegularFeatureEffect(
    const TFullModel& model,
    const TDataProviderPtr dataset,
    EFstrType type,
    NPar::ILocalExecutor* localExecutor,
    ECalcTypeShapValues calcType)
{
    return GetFeatureEffectForLinearIndices(
        CalcFeatureEffect(model, dataset, type, localExecutor, calcType),
        model
    );
}

TVector<TInternalFeatureInteraction> CalcInternalFeatureInteraction(const TFullModel& model) {
    if (model.GetTreeCount() == 0) {
        return TVector<TInternalFeatureInteraction>();
    }
    CB_ENSURE_SCALE_IDENTITY(model.GetScaleAndBias(), "feature interaction");

    TVector<TFeature> features;
    THashMap<TFeature, int, TFeatureHash> featureToIdx = GetFeatureToIdxMap(model, &features);

    TVector<TFeaturePairInteractionInfo> pairwiseEffect;

    if (model.IsOblivious()) {
        TVector<TMxTree> trees = BuildTrees(featureToIdx, model);
        pairwiseEffect = CalcMostInteractingFeatures(trees);
    } else {
        pairwiseEffect = CalcMostInteractingFeatures(
            model,
            featureToIdx
        );
    }

    TVector<TInternalFeatureInteraction> result;
    result.reserve(pairwiseEffect.size());
    for (const auto& efffect : pairwiseEffect) {
        result.emplace_back(efffect.Score, features[efffect.Feature1], features[efffect.Feature2]);
    }
    return result;
}

TVector<TFeatureInteraction> CalcFeatureInteraction(
    const TVector<TInternalFeatureInteraction>& internalFeatureInteraction,
    const NCB::TFeaturesLayout& layout)
{
    THashMap<std::pair<int, int>, double> sumInteraction;
    double totalEffect = 0;

    for (const auto& effectWithFeaturePair : internalFeatureInteraction) {
        TVector<TFeature> features{effectWithFeaturePair.FirstFeature, effectWithFeaturePair.SecondFeature};

        TVector<TVector<int>> internalToRegular;
        for (const auto& internalFeature : features) {
            TVector<int> regularFeatures;
            if (internalFeature.Type == ESplitType::FloatFeature) {
                regularFeatures.push_back(
                    layout.GetExternalFeatureIdx(internalFeature.FeatureIdx, EFeatureType::Float)
                );
            } else {
                auto proj = internalFeature.Ctr.Base.Projection;
                for (auto& binFeature : proj.BinFeatures) {
                    regularFeatures.push_back(
                        layout.GetExternalFeatureIdx(binFeature.FloatFeature, EFeatureType::Float)
                    );
                }
                for (auto catFeature : proj.CatFeatures) {
                    regularFeatures.push_back(
                        layout.GetExternalFeatureIdx(catFeature, EFeatureType::Categorical)
                    );
                }
            }
            internalToRegular.push_back(regularFeatures);
        }

        double effect = effectWithFeaturePair.Score;
        for (int f0 : internalToRegular[0]) {
            for (int f1 : internalToRegular[1]) {
                if (f0 == f1) {
                    continue;
                }
                if (f1 < f0) {
                    DoSwap(f0, f1);
                }
                sumInteraction[std::make_pair(f0, f1)] += effect
                    / (internalToRegular[0].ysize() * internalToRegular[1].ysize());
            }
        }
        totalEffect += effect;
    }

    TVector<TFeatureInteraction> regularFeatureEffect;
    for (const auto& pairInteraction : sumInteraction) {
        int f0 = pairInteraction.first.first;
        int f1 = pairInteraction.first.second;
        regularFeatureEffect.push_back(
            TFeatureInteraction(
                sumInteraction[pairInteraction.first] / totalEffect * 100,
                layout.GetExternalFeatureType(f0),
                layout.GetInternalFeatureIdx(f0),
                layout.GetExternalFeatureType(f1),
                layout.GetInternalFeatureIdx(f1))
            );
    }

    StableSort(
        regularFeatureEffect.rbegin(),
        regularFeatureEffect.rend(),
        [](const TFeatureInteraction& left, const TFeatureInteraction& right) {
            return left.Score < right.Score;
        }
    );
    return regularFeatureEffect;
}

static TVector<TVector<double>> CalcFstr(
    const TFullModel& model,
    const TDataProviderPtr dataset,
    EFstrType type,
    NPar::ILocalExecutor* localExecutor,
    ECalcTypeShapValues calcType)
{
    CB_ENSURE(
        !model.ModelTrees->GetModelTreeData()->GetLeafWeights().empty() || (dataset != nullptr),
        "CalcFstr requires either non-empty LeafWeights in model or provided dataset"
    );

    TVector<double> regularEffect = CalcRegularFeatureEffect(model, dataset, type, localExecutor, calcType);
    TVector<TVector<double>> result;
    for (const auto& value : regularEffect){
        TVector<double> vec = {value};
        result.push_back(vec);
    }
    return result;
}

TVector<TVector<double>> CalcInteraction(const TFullModel& model) {
    const TFeaturesLayout layout = MakeFeaturesLayout(model);

    TVector<TInternalFeatureInteraction> internalInteraction = CalcInternalFeatureInteraction(model);
    TVector<TFeatureInteraction> interaction = CalcFeatureInteraction(internalInteraction, layout);
    TVector<TVector<double>> result;
    for (const auto& value : interaction){
        int featureIdxFirst = layout.GetExternalFeatureIdx(value.FirstFeature.Index, value.FirstFeature.Type);
        int featureIdxSecond = layout.GetExternalFeatureIdx(
            value.SecondFeature.Index,
            value.SecondFeature.Type
        );
        TVector<double> vec = {
            static_cast<double>(featureIdxFirst),
            static_cast<double>(featureIdxSecond),
            value.Score
        };
        result.push_back(vec);
    }
    return result;
}


static bool AllFeatureIdsEmpty(TConstArrayRef<TFeatureMetaInfo> featuresMetaInfo) {
    return AllOf(
        featuresMetaInfo.begin(),
        featuresMetaInfo.end(),
        [](const auto& featureMetaInfo) { return featureMetaInfo.Name.empty(); }
    );
}


TVector<TVector<double>> GetFeatureImportances(
    const EFstrType fstrType,
    const TFullModel& model,
    const TDataProviderPtr dataset, // can be nullptr
    const TDataProviderPtr referenceDataset, // can be nullptr
    int threadCount,
    EPreCalcShapValues mode,
    int logPeriod,
    ECalcTypeShapValues calcType,
    EExplainableModelOutput modelOutputType,
    size_t sageNSamples,
    size_t sageBatchSize,
    bool sageDetectConvergence)
{
    TSetLoggingVerboseOrSilent inThisScope(logPeriod);
    CB_ENSURE(model.GetTreeCount(), "Model is not trained");
    if (dataset) {
        CheckModelAndDatasetCompatibility(model, *dataset->ObjectsData.Get());
    }
    if (fstrType != EFstrType::PredictionValuesChange) {
        CB_ENSURE_SCALE_IDENTITY(model.GetScaleAndBias(), "feature importance");
    }
    switch (fstrType) {
        case EFstrType::PredictionValuesChange:
        case EFstrType::LossFunctionChange:
        case EFstrType::FeatureImportance: {
            NPar::TLocalExecutor localExecutor;
            localExecutor.RunAdditionalThreads(threadCount - 1);

            return CalcFstr(model, dataset, fstrType, &localExecutor, calcType);
        }
        case EFstrType::Interaction:
            if (dataset) {
                CATBOOST_NOTICE_LOG << "Dataset is provided, but not used, because importance values are"
                    " cached in the model." << Endl;
            }
            return CalcInteraction(model);
        case EFstrType::ShapValues: {
            CB_ENSURE(dataset, "Dataset is not provided");

            NPar::TLocalExecutor localExecutor;
            localExecutor.RunAdditionalThreads(threadCount - 1);

            return CalcShapValues(
                model,
                *dataset,
                referenceDataset,
                /*fixedFeatureParams*/ Nothing(),
                logPeriod,
                mode,
                &localExecutor,
                calcType,
                modelOutputType
            );
        }
        case EFstrType::SageValues: {
            CB_ENSURE(dataset, "Dataset is not provided");

            NPar::TLocalExecutor localExecutor;
            localExecutor.RunAdditionalThreads(threadCount - 1);

            return CalcSageValues(
                model,
                *dataset,
                logPeriod,
                &localExecutor,
                sageNSamples,
                sageBatchSize,
                sageDetectConvergence
            );
        }
        case EFstrType::PredictionDiff: {
            NPar::TLocalExecutor localExecutor;
            localExecutor.RunAdditionalThreads(threadCount - 1);

            CB_ENSURE(dataset, "Documents for comparison are not provided");
            return GetPredictionDiff(model, *dataset, &localExecutor);
        }
        default:
            CB_ENSURE(false, "Unexpected Fstr type");
    }
}

TVector<TVector<TVector<double>>> GetFeatureImportancesMulti(
    const EFstrType fstrType,
    const TFullModel& model,
    const TDataProviderPtr dataset,
    const TDataProviderPtr referenceDataset, // can be nullptr
    int threadCount,
    EPreCalcShapValues mode,
    int logPeriod,
    ECalcTypeShapValues calcType,
    EExplainableModelOutput modelOutputType)
{
    TSetLoggingVerboseOrSilent inThisScope(logPeriod);
    CB_ENSURE(model.GetTreeCount(), "Model is not trained");

    CB_ENSURE(fstrType == EFstrType::ShapValues, "Only shap values can provide multi approxes.");

    CB_ENSURE(dataset, "Dataset is not provided");
    CheckModelAndDatasetCompatibility(model, *dataset->ObjectsData.Get());

    NPar::TLocalExecutor localExecutor;
    localExecutor.RunAdditionalThreads(threadCount - 1);

    return CalcShapValuesMulti(
        model,
        *dataset,
        referenceDataset,
        /*fixedFeatureParams*/ Nothing(),
        logPeriod,
        mode,
        &localExecutor,
        calcType,
        modelOutputType
    );
}

TVector<TVector<TVector<TVector<double>>>> CalcShapFeatureInteractionMulti(
    const EFstrType fstrType,
    const TFullModel& model,
    const NCB::TDataProviderPtr dataset,
    const TMaybe<std::pair<int, int>>& pairOfFeatures,
    int threadCount,
    EPreCalcShapValues mode,
    int logPeriod,
    ECalcTypeShapValues calcType)
{
    ValidateFeatureInteractionParams(fstrType, model, dataset, calcType);
    if (pairOfFeatures.Defined()) {
        const int flatFeatureCount = SafeIntegerCast<int>(dataset->MetaInfo.GetFeatureCount());
        ValidateFeaturePair(flatFeatureCount, pairOfFeatures.GetRef());
    }

    NPar::TLocalExecutor localExecutor;
    localExecutor.RunAdditionalThreads(threadCount - 1);

    return CalcShapInteractionValuesMulti(
        model,
        *dataset,
        pairOfFeatures,
        logPeriod,
        mode,
        &localExecutor,
        calcType
    );
}

TVector<TString> GetMaybeGeneratedModelFeatureIds(
    const TFullModel& model,
    const TFeaturesLayoutPtr datasetFeaturesLayout)
{
    const NCB::TFeaturesLayout modelFeaturesLayout = MakeFeaturesLayout(model);
    TVector<TString> modelFeatureIds(modelFeaturesLayout.GetExternalFeatureCount());
    if (AllFeatureIdsEmpty(modelFeaturesLayout.GetExternalFeaturesMetaInfo())) {
        if (datasetFeaturesLayout) {
            THashMap<ui32, ui32> columnIndexesReorderMap; // unused
            CheckModelAndDatasetCompatibility(model, *datasetFeaturesLayout, &columnIndexesReorderMap);
            const auto datasetFeaturesMetaInfo = datasetFeaturesLayout->GetExternalFeaturesMetaInfo();
            if (!AllFeatureIdsEmpty(datasetFeaturesMetaInfo)) {
                CB_ENSURE(
                    datasetFeaturesMetaInfo.size() >= modelFeatureIds.size(),
                    "Dataset has less features than the model"
                );
                for (auto i : xrange(modelFeatureIds.size())) {
                    modelFeatureIds[i] = datasetFeaturesMetaInfo[i].Name;
                }
            }
        }
    } else {
        modelFeatureIds = modelFeaturesLayout.GetExternalFeatureIds();
    }
    for (size_t i = 0; i < modelFeatureIds.size(); ++i) {
        if (modelFeatureIds[i].empty()) {
            modelFeatureIds[i] = ToString(i);
        }
    }
    return modelFeatureIds;
}

TVector<TString> GetMaybeGeneratedModelFeatureIds(const TFullModel& model, const TDataProviderPtr dataset) {
    TFeaturesLayoutPtr datasetFeaturesLayout;
    if (dataset) {
        datasetFeaturesLayout = dataset->MetaInfo.FeaturesLayout;
    }
    return GetMaybeGeneratedModelFeatureIds(model, std::move(datasetFeaturesLayout));
}

