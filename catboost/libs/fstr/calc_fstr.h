#pragma once

#include "feature_str.h"
#include "loss_change_fstr.h"

#include <catboost/private/libs/algo/split.h>
#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/metrics/metric_holder.h>
#include <catboost/libs/model/model.h>
#include <catboost/private/libs/options/enums.h>
#include <catboost/private/libs/options/enum_helpers.h>
#include <catboost/private/libs/options/loss_description.h>

#include <library/cpp/threading/local_executor/local_executor.h>

#include <util/digest/multi.h>
#include <util/system/types.h>
#include <util/system/yassert.h>

#include <utility>


struct TRegularFeature {
    EFeatureType Type;
    int Index;

public:
    TRegularFeature(EFeatureType type, int index)
        : Type(type)
        , Index(index)
    {}
};

struct TFeatureEffect {
    double Score = 0;
    TRegularFeature Feature;

public:
    TFeatureEffect() = default;

    TFeatureEffect(double score, EFeatureType type, int index)
        : Score(score)
        , Feature{type, index}
    {}
};

struct TFeatureInteraction {
    double Score = 0;
    TRegularFeature FirstFeature, SecondFeature;

public:
    TFeatureInteraction() = default;

    TFeatureInteraction(
        double score,
        EFeatureType firstFeatureType,
        int firstFeatureIndex,
        EFeatureType secondFeatureType,
        int secondFeatureIndex)
        : Score(score)
        , FirstFeature{firstFeatureType, firstFeatureIndex}
        , SecondFeature{secondFeatureType, secondFeatureIndex}
    {}
};

struct TInternalFeatureInteraction {
    double Score = 0;
    TFeature FirstFeature, SecondFeature;

public:
    TInternalFeatureInteraction(double score, const TFeature& firstFeature, const TFeature& secondFeature)
        : Score(score)
        , FirstFeature(firstFeature)
        , SecondFeature(secondFeature)
    {}
};

TVector<std::pair<double, TFeature>> CalcFeatureEffectAverageChange(
    const TFullModel& model,
    TConstArrayRef<double> weights
);

TVector<std::pair<double, TFeature>> CalcFeatureEffect(
    const TFullModel& model,
    const NCB::TDataProviderPtr dataset, // can be nullptr
    EFstrType type,
    NPar::ILocalExecutor* localExecutor,
    ECalcTypeShapValues calcType = ECalcTypeShapValues::Regular
);

TVector<TFeatureEffect> CalcRegularFeatureEffect(
    const TVector<std::pair<double, TFeature>>& effect,
    const TFullModel& model
);

TVector<double> GetFeatureEffectForLinearIndices(
    const TVector<std::pair<double, TFeature>>& featureEffect,
    const TFullModel& model
);

TVector<double> CalcRegularFeatureEffect(
    const TFullModel& model,
    const NCB::TDataProviderPtr dataset, // can be nullptr
    EFstrType type,
    NPar::ILocalExecutor* localExecutor,
    ECalcTypeShapValues calcType = ECalcTypeShapValues::Regular
);

TVector<TInternalFeatureInteraction> CalcInternalFeatureInteraction(const TFullModel& model);
TVector<TFeatureInteraction> CalcFeatureInteraction(
    const TVector<TInternalFeatureInteraction>& internalFeatureInteraction,
    const NCB::TFeaturesLayout& layout
);

TVector<TVector<double>> CalcInteraction(const TFullModel& model);
TVector<TVector<double>> GetFeatureImportances(
    const EFstrType type,
    const TFullModel& model,
    const NCB::TDataProviderPtr dataset, // can be nullptr
    const NCB::TDataProviderPtr referenceDataset, // can be nullptr
    int threadCount,
    EPreCalcShapValues mode,
    int logPeriod = 0,
    ECalcTypeShapValues calcType = ECalcTypeShapValues::Regular,
    EExplainableModelOutput modelOutputType = EExplainableModelOutput::Raw,
    size_t sageNSamples = 128,
    size_t sageBatchSize = 512,
    bool sageDetectConvergence = true
);

TVector<TVector<TVector<double>>> GetFeatureImportancesMulti(
    const EFstrType type,
    const TFullModel& model,
    const NCB::TDataProviderPtr dataset,
    const NCB::TDataProviderPtr referenceDataset, // can be nullptr
    int threadCount,
    EPreCalcShapValues mode,
    int logPeriod = 0,
    ECalcTypeShapValues calcType = ECalcTypeShapValues::Regular,
    EExplainableModelOutput modelOutputType = EExplainableModelOutput::Raw
);

TVector<TVector<TVector<TVector<double>>>> CalcShapFeatureInteractionMulti(
    const EFstrType fstrType,
    const TFullModel& model,
    const NCB::TDataProviderPtr dataset,
    const TMaybe<std::pair<int, int>>& pairOfFeatures,
    int threadCount,
    EPreCalcShapValues mode,
    int logPeriod = 0,
    ECalcTypeShapValues calcType = ECalcTypeShapValues::Regular
);

/*
 * model is the primary source of featureIds,
 * if model does not contain featureIds data then try to get this data from dataset (if provided (non nullptr))
 * for all remaining features without id generated featureIds will be just their external indices
 * (indices in original training dataset)
 */
TVector<TString> GetMaybeGeneratedModelFeatureIds(
    const TFullModel& model,
    const NCB::TFeaturesLayoutPtr datasetFeaturesLayout // can be nullptr
);

TVector<TString> GetMaybeGeneratedModelFeatureIds(
    const TFullModel& model,
    const NCB::TDataProviderPtr dataset // can be nullptr
);
