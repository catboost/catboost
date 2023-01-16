#pragma once

#include "feature_str.h"
#include "shap_values.h"

#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/model/model.h>
#include <catboost/private/libs/options/enums.h>

#include <library/cpp/threading/local_executor/local_executor.h>

struct TCombinationClassFeatures : public TVector<TFeature> {};

TCombinationClassFeatures GetCombinationClassFeatures(const TFullModel& model);

TVector<std::pair<double, TFeature>> CalcFeatureEffectLossChange(
    const TFullModel& model,
    const NCB::TDataProviderPtr dataProvider,
    NPar::ILocalExecutor* localExecutor,
    ECalcTypeShapValues calcType
);

const NCB::TDataProviderPtr GetSubsetForFstrCalc(
    const NCB::TDataProviderPtr dataset,
    NPar::ILocalExecutor* localExecutor
);

TVector<TMetricHolder> CalcFeatureEffectLossChangeMetricStats(
    const TFullModel& model,
    const int featuresCount,
    const TShapPreparedTrees& preparedTrees,
    const NCB::TDataProviderPtr dataset,
    ECalcTypeShapValues calcType,
    ui64 randomSeed,
    NPar::ILocalExecutor* localExecutor
);

TVector<std::pair<double, TFeature>> CalcFeatureEffectLossChangeFromScores(
    const TCombinationClassFeatures& combinationClassFeatures,
    const IMetric& metric,
    const TVector<TMetricHolder>& scores);

void CreateMetricAndLossDescriptionForLossChange(
    const TFullModel& model,
    NCatboostOptions::TLossDescription* metricDescription,
    NCatboostOptions::TLossDescription* lossDescription,
    bool* needYetiRankPairs,
    THolder<IMetric>* metric);

i64 GetMaxObjectCountForFstrCalc(i64 objectCount, i32 featureCount);
