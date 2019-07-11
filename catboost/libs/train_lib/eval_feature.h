#pragma once

#include <catboost/libs/algo/custom_objective_descriptor.h>
#include <catboost/libs/data_new/data_provider.h>
#include <catboost/libs/options/feature_eval_options.h>
#include <catboost/libs/train_lib/cross_validation.h>

#include <library/json/json_value.h>

#include <util/generic/maybe.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>


struct TFeatureEvaluationSummary {
    TVector<TString> Metrics; // [metric count]
    TVector<TVector<ui32>> FeatureSets; // [feature set count][]
    TVector<double> WxTest; // [feature set count]
    TVector<TVector<ui32>> BestBaselineIterations; // [feature set count][fold Count]
    TVector<TVector<double>> MetricDelta; // [feature set count][metric count]

public:
    void PushBackWxTestAndDelta(
        const TVector<THolder<IMetric>>& metrics,
        const TVector<TFoldContext>& baselineFoldContexts,
        const TVector<TFoldContext>& testedFoldContexts);
};

TString ToString(const TFeatureEvaluationSummary& summary);

void EvaluateFeatures(
    const NJson::TJsonValue& plainJsonParams,
    const NCatboostOptions::TFeatureEvalOptions& featureEvalOptions,
    const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
    const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
    const TCvDataPartitionParams& cvParams,
    NCB::TDataProviderPtr data,
    TFeatureEvaluationSummary* results);
