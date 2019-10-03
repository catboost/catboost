#pragma once

#include <catboost/private/libs/algo_helpers/custom_objective_descriptor.h>
#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/metrics/metric.h>
#include <catboost/private/libs/options/feature_eval_options.h>
#include <catboost/libs/train_lib/cross_validation.h>

#include <library/json/json_value.h>

#include <util/generic/maybe.h>
#include <util/generic/ptr.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>


struct TFeatureEvaluationSummary {
    TVector<EMetricBestValue> MetricTypes; // [metric count]
    TVector<TString> MetricNames; // [metric count]
    TVector<TVector<ui32>> FeatureSets; // [feature set count][]

    TVector<TVector<TVector<double>>> BestBaselineMetrics; // [feature set count][metric count][fold count]
    TVector<TVector<TVector<double>>> BestTestedMetrics; // [feature set count][metric count][fold count]
    TVector<TVector<ui32>> BestBaselineIterations; // [feature set count][fold count]

    TVector<double> WxTest; // [feature set count]
    TVector<TVector<double>> AverageMetricDelta; // [feature set count][metric count]

    ui32 FoldRangeOffset;

public:
    bool HasHeaderInfo() const;

    void SetHeaderInfo(
        const TVector<THolder<IMetric>>& metrics,
        const TVector<TVector<ui32>>& featureSets);

    void AppendFeatureSetMetrics(
        ui32 featureSetIdx,
        const TVector<TFoldContext>& baselineFoldContexts,
        const TVector<TFoldContext>& testedFoldContexts);

    void CalcWxTestAndAverageDelta();
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
