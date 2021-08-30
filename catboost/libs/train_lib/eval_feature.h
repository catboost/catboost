#pragma once

#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/train_lib/cross_validation.h>
#include <catboost/private/libs/algo_helpers/custom_objective_descriptor.h>
#include <catboost/private/libs/options/feature_eval_options.h>

#include <library/cpp/json/json_value.h>

#include <util/generic/maybe.h>
#include <util/generic/ptr.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>

#include <tuple>

struct TFeatureEvaluationSummary {
    TVector<EMetricBestValue> MetricTypes; // [metric count]
    TVector<TString> MetricNames; // [metric count]
    TVector<TVector<ui32>> FeatureSets; // [feature set count][]

    using TMetricsHistory = TVector<TVector<double>>; // [iteration count][metric count]
    TVector<TVector<TVector<TMetricsHistory>>> MetricsHistory; // [is test][feature set count][fold count]
    TVector<TVector<TVector<TVector<std::pair<double, TString>>>>> FeatureStrengths; // [is test][feature set count][fold count][feature index]
    TVector<TVector<TVector<TVector<std::pair<double, TString>>>>> RegularFeatureStrengths; // [is test][feature set count][fold count][feature index]
    TVector<TVector<TVector<TFullModel>>> Models; // [is test][feature set count][fold count]
    struct TProcessorsUsage {
        float Time;
        ui32 Iteration;
        NJson::TJsonValue Processors;
        Y_SAVELOAD_DEFINE(
            Time,
            Iteration,
            Processors);
    };
    TVector<TProcessorsUsage> ProcessorsUsage; // [snapshot idx]

    TVector<TVector<TVector<TVector<double>>>> BestMetrics; // [is test][feature set count][metric count][fold count]
    TVector<TVector<ui32>> BestBaselineIterations; // [feature set count][fold count]

    TVector<double> WxTest; // [feature set count]
    TVector<TVector<double>> AverageMetricDelta; // [feature set count][metric count]

public:
    size_t GetFeatureSetCount() const;

    bool HasHeaderInfo() const;

    void SetHeaderInfo(
        const TVector<THolder<IMetric>>& metrics,
        const TVector<TVector<ui32>>& featureSets);

    void AppendFeatureSetMetrics(
        bool isBaseline,
        ui32 featureSetIdx,
        const TVector<TVector<double>>& metricValuesOnTest);

    NJson::TJsonValue CalcProcessorsSummary() const;
    void CalcWxTestAndAverageDelta();
    void CreateLogs(
        const NCatboostOptions::TOutputFilesOptions& outputFileOptions,
        const NCatboostOptions::TFeatureEvalOptions& featureEvalOptions,
        const TVector<THolder<IMetric>>& metrics,
        ui32 iterationCount,
        bool isTest,
        ui32 foldRangeOffset,
        ui32 offsetFromOptions);

    Y_SAVELOAD_DEFINE(
        MetricTypes,
        MetricNames,
        FeatureSets,
        MetricsHistory,
        FeatureStrengths,
        RegularFeatureStrengths,
        Models,
        ProcessorsUsage,
        BestMetrics,
        BestBaselineIterations,
        WxTest,
        AverageMetricDelta);
};

TString ToString(const TFeatureEvaluationSummary& summary);

TFeatureEvaluationSummary EvaluateFeatures(
    const NJson::TJsonValue& plainJsonParams,
    const NCatboostOptions::TFeatureEvalOptions& featureEvalOptions,
    const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
    const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
    const TCvDataPartitionParams& cvParams,
    NCB::TDataProviderPtr data);
