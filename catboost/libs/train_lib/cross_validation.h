#pragma once

#include <catboost/libs/algo/custom_objective_descriptor.h>
#include <catboost/libs/algo/learn_context.h>
#include <catboost/libs/data_new/data_provider.h>
#include <catboost/libs/eval_result/eval_result.h>
#include <catboost/libs/helpers/parallel_tasks.h>
#include <catboost/libs/helpers/restorable_rng.h>
#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/options/cross_validation_params.h>
#include <catboost/libs/options/output_file_options.h>
#include <catboost/libs/train_lib/train_model.h>

#include <library/json/json_value.h>

#include <util/folder/tempdir.h>
#include <util/generic/array_ref.h>
#include <util/generic/maybe.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/generic/xrange.h>

#include <numeric>


struct TCVIterationResults {
    TMaybe<double> AverageTrain;
    TMaybe<double> StdDevTrain;
    double AverageTest;
    double StdDevTest;
};

struct TCVResult {
    TString Metric;
    TVector<ui32> Iterations;
    TVector<double> AverageTrain;
    TVector<double> StdDevTrain;
    TVector<double> AverageTest;
    TVector<double> StdDevTest;

    void AppendOneIterationResults(ui32 iteration, const TCVIterationResults& results) {
        Iterations.push_back(iteration);
        if (results.AverageTrain.Defined()) {
            AverageTrain.push_back(results.AverageTrain.GetRef());
        }

        if (results.StdDevTrain.Defined()) {
            StdDevTrain.push_back(results.StdDevTrain.GetRef());
        }

        AverageTest.push_back(results.AverageTest);
        StdDevTest.push_back(results.StdDevTest);
    }
};

TConstArrayRef<TString> GetTargetForStratifiedSplit(const NCB::TDataProvider& dataProvider);
TConstArrayRef<float> GetTargetForStratifiedSplit(const NCB::TTrainingDataProvider& dataProvider);

TVector<NCB::TArraySubsetIndexing<ui32>> CalcTrainSubsets(
    const TVector<NCB::TArraySubsetIndexing<ui32>>& testSubsets,
    ui32 groupCount);


template <class TDataProvidersTemplate> // TDataProvidersTemplate<...> or TTrainingDataProvidersTemplate<...>
TVector<TDataProvidersTemplate> PrepareCvFolds(
    typename TDataProvidersTemplate::TDataPtr srcData,
    const TCrossValidationParams& cvParams,
    TMaybe<ui32> foldIdx, // if Nothing() - return data for all folds, if defined - return only one fold
    bool oldCvStyleSplit,
    NPar::TLocalExecutor* localExecutor) {

    // group subsets, groups maybe trivial
    TVector<NCB::TArraySubsetIndexing<ui32>> testSubsets;

    // both NCB::Split and NCB::StratifiedSplit keep objects order
    NCB::EObjectsOrder objectsOrder = NCB::EObjectsOrder::Ordered;

    if (cvParams.Stratified) {
        testSubsets = NCB::StratifiedSplit(
            *srcData->ObjectsGrouping,
            GetTargetForStratifiedSplit(*srcData),
            cvParams.FoldCount);
    } else {
        testSubsets = NCB::Split(*srcData->ObjectsGrouping, cvParams.FoldCount, oldCvStyleSplit);
    }

    // group subsets, maybe trivial
    TVector<NCB::TArraySubsetIndexing<ui32>> trainSubsets
        = CalcTrainSubsets(testSubsets, srcData->ObjectsGrouping->GetGroupCount());

    if (cvParams.Inverted) {
        testSubsets.swap(trainSubsets);
    }

    TVector<ui32> resultFolds;

    if (foldIdx) {
        resultFolds.assign(1, *foldIdx);
    } else {
        resultFolds.resize(cvParams.FoldCount);
        std::iota(resultFolds.begin(), resultFolds.end(), 0);
    }

    TVector<TDataProvidersTemplate> result(resultFolds.size());

    TVector<std::function<void()>> tasks;

    for (ui32 resultIdx : xrange(resultFolds.size())) {
        tasks.emplace_back(
            [&, resultIdx]() {
                result[resultIdx].Learn = srcData->GetSubset(
                    GetSubset(
                        srcData->ObjectsGrouping,
                        std::move(trainSubsets[resultFolds[resultIdx]]),
                        objectsOrder
                    ),
                    localExecutor
                );
            }
        );
        tasks.emplace_back(
            [&, resultIdx]() {
                result[resultIdx].Test.emplace_back(
                    srcData->GetSubset(
                        GetSubset(
                            srcData->ObjectsGrouping,
                            std::move(testSubsets[resultFolds[resultIdx]]),
                            objectsOrder
                        ),
                        localExecutor
                    )
                );
            }
        );
    }

    NCB::ExecuteTasksInParallel(&tasks, localExecutor);

    return result;
}

void CrossValidate(
    NJson::TJsonValue plainJsonParams,
    NCB::TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
    const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
    const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
    NCB::TDataProviderPtr data,
    const TCrossValidationParams& cvParams,
    TVector<TCVResult>* results);

struct TFoldContext {
    ui32 FoldIdx;

    ETaskType TaskType;

    THolder<TTempDir> TempDir; // THolder because of bugs with move semantics of TTempDir
    NCatboostOptions::TOutputFilesOptions OutputOptions; // with modified Overfitting params, TrainDir
    NCB::TTrainingDataProviders TrainingData;

    THolder<TLearnProgress> LearnProgress;
    TMaybe<TFullModel> FullModel;

    TVector<TVector<double>> MetricValuesOnTrain; // [iter][metricIdx]
    TVector<TVector<double>> MetricValuesOnTest;  // [iter][metricIdx]

    NCB::TEvalResult LastUpdateEvalResult;

    TRestorableFastRng64 Rand;

public:
    TFoldContext(
        size_t foldIdx,
        ETaskType taskType,
        const NCatboostOptions::TOutputFilesOptions& commonOutputOptions,
        NCB::TTrainingDataProviders&& trainingData,
        ui64 randomSeed,
        bool hasFullModel = false);

};

void TrainBatch(
    const NCatboostOptions::TCatBoostOptions& catboostOption,
    const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
    const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
    const TLabelConverter& labelConverter,
    TConstArrayRef<THolder<IMetric>> metrics,
    TConstArrayRef<bool> skipMetricOnTrain,
    double maxTimeSpentOnFixedCostRatio,
    ui32 maxIterationsBatchSize,
    size_t globalMaxIteration,
    bool isErrorTrackerActive,
    ELoggingLevel loggingLevel,
    TFoldContext* foldContext,
    IModelTrainer* modelTrainer,
    NPar::TLocalExecutor* localExecutor,
    TMaybe<ui32>* upToIteration);

void Train(
    const NCatboostOptions::TCatBoostOptions& catboostOption,
    const TString& trainDir,
    const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
    const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
    const TLabelConverter& labelConverter,
    const TVector<THolder<IMetric>>& metrics,
    bool isErrorTrackerActive,
    TFoldContext* foldContext,
    IModelTrainer* modelTrainer,
    NPar::TLocalExecutor* localExecutor);

void UpdateMetricsAfterIteration(
    size_t iteration,
    bool calcMetric,
    bool isErrorTrackerActive,
    TConstArrayRef<THolder<IMetric>> metrics,
    TConstArrayRef<bool> skipMetricOnTrain,
    const TMetricsAndTimeLeftHistory& metricsAndTimeHistory,
    TVector<TVector<double>>* metricValuesOnTrain,
    TVector<TVector<double>>* metricValuesOnTest);

void UpdatePermutationBlockSize(
    ETaskType taskType,
    TConstArrayRef<NCB::TTrainingDataProviders> foldsData,
    NCatboostOptions::TCatBoostOptions* catboostOptions);
