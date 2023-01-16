#pragma once

#include <catboost/private/libs/algo/data.h>
#include <catboost/private/libs/algo/learn_context.h>
#include <catboost/private/libs/algo_helpers/custom_objective_descriptor.h>
#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/eval_result/eval_result.h>
#include <catboost/libs/helpers/parallel_tasks.h>
#include <catboost/libs/helpers/restorable_rng.h>
#include <catboost/libs/metrics/metric.h>
#include <catboost/private/libs/options/cross_validation_params.h>
#include <catboost/private/libs/options/output_file_options.h>
#include <catboost/libs/train_lib/train_model.h>

#include <library/cpp/json/json_value.h>

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

    TVector<TFullModel> CVFullModels;

    //for painting
    TVector<double> LastTrainEvalMetric;//[foldIdx]
    TVector<double> LastTestEvalMetric;//[foldIdx]

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

TVector<NCB::TArraySubsetIndexing<ui32>> CalcTrainSubsets(
    const TVector<NCB::TArraySubsetIndexing<ui32>>& testSubsets,
    ui32 groupCount);

TVector<NCB::TArraySubsetIndexing<ui32>> CalcTrainSubsetsRange(
    const TVector<NCB::TArraySubsetIndexing<ui32>>& testSubsets,
    ui32 groupCount,
    const NCB::TIndexRange<ui32>& trainSubsetsRange);

TVector<NCB::TArraySubsetIndexing<ui32>> TransformToVectorArrayIndexing(const TVector<TVector<ui32>>& vectorData);

TVector<NCB::TArraySubsetIndexing<ui32>> StratifiedSplitToFolds(
    const NCB::TDataProvider& dataProvider,
    ui32 partCount);

TVector<NCB::TArraySubsetIndexing<ui32>> StratifiedSplitToFolds(
    const NCB::TTrainingDataProvider& trainingDataProvider,
    ui32 partCount);


template <class TDataProvidersTemplate> // TDataProvidersTemplate<...> or TTrainingDataProvidersTemplate<...>
TVector<TDataProvidersTemplate> PrepareCvFolds(
    typename TDataProvidersTemplate::TDataPtr srcData,
    const TCrossValidationParams& cvParams,
    TMaybe<ui32> foldIdx, // if Nothing() - return data for all folds, if defined - return only one fold
    bool oldCvStyleSplit,
    ui64 cpuUsedRamLimit,
    NPar::ILocalExecutor* localExecutor) {

    // group subsets, groups maybe trivial
    TVector<NCB::TArraySubsetIndexing<ui32>> testSubsets;

    // group subsets, maybe trivial
    TVector<NCB::TArraySubsetIndexing<ui32>> trainSubsets;

    if (cvParams.customTrainSubsets) {
        trainSubsets = TransformToVectorArrayIndexing(cvParams.customTrainSubsets.GetRef());
        testSubsets = TransformToVectorArrayIndexing(cvParams.customTestSubsets.GetRef());

        CB_ENSURE(
            cvParams.FoldCount == trainSubsets.size() &&
            testSubsets.size() == trainSubsets.size(),
            "Fold count must be equal to number of custom subsets"
        );
    } else if (cvParams.Type == ECrossValidation::TimeSeries) {
        const auto trainAndTestSubsets = NCB::TimeSeriesSplit(
            *srcData->ObjectsGrouping,
            cvParams.FoldCount,
            oldCvStyleSplit
        );
        trainSubsets = trainAndTestSubsets.first;
        testSubsets = trainAndTestSubsets.second;
    } else {
        if (cvParams.Stratified) {
            testSubsets = StratifiedSplitToFolds(*srcData, cvParams.FoldCount);
        } else {
            testSubsets = NCB::Split(*srcData->ObjectsGrouping, cvParams.FoldCount, oldCvStyleSplit);
        }
        trainSubsets = CalcTrainSubsets(testSubsets, srcData->ObjectsGrouping->GetGroupCount());
    }


    if (cvParams.Type == ECrossValidation::Inverted) {
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

    const ui64 perTaskCpuUsedRamLimit = cpuUsedRamLimit / resultFolds.size();

    for (ui32 resultIdx : xrange(resultFolds.size())) {
        tasks.emplace_back(
            [&, resultIdx]() {
                result[resultIdx] = NCB::CreateTrainTestSubsets<TDataProvidersTemplate>(
                    srcData,
                    std::move(trainSubsets[resultFolds[resultIdx]]),
                    std::move(testSubsets[resultFolds[resultIdx]]),
                    perTaskCpuUsedRamLimit,
                    localExecutor
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

void CrossValidate(
    NJson::TJsonValue plainJsonParams,
    NCB::TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
    const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
    const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
    TLabelConverter& labelConverter,
    NCB::TDataProviderPtr data,
    const TCrossValidationParams& cvParams,
    NPar::ILocalExecutor* localExecutor,
    TVector<TCVResult>* results,
    bool isAlreadyShuffled = false);

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

void Train(
    const NCatboostOptions::TCatBoostOptions& catboostOption,
    const TString& trainDir,
    const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
    const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
    const TLabelConverter& labelConverter,
    const TVector<THolder<IMetric>>& metrics,
    bool isErrorTrackerActive,
    ITrainingCallbacks* trainingCallbacks,
    TFoldContext* foldContext,
    IModelTrainer* modelTrainer,
    NPar::ILocalExecutor* localExecutor);

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
