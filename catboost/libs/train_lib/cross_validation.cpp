#include "cross_validation.h"
#include "dir_helper.h"
#include "train_model.h"
#include "options_helper.h"
#include "trainer_env.h"

#include <catboost/private/libs/algo/approx_dimension.h>
#include <catboost/private/libs/algo/calc_score_cache.h>
#include <catboost/private/libs/algo/data.h>
#include <catboost/private/libs/algo/helpers.h>
#include <catboost/private/libs/algo/preprocess.h>
#include <catboost/private/libs/algo/roc_curve.h>
#include <catboost/private/libs/algo/train.h>
#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/data/feature_names_converter.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/vector_helpers.h>
#include <catboost/libs/loggers/catboost_logger_helpers.h>
#include <catboost/libs/loggers/logger.h>
#include <catboost/libs/logging/logging.h>
#include <catboost/libs/logging/profile_info.h>
#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/model/features.h>
#include <catboost/private/libs/options/defaults_helper.h>
#include <catboost/private/libs/options/enum_helpers.h>
#include <catboost/private/libs/options/plain_options_helper.h>
#include <catboost/private/libs/target/util.h>

#include <util/generic/algorithm.h>
#include <util/generic/mapfindptr.h>
#include <util/generic/scope.h>
#include <util/generic/ymath.h>
#include <util/generic/maybe.h>
#include <util/stream/labeled.h>
#include <util/string/cast.h>
#include <util/system/compiler.h>
#include <util/system/hp_timer.h>

#include <cmath>
#include <numeric>


using namespace NCB;


TVector<TArraySubsetIndexing<ui32>> CalcTrainSubsetsRange(
    const TVector<TArraySubsetIndexing<ui32>>& testSubsets,
    ui32 groupCount,
    const TIndexRange<ui32>& trainSubsetsRange
) {

    CB_ENSURE_INTERNAL(trainSubsetsRange.End <= testSubsets.size(), "Too many train subsets are requested");
    TVector<TVector<ui32>> trainSubsetIndices(testSubsets.size());
    for (ui32 fold : trainSubsetsRange.Iter()) {
        trainSubsetIndices[fold].reserve(groupCount - testSubsets[fold].Size());
    }
    for (ui32 testFold = 0; testFold < testSubsets.size(); ++testFold) {
        testSubsets[testFold].ForEach(
            [&](ui32 /*idx*/, ui32 srcIdx) {
                for (ui32 fold : trainSubsetsRange.Iter()) {
                    if (testFold == fold) {
                        continue;
                    }
                    trainSubsetIndices[fold].push_back(srcIdx);
                }
            }
        );
    }

    TVector<TArraySubsetIndexing<ui32>> result;
    for (auto& foldIndices : trainSubsetIndices) {
        result.push_back( TArraySubsetIndexing<ui32>(std::move(foldIndices)) );
    }

    return result;
}

TVector<TArraySubsetIndexing<ui32>> CalcTrainSubsets(
    const TVector<TArraySubsetIndexing<ui32>>& testSubsets,
    ui32 groupCount
) {
    return CalcTrainSubsetsRange(testSubsets, groupCount, TIndexRange<ui32>(testSubsets.size()));
}

static void CheckCrossValidationOptions(
    NCB::TDataProviderPtr data,
    const TVector<THolder<IMetric>>& metrics,
    const NCatboostOptions::TCatBoostOptions& catBoostOptions,
    const NCatboostOptions::TOutputFilesOptions& outputFileOptions,
    const TCrossValidationParams& cvParams
) {

    CB_ENSURE(
            catBoostOptions.DataProcessingOptions->ClassLabels.Get() == data->MetaInfo.ClassLabels,
            "ClassLabels in dataprocessing options and in training data must match");
    // TODO(akhropov): implement snapshots in CV. MLTOOLS-3439.
    CB_ENSURE(!outputFileOptions.SaveSnapshot(), "Saving snapshots in Cross-validation is not supported yet");

    const ui32 allDataObjectCount = data->ObjectsData->GetObjectCount();

    CB_ENSURE(allDataObjectCount != 0, "Pool is empty");
    CB_ENSURE(allDataObjectCount > cvParams.FoldCount, "Pool is too small to be split into folds");

    // TODO(akhropov): implement ordered split. MLTOOLS-2486.
    CB_ENSURE(
        data->ObjectsData->GetOrder() != EObjectsOrder::Ordered,
        "Cross-validation for Ordered objects data is not yet implemented"
    );
    if (catBoostOptions.GetTaskType() == ETaskType::GPU) {
        CB_ENSURE(
            TTrainerFactory::Has(ETaskType::GPU),
            "Can't load GPU learning library. "
            "Module was not compiled or driver  is incompatible with package. "
            "Please install latest NVDIA driver and check again");
    }

    bool hasQuerywiseMetric = false;
    for (const auto& metric : metrics) {
        if (metric.Get()->GetErrorType() == EErrorType::QuerywiseError) {
            hasQuerywiseMetric = true;
        }
    }
    if (hasQuerywiseMetric) {
        CB_ENSURE(!cvParams.Stratified, "Stratified split is incompatible with groupwise metrics");
    }
}

static double ComputeStdDev(const TVector<double>& values, double avg) {
    double sqrSum = 0.0;
    for (double value : values) {
        sqrSum += Sqr(value - avg);
    }
    return std::sqrt(sqrSum / (values.size() - 1));
}

static TCVIterationResults ComputeIterationResults(
    const TVector<double>& trainErrors,
    const TVector<double>& testErrors,
    size_t foldCount,
    bool skipTrain
) {
    TCVIterationResults cvResults;
    if (!skipTrain) {
        cvResults.AverageTrain = Accumulate(trainErrors.begin(), trainErrors.end(), 0.0) / foldCount;
        cvResults.StdDevTrain = ComputeStdDev(trainErrors, cvResults.AverageTrain.GetRef());
    }
    cvResults.AverageTest = Accumulate(testErrors.begin(), testErrors.end(), 0.0) / foldCount;
    cvResults.StdDevTest = ComputeStdDev(testErrors, cvResults.AverageTest);
    return cvResults;
}


inline bool DivisibleOrLastIteration(int currentIteration, int iterationsCount, int period) {
    return currentIteration % period == 0 || currentIteration == iterationsCount - 1;
}

TFoldContext::TFoldContext(
    size_t foldIdx,
    ETaskType taskType,
    const NCatboostOptions::TOutputFilesOptions& commonOutputOptions,
    TTrainingDataProviders&& trainingData,
    ui64 randomSeed,
    bool hasFullModel)
    : FoldIdx(foldIdx)
    , TaskType(taskType)
    , TempDir(MakeHolder<TTempDir>())
    , OutputOptions(commonOutputOptions)
    , TrainingData(std::move(trainingData))
    , Rand(randomSeed)
{
    OutputOptions.UseBestModel = false;
    if (hasFullModel) {
        FullModel = TFullModel();
    }
}

class TCrossValidationCallbacks : public ITrainingCallbacks {
public:
    TCrossValidationCallbacks(
        size_t globalMaxIteration,
        TErrorTracker* errorTracker,
        TConstArrayRef<THolder<IMetric>> metrics,
        TFoldContext* foldContext)
    : GlobalMaxIteration(globalMaxIteration)
    , ErrorTracker(errorTracker)
    , Metrics(metrics)
    , FoldContext(foldContext)
    {
    }

    bool IsContinueTraining(const TMetricsAndTimeLeftHistory& metricsAndTimeHistory) override {
        CB_ENSURE(metricsAndTimeHistory.TimeHistory.size() > 0, "Training time history is empty");
        size_t iteration = (FoldContext->TaskType == ETaskType::CPU) ?
              metricsAndTimeHistory.TimeHistory.size() - 1
            : (metricsAndTimeHistory.TimeHistory.size() - 1);

        const bool calcMetrics = DivisibleOrLastIteration(
            iteration,
            GlobalMaxIteration,
            FoldContext->OutputOptions.GetMetricPeriod());
        if (calcMetrics || ErrorTracker->IsActive()) {
            TVector<double> valuesToLog;
            ErrorTracker->AddError(metricsAndTimeHistory.TestMetricsHistory[iteration][0].at(Metrics[0]->GetDescription()),
                                   iteration,
                                   &valuesToLog);
        }
        if (ErrorTracker->IsActive() && ErrorTracker -> GetIsNeedStop()) {
            return false;
        }
        return (iteration + 1) < GlobalMaxIteration;
    }

private:
    size_t GlobalMaxIteration;
    TErrorTracker* ErrorTracker;
    TConstArrayRef<THolder<IMetric>> Metrics;
    TFoldContext* const FoldContext;
};

static TVector<double> GetMetricValues(
    TConstArrayRef<THolder<IMetric>> metrics,
    TConstArrayRef<bool> skipMetric,
    const THashMap<TString, double>& iterationMetrics
) {
    TVector<double> result;
    for (auto metricIdx : xrange(metrics.size())) {
        const auto& metricDescription = metrics[metricIdx]->GetDescription();
        const auto isMetricAvailable = skipMetric.empty() || !skipMetric[metricIdx];
        const auto haveMetricValue = isMetricAvailable && iterationMetrics.contains(metricDescription);
        if (haveMetricValue) {
            result.push_back(iterationMetrics.at(metricDescription));
        } else {
            result.push_back(std::numeric_limits<double>::quiet_NaN());
        }
    }
    return result;
}

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
    NPar::ILocalExecutor* localExecutor
) {
    TTrainModelInternalOptions internalOptions;
    internalOptions.CalcMetricsOnly = !foldContext->FullModel.Defined();
    internalOptions.ForceCalcEvalMetricOnEveryIteration = isErrorTrackerActive;

    auto foldOutputOptions = foldContext->OutputOptions;
    foldOutputOptions.SetTrainDir(trainDir);
    if (foldContext->FullModel.Defined()) {
        // TrainModel saves model either to memory pointed by dstModel, or to ResultModelPath
        foldOutputOptions.ResultModelPath = NCatboostOptions::TOption<TString>("result_model_file", "model");
    }
    TMetricsAndTimeLeftHistory metricsAndTimeHistory;
    const auto defaultCustomCallbacks = MakeHolder<TCustomCallbacks>(Nothing());
    modelTrainer->TrainModel(
        internalOptions,
        catboostOption,
        foldOutputOptions,
        objectiveDescriptor,
        evalMetricDescriptor,
        foldContext->TrainingData,
        /*precomputedSingleOnlineCtrDataForSingleFold*/ Nothing(),
        labelConverter,
        trainingCallbacks,
        defaultCustomCallbacks.Get(),
        /*initModel*/ Nothing(),
        THolder<TLearnProgress>(),
        /*initModelApplyCompatiblePools*/ TDataProviders(),
        localExecutor,
        /*rand*/ Nothing(),
        foldContext->FullModel.Defined() ? foldContext->FullModel.Get() : nullptr,
        TVector<TEvalResult*>{&foldContext->LastUpdateEvalResult},
        &metricsAndTimeHistory,
        (foldContext->TaskType == ETaskType::CPU) ? &foldContext->LearnProgress : nullptr
    );
    if (foldContext->FullModel.Defined()) {
        TFileOutput modelFile(JoinFsPaths(trainDir, foldContext->OutputOptions.ResultModelPath.Get()));
        foldContext->FullModel->Save(&modelFile);
    }
    const auto skipMetricOnTrain = GetSkipMetricOnTrain(metrics);
    for (const auto& trainMetrics : metricsAndTimeHistory.LearnMetricsHistory) {
            foldContext->MetricValuesOnTrain.emplace_back(GetMetricValues(metrics, skipMetricOnTrain, trainMetrics));
    }
    for (const auto& testMetrics : metricsAndTimeHistory.TestMetricsHistory) {
        CB_ENSURE(testMetrics.size() <= 1, "Expect only one test dataset");
        if (!testMetrics.empty()) {
            foldContext->MetricValuesOnTest.emplace_back(GetMetricValues(metrics, /*skipMetric*/{}, testMetrics[0]));
        } else {
            foldContext->MetricValuesOnTest.emplace_back(TVector<double>(metrics.size(),
                                                         std::numeric_limits<double>::quiet_NaN()));
        }
    }
}

void UpdatePermutationBlockSize(
    ETaskType taskType,
    TConstArrayRef<TTrainingDataProviders> foldsData,
    NCatboostOptions::TCatBoostOptions* catboostOptions
) {
    if (taskType == ETaskType::GPU) {
        return;
    }

    const auto isConsecutiveLearnFeaturesData = [&] (const TTrainingDataProviders& foldData) {
        const auto& learnObjectsDataProvider
            = dynamic_cast<const TQuantizedObjectsDataProvider&>(*foldData.Learn->ObjectsData);
        return learnObjectsDataProvider.GetFeaturesArraySubsetIndexing().IsConsecutive();
    };
    if (!AllOf(foldsData, isConsecutiveLearnFeaturesData)) {
        catboostOptions->BoostingOptions->PermutationBlockSize = 1;
    }
}

static void UpdateYetiRankEvalMetric(
    NCB::TDataProviderPtr data,
    NPar::ILocalExecutor* localExecutor,
    NCatboostOptions::TCatBoostOptions* catBoostOptions) {

    TTargetStats targetStats;
    if (data->MetaInfo.TargetStats.Defined()) {
        targetStats = *(data->MetaInfo.TargetStats);
    } else {
        targetStats = ComputeTargetStatsForYetiRank(
            data->RawTargetData,
            catBoostOptions->LossFunctionDescription.Get().LossFunction,
            localExecutor
        );
    }
    UpdateYetiRankEvalMetric(targetStats, Nothing(), catBoostOptions);
}


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
    bool isAlreadyShuffled) {

    cvParams.Check();

    NJson::TJsonValue jsonParams;
    NJson::TJsonValue outputJsonParams;
    ConvertIgnoredFeaturesFromStringToIndices(data.Get()->MetaInfo, &plainJsonParams);
    NCatboostOptions::PlainJsonToOptions(plainJsonParams, &jsonParams, &outputJsonParams);
    ConvertParamsToCanonicalFormat(data.Get()->MetaInfo, &jsonParams);
    NCatboostOptions::TCatBoostOptions catBoostOptions(NCatboostOptions::LoadOptions(jsonParams));
    NCatboostOptions::TOutputFilesOptions outputFileOptions;
    outputFileOptions.Load(outputJsonParams);

    if (catBoostOptions.DataProcessingOptions->ClassLabels->empty()) {
        catBoostOptions.DataProcessingOptions->ClassLabels = data->MetaInfo.ClassLabels;
    }
    ui32 approxDimension = GetApproxDimension(catBoostOptions,
                                              labelConverter,
                                              data->RawTargetData.GetTargetDimension());

    if (IsYetiRankLossFunction(catBoostOptions.LossFunctionDescription.Get().LossFunction)) {
        // Can't use standard UpdateYetiRankEvalMetric because for raw data TargetStats might not be available
        UpdateYetiRankEvalMetric(data, localExecutor, &catBoostOptions);
    }

    UpdateSampleRateOption(data->ObjectsData->GetObjectCount(), &catBoostOptions);

    InitializeEvalMetricIfNotSet(catBoostOptions.MetricOptions->ObjectiveMetric,
                                 &catBoostOptions.MetricOptions->EvalMetric);

    UpdateMetricPeriodOption(catBoostOptions, &outputFileOptions);

    TVector<THolder<IMetric>> metrics = CreateMetrics(
        catBoostOptions.MetricOptions,
        evalMetricDescriptor,
        approxDimension,
        data->MetaInfo.HasWeights
    );

    CheckMetrics(metrics, catBoostOptions.LossFunctionDescription.Get().GetLossFunction());
    CheckCrossValidationOptions(data, metrics, catBoostOptions, outputFileOptions, cvParams);

    const ui64 cpuUsedRamLimit =
        ParseMemorySizeDescription(catBoostOptions.SystemOptions->CpuUsedRamLimit.Get());

    TRestorableFastRng64 rand(cvParams.PartitionRandSeed);
    if (cvParams.Shuffle && !isAlreadyShuffled) {
        auto objectsGroupingSubset = NCB::Shuffle(data->ObjectsGrouping, 1, &rand);
        data = data->GetSubset(objectsGroupingSubset, cpuUsedRamLimit, localExecutor);
    }

    const auto overfittingDetectorOptions = catBoostOptions.BoostingOptions->OverfittingDetector;
    catBoostOptions.BoostingOptions->OverfittingDetector->OverfittingDetectorType = EOverfittingDetectorType::None;

    const ETaskType taskType = catBoostOptions.GetTaskType();
    THolder<IModelTrainer> modelTrainerHolder = THolder<IModelTrainer>(TTrainerFactory::Construct(taskType));

    TSetLogging inThisScope(catBoostOptions.LoggingLevel);
    TVector<bool> skipMetricOnTrain = GetSkipMetricOnTrain(metrics);

    TString tmpDir;
    if (outputFileOptions.AllowWriteFiles()) {
        NCB::NPrivate::CreateTrainDirWithTmpDirIfNotExist(outputFileOptions.GetTrainDir(), &tmpDir);
    }

    EMetricBestValue bestValueType;
    float bestPossibleValue;
    metrics.front()->GetBestValue(&bestValueType, &bestPossibleValue);

    results->reserve(metrics.size());
    for (const auto& metric : metrics) {
        TCVResult result;
        result.Metric = metric->GetDescription();
        results->push_back(result);
    }

    if (cvParams.IsCalledFromSearchHyperparameters) {
        outputFileOptions.SetAllowWriteFiles(false);
    }

    ui32 globalMaxIteration = catBoostOptions.BoostingOptions->IterationCount;

    TVector<TVector<TVector<double>>> trainData(cvParams.FoldCount); // [foldId, iteration, MetricId]
    TVector<TVector<TVector<double>>> testData(cvParams.FoldCount); // [foldId, iteration, MetricId]
    TVector<TVector<double>> allApproxes;
    TVector<TVector<float>> labels;

    int metricPeriod = outputFileOptions.GetMetricPeriod();

    for (auto foldIdx : xrange(cvParams.FoldCount)) {
        TErrorTracker errorTracker = CreateErrorTracker(
            overfittingDetectorOptions,
            bestPossibleValue,
            bestValueType,
            /* hasTest */ true
        );
        if (foldIdx == 0 && (metricPeriod > 1 && errorTracker.IsActive())) {
            CATBOOST_WARNING_LOG << "Warning: Overfitting detector is active, thus evaluation metric is " <<
                    "calculated on every iteration. 'metric_period' is ignored for evaluation metric." << Endl;
        }
        if (catBoostOptions.LoggingLevel != ELoggingLevel::Silent) {
            CATBOOST_NOTICE_LOG << "Training on fold [" << foldIdx << "/" << cvParams.FoldCount << "]" << Endl;
        }
        TDataProviders foldRawData = PrepareCvFolds<TDataProviders>(
            data,
            cvParams,
            foldIdx,
            /* oldCvStyleSplit */ false,
            cpuUsedRamLimit,
            localExecutor
        )[0];
        TTrainingDataProviders foldData = GetTrainingData(
            std::move(foldRawData),
            /*trainDataCanByEmpty*/ false,
            Nothing(),
            /*ensureConsecutiveLearnFeaturesDataForCpu*/ false,
            /*unloadCatFeaturePerfectHashFromRam*/ outputFileOptions.AllowWriteFiles(),
            tmpDir,
            quantizedFeaturesInfo,
            &catBoostOptions,
            &labelConverter,
            localExecutor,
            &rand,
            Nothing()
        );
        auto foldOutputFileOptions = outputFileOptions;
        foldOutputFileOptions.SetTrainDir(outputFileOptions.GetTrainDir() + "/fold-" + ToString(foldIdx));
        TFoldContext foldContext(
            foldIdx,
            taskType,
            foldOutputFileOptions,
            std::move(foldData),
            catBoostOptions.RandomSeed,
            cvParams.ReturnModels
        );
        const THolder<ITrainingCallbacks> cvCallbacks = MakeHolder<TCrossValidationCallbacks>(
            globalMaxIteration,
            &errorTracker,
            metrics,
            &foldContext);
        Train(
            catBoostOptions,
            foldContext.OutputOptions.GetTrainDir(),
            objectiveDescriptor,
            evalMetricDescriptor,
            labelConverter,
            metrics,
            errorTracker.IsActive(),
            cvCallbacks.Get(),
            &foldContext,
            modelTrainerHolder.Get(),
            localExecutor
        );
        for (auto iteration : xrange(foldContext.MetricValuesOnTrain.size())) {
            trainData[foldIdx].push_back(foldContext.MetricValuesOnTrain[iteration]);
            testData[foldIdx].push_back(foldContext.MetricValuesOnTest[iteration]);
        }
        if (!outputFileOptions.GetRocOutputPath().empty()) {
            allApproxes.push_back(std::move(foldContext.LastUpdateEvalResult.GetRawValuesRef()[0][0]));
            auto foldLabels = *foldContext.TrainingData.Test[0]->TargetData->GetOneDimensionalTarget();
            labels.emplace_back(foldLabels.begin(), foldLabels.end());
        }
        if (cvParams.ReturnModels) {
            results->front().CVFullModels.push_back(*foldContext.FullModel.Get());
        }
        if (cvParams.IsCalledFromSearchHyperparameters) {
            const int lastIteration = foldContext.MetricValuesOnTrain.size() - 1;
            for (auto metricIdx : xrange(metrics.size())) {
                (*results)[metricIdx].LastTrainEvalMetric.push_back(foldContext.MetricValuesOnTrain[lastIteration][metricIdx]);
                (*results)[metricIdx].LastTestEvalMetric.push_back(foldContext.MetricValuesOnTest[lastIteration][metricIdx]);
            }
        }
    }
    TVector<double> trainFoldsMetric(cvParams.FoldCount), testFoldsMetric(cvParams.FoldCount);
    size_t lastRow = 0;
    for (auto foldIdx : xrange(cvParams.FoldCount)) {
        lastRow = Max(lastRow, testData[foldIdx].size());
    }
    for (auto metricIdx : xrange(metrics.size())) {
        size_t rowIdx = 0;
        while (rowIdx < lastRow) {
            for (auto foldIdx : xrange(cvParams.FoldCount)) {
                if (rowIdx < trainData[foldIdx].size()) {
                    trainFoldsMetric[foldIdx] = trainData[foldIdx][rowIdx][metricIdx];
                    testFoldsMetric[foldIdx] = testData[foldIdx][rowIdx][metricIdx];
                }
            }
            TCVIterationResults cvResults = ComputeIterationResults(trainFoldsMetric, testFoldsMetric, cvParams.FoldCount, skipMetricOnTrain[metricIdx]);
            (*results)[metricIdx].AppendOneIterationResults(rowIdx, cvResults);
            rowIdx = Max(rowIdx + 1, Min(rowIdx + metricPeriod, lastRow - 1u));
        }
    }

    if (!outputFileOptions.GetRocOutputPath().empty()) {
        CB_ENSURE(
            catBoostOptions.LossFunctionDescription->GetLossFunction() == ELossFunction::Logloss,
            "For ROC curve loss function must be Logloss."
        );
        TRocCurve rocCurve(allApproxes, TVector<TConstArrayRef<float>>(labels.begin(), labels.end()), catBoostOptions.SystemOptions.Get().NumThreads);
        rocCurve.OutputRocCurve(outputFileOptions.GetRocOutputPath());
    }
}

void CrossValidate(
    NJson::TJsonValue plainJsonParams,
    TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
    const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
    const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
    TDataProviderPtr data,
    const TCrossValidationParams& cvParams,
    TVector<TCVResult>* results
) {
    cvParams.Check();

    NJson::TJsonValue jsonParams;
    NJson::TJsonValue outputJsonParams;
    ConvertIgnoredFeaturesFromStringToIndices(data.Get()->MetaInfo, &plainJsonParams);
    NCatboostOptions::PlainJsonToOptions(plainJsonParams, &jsonParams, &outputJsonParams);
    ConvertParamsToCanonicalFormat(data.Get()->MetaInfo, &jsonParams);
    NCatboostOptions::TCatBoostOptions catBoostOptions(NCatboostOptions::LoadOptions(jsonParams));
    NCatboostOptions::TOutputFilesOptions outputFileOptions;
    outputFileOptions.Load(outputJsonParams);

    auto trainerEnv = NCB::CreateTrainerEnv(NCatboostOptions::LoadOptions(jsonParams));

    TRestorableFastRng64 rand(cvParams.PartitionRandSeed);

    NPar::TLocalExecutor localExecutor;
    localExecutor.RunAdditionalThreads(catBoostOptions.SystemOptions->NumThreads.Get() - 1);

    const ui64 cpuUsedRamLimit =
        ParseMemorySizeDescription(catBoostOptions.SystemOptions->CpuUsedRamLimit.Get());

    if (cvParams.Shuffle) {
        auto objectsGroupingSubset = NCB::Shuffle(data->ObjectsGrouping, 1, &rand);
        data = data->GetSubset(objectsGroupingSubset, cpuUsedRamLimit, &localExecutor);
    }

    TLabelConverter labelConverter;

    CrossValidate(
        plainJsonParams,
        quantizedFeaturesInfo,
        objectiveDescriptor,
        evalMetricDescriptor,
        labelConverter,
        data,
        cvParams,
        &localExecutor,
        results,
        true);
}

TVector<NCB::TArraySubsetIndexing<ui32>> TransformToVectorArrayIndexing(
    const TVector<TVector<ui32>>& vectorData) {
    TVector<NCB::TArraySubsetIndexing<ui32>> result;
    result.reserve(vectorData.size());
    for (const auto& block: vectorData) {
        result.push_back(
            NCB::TArraySubsetIndexing<ui32>(
                NCB::TIndexedSubset<ui32>(block))
        );
    }
    return result;
}

TVector<TArraySubsetIndexing<ui32>> StratifiedSplitToFolds(
    const TDataProvider& dataProvider,
    ui32 partCount
) {
    CB_ENSURE(
        dataProvider.MetaInfo.TargetCount > 0,
        "Cannot do stratified split: Target data is unavailable"
    );
    CB_ENSURE(
        dataProvider.MetaInfo.TargetCount == 1,
        "Cannot do stratified split: Target data is multi-dimensional"
    );

    switch (dataProvider.RawTargetData.GetTargetType()) {
        case ERawTargetType::Boolean:
        case ERawTargetType::Integer:
        case ERawTargetType::Float: {
            TVector<float> rawTargetData;
            rawTargetData.yresize(dataProvider.GetObjectCount());
            TArrayRef<float> rawTargetDataRef = rawTargetData;
            dataProvider.RawTargetData.GetNumericTarget(TArrayRef<TArrayRef<float>>(&rawTargetDataRef, 1));
            return NCB::StratifiedSplitToFolds<float>(
                *dataProvider.ObjectsGrouping,
                rawTargetData,
                partCount);
        }
        case ERawTargetType::String: {
            TVector<TConstArrayRef<TString>> rawTargetData;
            dataProvider.RawTargetData.GetStringTargetRef(&rawTargetData);
            return NCB::StratifiedSplitToFolds(*dataProvider.ObjectsGrouping, rawTargetData[0], partCount);
        }
        default:
            CB_ENSURE(false, "Unexpected raw target type");
    }
}

TVector<TArraySubsetIndexing<ui32>> StratifiedSplitToFolds(
    const NCB::TTrainingDataProvider& trainingDataProvider,
    ui32 partCount
) {
    TMaybeData<TConstArrayRef<float>> maybeTarget
        = trainingDataProvider.TargetData->GetOneDimensionalTarget();
    CB_ENSURE(maybeTarget, "Cannot do stratified split: Target data is unavailable");
    return NCB::StratifiedSplitToFolds(*trainingDataProvider.ObjectsGrouping, *maybeTarget, partCount);
}
