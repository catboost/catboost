#include "cross_validation.h"
#include "dir_helper.h"
#include "train_model.h"
#include "options_helper.h"

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
#include <catboost/private/libs/options/enum_helpers.h>
#include <catboost/private/libs/options/plain_options_helper.h>

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


TVector<TArraySubsetIndexing<ui32>> CalcTrainSubsets(
    const TVector<TArraySubsetIndexing<ui32>>& testSubsets,
    ui32 groupCount
) {

    TVector<TVector<ui32>> trainSubsetIndices(testSubsets.size());
    for (ui32 fold = 0; fold < testSubsets.size(); ++fold) {
        trainSubsetIndices[fold].reserve(groupCount - testSubsets[fold].Size());
    }
    for (ui32 testFold = 0; testFold < testSubsets.size(); ++testFold) {
        testSubsets[testFold].ForEach(
            [&](ui32 /*idx*/, ui32 srcIdx) {
                for (ui32 fold = 0; fold < trainSubsetIndices.size(); ++fold) {
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


/*
 * For CPU foldContexts contain LearnProgress that allows quick learning resume when switching
 *  between folds, so use one iteration batches.
 *
 * For GPU where fold context switch is relatively expensive init batch size based on profile data
 */
static ui32 CalcBatchSize(
    ETaskType taskType,
    ui32 lastIteration,
    ui32 batchStartIteration,
    double batchIterationsTime, // in sec
    double batchIterationsPlusTrainInitializationTime, // in sec
    double maxTimeSpentOnFixedCostRatio,
    ui32 maxIterationsBatchSize,
    ui32 globalMaxIteration
) {
    CATBOOST_DEBUG_LOG << "CalcBatchSize:\n\t" << LabeledOutput(taskType) << "\n\t";

    if (taskType == ETaskType::CPU) {
        CATBOOST_DEBUG_LOG << "set batch size to 1\n";
        return 1;
    }

    CB_ENSURE_INTERNAL(batchIterationsTime > 0.0, "batchIterationTime <= 0.0");
    CB_ENSURE_INTERNAL(
        batchIterationsPlusTrainInitializationTime > batchIterationsTime,
        "batchIterationsPlusTrainInitializationTime <= batchIterationsTime"
    );

    double timeSpentOnFixedCost = batchIterationsPlusTrainInitializationTime - batchIterationsTime;
    double averageIterationTime = batchIterationsTime / double(lastIteration - batchStartIteration + 1);

    // estimated to be under fixed cost limit
    double estimatedBatchSize =
        std::ceil((1.0 - maxTimeSpentOnFixedCostRatio)*timeSpentOnFixedCost
           / (maxTimeSpentOnFixedCostRatio*averageIterationTime));

    CATBOOST_DEBUG_LOG
        << LabeledOutput(
            lastIteration,
            batchIterationsTime,
            batchIterationsPlusTrainInitializationTime,
            averageIterationTime) << Endl
        << '\t' << LabeledOutput(timeSpentOnFixedCost, maxTimeSpentOnFixedCostRatio, globalMaxIteration)
        << "\n\testimated batch size to be under fixed cost ratio limit: "
        << estimatedBatchSize << Endl;

    // cast to ui32 from double is safe after taking Min
    ui32 batchSize = Min((double)maxIterationsBatchSize, estimatedBatchSize);

    if ((batchStartIteration + batchSize) <= lastIteration) {
        return lastIteration - batchStartIteration + 1;
    } else if ((batchStartIteration + batchSize) > globalMaxIteration) {
        return globalMaxIteration - batchStartIteration;
    } else {
        return batchSize;
    }
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
    OutputOptions.SetTrainDir(TempDir->Name());
    OutputOptions.UseBestModel = false;
    // TODO(akhropov): implement learning continuation for GPU, do not rely on snapshots. MLTOOLS-3735.
    OutputOptions.SetSaveSnapshotFlag(taskType == ETaskType::GPU);
    if (hasFullModel) {
        FullModel = TFullModel();
    }
}

class TCrossValidationCallbacks : public ITrainingCallbacks {
public:
    TCrossValidationCallbacks(
        ELoggingLevel loggingLevel,
        double maxTimeSpentOnFixedCostRatio,
        ui32 maxIterationsBatchSize,
        size_t globalMaxIteration,
        bool isErrorTrackerActive,
        TConstArrayRef<THolder<IMetric>> metrics,
        TConstArrayRef<bool> skipMetricOnTrain,
        TMaybe<ui32>* upToIteration,
        TFoldContext* foldContext)
    : BatchStartIteration(foldContext->MetricValuesOnTest.size())
    , LoggingLevel(loggingLevel)
    , MaxTimeSpentOnFixedCostRatio(maxTimeSpentOnFixedCostRatio)
    , MaxIterationsBatchSize(maxIterationsBatchSize)
    , GlobalMaxIteration(globalMaxIteration)
    , IsErrorTrackerActive(isErrorTrackerActive)
    , Metrics(metrics)
    , SkipMetricOnTrain(skipMetricOnTrain)
    , UpToIteration(upToIteration)
    , FoldContext(foldContext)
    {
    }

    bool IsContinueTraining(const TMetricsAndTimeLeftHistory& metricsAndTimeHistory) override {
        Y_VERIFY(metricsAndTimeHistory.TimeHistory.size() > 0);
        size_t iteration = (FoldContext->TaskType == ETaskType::CPU) ?
              BatchStartIteration + (metricsAndTimeHistory.TimeHistory.size() - 1)
            : (metricsAndTimeHistory.TimeHistory.size() - 1);

        // replay
        if (iteration < FoldContext->MetricValuesOnTest.size()) {
            return true;
        }

        if (!*UpToIteration) {
            TSetLogging inThisScope(LoggingLevel);

            BatchIterationsTime += metricsAndTimeHistory.TimeHistory.back().IterationTime;

            TMaybe<ui32> prevUpToIteration = *UpToIteration;

            *UpToIteration
                = BatchStartIteration + CalcBatchSize(
                    FoldContext->TaskType,
                    iteration,
                    BatchStartIteration,
                    BatchIterationsTime,
                    TrainTimer.Passed(),
                    MaxTimeSpentOnFixedCostRatio,
                    MaxIterationsBatchSize,
                    GlobalMaxIteration);

            if (*UpToIteration != prevUpToIteration) {
                CATBOOST_INFO_LOG << "CrossValidation: batch iterations upper bound estimate = "
                    << **UpToIteration << Endl;
            }
        }

        const bool calcMetrics = DivisibleOrLastIteration(
            iteration,
            GlobalMaxIteration,
            FoldContext->OutputOptions.GetMetricPeriod());

        UpdateMetricsAfterIteration(
            iteration,
            calcMetrics,
            IsErrorTrackerActive,
            Metrics,
            SkipMetricOnTrain,
            metricsAndTimeHistory,
            &FoldContext->MetricValuesOnTrain,
            &FoldContext->MetricValuesOnTest);

        return (iteration + 1) < **UpToIteration;
    }

private:
    size_t BatchStartIteration;
    ELoggingLevel LoggingLevel;
    double MaxTimeSpentOnFixedCostRatio;
    ui32 MaxIterationsBatchSize;
    size_t GlobalMaxIteration;
    bool IsErrorTrackerActive;
    TConstArrayRef<THolder<IMetric>> Metrics;
    TConstArrayRef<bool> SkipMetricOnTrain;
    THPTimer TrainTimer;
    double BatchIterationsTime = 0.0;
    TMaybe<ui32>* const UpToIteration;
    TFoldContext* const FoldContext;
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
    TMaybe<ui32>* upToIteration) { // exclusive bound, if not inited - init from profile data

    // don't output data from folds training
    TSetLoggingSilent silentMode;

    const size_t batchStartIteration = foldContext->MetricValuesOnTest.size();
    Y_ASSERT(
        !batchStartIteration ||
        (foldContext->TaskType != ETaskType::CPU) ||
        (foldContext->LearnProgress && (foldContext->LearnProgress->GetInitModelTreesSize() == batchStartIteration))
    );

    TTrainModelInternalOptions internalOptions;
    internalOptions.CalcMetricsOnly = true;
    internalOptions.ForceCalcEvalMetricOnEveryIteration = isErrorTrackerActive;
    internalOptions.OffsetMetricPeriodByInitModelSize = true;

    THolder<TLearnProgress> dstLearnProgress;

    const THolder<ITrainingCallbacks> cvCallbacks = MakeHolder<TCrossValidationCallbacks>(
        loggingLevel,
        maxTimeSpentOnFixedCostRatio,
        maxIterationsBatchSize,
        globalMaxIteration,
        isErrorTrackerActive,
        metrics,
        skipMetricOnTrain,
        upToIteration,
        foldContext);

    modelTrainer->TrainModel(
        internalOptions,
        catboostOption,
        foldContext->OutputOptions,
        objectiveDescriptor,
        evalMetricDescriptor,
        foldContext->TrainingData,
        labelConverter,
        cvCallbacks.Get(),
        /*initModel*/ Nothing(),
        std::move(foldContext->LearnProgress),
        /*initModelApplyCompatiblePools*/ TDataProviders(),
        localExecutor,
        /*rand*/ Nothing(),
        /*model*/ nullptr,
        TVector<TEvalResult*>{&foldContext->LastUpdateEvalResult},
        /*metricsAndTimeHistory*/nullptr,
        (foldContext->TaskType == ETaskType::CPU) ? &dstLearnProgress : nullptr
    );
    foldContext->LearnProgress = std::move(dstLearnProgress);
}


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

static TString GetNamesPrefix(ui32 foldIdx) {
    return "fold_" + ToString(foldIdx) + "_";
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
    NPar::TLocalExecutor* localExecutor
) {
    // don't output data from folds training
    TSetLoggingSilent silentMode;

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
    modelTrainer->TrainModel(
        internalOptions,
        catboostOption,
        foldOutputOptions,
        objectiveDescriptor,
        evalMetricDescriptor,
        foldContext->TrainingData,
        labelConverter,
        trainingCallbacks,
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
        CB_ENSURE(testMetrics.size() == 1, "Expect only one test dataset");
        foldContext->MetricValuesOnTest.emplace_back(GetMetricValues(metrics, /*skipMetric*/{}, testMetrics[0]));
    }
}


void UpdateMetricsAfterIteration(
    size_t iteration,
    bool calcMetrics,
    bool isErrorTrackerActive,
    TConstArrayRef<THolder<IMetric>> metrics,
    TConstArrayRef<bool> skipMetricOnTrain,
    const TMetricsAndTimeLeftHistory& metricsAndTimeHistory,
    TVector<TVector<double>>* metricValuesOnTrain,
    TVector<TVector<double>>* metricValuesOnTest
) {
    const bool calcErrorTrackerMetric = calcMetrics || isErrorTrackerActive;
    const int errorTrackerMetricIdx = calcErrorTrackerMetric ? 0 : -1;

    metricValuesOnTrain->resize(iteration + 1);
    metricValuesOnTest->resize(iteration + 1);

    for (auto metricIdx : xrange((int)metrics.size())) {
        if (!calcMetrics && (metricIdx != errorTrackerMetricIdx)) {
            continue;
        }
        const auto& metric = metrics[metricIdx];
        const TString& metricDescription = metric->GetDescription();

        const auto* metricValueOnTrain
            = MapFindPtr(metricsAndTimeHistory.LearnMetricsHistory.back(), metricDescription);
        (*metricValuesOnTrain)[iteration].push_back(
            (skipMetricOnTrain[metricIdx] || (metricValueOnTrain == nullptr)) ?
                std::numeric_limits<double>::quiet_NaN() :
                *metricValueOnTrain);

        (*metricValuesOnTest)[iteration].push_back(
            metricsAndTimeHistory.TestMetricsHistory.back()[0].at(metricDescription));
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
            = dynamic_cast<const TQuantizedForCPUObjectsDataProvider&>(*foldData.Learn->ObjectsData);
        return learnObjectsDataProvider.GetFeaturesArraySubsetIndexing().IsConsecutive();
    };
    if (!AllOf(foldsData, isConsecutiveLearnFeaturesData)) {
        catboostOptions->BoostingOptions->PermutationBlockSize = 1;
    }
}

void CrossValidate(
    NJson::TJsonValue plainJsonParams,
    const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
    const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
    const TLabelConverter& labelConverter,
    NCB::TTrainingDataProviderPtr trainingData,
    const TCrossValidationParams& cvParams,
    NPar::TLocalExecutor* localExecutor,
    TVector<TCVResult>* results,
    bool isAlreadyShuffled) {

    cvParams.Check();

    NJson::TJsonValue jsonParams;
    NJson::TJsonValue outputJsonParams;
    ConvertIgnoredFeaturesFromStringToIndices(trainingData.Get()->MetaInfo, &plainJsonParams);
    NCatboostOptions::PlainJsonToOptions(plainJsonParams, &jsonParams, &outputJsonParams);
    ConvertParamsToCanonicalFormat(trainingData.Get()->MetaInfo, &jsonParams);
    NCatboostOptions::TCatBoostOptions catBoostOptions(NCatboostOptions::LoadOptions(jsonParams));
    NCatboostOptions::TOutputFilesOptions outputFileOptions;
    outputFileOptions.Load(outputJsonParams);

    // TODO(akhropov): implement snapshots in CV. MLTOOLS-3439.
    CB_ENSURE(!outputFileOptions.SaveSnapshot(), "Saving snapshots in Cross-validation is not supported yet");

    const ETaskType taskType = catBoostOptions.GetTaskType();

    // TODO(akhropov): implement learning continuation for GPU, do not rely on snapshots. MLTOOLS-3735.
    CB_ENSURE(
        (taskType == ETaskType::CPU) || outputFileOptions.AllowWriteFiles(),
        "Cross-validation on GPU relies on writing files, so it must be allowed"
    );

    const ui32 allDataObjectCount = trainingData->ObjectsData->GetObjectCount();

    CB_ENSURE(allDataObjectCount != 0, "Pool is empty");
    CB_ENSURE(allDataObjectCount > cvParams.FoldCount, "Pool is too small to be split into folds");

    // TODO(akhropov): implement ordered split. MLTOOLS-2486.
    CB_ENSURE(
        trainingData->ObjectsData->GetOrder() != EObjectsOrder::Ordered,
        "Cross-validation for Ordered objects data is not yet implemented"
    );

    const ui64 cpuUsedRamLimit =
        ParseMemorySizeDescription(catBoostOptions.SystemOptions->CpuUsedRamLimit.Get());

    TRestorableFastRng64 rand(cvParams.PartitionRandSeed);

    if (cvParams.Shuffle && !isAlreadyShuffled) {
        auto objectsGroupingSubset = NCB::Shuffle(trainingData->ObjectsGrouping, 1, &rand);
        trainingData = trainingData->GetSubset(objectsGroupingSubset, cpuUsedRamLimit, localExecutor);
    }

    UpdateYetiRankEvalMetric(trainingData->MetaInfo.TargetStats, Nothing(), &catBoostOptions);
    UpdateSampleRateOption(allDataObjectCount, &catBoostOptions);

    InitializeEvalMetricIfNotSet(catBoostOptions.MetricOptions->ObjectiveMetric,
                                 &catBoostOptions.MetricOptions->EvalMetric);
    NJson::TJsonValue updatedTrainOptionsJson = jsonParams;
    // disable overfitting detector on folds training, it will work on average values
    const auto overfittingDetectorOptions = catBoostOptions.BoostingOptions->OverfittingDetector;
    catBoostOptions.BoostingOptions->OverfittingDetector->OverfittingDetectorType = EOverfittingDetectorType::None;

    // internal training output shouldn't interfere with main stdout
    const auto loggingLevel = catBoostOptions.LoggingLevel;
    catBoostOptions.LoggingLevel = ELoggingLevel::Silent;

    THolder<IModelTrainer> modelTrainerHolder;

    const bool isGpuDeviceType = taskType == ETaskType::GPU;
    if (isGpuDeviceType && TTrainerFactory::Has(ETaskType::GPU)) {
        modelTrainerHolder = TTrainerFactory::Construct(ETaskType::GPU);
    } else {
        CB_ENSURE(
            !isGpuDeviceType,
            "Can't load GPU learning library. "
            "Module was not compiled or driver  is incompatible with package. "
            "Please install latest NVDIA driver and check again");
        modelTrainerHolder = TTrainerFactory::Construct(ETaskType::CPU);
    }

    TSetLogging inThisScope(loggingLevel);

    ui32 approxDimension = GetApproxDimension(catBoostOptions, labelConverter, trainingData->TargetData->GetTargetDimension());


    TVector<THolder<IMetric>> metrics = CreateMetrics(
        catBoostOptions.MetricOptions,
        evalMetricDescriptor,
        approxDimension,
        trainingData->MetaInfo.HasWeights
    );
    CheckMetrics(metrics, catBoostOptions.LossFunctionDescription.Get().GetLossFunction());

    TVector<bool> skipMetricOnTrain = GetSkipMetricOnTrain(metrics);

    bool hasQuerywiseMetric = false;
    for (const auto& metric : metrics) {
        if (metric.Get()->GetErrorType() == EErrorType::QuerywiseError) {
            hasQuerywiseMetric = true;
        }
    }

    if (hasQuerywiseMetric) {
        CB_ENSURE(!cvParams.Stratified, "Stratified split is incompatible with groupwise metrics");
    }


    TVector<TTrainingDataProviders> foldsData = PrepareCvFolds<TTrainingDataProviders>(
        std::move(trainingData),
        cvParams,
        Nothing(),
        /* oldCvStyleSplit */ false,
        cpuUsedRamLimit,
        localExecutor);

    /* ensure that all folds have the same permutation block size because some of them might be consecutive
       and some might not
    */
    UpdatePermutationBlockSize(taskType, foldsData, &catBoostOptions);

    TVector<TFoldContext> foldContexts;

    for (auto foldIdx : xrange((size_t)cvParams.FoldCount)) {
        foldContexts.emplace_back(
            foldIdx,
            taskType,
            outputFileOptions,
            std::move(foldsData[foldIdx]),
            catBoostOptions.RandomSeed);
    }


    EMetricBestValue bestValueType;
    float bestPossibleValue;
    metrics.front()->GetBestValue(&bestValueType, &bestPossibleValue);

    TErrorTracker errorTracker = CreateErrorTracker(
        overfittingDetectorOptions,
        bestPossibleValue,
        bestValueType,
        /* hasTest */ true);

    if (outputFileOptions.GetMetricPeriod() > 1 && errorTracker.IsActive()) {
        CATBOOST_WARNING_LOG << "Warning: Overfitting detector is active, thus evaluation metric is " <<
            "calculated on every iteration. 'metric_period' is ignored for evaluation metric." << Endl;
    }

    results->reserve(metrics.size());
    for (const auto& metric : metrics) {
        TCVResult result;
        result.Metric = metric->GetDescription();
        results->push_back(result);
    }

    TLogger logger;
    TString learnToken = "learn";
    TString testToken = "test";

    if (cvParams.IsCalledFromSearchHyperparameters) {
        outputFileOptions.SetAllowWriteFiles(false);
    }

    if (outputFileOptions.AllowWriteFiles()) {
        TVector<TString> learnSetNames, testSetNames;
        for (auto foldIdx : xrange(cvParams.FoldCount)) {
            learnSetNames.push_back("fold_" + ToString(foldIdx) + "_learn");
            testSetNames.push_back("fold_" + ToString(foldIdx) + "_test");
        }
        const auto& metricsMetaJson = GetJsonMeta(
            catBoostOptions.BoostingOptions->IterationCount.Get(),
            outputFileOptions.GetName(),
            GetConstPointers(metrics),
            learnSetNames,
            testSetNames,
            /*parametersName=*/ "",
            ELaunchMode::CV);
        // TODO(akhropov): compatibility name
        const TString namesPrefix = "fold_0_";
        InitializeFileLoggers(
            outputFileOptions,
            metricsMetaJson,
            namesPrefix,
            /*isDetailedProfile*/false,
            &logger);
    }

    AddConsoleLogger(
        learnToken,
        {testToken},
        /*hasTrain=*/true,
        outputFileOptions.GetVerbosePeriod(),
        catBoostOptions.BoostingOptions->IterationCount,
        &logger
    );

    ui32 globalMaxIteration = catBoostOptions.BoostingOptions->IterationCount;

    TProfileInfo profile(globalMaxIteration);

    ui32 iteration = 0;
    ui32 batchStartIteration = 0;

    while (!errorTracker.GetIsNeedStop() && (batchStartIteration < globalMaxIteration)) {
        profile.StartIterationBlock();

        /* Inited using profile data after first iteration
         *
         * TODO(akhropov): assuming all folds have approximately the same fixed cost/iteration cost ratio
         * this might not be the case in the future when time-split CV folds or custom CV folds
         * will be of different size
         */
        TMaybe<ui32> batchEndIteration;

        for (auto foldIdx : xrange(foldContexts.size())) {
            THPTimer timer;

            TrainBatch(
                catBoostOptions,
                objectiveDescriptor,
                evalMetricDescriptor,
                labelConverter,
                metrics,
                skipMetricOnTrain,
                cvParams.MaxTimeSpentOnFixedCostRatio,
                cvParams.DevMaxIterationsBatchSize,
                globalMaxIteration,
                errorTracker.IsActive(),
                loggingLevel,
                &foldContexts[foldIdx],
                modelTrainerHolder.Get(),
                localExecutor,
                &batchEndIteration);

            Y_ASSERT(batchEndIteration); // should be inited right after the first iteration of the first fold
            CATBOOST_INFO_LOG << "CrossValidation: Processed batch of iterations [" << batchStartIteration
                << ',' << *batchEndIteration << ") for fold " << foldIdx << '/' << cvParams.FoldCount
                << " in " << FloatToString(timer.Passed(), PREC_NDIGITS, 2) << " sec" << Endl;
        }

        while (true) {
            bool calcMetrics = DivisibleOrLastIteration(
                iteration,
                catBoostOptions.BoostingOptions->IterationCount,
                outputFileOptions.GetMetricPeriod()
            );

            const bool calcErrorTrackerMetric = calcMetrics || errorTracker.IsActive();
            const int errorTrackerMetricIdx = calcErrorTrackerMetric ? 0 : -1;

            TOneInterationLogger oneIterLogger(logger);

            for (int metricIdx = 0; metricIdx < metrics.ysize(); ++metricIdx) {
                if (!calcMetrics && metricIdx != errorTrackerMetricIdx) {
                    continue;
                }
                const auto& metric = metrics[metricIdx];

                TVector<double> trainFoldsMetric; // [foldIdx]
                TVector<double> testFoldsMetric; // [foldIdx]
                for (const auto& foldContext : foldContexts) {
                    trainFoldsMetric.push_back(foldContext.MetricValuesOnTrain[iteration][metricIdx]);
                    if (!skipMetricOnTrain[metricIdx]) {
                        oneIterLogger.OutputMetric(
                            GetNamesPrefix(foldContext.FoldIdx) + learnToken,
                            TMetricEvalResult(metric->GetDescription(), trainFoldsMetric.back(), metricIdx == errorTrackerMetricIdx)
                        );
                    }
                    testFoldsMetric.push_back(foldContext.MetricValuesOnTest[iteration][metricIdx]);
                    oneIterLogger.OutputMetric(
                        GetNamesPrefix(foldContext.FoldIdx) + testToken,
                        TMetricEvalResult(metric->GetDescription(), testFoldsMetric.back(), metricIdx == errorTrackerMetricIdx)
                    );
                }

                TCVIterationResults cvResults = ComputeIterationResults(trainFoldsMetric, testFoldsMetric, cvParams.FoldCount, skipMetricOnTrain[metricIdx]);

                if (calcMetrics) {
                    (*results)[metricIdx].AppendOneIterationResults(iteration, cvResults);
                }

                if (metricIdx == errorTrackerMetricIdx) {
                    TVector<double> valuesToLog;
                    errorTracker.AddError(cvResults.AverageTest, iteration, &valuesToLog);
                }

                if (!skipMetricOnTrain[metricIdx]) {
                    oneIterLogger.OutputMetric(
                        learnToken,
                        TMetricEvalResult(metric->GetDescription(),
                        cvResults.AverageTrain.GetRef(),
                        metricIdx == errorTrackerMetricIdx));
                }
                oneIterLogger.OutputMetric(
                    testToken,
                    TMetricEvalResult(
                        metric->GetDescription(),
                        cvResults.AverageTest,
                        errorTracker.GetBestError(),
                        errorTracker.GetBestIteration(),
                        metricIdx == errorTrackerMetricIdx
                    )
                );
            }

            bool lastIterInBatch = false;
            if (errorTracker.GetIsNeedStop()) {
                CATBOOST_NOTICE_LOG << "Stopped by overfitting detector "
                    << " (" << errorTracker.GetOverfittingDetectorIterationsWait() << " iterations wait)" << Endl;
                lastIterInBatch = true;
            }
            ++iteration;
            if (iteration == *batchEndIteration) {
                lastIterInBatch = true;
            }
            if (lastIterInBatch) {
                profile.FinishIterationBlock(iteration - batchStartIteration);
                oneIterLogger.OutputProfile(profile.GetProfileResults());
                batchStartIteration = iteration;
                break;
            }
        }
    }

    if (cvParams.IsCalledFromSearchHyperparameters) {
        const int lastIteration = iteration - 1;
        for (int metricIdx = 0; metricIdx < metrics.ysize(); ++metricIdx) {
            for (const auto& foldContext : foldContexts) {
                (*results)[metricIdx].LastTrainEvalMetric.push_back(foldContext.MetricValuesOnTrain[lastIteration][metricIdx]);
                (*results)[metricIdx].LastTestEvalMetric.push_back(foldContext.MetricValuesOnTest[lastIteration][metricIdx]);;
            }
        }
    }

    if (!outputFileOptions.GetRocOutputPath().empty()) {
        CB_ENSURE(
            catBoostOptions.LossFunctionDescription->GetLossFunction() == ELossFunction::Logloss,
            "For ROC curve loss function must be Logloss."
        );
        TVector<TVector<double>> allApproxes;
        TVector<TConstArrayRef<float>> labels;
        for (auto& foldContext : foldContexts) {
            allApproxes.push_back(std::move(foldContext.LastUpdateEvalResult.GetRawValuesRef()[0][0]));
            labels.push_back(*foldContext.TrainingData.Test[0]->TargetData->GetOneDimensionalTarget());
        }

        TRocCurve rocCurve(allApproxes, labels, catBoostOptions.SystemOptions.Get().NumThreads);
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

    // TODO(akhropov): implement snapshots in CV. MLTOOLS-3439.
    CB_ENSURE(!outputFileOptions.SaveSnapshot(), "Saving snapshots in Cross-validation is not supported yet");

    const ETaskType taskType = catBoostOptions.GetTaskType();

    // TODO(akhropov): implement learning continuation for GPU, do not rely on snapshots. MLTOOLS-3735.
    CB_ENSURE(
        (taskType == ETaskType::CPU) || outputFileOptions.AllowWriteFiles(),
        "Cross-validation on GPU relies on writing files, so it must be allowed"
    );

    const ui32 allDataObjectCount = data->ObjectsData->GetObjectCount();

    CB_ENSURE(allDataObjectCount != 0, "Pool is empty");
    CB_ENSURE(allDataObjectCount > cvParams.FoldCount, "Pool is too small to be split into folds");

    // TODO(akhropov): implement ordered split. MLTOOLS-2486.
    CB_ENSURE(
        data->ObjectsData->GetOrder() != EObjectsOrder::Ordered,
        "Cross-validation for Ordered objects data is not yet implemented"
    );

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
    TMaybe<float> targetBorder = catBoostOptions.DataProcessingOptions->TargetBorder;

    TString tmpDir;
    if (outputFileOptions.AllowWriteFiles()) {
        NCB::NPrivate::CreateTrainDirWithTmpDirIfNotExist(outputFileOptions.GetTrainDir(), &tmpDir);
    }

    TTrainingDataProviderPtr trainingData = GetTrainingData(
        std::move(data),
        /*isLearnData*/ true,
        TStringBuf(),
        Nothing(), // TODO(akhropov): allow loading borders and nanModes in CV?
        /*unloadCatFeaturePerfectHashFromRam*/ outputFileOptions.AllowWriteFiles(),
        /*ensureConsecutiveLearnFeaturesDataForCpu*/ false,
        tmpDir,
        quantizedFeaturesInfo,
        &catBoostOptions,
        &labelConverter,
        &targetBorder,
        &localExecutor,
        &rand);

    CrossValidate(
        plainJsonParams,
        objectiveDescriptor,
        evalMetricDescriptor,
        labelConverter,
        trainingData,
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
            Y_UNREACHABLE();
    }
    Y_UNREACHABLE();
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
