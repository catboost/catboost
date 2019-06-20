#include "cross_validation.h"
#include "train_model.h"
#include "options_helper.h"

#include <catboost/libs/algo/approx_dimension.h>
#include <catboost/libs/algo/calc_score_cache.h>
#include <catboost/libs/algo/data.h>
#include <catboost/libs/algo/helpers.h>
#include <catboost/libs/algo/learn_context.h>
#include <catboost/libs/algo/preprocess.h>
#include <catboost/libs/algo/roc_curve.h>
#include <catboost/libs/algo/train.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/restorable_rng.h>
#include <catboost/libs/helpers/vector_helpers.h>
#include <catboost/libs/loggers/catboost_logger_helpers.h>
#include <catboost/libs/loggers/logger.h>
#include <catboost/libs/logging/logging.h>
#include <catboost/libs/logging/profile_info.h>
#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/model/features.h>
#include <catboost/libs/options/enum_helpers.h>
#include <catboost/libs/options/output_file_options.h>
#include <catboost/libs/options/plain_options_helper.h>

#include <util/folder/tempdir.h>
#include <util/generic/algorithm.h>
#include <util/generic/mapfindptr.h>
#include <util/generic/scope.h>
#include <util/generic/ymath.h>
#include <util/generic/maybe.h>
#include <util/stream/labeled.h>
#include <util/string/cast.h>
#include <util/system/hp_timer.h>

#include <cmath>
#include <numeric>


using namespace NCB;


TConstArrayRef<TString> GetTargetForStratifiedSplit(const TDataProvider& dataProvider) {
    auto maybeTarget = dataProvider.RawTargetData.GetTarget();
    CB_ENSURE(maybeTarget, "Cannot do stratified split: Target data is unavailable");
    return *maybeTarget;
}


TConstArrayRef<float> GetTargetForStratifiedSplit(const TTrainingDataProvider& dataProvider) {
    return *dataProvider.TargetData->GetTarget();
}


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


struct TFoldContext {
    TString NamesPrefix;

    ETaskType TaskType;

    THolder<TTempDir> TempDir; // THolder because of bugs with move semantics of TTempDir
    NCatboostOptions::TOutputFilesOptions OutputOptions; // with modified Overfitting params, TrainDir
    TTrainingDataProviders TrainingData;

    THolder<TLearnProgress> LearnProgress;

    TVector<TVector<double>> MetricValuesOnTrain; // [iter][metricIdx]
    TVector<TVector<double>> MetricValuesOnTest;  // [iter][metricIdx]

    TEvalResult LastUpdateEvalResult;

    TRestorableFastRng64 Rand;

public:
    TFoldContext(
        size_t foldIdx,
        ETaskType taskType,
        const NJson::TJsonValue& commonOutputJsonParams,
        TTrainingDataProviders&& trainingData,
        ui64 randomSeed)
        : NamesPrefix("fold_" + ToString(foldIdx) + "_")
        , TaskType(taskType)
        , TempDir(MakeHolder<TTempDir>())
        , TrainingData(std::move(trainingData))
        , Rand(randomSeed)
    {
        NJson::TJsonValue outputJsonParams = commonOutputJsonParams;
        outputJsonParams["train_dir"] = TempDir->Name();
        outputJsonParams["use_best_model"] = false;

        // TODO(akhropov): implement learning continuation for GPU, do not rely on snapshots. MLTOOLS-3735.
        outputJsonParams["save_snapshot"] = (taskType == ETaskType::GPU);
        OutputOptions.Load(outputJsonParams);
    }

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
        IModelTrainer* modelTrainer,
        NPar::TLocalExecutor* localExecutor,
        TMaybe<ui32>* upToIteration) { // exclusive bound, if not inited - init from profile data

        // don't output data from folds training
        TSetLoggingSilent silentMode;

        const size_t batchStartIteration = MetricValuesOnTest.size();
        Y_ASSERT(
            !batchStartIteration ||
            (TaskType != ETaskType::CPU) ||
            (LearnProgress && (LearnProgress->GetInitModelTreesSize() == batchStartIteration))
        );

        double batchIterationsTime = 0.0; // without initialization time

        TTrainModelInternalOptions internalOptions;
        internalOptions.CalcMetricsOnly = true;
        internalOptions.ForceCalcEvalMetricOnEveryIteration = isErrorTrackerActive;
        internalOptions.OffsetMetricPeriodByInitModelSize = true;

        THPTimer trainTimer;
        THolder<TLearnProgress> dstLearnProgress;

        TOnEndIterationCallback onEndIterationCallback
            = [
                batchStartIteration,
                loggingLevel,
                &upToIteration,
                &batchIterationsTime,
                &trainTimer,
                maxTimeSpentOnFixedCostRatio,
                maxIterationsBatchSize,
                globalMaxIteration,
                isErrorTrackerActive,
                &metrics,
                skipMetricOnTrain,
                this
              ] (const TMetricsAndTimeLeftHistory& metricsAndTimeHistory) -> bool {
                    Y_VERIFY(metricsAndTimeHistory.TimeHistory.size() > 0);
                    size_t iteration = (TaskType == ETaskType::CPU) ?
                          batchStartIteration + (metricsAndTimeHistory.TimeHistory.size() - 1)
                        : (metricsAndTimeHistory.TimeHistory.size() - 1);

                    // replay
                    if (iteration < MetricValuesOnTest.size()) {
                        return true;
                    }

                    if (!*upToIteration) {
                        TSetLogging inThisScope(loggingLevel);

                        batchIterationsTime += metricsAndTimeHistory.TimeHistory.back().IterationTime;

                        TMaybe<ui32> prevUpToIteration = *upToIteration;

                        *upToIteration
                            = batchStartIteration + CalcBatchSize(
                                TaskType,
                                iteration,
                                batchStartIteration,
                                batchIterationsTime,
                                trainTimer.Passed(),
                                maxTimeSpentOnFixedCostRatio,
                                maxIterationsBatchSize,
                                globalMaxIteration);

                        if (*upToIteration != prevUpToIteration) {
                            CATBOOST_INFO_LOG << "CrossValidation: batch iterations upper bound estimate = "
                                << **upToIteration << Endl;
                        }
                    }

                    UpdateMetricsAfterIteration(
                        iteration,
                        globalMaxIteration,
                        isErrorTrackerActive,
                        metrics,
                        skipMetricOnTrain,
                        metricsAndTimeHistory);

                    return (iteration + 1) < **upToIteration;
                };

        modelTrainer->TrainModel(
            internalOptions,
            catboostOption,
            OutputOptions,
            objectiveDescriptor,
            evalMetricDescriptor,
            onEndIterationCallback,
            TrainingData,
            labelConverter,
            /*initModel*/ Nothing(),
            std::move(LearnProgress),
            /*initModelApplyCompatiblePools*/ TDataProviders(),
            localExecutor,
            /*rand*/ Nothing(),
            /*model*/ nullptr,
            TVector<TEvalResult*>{&LastUpdateEvalResult},
            /*metricsAndTimeHistory*/nullptr,
            (TaskType == ETaskType::CPU) ? &dstLearnProgress : nullptr
        );
        LearnProgress = std::move(dstLearnProgress);
    }

private:
    void UpdateMetricsAfterIteration(
        size_t iteration,
        size_t globalMaxIteration,
        bool isErrorTrackerActive,
        TConstArrayRef<THolder<IMetric>> metrics,
        TConstArrayRef<bool> skipMetricOnTrain,
        const TMetricsAndTimeLeftHistory& metricsAndTimeHistory
    ) {
        bool calcMetrics = DivisibleOrLastIteration(
            iteration,
            globalMaxIteration,
            OutputOptions.GetMetricPeriod()
        );

        const bool calcErrorTrackerMetric = calcMetrics || isErrorTrackerActive;
        const int errorTrackerMetricIdx = calcErrorTrackerMetric ? 0 : -1;

        MetricValuesOnTrain.resize(iteration + 1);
        MetricValuesOnTest.resize(iteration + 1);

        for (auto metricIdx : xrange((int)metrics.size())) {
            if (!calcMetrics && (metricIdx != errorTrackerMetricIdx)) {
                continue;
            }
            const auto& metric = metrics[metricIdx];
            const TString& metricDescription = metric->GetDescription();

            const auto* metricValueOnTrain
                = MapFindPtr(metricsAndTimeHistory.LearnMetricsHistory.back(), metricDescription);
            MetricValuesOnTrain[iteration].push_back(
                (skipMetricOnTrain[metricIdx] || (metricValueOnTrain == nullptr)) ?
                    std::numeric_limits<double>::quiet_NaN() :
                    *metricValueOnTrain);

            MetricValuesOnTest[iteration].push_back(
                metricsAndTimeHistory.TestMetricsHistory.back()[0].at(metricDescription));
        }
    }

};


static void UpdatePermutationBlockSize(
    ETaskType taskType,
    TConstArrayRef<TTrainingDataProviders> foldsData,
    NCatboostOptions::TCatBoostOptions* catboostOptions
) {
    if (taskType == ETaskType::GPU) {
        return;
    }

    bool isAnyFoldHasNonConsecutiveLearnFeaturesData = AnyOf(
        foldsData,
        [&] (const TTrainingDataProviders& foldData) {
            const auto& learnObjectsDataProvider
                = dynamic_cast<const TQuantizedForCPUObjectsDataProvider&>(*foldData.Learn->ObjectsData);

            return !learnObjectsDataProvider.GetFeaturesArraySubsetIndexing().IsConsecutive();
        }
    );

    if (isAnyFoldHasNonConsecutiveLearnFeaturesData) {
        catboostOptions->BoostingOptions->PermutationBlockSize = 1;
    }
}


void CrossValidate(
    const NJson::TJsonValue& plainJsonParams,
    const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
    const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
    TDataProviderPtr data,
    const TCrossValidationParams& cvParams,
    TVector<TCVResult>* results
) {
    cvParams.Check();

    NJson::TJsonValue jsonParams;
    NJson::TJsonValue outputJsonParams;
    NCatboostOptions::PlainJsonToOptions(plainJsonParams, &jsonParams, &outputJsonParams);
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

    if (cvParams.Shuffle) {
        auto objectsGroupingSubset = NCB::Shuffle(data->ObjectsGrouping, 1, &rand);
        data = data->GetSubset(objectsGroupingSubset, &localExecutor);
    }

    TLabelConverter labelConverter;
    TMaybe<float> targetBorder = catBoostOptions.DataProcessingOptions->TargetBorder;

    TTrainingDataProviderPtr trainingData = GetTrainingData(
        std::move(data),
        /*isLearnData*/ true,
        TStringBuf(),
        Nothing(), // TODO(akhropov): allow loading borders and nanModes in CV?
        /*unloadCatFeaturePerfectHashFromRamIfPossible*/ true,
        /*ensureConsecutiveLearnFeaturesDataForCpu*/ false,
        outputFileOptions.AllowWriteFiles(),
        /*quantizedFeaturesInfo*/ nullptr,
        &catBoostOptions,
        &labelConverter,
        &targetBorder,
        &localExecutor,
        &rand);

    UpdateYetiRankEvalMetric(trainingData->MetaInfo.TargetStats, Nothing(), &catBoostOptions);

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

    ui32 approxDimension = GetApproxDimension(catBoostOptions, labelConverter);


    TVector<THolder<IMetric>> metrics = CreateMetrics(
        catBoostOptions.MetricOptions,
        evalMetricDescriptor,
        approxDimension
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
        &localExecutor);

    /* ensure that all folds have the same permutation block size because some of them might be consecutive
       and some might not
    */
    UpdatePermutationBlockSize(taskType, foldsData, &catBoostOptions);

    TVector<TFoldContext> foldContexts;

    for (auto foldIdx : xrange((size_t)cvParams.FoldCount)) {
        foldContexts.emplace_back(
            foldIdx,
            taskType,
            outputJsonParams,
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

    if (outputFileOptions.AllowWriteFiles()) {
        // TODO(akhropov): compatibility name
        TString namesPrefix = "fold_0_";

        TOutputFiles outputFiles(outputFileOptions, namesPrefix);

        TVector<TString> learnSetNames, testSetNames;
        for (auto foldIdx : xrange(cvParams.FoldCount)) {
            learnSetNames.push_back("fold_" + ToString(foldIdx) + "_learn");
            testSetNames.push_back("fold_" + ToString(foldIdx) + "_test");
        }
        AddFileLoggers(
            /*detailedProfile*/false,
            outputFiles.LearnErrorLogFile,
            outputFiles.TestErrorLogFile,
            outputFiles.TimeLeftLogFile,
            outputFiles.JsonLogFile,
            outputFiles.ProfileLogFile,
            outputFileOptions.GetTrainDir(),
            GetJsonMeta(
                catBoostOptions.BoostingOptions->IterationCount.Get(),
                outputFileOptions.GetName(),
                GetConstPointers(metrics),
                learnSetNames,
                testSetNames,
                ELaunchMode::CV),
            outputFileOptions.GetMetricPeriod(),
            &logger
        );
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

            foldContexts[foldIdx].TrainBatch(
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
                modelTrainerHolder.Get(),
                &localExecutor,
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
                            foldContext.NamesPrefix + learnToken,
                            TMetricEvalResult(metric->GetDescription(), trainFoldsMetric.back(), metricIdx == errorTrackerMetricIdx)
                        );
                    }
                    testFoldsMetric.push_back(foldContext.MetricValuesOnTest[iteration][metricIdx]);
                    oneIterLogger.OutputMetric(
                        foldContext.NamesPrefix + testToken,
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

    if (!outputFileOptions.GetRocOutputPath().empty()) {
        CB_ENSURE(
            catBoostOptions.LossFunctionDescription->GetLossFunction() == ELossFunction::Logloss,
            "For ROC curve loss function must be Logloss."
        );
        TVector<TVector<double>> allApproxes;
        TVector<TConstArrayRef<float>> labels;
        for (auto& foldContext : foldContexts) {
            allApproxes.push_back(std::move(foldContext.LastUpdateEvalResult.GetRawValuesRef()[0][0]));
            labels.push_back(*foldContext.TrainingData.Test[0]->TargetData->GetTarget());
        }

        TRocCurve rocCurve(allApproxes, labels, catBoostOptions.SystemOptions.Get().NumThreads);
        rocCurve.OutputRocCurve(outputFileOptions.GetRocOutputPath());
    }
}
