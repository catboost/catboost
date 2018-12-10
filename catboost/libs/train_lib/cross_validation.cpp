#include "cross_validation.h"

#include "approx_dimension.h"
#include "data.h"
#include "preprocess.h"
#include "train_model.h"

#include <catboost/libs/algo/calc_score_cache.h>
#include <catboost/libs/algo/helpers.h>
#include <catboost/libs/algo/learn_context.h>
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
#include <catboost/libs/options/defaults_helper.h>
#include <catboost/libs/options/enum_helpers.h>
#include <catboost/libs/options/output_file_options.h>
#include <catboost/libs/options/plain_options_helper.h>

#include <util/generic/algorithm.h>
#include <util/generic/scope.h>
#include <util/generic/ymath.h>

#include <cmath>
#include <numeric>


using namespace NCB;


TConstArrayRef<TString> GetTargetForStratifiedSplit(const TDataProvider& dataProvider) {
    auto maybeTarget = dataProvider.RawTargetData.GetTarget();
    CB_ENSURE(maybeTarget, "Cannot do stratified split: Target data is unavailable");
    return *maybeTarget;
}


TConstArrayRef<float> GetTargetForStratifiedSplit(const TTrainingDataProvider& dataProvider) {
    return NCB::GetTarget(dataProvider.TargetData);
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
    size_t foldCount
) {
    TCVIterationResults cvResults;
    cvResults.AverageTrain = Accumulate(trainErrors.begin(), trainErrors.end(), 0.0) / foldCount;
    cvResults.StdDevTrain = ComputeStdDev(trainErrors, cvResults.AverageTrain);
    cvResults.AverageTest = Accumulate(testErrors.begin(), testErrors.end(), 0.0) / foldCount;
    cvResults.StdDevTest = ComputeStdDev(testErrors, cvResults.AverageTest);
    return cvResults;
}

static TVector<const TLearnContext*> GetRawPointers(const TVector<THolder<TLearnContext>>& contexts) {
    TVector<const TLearnContext*> pointerContexts;
    pointerContexts.reserve(contexts.size());
    for (auto& ctx : contexts) {
        pointerContexts.push_back(ctx.Get());
    }
    return pointerContexts;
}

inline bool DivisibleOrLastIteration(int currentIteration, int iterationsCount, int period) {
    return currentIteration % period == 0 || currentIteration == iterationsCount - 1;
}

void CrossValidate(
    const NJson::TJsonValue& plainJsonParams,
    const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
    const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
    TDataProviderPtr data,
    const TCrossValidationParams& cvParams,
    TVector<TCVResult>* results
) {
    NJson::TJsonValue jsonParams;
    NJson::TJsonValue outputJsonParams;
    NCatboostOptions::PlainJsonToOptions(plainJsonParams, &jsonParams, &outputJsonParams);
    NCatboostOptions::TCatBoostOptions params(NCatboostOptions::LoadOptions(jsonParams));
    NCatboostOptions::TOutputFilesOptions outputFileOptions(params.GetTaskType());
    outputFileOptions.Load(outputJsonParams);


    const ui32 allDataObjectCount = data->ObjectsData->GetObjectCount();

    CB_ENSURE(allDataObjectCount != 0, "Pool is empty");
    CB_ENSURE(allDataObjectCount > cvParams.FoldCount, "Pool is too small to be split into folds");

    // TODO(akhropov): implement ordered split. MLTOOLS-2486.
    CB_ENSURE(
        data->ObjectsData->GetOrder() != EObjectsOrder::Ordered,
        "Cross-validation for Ordered objects data is not yet implemented"
    );

    const ui32 oneFoldSize = allDataObjectCount / cvParams.FoldCount;
    const ui32 cvTrainSize = cvParams.Inverted ? oneFoldSize : oneFoldSize * (cvParams.FoldCount - 1);
    SetDataDependentDefaults(
        cvTrainSize,
        /*testPoolSize=*/allDataObjectCount - cvTrainSize,
        /*hasTestLabels=*/data->MetaInfo.HasTarget,
        /*hasTestPairs*/data->MetaInfo.HasPairs,
        &outputFileOptions.UseBestModel,
        &params
    );


    TRestorableFastRng64 rand(cvParams.PartitionRandSeed);

    NPar::TLocalExecutor localExecutor;
    localExecutor.RunAdditionalThreads(params.SystemOptions->NumThreads.Get() - 1);

    if (cvParams.Shuffle) {
        auto objectsGroupingSubset = NCB::Shuffle(data->ObjectsGrouping, 1, &rand);
        data = data->GetSubset(objectsGroupingSubset, &localExecutor);
    }

    TLabelConverter labelConverter;

    TTrainingDataProviderPtr trainingData = GetTrainingData(
        std::move(data),
        /*isLearnData*/ true,
        TStringBuf(),
        Nothing(), // TODO(akhropov): allow loading borders and nanModes in CV?
        /*unloadCatFeaturePerfectHashFromRam*/ true,
        /*quantizedFeaturesInfo*/ nullptr,
        &params,
        &labelConverter,
        &localExecutor,
        &rand);

    TVector<TFloatFeature> floatFeatures = CreateFloatFeatures(
        *trainingData->ObjectsData->GetQuantizedFeaturesInfo());

    ui32 approxDimension = GetApproxDimension(params, labelConverter);


    TVector<THolder<TLearnContext>> contexts;
    contexts.reserve(cvParams.FoldCount);

    for (size_t idx = 0; idx < cvParams.FoldCount; ++idx) {
        contexts.emplace_back(new TLearnContext(
            params,
            objectiveDescriptor,
            evalMetricDescriptor,
            outputFileOptions,
            trainingData->MetaInfo.FeaturesLayout,
            /*rand*/ Nothing(),
            &localExecutor,
            "fold_" + ToString(idx) + "_"
        ));

        contexts.back()->LearnProgress.ApproxDimension = (int)approxDimension;
        if (approxDimension > 1) {
            contexts.back()->LearnProgress.LabelConverter = labelConverter;
        }
        contexts.back()->LearnProgress.FloatFeatures = floatFeatures;
    }

    // TODO(kirillovs): All contexts are created equally, the difference is only in
    // learn progress. Its better to have TCommonContext as a field in TLearnContext
    // without fields duplication.
    auto& ctx = contexts.front();

    TSetLogging inThisScope(ctx->Params.LoggingLevel);


    TVector<THolder<IMetric>> metrics = CreateMetrics(
        ctx->Params.LossFunctionDescription,
        ctx->Params.MetricOptions,
        ctx->EvalMetricDescriptor,
        ctx->LearnProgress.ApproxDimension
    );
    CheckMetrics(metrics, ctx->Params.LossFunctionDescription.Get().GetLossFunction());

    // TODO(nikitxskv): Remove this hot-fix and make correct skip-metrics support in cv.
    for (THolder<IMetric>& metric : metrics) {
        metric->AddHint("skip_train", "false");
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


    TVector<TTrainingDataProviders> folds = PrepareCvFolds<TTrainingDataProviders>(
        std::move(trainingData),
        cvParams,
        Nothing(),
        /* oldCvStyleSplit */ false,
        &localExecutor);

    TVector<TTrainingForCPUDataProviders> foldsForCpu;
    for (auto& fold : folds) {
        foldsForCpu.emplace_back(fold.Cast<TQuantizedForCPUObjectsDataProvider>());
    }


    for (size_t foldIdx = 0; foldIdx < foldsForCpu.size(); ++foldIdx) {
        contexts[foldIdx]->InitContext(foldsForCpu[foldIdx]);
    }

    const bool isPairwiseScoring = IsPairwiseScoring(ctx->Params.LossFunctionDescription->GetLossFunction());
    for (size_t foldIdx = 0; foldIdx < foldsForCpu.size(); ++foldIdx) {
        TLearnContext& ctx = *contexts[foldIdx];
        const int defaultCalcStatsObjBlockSize = static_cast<int>(ctx.Params.ObliviousTreeOptions->DevScoreCalcObjBlockSize);
        if (ctx.UseTreeLevelCaching()) {
            ctx.SmallestSplitSideDocs.Create(ctx.LearnProgress.Folds, isPairwiseScoring, defaultCalcStatsObjBlockSize);
            ctx.PrevTreeLevelStats.Create(
                ctx.LearnProgress.Folds,
                CountNonCtrBuckets(
                    CountSplits(ctx.LearnProgress.FloatFeatures),
                    *foldsForCpu[foldIdx].Learn->ObjectsData->GetQuantizedFeaturesInfo(),
                    params.CatFeatureParams->OneHotMaxSize),
                static_cast<int>(ctx.Params.ObliviousTreeOptions->MaxDepth));
        }
        ctx.SampledDocs.Create(
            ctx.LearnProgress.Folds,
            isPairwiseScoring,
            defaultCalcStatsObjBlockSize,
            GetBernoulliSampleRate(ctx.Params.ObliviousTreeOptions->BootstrapConfig)
        ); // TODO(espetrov): create only if sample rate < 1
    }

    EMetricBestValue bestValueType;
    float bestPossibleValue;
    metrics.front()->GetBestValue(&bestValueType, &bestPossibleValue);
    TErrorTracker errorTracker = BuildErrorTracker(bestValueType, bestPossibleValue, /* hasTest */ true, ctx.Get());

    results->reserve(metrics.size());
    for (const auto& metric : metrics) {
        TCVResult result;
        result.Metric = metric->GetDescription();
        results->push_back(result);
    }

    TLogger logger;
    TString learnToken = "learn";
    TString testToken = "test";
    if (ctx->OutputOptions.AllowWriteFiles()) {
        TVector<TString> learnSetNames, testSetNames;
        for (const auto& x : GetRawPointers(contexts)) {
            learnSetNames.push_back(x->Files.NamesPrefix + learnToken);
            testSetNames.push_back(x->Files.NamesPrefix + testToken);
        }
        AddFileLoggers(
            /*detailedProfile=*/false,
            ctx->Files.LearnErrorLogFile,
            ctx->Files.TestErrorLogFile,
            ctx->Files.TimeLeftLogFile,
            ctx->Files.JsonLogFile,
            ctx->Files.ProfileLogFile,
            ctx->OutputOptions.GetTrainDir(),
            GetJsonMeta(
                ctx->Params.BoostingOptions->IterationCount.Get(),
                ctx->OutputOptions.GetName(),
                GetConstPointers(metrics),
                learnSetNames,
                testSetNames,
                ELaunchMode::CV),
            ctx->OutputOptions.GetMetricPeriod(),
            &logger
        );
    }

    AddConsoleLogger(
        learnToken,
        {testToken},
        /*hasTrain=*/true,
        ctx->OutputOptions.GetVerbosePeriod(),
        ctx->Params.BoostingOptions->IterationCount,
        &logger
    );

    TProfileInfo& profile = ctx->Profile;
    for (ui32 iteration = 0; iteration < ctx->Params.BoostingOptions->IterationCount; ++iteration) {
        profile.StartNextIteration();

        bool calcMetrics = DivisibleOrLastIteration(
            iteration,
            ctx->Params.BoostingOptions->IterationCount,
            ctx->OutputOptions.GetMetricPeriod()
        );

        const bool calcErrorTrackerMetric = calcMetrics || errorTracker.IsActive();
        const int errorTrackerMetricIdx = calcErrorTrackerMetric ? 0 : -1;

        for (size_t foldIdx = 0; foldIdx < foldsForCpu.size(); ++foldIdx) {
            {
                TSetLogging inThisScope(ctx->Params.LoggingLevel);
                TrainOneIteration(foldsForCpu[foldIdx], contexts[foldIdx].Get());
            }
            CalcErrors(
                foldsForCpu[foldIdx],
                metrics,
                calcMetrics,
                calcErrorTrackerMetric,
                contexts[foldIdx].Get()
            );
        }

        TOneInterationLogger oneIterLogger(logger);

        for (int metricIdx = 0; metricIdx < metrics.ysize(); ++metricIdx) {
            if (!calcMetrics && metricIdx != errorTrackerMetricIdx) {
                continue;
            }
            const auto& metric = metrics[metricIdx];
            const TString& metricDescription = metric->GetDescription();
            TVector<double> trainFoldsMetric;
            TVector<double> testFoldsMetric;
            for (size_t foldIdx = 0; foldIdx < foldsForCpu.size(); ++foldIdx) {
                trainFoldsMetric.push_back(contexts[foldIdx]->LearnProgress.MetricsAndTimeHistory.LearnMetricsHistory.back().at(metricDescription));
                oneIterLogger.OutputMetric(
                    contexts[foldIdx]->Files.NamesPrefix + learnToken,
                    TMetricEvalResult(metric->GetDescription(), trainFoldsMetric.back(), metricIdx == errorTrackerMetricIdx)
                );
                testFoldsMetric.push_back(contexts[foldIdx]->LearnProgress.MetricsAndTimeHistory.TestMetricsHistory.back()[0].at(metricDescription));
                oneIterLogger.OutputMetric(
                    contexts[foldIdx]->Files.NamesPrefix + testToken,
                    TMetricEvalResult(metric->GetDescription(), testFoldsMetric.back(), metricIdx == errorTrackerMetricIdx)
                );
            }

            TCVIterationResults cvResults = ComputeIterationResults(trainFoldsMetric, testFoldsMetric, foldsForCpu.size());

            (*results)[metricIdx].AppendOneIterationResults(cvResults);

            if (metricIdx == errorTrackerMetricIdx) {
                TVector<double> valuesToLog;
                errorTracker.AddError(cvResults.AverageTest, iteration, &valuesToLog);
            }

            oneIterLogger.OutputMetric(learnToken, TMetricEvalResult(metric->GetDescription(), cvResults.AverageTrain, metricIdx == errorTrackerMetricIdx));
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

        profile.FinishIteration();
        oneIterLogger.OutputProfile(profile.GetProfileResults());

        if (errorTracker.GetIsNeedStop()) {
            CATBOOST_NOTICE_LOG << "Stopped by overfitting detector "
                << " (" << errorTracker.GetOverfittingDetectorIterationsWait() << " iterations wait)" << Endl;
            break;
        }
    }

    if (!ctx->OutputOptions.GetRocOutputPath().empty()) {
        CB_ENSURE(
            ctx->Params.LossFunctionDescription->GetLossFunction() == ELossFunction::Logloss,
            "For ROC curve loss function must be Logloss."
        );
        TVector<TVector<double>> allApproxes;
        TVector<TConstArrayRef<float>> labels;
        for (auto foldIdx : xrange(folds.size())) {
            allApproxes.push_back(std::move(contexts[foldIdx]->LearnProgress.TestApprox[0][0]));
            labels.push_back(GetTarget(folds[foldIdx].Test[0]->TargetData));
        }

        TRocCurve rocCurve(allApproxes, labels, params.SystemOptions.Get().NumThreads);
        rocCurve.OutputRocCurve(ctx->OutputOptions.GetRocOutputPath());
    }
}
