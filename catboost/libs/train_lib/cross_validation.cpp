#include "cross_validation.h"
#include "train_model.h"
#include "preprocess.h"

#include <catboost/libs/algo/train.h>
#include <catboost/libs/algo/helpers.h>
#include <catboost/libs/algo/roc_curve.h>
#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/loggers/logger.h>
#include <catboost/libs/helpers/data_split.h>
#include <catboost/libs/helpers/int_cast.h>
#include <catboost/libs/helpers/query_info_helper.h>
#include <catboost/libs/helpers/element_range.h>
#include <catboost/libs/helpers/binarize_target.h>
#include <catboost/libs/helpers/restorable_rng.h>
#include <catboost/libs/helpers/permutation.h>
#include <catboost/libs/helpers/vector_helpers.h>
#include <catboost/libs/overfitting_detector/error_tracker.h>
#include <catboost/libs/options/plain_options_helper.h>

#include <library/threading/local_executor/local_executor.h>

#include <util/generic/scope.h>
#include <util/random/shuffle.h>

#include <limits>
#include <cmath>

static TVector<TVector<ui32>> GetSplittedDocs(const TVector<std::pair<ui32, ui32>>& startEnd) {
    TVector<TVector<ui32>> result(startEnd.size());
    for (ui32 fold = 0; fold < result.size(); ++fold) {
        ui32 foldStartIndex = startEnd[fold].first;
        ui32 foldEndIndex = startEnd[fold].second;
        result[fold].reserve(foldEndIndex - foldStartIndex);
        for (ui32 idx = foldStartIndex; idx < foldEndIndex; ++idx) {
            result[fold].push_back(idx);
        }
    }
    return result;
}

static TVector<TVector<ui32>> CalcTrainDocs(const TVector<TVector<ui32>>& testDocs, ui32 docCount) {
    TVector<TVector<ui32>> result(testDocs.size());
    for (ui32 fold = 0; fold < result.size(); ++fold) {
        result[fold].reserve(docCount - testDocs[fold].size());
        for (ui32 testFold = 0; testFold < testDocs.size(); ++testFold) {
            if (testFold == fold) {
                continue;
            }
            for (auto doc : testDocs[testFold]) {
                result[fold].push_back(doc);
            }
        }
    }
    return result;
}

static void PopulateData(const TPool& pool,
                         const TVector<ui32>& indices,
                         TDataset* learnOrTestData) {
    auto& data = *learnOrTestData;
    const TDocumentStorage& docStorage = pool.Docs;
    data.Target.yresize(indices.size());
    data.Weights.yresize(indices.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        data.Target[i] = docStorage.Target[indices[i]];
        data.Weights[i] = docStorage.Weight[indices[i]];
    }
    for (int dim = 0; dim < docStorage.GetBaselineDimension(); ++dim) {
        data.Baseline[dim].yresize(indices.size());
        for (size_t i = 0; i < indices.size(); ++i) {
            data.Baseline[dim][i] = docStorage.Baseline[dim][indices[i]];
        }
    }

    if (!docStorage.QueryId.empty()) {
        data.QueryId.yresize(indices.size());
        for (size_t i = 0; i < indices.size(); ++i) {
            data.QueryId[i] = docStorage.QueryId[indices[i]];
        }
    }
    if (!docStorage.SubgroupId.empty()) {
        data.SubgroupId.yresize(indices.size());
        for (size_t i = 0; i < indices.size(); ++i) {
            data.SubgroupId[i] = docStorage.SubgroupId[indices[i]];
        }
    }
    learnOrTestData->HasGroupWeight = pool.MetaInfo.HasGroupWeight;
    const TVector<float>& groupWeight = data.HasGroupWeight ? data.Weights : TVector<float>();
    UpdateQueriesInfo(data.QueryId, groupWeight, data.SubgroupId, 0, data.GetSampleCount(), &data.QueryInfo);
};

static void PrepareFolds(
    const NCatboostOptions::TLossDescription& lossDescription,
    bool allowConstLabel,
    const TPool& pool,
    const TVector<THolder<TLearnContext>>& contexts,
    const TCrossValidationParams& cvParams,
    TVector<TDataset>* folds,
    TVector<TDataset>* testFolds
) {
    bool hasQuery = !pool.Docs.QueryId.empty();
    TVector<TVector<ui32>> docsInTest;
    TVector<std::pair<ui32, ui32>> testDocsStartEndIndices;
    if (cvParams.Stratified) {
        docsInTest = StratifiedSplit(pool.Docs.Target, cvParams.FoldCount);
    } else {
        testDocsStartEndIndices = hasQuery
            ? Split(pool.Docs.GetDocCount(), pool.Docs.QueryId, cvParams.FoldCount)
            : Split(pool.Docs.GetDocCount(), cvParams.FoldCount);
        docsInTest = GetSplittedDocs(testDocsStartEndIndices);
    }

    const int docCount = pool.Docs.GetDocCount();
    TVector<TVector<ui32>> docsInTrain = CalcTrainDocs(docsInTest, docCount);

    if (cvParams.Inverted) {
        docsInTest.swap(docsInTrain);
    }

    TVector<ui32> docIndices;
    docIndices.reserve(docCount);
    for (ui32 foldIdx = 0; foldIdx < cvParams.FoldCount; ++foldIdx) {
        TDataset learnData;
        TDataset testData;

        docIndices.clear();
        docIndices.insert(docIndices.end(), docsInTrain[foldIdx].begin(), docsInTrain[foldIdx].end());
        docIndices.insert(docIndices.end(), docsInTest[foldIdx].begin(), docsInTest[foldIdx].end());

        PopulateData(pool, docsInTrain[foldIdx], &learnData);
        PopulateData(pool, docsInTest[foldIdx], &testData);

        if (!pool.Pairs.empty()) {
            int testDocsBegin = testDocsStartEndIndices[foldIdx].first;
            int testDocsEnd = testDocsStartEndIndices[foldIdx].second;
            SplitPairsAndReindex(pool.Pairs, testDocsBegin, testDocsEnd, &learnData.Pairs, &testData.Pairs);
        }

        UpdateQueriesPairs(learnData.Pairs, /*invertedPermutation=*/{}, &learnData.QueryInfo);
        UpdateQueriesPairs(testData.Pairs, /*invertedPermutation=*/{}, &testData.QueryInfo);

        const auto& classWeights = contexts[foldIdx]->Params.DataProcessingOptions->ClassWeights;
        const auto& labelConverter =  contexts[foldIdx]->LearnProgress.LabelConverter;
        Preprocess(lossDescription, classWeights, labelConverter, learnData);
        Preprocess(lossDescription, classWeights, labelConverter, testData);


        // TODO(akhropov): cast will be removed after switch to new Pool format. MLTOOLS-140.
        THashSet<int> catFeatures = ToSigned(contexts[foldIdx]->CatFeatures);

        PrepareAllFeaturesLearn(
            catFeatures,
            contexts[foldIdx]->LearnProgress.FloatFeatures,
            Nothing(),
            contexts[foldIdx]->Params.DataProcessingOptions->IgnoredFeatures,
            /*ignoreRedundantFeatures=*/true,
            (size_t)contexts[foldIdx]->Params.CatFeatureParams->OneHotMaxSize,
            /*clearPool=*/false,
            contexts[foldIdx]->LocalExecutor,
            docsInTrain[foldIdx],
            &pool.Docs,
            &learnData.AllFeatures
        );

        PrepareAllFeaturesTest(
            catFeatures,
            contexts[foldIdx]->LearnProgress.FloatFeatures,
            learnData.AllFeatures,
            /*allowNansOnlyInTest=*/true,
            /*clearPool=*/false,
            contexts[foldIdx]->LocalExecutor,
            docsInTest[foldIdx],
            &pool.Docs,
            &testData.AllFeatures
        );

        CheckLearnConsistency(lossDescription, allowConstLabel, learnData);
        CheckTestConsistency(lossDescription, learnData, testData);

        folds->push_back(learnData);
        testFolds->push_back(testData);
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
    TPool& pool,
    const TCrossValidationParams& cvParams,
    TVector<TCVResult>* results
) {
    NJson::TJsonValue jsonParams;
    NJson::TJsonValue outputJsonParams;
    NCatboostOptions::PlainJsonToOptions(plainJsonParams, &jsonParams, &outputJsonParams);
    NCatboostOptions::TOutputFilesOptions outputFileOptions(ETaskType::CPU);
    outputFileOptions.Load(outputJsonParams);
    NCatboostOptions::TCatBoostOptions params(NCatboostOptions::LoadOptions(jsonParams));

    CB_ENSURE(pool.Docs.GetDocCount() != 0, "Pool is empty");
    CB_ENSURE(pool.Docs.GetDocCount() > cvParams.FoldCount, "Pool is too small to be split into folds");

    const int featureCount = pool.Docs.GetEffectiveFactorCount();

    TVector<THolder<TLearnContext>> contexts;
    contexts.reserve(cvParams.FoldCount);

    const int oneFoldSize = pool.Docs.GetDocCount() / cvParams.FoldCount;
    const int cvTrainSize = cvParams.Inverted ? oneFoldSize : oneFoldSize * (cvParams.FoldCount - 1);
    SetDataDependantDefaults(
        cvTrainSize,
        /*testPoolSize=*/pool.Docs.GetDocCount() - cvTrainSize,
        /*hasTestLabels=*/true,
        !pool.IsTrivialWeights(),
        &outputFileOptions.UseBestModel,
        &params
    );
    for (size_t idx = 0; idx < cvParams.FoldCount; ++idx) {
        contexts.emplace_back(new TLearnContext(
            params,
            objectiveDescriptor,
            evalMetricDescriptor,
            outputFileOptions,
            featureCount,
            // TODO(akhropov): cast will be removed after switch to new Pool format. MLTOOLS-140.
            ToUnsigned(pool.CatFeatures),
            pool.FeatureId,
            "fold_" + ToString(idx) + "_"
        ));
    }

    // TODO(kirillovs): All contexts are created equally, the difference is only in
    // learn progress. Its better to have TCommonContext as a field in TLearnContext
    // without fields duplication.
    auto& ctx = contexts.front();

    SetLogingLevel(ctx->Params.LoggingLevel);

    Y_DEFER {
        SetSilentLogingMode();
    };

    if (IsMultiClassMetric(ctx->Params.LossFunctionDescription->GetLossFunction())) {
        for (const auto& context : contexts) {
            int classesCount = GetClassesCount(
                    context->Params.DataProcessingOptions->ClassesCount,
                    context->Params.DataProcessingOptions->ClassNames
            );
            context->LearnProgress.LabelConverter.Initialize(pool.Docs.Target, classesCount);
            context->LearnProgress.ApproxDimension =  context->LearnProgress.LabelConverter.GetApproxDimension();
        }
    }

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
        CB_ENSURE(pool.Docs.QueryId.size() == pool.Docs.Target.size(), "Query ids not provided for querywise metric.");
        CB_ENSURE(!cvParams.Stratified, "Stratified split is incompatible with groupwise metrics");
    }

    TRestorableFastRng64 rand(cvParams.PartitionRandSeed);

    TVector<ui64> indices(pool.Docs.GetDocCount(), 0);
    std::iota(indices.begin(), indices.end(), 0);
    if (cvParams.Shuffle) {
        Shuffle(pool.Docs.QueryId, rand, &indices);
    }

    ApplyPermutation(InvertPermutation(indices), &pool, &ctx->LocalExecutor);
    Y_DEFER {
        ApplyPermutation(indices, &pool, &ctx->LocalExecutor);
    };
    TVector<TFloatFeature> floatFeatures;
    GenerateBorders(pool, ctx.Get(), &floatFeatures);

    for (size_t i = 0; i < cvParams.FoldCount; ++i) {
        contexts[i]->LearnProgress.FloatFeatures = floatFeatures;
    }

    TVector<TDataset> learnFolds;
    TVector<TDataset> testFolds;
    PrepareFolds(ctx->Params.LossFunctionDescription.Get(), ctx->Params.DataProcessingOptions->AllowConstLabel, pool, contexts, cvParams, &learnFolds, &testFolds);

    for (size_t foldIdx = 0; foldIdx < learnFolds.size(); ++foldIdx) {
        contexts[foldIdx]->InitContext(learnFolds[foldIdx], {&testFolds[foldIdx]});
    }

    const bool isPairwiseScoring = IsPairwiseScoring(ctx->Params.LossFunctionDescription->GetLossFunction());
    for (size_t foldIdx = 0; foldIdx < learnFolds.size(); ++foldIdx) {
        TLearnContext& ctx = *contexts[foldIdx];
        const int defaultCalcStatsObjBlockSize = static_cast<int>(ctx.Params.ObliviousTreeOptions->DevScoreCalcObjBlockSize);
        if (IsSamplingPerTree(ctx.Params.ObliviousTreeOptions.Get())) {
            ctx.SmallestSplitSideDocs.Create(ctx.LearnProgress.Folds, isPairwiseScoring, defaultCalcStatsObjBlockSize);
            ctx.PrevTreeLevelStats.Create(
                ctx.LearnProgress.Folds,
                CountNonCtrBuckets(CountSplits(ctx.LearnProgress.FloatFeatures), learnFolds[foldIdx].AllFeatures),
                static_cast<int>(ctx.Params.ObliviousTreeOptions->MaxDepth)
            );
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

        for (size_t foldIdx = 0; foldIdx < learnFolds.size(); ++foldIdx) {
            TrainOneIteration(learnFolds[foldIdx], &testFolds[foldIdx], contexts[foldIdx].Get());
            CalcErrors(
                learnFolds[foldIdx],
                {&testFolds[foldIdx]},
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
            for (size_t foldIdx = 0; foldIdx < learnFolds.size(); ++foldIdx) {
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

            TCVIterationResults cvResults = ComputeIterationResults(trainFoldsMetric, testFoldsMetric, learnFolds.size());

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
        TVector<TVector<double>> allApproxes(1, TVector<double>(pool.Docs.GetDocCount()));
        size_t documentOffset = 0;
        for (size_t foldIdx = 0; foldIdx < testFolds.size(); ++foldIdx) {
            for (size_t documentIdx = 0; documentIdx < testFolds[foldIdx].GetSampleCount(); ++documentIdx) {
                allApproxes[0][documentOffset + documentIdx] =
                    contexts[foldIdx]->LearnProgress.TestApprox[0][0][documentIdx];
            }
            documentOffset += testFolds[foldIdx].GetSampleCount();
        }
        TVector<TPool> pools(1, pool);
        TRocCurve rocCurve(allApproxes, pools, &ctx->LocalExecutor);
        rocCurve.OutputRocCurve(ctx->OutputOptions.GetRocOutputPath());
    }
}
