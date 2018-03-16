#include "cross_validation.h"
#include "train_model.h"
#include "preprocess.h"

#include <catboost/libs/algo/train.h>
#include <catboost/libs/algo/helpers.h>
#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/loggers/logger.h>
#include <catboost/libs/helpers/data_split.h>
#include <catboost/libs/helpers/query_info_helper.h>
#include <catboost/libs/helpers/element_range.h>
#include <catboost/libs/helpers/binarize_target.h>
#include <catboost/libs/helpers/restorable_rng.h>
#include <catboost/libs/helpers/permutation.h>
#include <catboost/libs/helpers/vector_helpers.h>
#include <catboost/libs/overfitting_detector/error_tracker.h>
#include <catboost/libs/options/plain_options_helper.h>

#include <util/random/shuffle.h>

#include <limits>
#include <cmath>

static TVector<TVector<size_t>> GetSplittedDocs(const TVector<std::pair<size_t, size_t>>& startEnd) {
    TVector<TVector<size_t>> result(startEnd.ysize());
    for (int fold = 0; fold < result.ysize(); ++fold) {
        int foldStartIndex = startEnd[fold].first;
        int foldEndIndex = startEnd[fold].second;
        result[fold].reserve(foldEndIndex - foldStartIndex);
        for (int idx = foldStartIndex; idx < foldEndIndex; ++idx) {
            result[fold].push_back(idx);
        }
    }
    return result;
}

static TVector<TVector<size_t>> CalcTrainDocs(const TVector<TVector<size_t>>& testDocs, int docCount) {
    TVector<TVector<size_t>> result(testDocs.size());
    for (int fold = 0; fold < result.ysize(); ++fold) {
        result[fold].reserve(docCount - testDocs[fold].ysize());
        for (int testFold = 0; testFold < testDocs.ysize(); ++testFold) {
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
                         const TVector<size_t>& indices,
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
    UpdateQueriesInfo(data.QueryId, data.SubgroupId, 0, data.GetSampleCount(), &data.QueryInfo);
};

static void PrepareFolds(
    const NCatboostOptions::TLossDescription& lossDescription,
    const TPool& pool,
    const TVector<THolder<TLearnContext>>& contexts,
    const TCrossValidationParams& cvParams,
    TVector<TDataset>* folds,
    TVector<TDataset>* testFolds
) {
    bool hasQuery = !pool.Docs.QueryId.empty();
    if (hasQuery) {
        CB_ENSURE(!cvParams.Stratified, "Stratified cross validation is not supported for datasets with query id.");
    }

    TVector<TVector<size_t>> docsInTest;
    TVector<std::pair<size_t, size_t>> testDocsStartEndIndices;
    if (cvParams.Stratified) {
        CB_ENSURE(!IsQuerywiseError(lossDescription.GetLossFunction()), "Stratified CV isn't supported for querywise errors");
        docsInTest = StratifiedSplit(pool.Docs.Target, cvParams.FoldCount);
    } else {
        testDocsStartEndIndices = hasQuery
            ? Split(pool.Docs.GetDocCount(), pool.Docs.QueryId, cvParams.FoldCount)
            : Split(pool.Docs.GetDocCount(), cvParams.FoldCount);
        docsInTest = GetSplittedDocs(testDocsStartEndIndices);
    }

    const int docCount = pool.Docs.GetDocCount();
    TVector<TVector<size_t>> docsInTrain = CalcTrainDocs(docsInTest, docCount);

    if (cvParams.Inverted) {
        docsInTest.swap(docsInTrain);
    }

    TVector<size_t> docIndices;
    docIndices.reserve(docCount);
    for (size_t foldIdx = 0; foldIdx < cvParams.FoldCount; ++foldIdx) {
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

        const TVector<float>& classWeights = contexts[foldIdx]->Params.DataProcessingOptions->ClassWeights;
        Preprocess(lossDescription, classWeights, learnData);
        Preprocess(lossDescription, classWeights, testData);

        PrepareAllFeaturesLearn(
            contexts[foldIdx]->CatFeatures,
            contexts[foldIdx]->LearnProgress.FloatFeatures,
            contexts[foldIdx]->Params.DataProcessingOptions->IgnoredFeatures,
            /*ignoreRedundantFeatures=*/true,
            (size_t)contexts[foldIdx]->Params.CatFeatureParams->OneHotMaxSize,
            contexts[foldIdx]->Params.DataProcessingOptions->FloatFeaturesBinarization->NanMode,
            /*clearPool=*/false,
            contexts[foldIdx]->LocalExecutor,
            docsInTrain[foldIdx],
            &pool.Docs,
            &learnData.AllFeatures
        );

        PrepareAllFeaturesTest(
            contexts[foldIdx]->CatFeatures,
            contexts[foldIdx]->LearnProgress.FloatFeatures,
            learnData.AllFeatures,
            contexts[foldIdx]->Params.DataProcessingOptions->FloatFeaturesBinarization->NanMode,
            /*clearPool=*/false,
            contexts[foldIdx]->LocalExecutor,
            docsInTest[foldIdx],
            &pool.Docs,
            &testData.AllFeatures
        );

        CheckConsistency(lossDescription, learnData, testData);

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

    CB_ENSURE(pool.Docs.GetDocCount() != 0, "Pool is empty");
    CB_ENSURE(pool.Docs.GetDocCount() > cvParams.FoldCount, "Pool is too small to be split into folds");

    const int featureCount = pool.Docs.GetFactorsCount();

    TVector<THolder<TLearnContext>> contexts;
    contexts.reserve(cvParams.FoldCount);

    for (size_t idx = 0; idx < cvParams.FoldCount; ++idx) {
        contexts.emplace_back(new TLearnContext(
            jsonParams,
            objectiveDescriptor,
            evalMetricDescriptor,
            outputFileOptions,
            featureCount,
            pool.CatFeatures,
            pool.FeatureId,
            "fold_" + ToString(idx) + "_"
        ));
    }

    // TODO(kirillovs): All contexts are created equally, the difference is only in
    // learn progress. Its better to have TCommonContext as a field in TLearnContext
    // without fields duplication.
    auto& ctx = contexts.front();

    SetLogingLevel(ctx->Params.LoggingLevel);

    auto loggingGuard = Finally([&] { SetSilentLogingMode(); });

    if (IsMultiClassError(ctx->Params.LossFunctionDescription->GetLossFunction())) {
        for (const auto& context : contexts) {
            context->LearnProgress.ApproxDimension = GetClassesCount(
                pool.Docs.Target,
                static_cast<int>(context->Params.DataProcessingOptions->ClassesCount)
            );
        }
    }

    TVector<THolder<IMetric>> metrics = CreateMetrics(
         ctx->Params.LossFunctionDescription,
         ctx->Params.MetricOptions,
         ctx->EvalMetricDescriptor,
         ctx->LearnProgress.ApproxDimension
    );

    bool hasQuerywiseMetric = false;
    for (const auto& metric : metrics) {
        if (metric.Get()->GetErrorType() == EErrorType::QuerywiseError) {
            hasQuerywiseMetric = true;
        }
    }
    if (hasQuerywiseMetric) {
        CB_ENSURE(pool.Docs.QueryId.size() == pool.Docs.Target.size(), "Query ids not provided for querywise metric.");
    }

    TRestorableFastRng64 rand(cvParams.PartitionRandSeed);

    TVector<ui64> indices(pool.Docs.GetDocCount(), 0);
    std::iota(indices.begin(), indices.end(), 0);
    if (cvParams.Shuffle) {
        Shuffle(pool.Docs.QueryId, rand, &indices);
    }

    ApplyPermutation(InvertPermutation(indices), &pool, &ctx->LocalExecutor);
    auto permutationGuard = Finally([&] { ApplyPermutation(indices, &pool, &ctx->LocalExecutor); });
    TVector<TFloatFeature> floatFeatures;
    GenerateBorders(pool, ctx.Get(), &floatFeatures);

    for (size_t i = 0; i < cvParams.FoldCount; ++i) {
        contexts[i]->LearnProgress.FloatFeatures = floatFeatures;
    }

    TVector<TDataset> learnFolds;
    TVector<TDataset> testFolds;
    PrepareFolds(ctx->Params.LossFunctionDescription.Get(), pool, contexts, cvParams, &learnFolds, &testFolds);

    for (size_t foldIdx = 0; foldIdx < learnFolds.size(); ++foldIdx) {
        contexts[foldIdx]->InitContext(learnFolds[foldIdx], &testFolds[foldIdx]);
    }

    for (size_t foldIdx = 0; foldIdx < learnFolds.size(); ++foldIdx) {
        TLearnContext& ctx = *contexts[foldIdx];
        if (IsSamplingPerTree(ctx.Params.ObliviousTreeOptions.Get())) {
            ctx.SmallestSplitSideDocs.Create(ctx.LearnProgress.Folds);
            ctx.PrevTreeLevelStats.Create(
                ctx.LearnProgress.Folds,
                CountNonCtrBuckets(CountSplits(ctx.LearnProgress.FloatFeatures), learnFolds[foldIdx].AllFeatures.OneHotValues),
                static_cast<int>(ctx.Params.ObliviousTreeOptions->MaxDepth)
            );
        }
        ctx.SampledDocs.Create(
            ctx.LearnProgress.Folds,
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
        testToken,
        /*hasTrain=*/true,
        /*hasTest=*/true,
        ctx->OutputOptions.GetMetricPeriod(),
        &logger
    );

    TProfileInfo& profile = ctx->Profile;
    for (ui32 iteration = 0; iteration < ctx->Params.BoostingOptions->IterationCount; ++iteration) {
        profile.StartNextIteration();

        for (size_t foldIdx = 0; foldIdx < learnFolds.size(); ++foldIdx) {
            TrainOneIteration(learnFolds[foldIdx], &testFolds[foldIdx], contexts[foldIdx].Get());
            CalcErrors(learnFolds[foldIdx], testFolds[foldIdx], metrics, contexts[foldIdx].Get());
        }

        TOneInterationLogger oneIterLogger(logger);
        for (size_t metricIdx = 0; metricIdx < metrics.size(); ++metricIdx) {
            const auto& metric = metrics[metricIdx];
            TVector<double> trainFoldsMetric;
            TVector<double> testFoldsMetric;
            for (size_t foldIdx = 0; foldIdx < learnFolds.size(); ++foldIdx) {
                trainFoldsMetric.push_back(contexts[foldIdx]->LearnProgress.LearnErrorsHistory.back()[metricIdx]);
                oneIterLogger.OutputMetric(
                    contexts[foldIdx]->Files.NamesPrefix + learnToken,
                    TMetricEvalResult(metric->GetDescription(), trainFoldsMetric.back(), metricIdx == 0)
                );
                testFoldsMetric.push_back(contexts[foldIdx]->LearnProgress.TestErrorsHistory.back()[metricIdx]);
                oneIterLogger.OutputMetric(
                    contexts[foldIdx]->Files.NamesPrefix + testToken,
                    TMetricEvalResult(metric->GetDescription(), testFoldsMetric.back(), metricIdx == 0)
                );
            }

            TCVIterationResults cvResults = ComputeIterationResults(trainFoldsMetric, testFoldsMetric, learnFolds.size());

            (*results)[metricIdx].AppendOneIterationResults(cvResults);

            if (metricIdx == 0) {
                TVector<double> valuesToLog;
                errorTracker.AddError(cvResults.AverageTest, iteration, &valuesToLog);
            }

            oneIterLogger.OutputMetric(learnToken, TMetricEvalResult(metric->GetDescription(), cvResults.AverageTrain, metricIdx == 0));
            oneIterLogger.OutputMetric(
                testToken,
                TMetricEvalResult(
                    metric->GetDescription(),
                    cvResults.AverageTest,
                    errorTracker.GetBestError(),
                    errorTracker.GetBestIteration(),
                    metricIdx == 0
                )
            );
        }

        profile.FinishIteration();
        oneIterLogger.OutputProfile(profile.GetProfileResults());

        if (errorTracker.GetIsNeedStop()) {
            MATRIXNET_NOTICE_LOG << "Stopped by overfitting detector "
                << " (" << errorTracker.GetOverfittingDetectorIterationsWait() << " iterations wait)" << Endl;
            break;
        }
    }
}
