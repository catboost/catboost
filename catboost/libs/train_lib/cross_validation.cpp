#include "cross_validation.h"
#include "train_model.h"

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

static void PrepareFolds(
    const NCatboostOptions::TLossDescription& lossDescription,
    const TPool& pool,
    const TVector<THolder<TLearnContext>>& contexts,
    const TCrossValidationParams& cvParams,
    TVector<TTrainData>* folds
) {
    bool hasQuery = !pool.Docs.QueryId.empty();
    if (hasQuery) {
        CB_ENSURE(!cvParams.Stratified, "Stratified cross validation is not supported for datasets with query id.");
    }

    TVector<TVector<size_t>> docsInTest;
    if (cvParams.Stratified) {
        docsInTest = StratifiedSplit(pool.Docs.Target, cvParams.FoldCount);
    } else {
        auto startEnd = hasQuery
            ? Split(pool.Docs.GetDocCount(), pool.Docs.QueryId, cvParams.FoldCount)
            : Split(pool.Docs.GetDocCount(), cvParams.FoldCount);
        docsInTest = GetSplittedDocs(startEnd);
    }

    const int docCount = pool.Docs.GetDocCount();
    TVector<TVector<size_t>> docsInTrain = CalcTrainDocs(docsInTest, docCount);

    if (cvParams.Inverted) {
        docsInTest.swap(docsInTrain);
    }

    TVector<size_t> docIndices;
    docIndices.reserve(docCount);
    for (size_t foldIdx = 0; foldIdx < cvParams.FoldCount; ++foldIdx) {
        TTrainData fold;
        fold.LearnSampleCount = docsInTrain[foldIdx].ysize();
        fold.Target.reserve(docCount);
        fold.Weights.reserve(docCount);

        docIndices.clear();
        docIndices.insert(docIndices.end(), docsInTrain[foldIdx].begin(), docsInTrain[foldIdx].end());
        docIndices.insert(docIndices.end(), docsInTest[foldIdx].begin(), docsInTest[foldIdx].end());
        for (auto idx : docIndices) {
            fold.Target.push_back(pool.Docs.Target[idx]);
            fold.Weights.push_back(pool.Docs.Weight[idx]);
        }

        if (lossDescription.GetLossFunction() == ELossFunction::Logloss) {
            PrepareTargetBinary(NCatboostOptions::GetLogLossBorder(lossDescription), &fold.Target);
            float minTarget = *MinElement(fold.Target.begin(), fold.Target.begin() + fold.LearnSampleCount);
            float maxTarget = *MaxElement(fold.Target.begin(), fold.Target.begin() + fold.LearnSampleCount);
            CB_ENSURE(minTarget == 0, "All targets are greater than border");
            CB_ENSURE(maxTarget == 1, "All targets are smaller than border");
        }

        for (int dim = 0; dim < pool.Docs.GetBaselineDimension(); ++dim) {
            fold.Baseline[dim].reserve(pool.Docs.GetDocCount());
            for (auto idx : docIndices) {
                fold.Baseline[dim].push_back(pool.Docs.Baseline[dim][idx]);
            }
        }
        if (hasQuery) {
            fold.QueryId.reserve(pool.Docs.GetDocCount());
            for (size_t idx = 0; idx < docIndices.size(); ++idx) {
                fold.QueryId.push_back(pool.Docs.QueryId[docIndices[idx]]);
            }
            UpdateQueriesInfo(fold.QueryId, 0, fold.LearnSampleCount, &fold.QueryInfo);
            fold.LearnQueryCount = fold.QueryInfo.ysize();
            UpdateQueriesInfo(fold.QueryId, fold.LearnSampleCount, fold.GetSampleCount(), &fold.QueryInfo);
        }

        PrepareAllFeaturesFromPermutedDocs(
            docIndices,
            contexts[foldIdx]->CatFeatures,
            contexts[foldIdx]->LearnProgress.FloatFeatures,
            contexts[foldIdx]->Params.DataProcessingOptions->IgnoredFeatures,
            fold.LearnSampleCount,
            (size_t)contexts[foldIdx]->Params.CatFeatureParams->OneHotMaxSize,
            contexts[foldIdx]->Params.DataProcessingOptions->FloatFeaturesBinarization->NanMode,
            /*allowClearPool*/ false,
            contexts[foldIdx]->LocalExecutor,
            &pool.Docs,
            &fold.AllFeatures
        );

        folds->push_back(fold);
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

    float minTarget = *MinElement(pool.Docs.Target.begin(), pool.Docs.Target.end());
    float maxTarget = *MaxElement(pool.Docs.Target.begin(), pool.Docs.Target.end());
    CB_ENSURE(minTarget != maxTarget, "All targets are equal");

    if (IsMultiClassError(ctx->Params.LossFunctionDescription->GetLossFunction())) {
        CB_ENSURE(AllOf(pool.Docs.Target, [] (float target) { return floor(target) == target && target >= 0; }),
                  "Each target label should be non-negative integer for Multiclass/MultiClassOneVsAll loss function");
        for (const auto& context : contexts) {
            context->LearnProgress.ApproxDimension = GetClassesCount(
                pool.Docs.Target,
                static_cast<int>(context->Params.DataProcessingOptions->ClassesCount)
            );
        }
        CB_ENSURE(ctx->LearnProgress.ApproxDimension > 1, "All targets are equal");
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

    TVector<TTrainData> folds;
    PrepareFolds(ctx->Params.LossFunctionDescription.Get(), pool, contexts, cvParams, &folds);

    for (size_t foldIdx = 0; foldIdx < folds.size(); ++foldIdx) {
        contexts[foldIdx]->InitData(folds[foldIdx]);

        TLearnContext& ctx = *contexts[foldIdx];
        TFold* learnFold;
        if (!ctx.LearnProgress.Folds.empty()) {
            learnFold = &ctx.LearnProgress.Folds[0];
        } else {
            learnFold = &ctx.LearnProgress.AveragingFold;
        }
        const TTrainData& data = folds[foldIdx];
        if (IsSamplingPerTree(ctx.Params.ObliviousTreeOptions.Get())) {
            ctx.SmallestSplitSideDocs.Create(*learnFold); // assume that all folds have the same shape
            const int approxDimension = learnFold->GetApproxDimension();
            const int bodyTailCount = learnFold->BodyTailArr.ysize();
            ctx.PrevTreeLevelStats.Create(
                CountNonCtrBuckets(CountSplits(ctx.LearnProgress.FloatFeatures), data.AllFeatures.OneHotValues),
                static_cast<int>(ctx.Params.ObliviousTreeOptions->MaxDepth),
                approxDimension,
                bodyTailCount
            );
        }
        ctx.SampledDocs.Create(*learnFold, GetBernoulliSampleRate(ctx.Params.ObliviousTreeOptions->BootstrapConfig)); // TODO(espetrov): create only if sample rate < 1
    }

    TErrorTracker errorTracker = BuildErrorTracker(metrics.front()->IsMaxOptimal(), /* hasTest */ true, ctx.Get());

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
        AddFileLoggers(
            /*detailedProfile=*/false,
            ctx->Params.BoostingOptions->IterationCount,
            ctx->Files.LearnErrorLogFile,
            ctx->Files.TestErrorLogFile,
            ctx->Files.TimeLeftLogFile,
            ctx->Files.JsonLogFile,
            ctx->Files.ProfileLogFile,
            ctx->OutputOptions.GetTrainDir(),
            GetJsonMeta(
                GetRawPointers(contexts),
                ELaunchMode::CV,
                learnToken,
                testToken,
                /*hasTrain=*/true,
                /*hasTest=*/true
            ),
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

        for (size_t foldIdx = 0; foldIdx < folds.size(); ++foldIdx) {
            TrainOneIteration(folds[foldIdx], contexts[foldIdx].Get());
            CalcErrors(folds[foldIdx], metrics, /*hasTrain=*/true, /*hasTest=*/true, contexts[foldIdx].Get());
        }

        TOneInterationLogger oneIterLogger(logger);
        for (size_t metricIdx = 0; metricIdx < metrics.size(); ++metricIdx) {
            const auto& metric = metrics[metricIdx];
            TVector<double> trainFoldsMetric;
            TVector<double> testFoldsMetric;
            for (size_t foldIdx = 0; foldIdx < folds.size(); ++foldIdx) {
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

            TCVIterationResults cvResults = ComputeIterationResults(trainFoldsMetric, testFoldsMetric, folds.size());

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
