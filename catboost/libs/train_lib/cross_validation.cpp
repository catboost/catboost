#include "cross_validation.h"
#include "train_model.h"

#include <catboost/libs/algo/train.h>
#include <catboost/libs/algo/helpers.h>
#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/loggers/logger.h>
#include <catboost/libs/helpers/permutation.h>
#include <catboost/libs/helpers/binarize_target.h>
#include <catboost/libs/overfitting_detector/error_tracker.h>
#include <catboost/libs/options/plain_options_helper.h>

#include <util/random/shuffle.h>

#include <limits>
#include <cmath>

static void PrepareFolds(
    const NCatboostOptions::TLossDescription& lossDescription,
    const TPool& pool,
    const TVector<THolder<TLearnContext>>& contexts,
    const TCrossValidationParams& cvParams,
    TVector<TTrainData>* folds
) {
    folds->reserve(cvParams.FoldCount);
    const size_t docCount = pool.Docs.GetDocCount();

    bool hasQuery = !pool.Docs.QueryId.empty();
    TVector<TQueryEndInfo> queryEndInfo;
    if (hasQuery) {
        TVector<TQueryInfo> queryInfo;
        UpdateQueriesInfo(pool.Docs.QueryId, /*begin=*/0, docCount, &queryInfo);
        queryEndInfo = GetQueryEndInfo(queryInfo, docCount);
    }

    size_t currentFoldEnd = 0;
    const size_t foldSize = docCount / cvParams.FoldCount;
    for (size_t testFoldIdx = 0; testFoldIdx < cvParams.FoldCount; ++testFoldIdx) {
        size_t foldStartIndex = currentFoldEnd;
        size_t foldEndIndex = Min(foldStartIndex + foldSize, pool.Docs.GetDocCount());
        if (hasQuery) {
            foldEndIndex = queryEndInfo[foldEndIndex - 1].QueryEnd;
        }
        currentFoldEnd = foldEndIndex;
        CB_ENSURE(foldEndIndex - foldStartIndex > 0, "Not enough documents for cross validataion");

        TVector<size_t> docIndices;

        auto appendTestIndices = [foldStartIndex, foldEndIndex, &docIndices] {
            for (size_t idx = foldStartIndex; idx < foldEndIndex; ++idx) {
                docIndices.push_back(idx);
            }
        };

        auto appendTrainIndices = [foldStartIndex, foldEndIndex, &pool, &docIndices] {
            for (size_t idx = 0; idx < pool.Docs.GetDocCount(); ++idx) {
                if (idx < foldStartIndex || idx >= foldEndIndex) {
                    docIndices.push_back(idx);
                }
            }
        };

        TTrainData fold;

        if (!cvParams.Inverted) {
            appendTrainIndices();
            appendTestIndices();
            fold.LearnSampleCount = pool.Docs.GetDocCount() - foldEndIndex + foldStartIndex;
        } else {
            appendTestIndices();
            appendTrainIndices();
            fold.LearnSampleCount = foldEndIndex - foldStartIndex;
        }

        fold.Target.reserve(pool.Docs.GetDocCount());
        fold.Weights.reserve(pool.Docs.GetDocCount());

        for (size_t idx = 0; idx < docIndices.size(); ++idx) {
            fold.Target.push_back(pool.Docs.Target[docIndices[idx]]);
            fold.Weights.push_back(pool.Docs.Weight[docIndices[idx]]);
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
            for (size_t idx = 0; idx < docIndices.size(); ++idx) {
                fold.Baseline[dim].push_back(pool.Docs.Baseline[dim][docIndices[idx]]);
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
            contexts[testFoldIdx]->CatFeatures,
            contexts[testFoldIdx]->LearnProgress.FloatFeatures,
            contexts[testFoldIdx]->Params.DataProcessingOptions->IgnoredFeatures,
            fold.LearnSampleCount,
            (size_t)contexts[testFoldIdx]->Params.CatFeatureParams->OneHotMaxSize,
            contexts[testFoldIdx]->Params.DataProcessingOptions->FloatFeaturesBinarization->NanMode,
            /*allowClearPool*/ false,
            contexts[testFoldIdx]->LocalExecutor,
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

            oneIterLogger.OutputMetric(learnToken, TMetricEvalResult(metric->GetDescription(), cvResults.AverageTrain, metricIdx == 0));
            oneIterLogger.OutputMetric(testToken, TMetricEvalResult(metric->GetDescription(), cvResults.AverageTest, metricIdx == 0));

            (*results)[metricIdx].AppendOneIterationResults(cvResults);

            if (metricIdx == 0) {
                TVector<double> valuesToLog;
                errorTracker.AddError(cvResults.AverageTest, iteration, &valuesToLog);
            }
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
