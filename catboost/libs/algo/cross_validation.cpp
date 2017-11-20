#include "cross_validation.h"
#include "train_model.h"
#include "train.h"
#include "helpers.h"

#include <catboost/libs/overfitting_detector/error_tracker.h>

#include <util/random/shuffle.h>

#include <limits>
#include <cmath>

static void PrepareFolds(
    const TPool& pool,
    const TVector<THolder<TLearnContext>>& contexts,
    const TCrossValidationParams& cvParams,
    TVector<TTrainData>* folds)
{
    folds->reserve(cvParams.FoldCount);
    const size_t foldSize = pool.Docs.GetDocCount() / cvParams.FoldCount;

    for (size_t testFoldIdx = 0; testFoldIdx < cvParams.FoldCount; ++testFoldIdx) {
        size_t foldStartIndex = testFoldIdx * foldSize;
        size_t foldEndIndex = Min(foldStartIndex + foldSize, pool.Docs.GetDocCount());

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
        for (int dim = 0; dim < pool.Docs.GetBaselineDimension(); ++dim) {
            fold.Baseline[dim].reserve(pool.Docs.GetDocCount());
            for (size_t idx = 0; idx < docIndices.size(); ++idx) {
                fold.Baseline[dim].push_back(pool.Docs.Baseline[dim][docIndices[idx]]);
            }
        }

        PrepareAllFeaturesFromPermutedDocs(
            docIndices,
            contexts[testFoldIdx]->CatFeatures,
            contexts[testFoldIdx]->LearnProgress.FloatFeatures,
            contexts[testFoldIdx]->Params.IgnoredFeatures,
            fold.LearnSampleCount,
            contexts[testFoldIdx]->Params.OneHotMaxSize,
            contexts[testFoldIdx]->Params.NanMode,
            /*allowClearPool*/ false,
            contexts[testFoldIdx]->LocalExecutor,
            &pool.Docs,
            &fold.AllFeatures);

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
    size_t foldCount)
{
    TCVIterationResults cvResults;
    cvResults.AverageTrain = Accumulate(trainErrors.begin(), trainErrors.end(), 0.0) / foldCount;
    cvResults.StdDevTrain = ComputeStdDev(trainErrors, cvResults.AverageTrain);
    cvResults.AverageTest = Accumulate(testErrors.begin(), testErrors.end(), 0.0) / foldCount;
    cvResults.StdDevTest = ComputeStdDev(testErrors, cvResults.AverageTest);
    return cvResults;
}

void CrossValidate(
    const NJson::TJsonValue& jsonParams,
    const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
    const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
    TPool& pool,
    const TCrossValidationParams& cvParams,
    TVector<TCVResult>* results)
{
    CB_ENSURE(pool.Docs.GetDocCount() != 0, "Pool is empty");
    CB_ENSURE(pool.Docs.GetDocCount() > cvParams.FoldCount, "Pool is too small to be split into folds");

    const int featureCount = pool.Docs.GetFactorsCount();

    TVector<THolder<TLearnContext>> contexts;
    contexts.reserve(cvParams.FoldCount);

    for (size_t idx = 0; idx < cvParams.FoldCount; ++idx) {
        contexts.emplace_back(
            new TLearnContext(
                jsonParams,
                objectiveDescriptor,
                evalMetricDescriptor,
                featureCount,
                pool.CatFeatures,
                pool.FeatureId,
                "fold_" + ToString(idx) + "_"));
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

    if (IsMultiClassError(ctx->Params.LossFunction)) {
        CB_ENSURE(AllOf(pool.Docs.Target, [] (float target) { return floor(target) == target && target >= 0; }),
                  "Each target label should be non-negative integer for Multiclass/MultiClassOneVsAll loss function");
        for (const auto& context : contexts) {
            context->LearnProgress.ApproxDimension = GetClassesCount(pool.Docs.Target, context->Params.ClassesCount);
        }
        CB_ENSURE(ctx->LearnProgress.ApproxDimension > 1, "All targets are equal");
    }

    TRestorableFastRng64 rand(cvParams.PartitionRandSeed);

    TVector<size_t> indices(pool.Docs.GetDocCount(), 0);
    std::iota(indices.begin(), indices.end(), 0);
    if (cvParams.Shuffle) {
        Shuffle(indices.begin(), indices.end(), rand);
    }

    ApplyPermutation(InvertPermutation(indices), &pool);
    auto permutationGuard = Finally([&] { ApplyPermutation(indices, &pool); });
    TVector<TFloatFeature> floatFeatures;
    GenerateBorders(pool, ctx.Get(), &floatFeatures);

    for (size_t i = 0; i < cvParams.FoldCount; ++i) {
        contexts[i]->LearnProgress.FloatFeatures = floatFeatures;
    }

    TVector<TTrainData> folds;
    PrepareFolds(pool, contexts, cvParams, &folds);

    for (size_t foldIdx = 0; foldIdx < folds.size(); ++foldIdx) {
        contexts[foldIdx]->InitData(folds[foldIdx]);
    }

    TVector<THolder<IMetric>> metrics = CreateMetrics(ctx->Params.EvalMetric, ctx->Params.EvalMetricDescriptor,
                                                      ctx->Params.CustomLoss, ctx->LearnProgress.ApproxDimension);
    TErrorTracker errorTracker = BuildErrorTracker(metrics.front()->IsMaxOptimal(), /* hasTest */ true, ctx.Get());

    auto calcFoldError = [&] (size_t foldIndex, int begin, int end, const THolder<IMetric>& metric) {
        return metric->GetFinalError(metric->Eval(
            contexts[foldIndex]->LearnProgress.AvrgApprox,
            folds[foldIndex].Target,
            folds[foldIndex].Weights,
            begin,
            end,
            contexts[foldIndex]->LocalExecutor));
    };

    results->reserve(metrics.size());
    for (const auto& metric : metrics) {
        TCVResult result;
        result.Metric = metric->GetDescription();
        results->push_back(result);
    }

    for (int iteration = 0; iteration < ctx->Params.Iterations; ++iteration) {
        for (size_t foldIdx = 0; foldIdx < folds.size(); ++foldIdx) {
            TrainOneIteration(folds[foldIdx], contexts[foldIdx].Get());
        }

        for (size_t metricIdx = 0; metricIdx < metrics.size(); ++metricIdx) {
            const auto& metric = metrics[metricIdx];

            TVector<double> trainErrors;
            TVector<double> testErrors;

            for (size_t foldIdx = 0; foldIdx < folds.size(); ++foldIdx) {
                trainErrors.push_back(calcFoldError(foldIdx, 0, folds[foldIdx].LearnSampleCount, metric));
                testErrors.push_back(calcFoldError(foldIdx, folds[foldIdx].LearnSampleCount, folds[foldIdx].GetSampleCount(), metric));
            }

            TCVIterationResults cvResults = ComputeIterationResults(trainErrors, testErrors, folds.size());

            MATRIXNET_INFO_LOG << "Iteration: " << iteration;
            MATRIXNET_INFO_LOG << "\ttrain avg\t" << cvResults.AverageTrain << "\ttrain stddev\t" << cvResults.StdDevTrain;
            MATRIXNET_INFO_LOG << "\ttest avg\t" << cvResults.AverageTest << "\ttest stddev\t" << cvResults.StdDevTest;

            (*results)[metricIdx].AppendOneIterationResults(cvResults);

            if (metricIdx == 0) {
                TVector<double> valuesToLog;
                errorTracker.AddError(cvResults.AverageTest, iteration, &valuesToLog);
            }
        }

        if (errorTracker.GetIsNeedStop()) {
            MATRIXNET_INFO_LOG << "Stopped by overfitting detector ("
                << "iteration: " << iteration << ", "
                << "iterations wait: " << errorTracker.GetOverfittingDetectorIterationsWait()
                << ")" << Endl;
            break;
        }
    }
}
