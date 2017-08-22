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
    const yvector<THolder<TLearnContext>>& contexts,
    const TCrossValidationParams& cvParams,
    yvector<TTrainData>* folds)
{
    folds->reserve(cvParams.FoldCount);
    const size_t foldSize = pool.Docs.size() / cvParams.FoldCount;

    for (size_t testFoldIdx = 0; testFoldIdx < cvParams.FoldCount; ++testFoldIdx) {
        size_t foldStartIndex = testFoldIdx * foldSize;
        size_t foldEndIndex = Min(foldStartIndex + foldSize, pool.Docs.size());

        yvector<size_t> docIndices;

        auto appendTestIndices = [foldStartIndex, foldEndIndex, &docIndices] {
            for (size_t idx = foldStartIndex; idx < foldEndIndex; ++idx) {
                docIndices.push_back(idx);
            }
        };

        auto appendTrainIndices = [foldStartIndex, foldEndIndex, &pool, &docIndices] {
            for (size_t idx = 0; idx < pool.Docs.size(); ++idx) {
                if (idx < foldStartIndex || idx >= foldEndIndex) {
                    docIndices.push_back(idx);
                }
            }
        };

        TTrainData fold;

        if (!cvParams.Inverted) {
            appendTrainIndices();
            appendTestIndices();
            fold.LearnSampleCount = pool.Docs.size() - foldEndIndex + foldStartIndex;
        } else {
            appendTestIndices();
            appendTrainIndices();
            fold.LearnSampleCount = foldEndIndex - foldStartIndex;
        }

        fold.Target.reserve(pool.Docs.size());
        fold.Weights.reserve(pool.Docs.size());
        fold.Baseline.reserve(pool.Docs.size());

        for (size_t idx = 0; idx < docIndices.size(); ++idx) {
            fold.Target.push_back(pool.Docs[docIndices[idx]].Target);
            fold.Weights.push_back(pool.Docs[docIndices[idx]].Weight);
            fold.Baseline.push_back(pool.Docs[docIndices[idx]].Baseline);
        }

        PrepareAllFeaturesFromPermutedDocs(
            pool.Docs,
            docIndices,
            contexts[testFoldIdx]->CatFeatures,
            contexts[testFoldIdx]->LearnProgress.Model.Borders,
            contexts[testFoldIdx]->Params.IgnoredFeatures,
            fold.LearnSampleCount,
            contexts[testFoldIdx]->Params.OneHotMaxSize,
            contexts[testFoldIdx]->LocalExecutor,
            &fold.AllFeatures);

        folds->push_back(fold);
    }
}

static double ComputeStdDev(const yvector<double>& values, double avg) {
    double sqrSum = 0.0;
    for (double value : values) {
        sqrSum += Sqr(value - avg);
    }
    return std::sqrt(sqrSum / (values.size() - 1));
}

static TCVIterationResults ComputeIterationResults(
    const yvector<double>& trainErrors,
    const yvector<double>& testErrors,
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
    yvector<TCVResult>* results)
{
    CB_ENSURE(!pool.Docs.empty(), "Pool is empty");
    CB_ENSURE(pool.Docs.size() > cvParams.FoldCount, "Pool is too small to be split into folds");

    const int featureCount = pool.Docs[0].Factors.ysize();

    yvector<THolder<TLearnContext>> contexts;
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

    if (ctx->Params.Verbose) {
        SetVerboseLogingMode();
    } else {
        SetSilentLogingMode();
    }

    auto loggingGuard = Finally([&] { SetSilentLogingMode(); });

    yvector<float> targets;
    targets.reserve(pool.Docs.size());

    for (const auto& doc : pool.Docs) {
        targets.push_back(doc.Target);
    }

    float minTarget = *MinElement(targets.begin(), targets.end());
    float maxTarget = *MaxElement(targets.begin(), targets.end());
    CB_ENSURE(minTarget != maxTarget, "All targets are equal");

    int approxDimension = 1;
    if (IsMultiClassError(ctx->Params.LossFunction)) {
        CB_ENSURE(AllOf(pool.Docs, [] (const TDocInfo& doc) { return floor(doc.Target) == doc.Target && doc.Target >= 0; }),
                  "Each target label should be non-negative integer for Multiclass/MultiClassOneVsAll loss function");
        approxDimension = GetClassesCount(targets, ctx->Params.ClassesCount);
        CB_ENSURE(approxDimension > 1, "All targets are equal");
    }

    TRestorableFastRng64 rand(cvParams.RandSeed);

    yvector<size_t> indices(pool.Docs.size(), 0);
    std::iota(indices.begin(), indices.end(), 0);
    if (cvParams.Shuffle) {
        Shuffle(indices.begin(), indices.end(), rand);
    }

    ApplyPermutation(InvertPermutation(indices), &pool);
    auto permutationGuard = Finally([&] { ApplyPermutation(indices, &pool); });

    auto borders = GenerateBorders(pool.Docs, ctx.Get());

    for (size_t i = 0; i < cvParams.FoldCount; ++i) {
        contexts[i]->LearnProgress.Model.Borders = borders;
    }

    yvector<TTrainData> folds;
    PrepareFolds(pool, contexts, cvParams, &folds);

    for (size_t foldIdx = 0; foldIdx < folds.size(); ++foldIdx) {
        contexts[foldIdx]->InitData(folds[foldIdx], approxDimension);
    }

    yvector<THolder<IMetric>> metrics = CreateMetrics(ctx->Params, approxDimension);
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

        if ((iteration % cvParams.EvalPeriod) != 0) {
            continue;
        }

        for (size_t metricIdx = 0; metricIdx < metrics.size(); ++metricIdx) {
            const auto& metric = metrics[metricIdx];

            yvector<double> trainErrors;
            yvector<double> testErrors;

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
                yvector<double> valuesToLog;
                errorTracker.AddError(cvResults.AverageTest, iteration, &valuesToLog);
            }
        }

        if (cvParams.EnableEarlyStopping && errorTracker.GetIsNeedStop()) {
            MATRIXNET_INFO_LOG << "Stopped by overfitting detector with threshold "
                               << errorTracker.GetOverfittingDetectorThreshold() << Endl;
            break;
        }
    }
}
