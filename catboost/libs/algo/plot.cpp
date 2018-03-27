#include "plot.h"

#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/helpers/query_info_helper.h>
#include <catboost/libs/options/loss_description.h>

#include <library/threading/local_executor/local_executor.h>

void TMetricsPlotCalcer::ProceedMetrics(const TVector<TVector<double>>& cursor,
                                        const TVector<float>& target,
                                        const TVector<float>& weights,
                                        const TVector<TQueryInfo>& queriesInfo,
                                        ui32 plotLineIndex,
                                        ui32 modelIterationIndex) {
    const ui32 plotSize = plotLineIndex + 1;
    MetricPlots.resize(Metrics.size());
    if (Iterations.size() < plotSize) {
        Iterations.push_back(modelIterationIndex);
        CB_ENSURE(Iterations.size() == plotSize);
    }

    for (ui32 metricId = 0; metricId < Metrics.size(); ++metricId) {
        if (MetricPlots[metricId].size() < plotSize) {
            MetricPlots[metricId].resize(plotSize);
        }
        if (Metrics[metricId]->IsAdditiveMetric()) {
            MetricPlots[metricId][plotLineIndex].Add(ComputeMetric(*Metrics[metricId], target, weights, queriesInfo, cursor));
        } else {
            CB_ENSURE(Metrics[metricId]->GetErrorType() == EErrorType::PerObjectError, "Error: we don't support non-additive pairwise metrics currenty");
        }
    }

    if (HasNonAdditiveMetric()) {
        const ui32 newPoolSize = NonAdditiveMetricsData.Target.size() + target.size();

        if (plotLineIndex == 0) {
            NonAdditiveMetricsData.Target.reserve(newPoolSize);
            NonAdditiveMetricsData.Weights.reserve(newPoolSize);

            NonAdditiveMetricsData.Target.insert(NonAdditiveMetricsData.Target.end(), target.begin(), target.end());
            NonAdditiveMetricsData.Weights.insert(NonAdditiveMetricsData.Weights.end(), weights.begin(), weights.end());
        }
        SaveApproxToFile(plotLineIndex, cursor);
    }
}

TMetricHolder TMetricsPlotCalcer::ComputeMetric(const IMetric& metric,
                                                const TVector<float>& target,
                                                const TVector<float>& weights,
                                                const TVector<TQueryInfo>& queriesInfo,
                                                const TVector<TVector<double>>& approx) {
    ELossFunction lossFunction = ParseLossType(metric.GetDescription());
    CheckTarget(target, lossFunction);

    const auto docCount = static_cast<int>(target.size());
    const auto queryCount = static_cast<int>(queriesInfo.size());
    if (metric.GetErrorType() == EErrorType::PerObjectError) {
        return metric.Eval(approx, target, weights, queriesInfo, 0, docCount, Executor);
    } else {
        CB_ENSURE(metric.GetErrorType() == EErrorType::QuerywiseError || metric.GetErrorType() == EErrorType::PairwiseError);
        return metric.Eval(approx, target, weights, queriesInfo, 0, queryCount, Executor);
    }
}

void TMetricsPlotCalcer::Append(const TVector<TVector<double>>& approx,
                                TVector<TVector<double>>* dst) {
    const ui32 docCount = approx[0].size();

    for (ui32 dim = 0; dim < approx.size(); ++dim) {
        NPar::ParallelFor(Executor, 0, docCount, [&](int i) {
            (*dst)[dim][i] += approx[dim][i];
        });
    };
}

static void ResizePool(int size, const TPool& basePool, TPool* pool) {
    pool->Docs.Resize(
        size,
        basePool.Docs.GetFactorsCount(),
        basePool.Docs.GetBaselineDimension(),
        !basePool.Docs.QueryId.empty(),
        !basePool.Docs.SubgroupId.empty()
    );
}

TPool TMetricsPlotCalcer::ProcessBoundaryGroups(const TPool& rawPool) {
    TPool resultPool;
    resultPool.Docs.Swap(LastGroupPool.Docs);

    const int offset = resultPool.Docs.GetDocCount();
    const int rawPoolSize = rawPool.Docs.GetDocCount();
    ResizePool(offset + rawPoolSize, rawPool, &resultPool);
    for (int docId = 0; docId < rawPoolSize; ++docId) {
        resultPool.Docs.AssignDoc(offset + docId, rawPool.Docs, docId);
    }

    int lastQuerySize = 0;
    const TGroupId lastQueryId = rawPool.Docs.QueryId.back();
    for (auto queryIdIt = rawPool.Docs.QueryId.rbegin(); queryIdIt != rawPool.Docs.QueryId.rend(); ++queryIdIt) {
        if (lastQueryId == *queryIdIt) {
            ++lastQuerySize;
        } else {
            break;
        }
    }
    ResizePool(lastQuerySize, resultPool, &LastGroupPool);
    const int newResultPoolSize = offset + rawPoolSize - lastQuerySize;
    for (int docId = 0; docId < lastQuerySize; ++docId) {
        LastGroupPool.Docs.AssignDoc(docId, resultPool.Docs, newResultPoolSize + docId);
    }

    ResizePool(newResultPoolSize, resultPool, &resultPool);
    CB_ENSURE(resultPool.Docs.GetDocCount() != 0, "The size of the queries should be less than block-size parameter.");
    return resultPool;
}

TMetricsPlotCalcer& TMetricsPlotCalcer::ProceedDataSet(const TPool& rawPool, bool isProcessBoundaryGroups) {
    const TPool* poolPointer;
    THolder<TPool> tmpPoolHolder;
    if (isProcessBoundaryGroups) {
        tmpPoolHolder = MakeHolder<TPool>(ProcessBoundaryGroups(rawPool));
        poolPointer = tmpPoolHolder.Get();
    } else {
        poolPointer = &rawPool;
    }
    const TPool& pool = *poolPointer;

    EnsureCorrectParams();
    const ui32 docCount = pool.Docs.GetDocCount();

    TVector<TVector<double>> cursor(Model.ObliviousTrees.ApproxDimension, TVector<double>(docCount));
    ui32 currentIter = 0;
    ui32 idx = 0;
    TModelCalcerOnPool modelCalcerOnPool(Model, pool, Executor);

    TVector<TVector<double>> approxBuffer;
    TVector<TVector<double>> nextBatchApprox;

    TVector<TQueryInfo> queriesInfo;
    UpdateQueriesInfo(pool.Docs.QueryId, pool.Docs.SubgroupId, 0, pool.Docs.GetDocCount(), &queriesInfo);
    UpdateQueriesPairs(pool.Pairs, 0, pool.Pairs.ysize(), /*invertedPermutation=*/{}, &queriesInfo);

    for (ui32 nextBatchStart = First; nextBatchStart < Last; nextBatchStart += Step) {
        ui32 nextBatchEnd = Min<ui32>(Last, nextBatchStart + Step);
        ProceedMetrics(cursor, pool.Docs.Target, pool.Docs.Weight, queriesInfo, idx, currentIter);
        modelCalcerOnPool.ApplyModelMulti(EPredictionType::RawFormulaVal,
                                          (int)nextBatchStart,
                                          (int)nextBatchEnd,
                                          &nextBatchApprox);
        Append(nextBatchApprox, &cursor);
        currentIter = nextBatchEnd;
        ++idx;
    }
    ProceedMetrics(cursor, pool.Docs.Target, pool.Docs.Weight, queriesInfo, idx, currentIter);
    return *this;
}

void TMetricsPlotCalcer::ComputeNonAdditiveMetrics() {
    const auto& target = NonAdditiveMetricsData.Target;
    const auto& weights = NonAdditiveMetricsData.Weights;

    for (ui32 idx = 0; idx < Iterations.size(); ++idx) {
        auto approx = LoadApprox(idx);
        for (ui32 metricId = 0; metricId < Metrics.size(); ++metricId) {
            if (!Metrics[metricId]->IsAdditiveMetric()) {
                MetricPlots[metricId][idx] = Metrics[metricId]->Eval(approx,
                                                                     target,
                                                                     weights,
                                                                     {},
                                                                     0,
                                                                     target.size(),
                                                                     Executor);
            }
        }
    }
}

TString TMetricsPlotCalcer::GetApproxFileName(ui32 plotLineIndex) {
    const ui32 plotSize = plotLineIndex + 1;
    if (NonAdditiveMetricsData.ApproxFiles.size() < plotSize) {
        NonAdditiveMetricsData.ApproxFiles.resize(plotSize);
    }
    if (NonAdditiveMetricsData.ApproxFiles[plotLineIndex].Empty()) {
        if (!NFs::Exists(TmpDir)) {
            NFs::MakeDirectory(TmpDir);
            DeleteTmpDirOnExitFlag = true;
        }
        TString name = TStringBuilder() << CreateGuidAsString() << "_approx_" << plotLineIndex << ".tmp";
        auto path = JoinFsPaths(TmpDir, name);
        if (NFs::Exists(path)) {
            MATRIXNET_INFO_LOG << "Path already exists " << path << ". Will overwrite file" << Endl;
            NFs::Remove(path);
        }
        NonAdditiveMetricsData.ApproxFiles[plotLineIndex] = path;
    }
    return NonAdditiveMetricsData.ApproxFiles[plotLineIndex];
}

void TMetricsPlotCalcer::SaveApproxToFile(ui32 plotLineIndex,
                                          const TVector<TVector<double>>& approx) {
    auto fileName = GetApproxFileName(plotLineIndex);
    ui32 docCount = approx[0].size();
    TFile file(fileName, EOpenModeFlag::ForAppend | EOpenModeFlag::OpenAlways);
    TOFStream out(file);
    TVector<double> line(approx.size());

    for (ui32 i = 0; i < docCount; ++i) {
        for (ui32 dim = 0; dim < approx.size(); ++dim) {
            line[dim] = approx[dim][i];
        }
        ::Save(&out, line);
    }
}

TVector<TVector<double>> TMetricsPlotCalcer::LoadApprox(ui32 plotLineIndex) {
    TIFStream input(GetApproxFileName(plotLineIndex));
    ui32 docCount = NonAdditiveMetricsData.Target.size();
    TVector<TVector<double>> result(Model.ObliviousTrees.ApproxDimension, TVector<double>(docCount));
    TVector<double> line;
    for (ui32 i = 0; i < docCount; ++i) {
        ::Load(&input, line);
        for (ui32 dim = 0; dim < result.size(); ++dim) {
            result[dim][i] = line[dim];
        }
    }
    return result;
}

TMetricsPlotCalcer CreateMetricCalcer(
    const TFullModel& model,
    int begin,
    int end,
    int evalPeriod,
    NPar::TLocalExecutor& executor,
    const TString& tmpDir,
    const TVector<THolder<IMetric>>& metrics
) {
    if (end == 0) {
        end = model.GetTreeCount();
    } else {
        end = Min<int>(end, model.GetTreeCount());
    }

    TMetricsPlotCalcer plotCalcer(model, executor, tmpDir);
    plotCalcer
        .SetFirstIteration(begin)
        .SetLastIteration(end)
        .SetCustomStep(evalPeriod);

    for (const auto& metric : metrics) {
        plotCalcer.AddMetric(*metric);
    }

    return plotCalcer;
}

TVector<TVector<double>> TMetricsPlotCalcer::GetMetricsScore() {
    if (LastGroupPool.Docs.GetDocCount() != 0) {
        ProceedDataSet(LastGroupPool, /*isProcessBoundaryGroups=*/false);
    }
    if (HasNonAdditiveMetric()) {
        ComputeNonAdditiveMetrics();
    }
    TVector<TVector<double>> metricsScore(Metrics.size(), TVector<double>(Iterations.size()));
    for (ui32 i = 0; i < Iterations.size(); ++i) {
        for (ui32 metricId = 0; metricId < Metrics.size(); ++metricId) {
            metricsScore[metricId][i] = Metrics[metricId]->GetFinalError(MetricPlots[metricId][i]);
        }
    }
    return metricsScore;
}

TMetricsPlotCalcer& TMetricsPlotCalcer::SaveResult(const TString& resultDir, const TString& metricsFile, bool saveOnlyLogFiles) {
    TFsPath trainDirPath(resultDir);
    if (!resultDir.empty() && !trainDirPath.Exists()) {
        trainDirPath.MkDir();
    }

    if (!saveOnlyLogFiles) {
        TOFStream statsStream(JoinFsPaths(resultDir, "partial_stats.tsv"));
        const char sep = '\t';
        WriteHeaderForPartialStats(&statsStream, sep);
        WritePartialStats(&statsStream, sep);
    }

    TString token = "eval_dataset";

    TLogger logger;
    if (!saveOnlyLogFiles) {
        logger.AddBackend(token, TIntrusivePtr<ILoggingBackend>(new TErrorFileLoggingBackend(JoinFsPaths(resultDir, metricsFile))));
    }
    logger.AddBackend(token, TIntrusivePtr<ILoggingBackend>(new TTensorBoardLoggingBackend(JoinFsPaths(resultDir, token))));

    auto metaJson = GetJsonMeta(Iterations.back() + 1, ""/*optionalExperimentName*/, Metrics, {}/*learnSetNames*/, {token}, ELaunchMode::Eval);
    logger.AddBackend(token, TIntrusivePtr<ILoggingBackend>(new TJsonLoggingBackend(JoinFsPaths(resultDir, "catboost_training.json"), metaJson)));

    TVector<TVector<double>> results = GetMetricsScore();
    for (int iteration = 0; iteration < results[0].ysize(); ++iteration) {
        TOneInterationLogger oneIterLogger(logger);
        for (int metricIdx = 0; metricIdx < results.ysize(); ++metricIdx) {
            oneIterLogger.OutputMetric(token, TMetricEvalResult(Metrics[metricIdx]->GetDescription(), results[metricIdx][iteration], false));
        }
    }
    return *this;
}
