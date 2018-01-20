#include "plot.h"

#include <library/threading/local_executor/local_executor.h>

void TMetricsPlotCalcer::ProceedMetrics(const TVector<TVector<double>>& cursor,
                                        const TPool& pool,
                                        const TVector<float>& target,
                                        const TVector<float>& weights,
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
            MetricPlots[metricId][plotLineIndex].Add(ComputeMetric(*Metrics[metricId], pool, target, weights, cursor));
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
                                                const TPool& pool,
                                                const TVector<float>& target,
                                                const TVector<float>& weights,
                                                const TVector<TVector<double>>& approx) {
    const auto docCount = static_cast<int>(target.size());

    if (metric.GetErrorType() == EErrorType::PerObjectError) {
        return metric.Eval(approx,
                           target,
                           weights,
                           0,
                           docCount,
                           Executor);
    } else {
        CB_ENSURE(pool.Pairs.size());
        return metric.EvalPairwise(approx,
                                   pool.Pairs,
                                   0,
                                   docCount);
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

TMetricsPlotCalcer& TMetricsPlotCalcer::ProceedDataSet(const TPool& pool) {
    EnsureCorrectParams();
    const ui32 docCount = pool.Docs.GetDocCount();

    TVector<TVector<double>> cursor(Model.ObliviousTrees.ApproxDimension, TVector<double>(docCount));
    ui32 currentIter = 0;
    ui32 idx = 0;
    TModelCalcerOnPool modelCalcerOnPool(Model, pool, Executor);

    TVector<TVector<double>> approxBuffer;
    TVector<TVector<double>> nextBatchApprox;

    for (ui32 nextBatchStart = First; nextBatchStart < Last; nextBatchStart += Step) {
        ui32 nextBatchEnd = Min<ui32>(Last, nextBatchStart + Step);
        ProceedMetrics(cursor, pool, pool.Docs.Target, pool.Docs.Weight, idx, currentIter);
        modelCalcerOnPool.ApplyModelMulti(EPredictionType::RawFormulaVal,
                                          (int)nextBatchStart,
                                          (int)nextBatchEnd,
                                          &nextBatchApprox);
        Append(nextBatchApprox, &cursor);
        currentIter = nextBatchEnd;
        ++idx;
    }
    ProceedMetrics(cursor, pool, pool.Docs.Target, pool.Docs.Weight, idx, currentIter);
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

static inline void CreateMetricsFromDescription(const TVector<TString>& description, int approxDim, TVector<THolder<IMetric>>* metrics) {
    for (const auto& metricDescription : description) {
        auto metricsBatch = CreateMetricFromDescription(metricDescription, approxDim);
        for (ui32 i = 0; i < metricsBatch.size(); ++i) {
            metrics->push_back(std::move(metricsBatch[i]));
        }
    }
}

TMetricsPlotCalcer CreateMetricCalcer(
    const TFullModel& model,
    const TVector<TString>& metricsDescription,
    int begin,
    int end,
    int evalPeriod,
    NPar::TLocalExecutor& executor,
    const TString& tmpDir,
    TVector<THolder<IMetric>>* metrics
) {
    if (end == 0) {
        end = model.GetTreeCount();
    } else {
        end = Min<int>(end, model.GetTreeCount());
    }

    CreateMetricsFromDescription(metricsDescription, model.ObliviousTrees.ApproxDimension, metrics);

    TMetricsPlotCalcer plotCalcer(model, executor, tmpDir);
    plotCalcer
        .SetFirstIteration(begin)
        .SetLastIteration(end)
        .SetCustomStep(evalPeriod);

    for (auto& metric : *metrics) {
        plotCalcer.AddMetric(*metric);
    }

    return plotCalcer;
}
