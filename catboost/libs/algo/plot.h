#pragma once

#include "apply.h"
#include "approx_calcer.h"

#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/data/pool.h>
#include <catboost/libs/model/model.h>
#include <catboost/libs/loggers/logger.h>

#include <util/string/builder.h>
#include <util/generic/guid.h>
#include <util/system/fs.h>

class TMetricsPlotCalcer {
public:
    TMetricsPlotCalcer(const TFullModel& model,
                       NPar::TLocalExecutor& executor,
                       const TString& tmpDir)
        : Model(model)
        , Executor(executor)
        , First(0)
        , Last(Model.ObliviousTrees.TreeSizes.size())
        , Step(1)
        , TmpDir(tmpDir)
    {
    }

    TMetricsPlotCalcer& SetCustomStep(ui32 step) {
        Step = step;
        EnsureCorrectParams();
        return *this;
    }

    TMetricsPlotCalcer& SetFirstIteration(ui32 first) {
        First = first;
        EnsureCorrectParams();
        return *this;
    }

    TMetricsPlotCalcer& SetLastIteration(ui32 last) {
        Last = last;
        EnsureCorrectParams();
        return *this;
    }

    TMetricsPlotCalcer& AddMetric(const IMetric& metric) {
        Metrics.push_back(&metric);
        return *this;
    }

    void SetDeleteTmpDirOnExit(bool flag) {
        DeleteTmpDirOnExitFlag = flag;
    }

    TMetricsPlotCalcer& ProceedDataSet(const TPool& rawPool, bool isProcessBoundaryGroups);
    TMetricsPlotCalcer& SaveResult(const TString& resultDir, const TString& metricsFile);
    TVector<TVector<double>> GetMetricsScore();

    void ClearTempFiles() {
        for (const auto& tmpFile : NonAdditiveMetricsData.ApproxFiles) {
            if (!tmpFile.Empty()) {
                NFs::Remove(tmpFile);
            }
        }
        if (DeleteTmpDirOnExitFlag) {
            NFs::RemoveRecursive(TmpDir);
        }
    }

private:

    template <class TOutput>
    void WritePartialStats(TOutput* output, const char sep) const
    {
        for (ui32 i = 0; i < Iterations.size(); ++i) {
            (*output) << Iterations[i] << sep;

            for (ui32 metricId = 0; metricId < Metrics.size(); ++metricId) {
                WriteMetricStats(MetricPlots[metricId][i], output);
                if ((metricId + 1) != Metrics.size()) {
                    (*output) << sep;
                } else {
                    (*output) << "\n";
                }
            }
        }
    }

    template <class TOutput>
    void WriteHeaderForPartialStats(TOutput* output, const char sep) const
    {
        // TODO(annaveronika): same name as in metrics file.
        // TODO(annaveronika): create logger that outputs partial stats.
        // TODO(annaveronika): loss before first iteration should have iteration index -1.
        (*output) << "iter" << sep;
        for (ui32 metricId = 0; metricId < Metrics.size(); ++metricId) {
            WriteMetricColumns(*Metrics[metricId], output);
            if ((metricId + 1) != Metrics.size()) {
                (*output) << sep;
            } else {
                (*output) << "\n";
            }
        }
    }

    void ComputeNonAdditiveMetrics();

    void ProceedMetrics(const TVector<TVector<double>>& cursor,
                        const TPool& pool,
                        const TVector<float>& target,
                        const TVector<float>& weights,
                        const TVector<TQueryInfo>& queriesInfo,
                        ui32 plotLineIndex,
                        ui32 modelIterationIndex);

    TMetricHolder ComputeMetric(const IMetric& metric,
                               const TPool& pool,
                               const TVector<float>& target,
                               const TVector<float>& weights,
                               const TVector<TQueryInfo>& queriesInfo,
                               const TVector<TVector<double>>& approx);

    void Append(const TVector<TVector<double>>& approx, TVector<TVector<double>>* dst);

    void EnsureCorrectParams() {
        CB_ENSURE(First < Last, "First iteration should be less, than last");
        CB_ENSURE(Step <= (Last - First), "Step should be less, then plot size");
    }

    template <class TWriter>
    static TWriter& WriteMetricColumns(const IMetric& metric, TWriter* writer, char sep = '\t') {
        CB_ENSURE(writer, "Writer should not be nullptr");
        (*writer) << metric.GetDescription() << "_sum" << sep << metric.GetDescription() << "_weight";
        return *writer;
    }

    template <class TWriter>
    static TWriter& WriteMetricStats(const TMetricHolder& errorHolder, TWriter* writer, char sep = '\t') {
        CB_ENSURE(writer, "Writer should not be nullptr");
        (*writer) << errorHolder.Error << sep << errorHolder.Weight;
        return *writer;
    }

    TPool ProcessBoundaryGroups(const TPool& rawPool);

private:

    struct TNonAdditiveMetricData {
        TVector<TString> ApproxFiles;
        TVector<float> Target;
        TVector<float> Weights;
    };

    TString GetApproxFileName(ui32 plotLineIndex);

    void SaveApproxToFile(ui32 plotLineIndex, const TVector<TVector<double>>& approx);

    TVector<TVector<double>> LoadApprox(ui32 plotLineIndex);

    bool HasNonAdditiveMetric() const {
        for (const auto& metric : Metrics)
        {
            if (!metric->IsAdditiveMetric()) {
                return true;
            }
        }
        return false;
    }

private:
    const TFullModel& Model;
    NPar::TLocalExecutor& Executor;

    ui32 First;
    ui32 Last;
    ui32 Step;
    TString TmpDir;
    bool DeleteTmpDirOnExitFlag = false;

    TVector<const IMetric*> Metrics;
    TVector<TVector<TMetricHolder>> MetricPlots;
    TVector<ui32> Iterations;

    TNonAdditiveMetricData NonAdditiveMetricsData;

    TPool LastGroupPool;
};

TMetricsPlotCalcer CreateMetricCalcer(
    const TFullModel& model,
    int begin,
    int end,
    int evalPeriod,
    NPar::TLocalExecutor& executor,
    const TString& tmpDir,
    const TVector<THolder<IMetric>>& metrics
);
