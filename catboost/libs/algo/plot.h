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
    TMetricsPlotCalcer(
        const TFullModel& model,
        const TVector<THolder<IMetric>>& metrics,
        NPar::TLocalExecutor& executor,
        const TString& tmpDir,
        ui32 first,
        ui32 last,
        ui32 step,
        ui32 processIterationStep = -1
    )
        : Model(model)
        , Executor(executor)
        , First(first)
        , Last(last)
        , Step(step)
        , TmpDir(tmpDir)
        , ProcessedIterationsCount(0)
        , ProcessedIterationsStep(processIterationStep)
    {
        EnsureCorrectParams();
        for (ui32 iteration = First; iteration < Last; iteration += Step) {
            Iterations.push_back(iteration);
        }
        if (Iterations.back() != Last - 1) {
            Iterations.push_back(Last - 1);
        }
        for (int metricIndex = 0; metricIndex < metrics.ysize(); ++metricIndex) {
            const auto& metric = metrics[metricIndex];
            if (metric->IsAdditiveMetric()) {
                AdditiveMetrics.push_back(metric.Get());
                AdditiveMetricsIndices.push_back(metricIndex);
            } else {
                NonAdditiveMetrics.push_back(metric.Get());
                NonAdditiveMetricsIndices.push_back(metricIndex);
                CB_ENSURE(metric->GetErrorType() == EErrorType::PerObjectError,
                    "Error: we don't support non-additive querywise and pairwise metrics currenty");
            }
        }
        AdditiveMetricPlots.resize(AdditiveMetrics.ysize(), TVector<TMetricHolder>(Iterations.ysize()));
        NonAdditiveMetricPlots.resize(NonAdditiveMetrics.ysize(), TVector<TMetricHolder>(Iterations.ysize()));
    }

    void SetDeleteTmpDirOnExit(bool flag) {
        DeleteTmpDirOnExitFlag = flag;
    }

    bool HasAdditiveMetric() const {
        return !AdditiveMetrics.empty();
    }

    bool HasNonAdditiveMetric() const {
        return !NonAdditiveMetrics.empty();
    }

    bool AreAllIterationsProcessed() const {
        return ProcessedIterationsCount == Iterations.size();
    }

    TMetricsPlotCalcer& ProceedDataSetForAdditiveMetrics(const TPool& pool, bool isProcessBoundaryGroups);
    TMetricsPlotCalcer& FinishProceedDataSetForAdditiveMetrics();
    TMetricsPlotCalcer& ProceedDataSetForNonAdditiveMetrics(const TPool& pool);
    TMetricsPlotCalcer& FinishProceedDataSetForNonAdditiveMetrics();

    TMetricsPlotCalcer& SaveResult(const TString& resultDir, const TString& metricsFile, bool saveOnlyLogFiles);
    TVector<TVector<double>> GetMetricsScore();

    void ClearTempFiles() {
        if (DeleteTmpDirOnExitFlag) {
            NFs::RemoveRecursive(TmpDir);
        }
    }

private:

    TMetricsPlotCalcer& ProceedDataSet(
        const TPool& rawPool,
        ui32 beginIterationIndex,
        ui32 endIterationIndex,
        bool isProcessBoundaryGroups,
        bool isAdditive
    );

    template <class TOutput>
    void WritePartialStats(TOutput* output, const char sep) const
    {
        for (ui32 i = 0; i < Iterations.size(); ++i) {
            (*output) << Iterations[i] << sep;

            for (ui32 metricId = 0; metricId < AdditiveMetrics.size(); ++metricId) {
                WriteMetricStats(AdditiveMetricPlots[metricId][i], output);
                if ((metricId + 1) != AdditiveMetrics.size()) {
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
        for (ui32 metricId = 0; metricId < AdditiveMetrics.size(); ++metricId) {
            WriteMetricColumns(*AdditiveMetrics[metricId], output);
            if ((metricId + 1) != AdditiveMetrics.size()) {
                (*output) << sep;
            } else {
                (*output) << "\n";
            }
        }
    }

    void ComputeNonAdditiveMetrics(ui32 begin, ui32 end);

    void ComputeAdditiveMetric(
        const TVector<TVector<double>>& approx,
        const TVector<float>& target,
        const TVector<float>& weights,
        const TVector<TQueryInfo>& queriesInfo,
        ui32 plotLineIndex
    );

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
    void DeleteApprox(ui32 plotLineIndex);

private:
    const TFullModel& Model;
    NPar::TLocalExecutor& Executor;

    ui32 First;
    ui32 Last;
    ui32 Step;
    TString TmpDir;
    bool DeleteTmpDirOnExitFlag = false;

    TVector<const IMetric*> AdditiveMetrics;
    TVector<const IMetric*> NonAdditiveMetrics;
    TVector<TVector<TMetricHolder>> AdditiveMetricPlots;
    TVector<TVector<TMetricHolder>> NonAdditiveMetricPlots;
    TVector<ui32> AdditiveMetricsIndices;
    TVector<ui32> NonAdditiveMetricsIndices;
    TVector<ui32> Iterations;

    ui32 ProcessedIterationsCount;
    ui32 ProcessedIterationsStep;
    THolder<IInputStream> LastApproxes;

    TNonAdditiveMetricData NonAdditiveMetricsData;

    TPool LastGroupPool;
};

TMetricsPlotCalcer CreateMetricCalcer(
    const TFullModel& model,
    int begin,
    int end,
    int evalPeriod,
    int processedIterationsStep,
    NPar::TLocalExecutor& executor,
    const TString& tmpDir,
    const TVector<THolder<IMetric>>& metrics
);
