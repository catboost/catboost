#pragma once

#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/metrics/metric_holder.h>
#include <catboost/libs/model/fwd.h>

#include <util/generic/fwd.h>
#include <util/generic/ptr.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/stream/input.h>
#include <util/system/fs.h>
#include <util/system/types.h>


namespace NPar {
    class ILocalExecutor;
}


class TMetricsPlotCalcer {
public:
    TMetricsPlotCalcer(
        const TFullModel& model,
        const TVector<THolder<IMetric>>& metrics,
        const TString& tmpDir,
        ui32 first,
        ui32 last,
        ui32 step,
        ui32 processIterationStep, // = -1
        NPar::ILocalExecutor* executor
    );

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

    TMetricsPlotCalcer& ProceedDataSetForAdditiveMetrics(const NCB::TProcessedDataProvider& processedData);
    TMetricsPlotCalcer& ProceedDataSetForNonAdditiveMetrics(const NCB::TProcessedDataProvider& processedData);
    TMetricsPlotCalcer& FinishProceedDataSetForNonAdditiveMetrics();

    void ComputeNonAdditiveMetrics(const TVector<NCB::TProcessedDataProvider>& datasetParts);

    TMetricsPlotCalcer& SaveResult(
        const TString& resultDir,
        const TString& metricsFile,
        bool saveMetrics,
        bool saveStats
    );
    TVector<TVector<double>> GetMetricsScore();

    void ClearTempFiles() {
        if (DeleteTmpDirOnExitFlag) {
            NFs::RemoveRecursive(TmpDir);
        }
    }

    const TFullModel& GetModel() const {
        return Model;
    }

private:
    TMetricsPlotCalcer& ProceedDataSet(
        const NCB::TProcessedDataProvider& processedData,
        ui32 beginIterationIndex,
        ui32 endIterationIndex,
        bool isAdditive
    );

    template <class TOutput>
    void WritePartialStats(TOutput* output, const char sep) const {
        for (ui32 i = 0; i < Iterations.size(); ++i) {
            (*output) << Iterations[i] << sep;

            for (ui32 metricId = 0; metricId < AdditiveMetrics.size(); ++metricId) {
                WriteMetricStats(AdditiveMetricPlots[metricId][i], output);
                if ((metricId + 1) != AdditiveMetrics.size()) {
                    (*output) << sep;
                }
            }
            (*output) << "\n";
        }
    }

    template <class TOutput>
    void WriteHeaderForPartialStats(TOutput* output, const char sep) const {
        // TODO(annaveronika): same name as in metrics file.
        // TODO(annaveronika): create logger that outputs partial stats.
        // TODO(annaveronika): loss before first iteration should have iteration index -1.
        (*output) << "iter" << sep;
        for (ui32 metricId = 0; metricId < AdditiveMetrics.size(); ++metricId) {
            WriteMetricColumns(*AdditiveMetrics[metricId], output);
            if ((metricId + 1) != AdditiveMetrics.size()) {
                (*output) << sep;
            }
        }
        (*output) << "\n";
    }

    void ComputeNonAdditiveMetrics(ui32 begin, ui32 end);

    void ComputeAdditiveMetric(
        const TVector<TVector<double>>& approx,
        NCB::TMaybeData<TConstArrayRef<TConstArrayRef<float>>> target,
        TConstArrayRef<float> weights,
        TConstArrayRef<TQueryInfo> queriesInfo,
        ui32 plotLineIndex
    );

    void Append(
        const TVector<TVector<double>>& approx,
        int dstStartDoc /*= 0*/,
        TVector<TVector<double>>* dst
    );

    void EnsureCorrectParams() {
        CB_ENSURE(First < Last, "First iteration should be less than last");
        CB_ENSURE(Step <= (Last - First), "Step should be less than plot size");
        CB_ENSURE(Step > 0, "Step should be more than zero");
    }

    template <class TWriter>
    static TWriter& WriteMetricColumns(const IMetric& metric, TWriter* writer, char sep = '\t') {
        CB_ENSURE(writer, "Writer should not be nullptr");
        // TODO(annaveronika): Each metric should provide stats description.
        auto statDescriptions = metric.GetStatDescriptions();
        for (int i = 0; i < statDescriptions.ysize(); ++i) {
            (*writer) << metric.GetDescription() << "_" << statDescriptions[i];
            if (i + 1 != statDescriptions.ysize()) {
                (*writer) << sep;
            }
        }
        return *writer;
    }

    template <class TWriter>
    static TWriter& WriteMetricStats(const TMetricHolder& errorHolder, TWriter* writer, char sep = '\t') {
        CB_ENSURE(writer, "Writer should not be nullptr");
        for (int i = 0; i < errorHolder.Stats.ysize(); ++i) {
            (*writer) << errorHolder.Stats[i];
            if (i + 1 != errorHolder.Stats.ysize()) {
                (*writer) << sep;
            }
        }
        return *writer;
    }

    TString GetApproxFileName(ui32 plotLineIndex);

    void SaveApproxToFile(ui32 plotLineIndex, const TVector<TVector<double>>& approx);

    TVector<TVector<double>> LoadApprox(ui32 plotLineIndex);
    void DeleteApprox(ui32 plotLineIndex);

private:
    struct TNonAdditiveMetricData {
        TVector<TString> ApproxFiles;
        TVector<TVector<float>> Target;
        TVector<float> Weights;
        ui32 CumulativePoolSize = 0;
    };

private:
    const TFullModel& Model;
    NPar::ILocalExecutor& Executor;

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

    TVector<double> FlatApproxBuffer;
    TVector<TVector<double>> CurApproxBuffer;
    TVector<TVector<double>> NextApproxBuffer;
};

TMetricsPlotCalcer CreateMetricCalcer(
    const TFullModel& model,
    int begin,
    int end,
    int evalPeriod,
    int processedIterationsStep,
    const TString& tmpDir,
    const TVector<THolder<IMetric>>& metrics,
    NPar::ILocalExecutor* executor
);
