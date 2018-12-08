#pragma once

#include <catboost/libs/algo/plot.h>
#include <catboost/libs/data_types/groupid.h>
#include <catboost/libs/target/data_providers.h>

#include <util/generic/noncopyable.h>

#include <Python.h>

class TGilGuard : public TNonCopyable {
public:
    TGilGuard()
        : State_(PyGILState_Ensure())
    { }

    ~TGilGuard() {
        PyGILState_Release(State_);
    }
private:
    PyGILState_STATE State_;
};

void ProcessException();
void SetPythonInterruptHandler();
void ResetPythonInterruptHandler();

TVector<TVector<double>> EvalMetrics(
    const TFullModel& model,
    const NCB::TDataProvider& srcData,
    const TVector<TString>& metricsDescription,
    int begin,
    int end,
    int evalPeriod,
    int threadCount,
    const TString& resultDir,
    const TString& tmpDir
);

TVector<TString> GetMetricNames(const TFullModel& model, const TVector<TString>& metricsDescription);

TVector<double> EvalMetricsForUtils(
    const TVector<float>& label,
    const TVector<TVector<double>>& approx,
    const TString& metricName,
    const TVector<float>& weight,
    const TVector<TGroupId>& groupId,
    int threadCount
);

class TMetricsPlotCalcerPythonWrapper {
public:
    TMetricsPlotCalcerPythonWrapper(const TVector<TString>& metricDescriptions,
                                    const TFullModel& model,
                                    int begin,
                                    int end,
                                    int evalPeriod,
                                    int threadCount,
                                    const TString& tmpDir,
                                    bool deleteTempDirOnExit = false)
    : Rand(0)
    , Metrics(CreateMetricsFromDescription(metricDescriptions, model.ObliviousTrees.ApproxDimension))
    , MetricPlotCalcer(CreateMetricCalcer(
            model,
            begin,
            end,
            evalPeriod,
            /*processedIterationsStep=*/-1,
            Executor,
            tmpDir,
            Metrics)) {
        Executor.RunAdditionalThreads(threadCount - 1);
        MetricPlotCalcer.SetDeleteTmpDirOnExit(deleteTempDirOnExit);
    }

    ~TMetricsPlotCalcerPythonWrapper() {
        MetricPlotCalcer.ClearTempFiles();
    }

    void AddPool(const NCB::TDataProvider& srcData) {
        auto processedDataProvider = NCB::CreateModelCompatibleProcessedDataProvider(
            srcData,
            MetricPlotCalcer.GetModel(),
            &Rand,
            &Executor
        );

        if (MetricPlotCalcer.HasAdditiveMetric()) {
            MetricPlotCalcer.ProceedDataSetForAdditiveMetrics(processedDataProvider);
        }
        if (MetricPlotCalcer.HasNonAdditiveMetric()) {
            MetricPlotCalcer.ProceedDataSetForNonAdditiveMetrics(processedDataProvider);
        }

    }

    TVector<const IMetric*> GetMetricRawPtrs() const {
        TVector<const IMetric*> ptrs;
        for (const auto& metric : Metrics) {
            ptrs.push_back(metric.Get());
        }
        return ptrs;
    }

    TVector<TVector<double>> ComputeScores()  {
        if (MetricPlotCalcer.HasNonAdditiveMetric()) {
            MetricPlotCalcer.FinishProceedDataSetForNonAdditiveMetrics();
        }
        return MetricPlotCalcer.GetMetricsScore();
    }

private:
    TRestorableFastRng64 Rand;
    NPar::TLocalExecutor Executor;
    TVector<THolder<IMetric>> Metrics;
    TMetricsPlotCalcer MetricPlotCalcer;
};
