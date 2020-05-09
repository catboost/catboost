#pragma once

#include <Python.h>

#include <catboost/private/libs/algo/plot.h>
#include <catboost/private/libs/data_types/groupid.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/mem_usage.h>
#include <catboost/libs/metrics/metric.h>
#include <catboost/private/libs/options/loss_description.h>
#include <catboost/private/libs/options/plain_options_helper.h>
#include <catboost/private/libs/target/data_providers.h>
#include <catboost/libs/train_lib/options_helper.h>

#include <library/cpp/json/json_value.h>

#include <util/generic/noncopyable.h>


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
void ThrowCppExceptionWithMessage(const TString& message);

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
    TConstArrayRef<TVector<float>> label,
    const TVector<TVector<double>>& approx,
    const TString& metricName,
    const TVector<float>& weight,
    const TVector<TGroupId>& groupId,
    const TVector<TSubgroupId>& subgroupId,
    const TVector<TPair>& pairs,
    int threadCount
);

inline TVector<NCatboostOptions::TLossDescription> CreateMetricLossDescriptions(
    const TVector<TString>& metricDescriptions) {

    CB_ENSURE(!metricDescriptions.empty(), "No metrics in metric descriptions");

    TVector<NCatboostOptions::TLossDescription> result;
    for (const auto& metricDescription : metricDescriptions) {
        result.emplace_back(NCatboostOptions::ParseLossDescription(metricDescription));
    }

    return result;
}


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
    , MetricLossDescriptions(CreateMetricLossDescriptions(metricDescriptions))
    , Metrics(CreateMetrics(MetricLossDescriptions, model.GetDimensionsCount()))
    , MetricPlotCalcer(CreateMetricCalcer(
            model,
            begin,
            end,
            evalPeriod,
            /*processedIterationsStep=*/-1,
            tmpDir,
            Metrics,
            &Executor)) {
        Executor.RunAdditionalThreads(threadCount - 1);
        MetricPlotCalcer.SetDeleteTmpDirOnExit(deleteTempDirOnExit);
    }

    ~TMetricsPlotCalcerPythonWrapper() {
        MetricPlotCalcer.ClearTempFiles();
    }

    void AddPool(const NCB::TDataProvider& srcData) {
        auto processedDataProvider = NCB::CreateModelCompatibleProcessedDataProvider(
            srcData,
            MetricLossDescriptions,
            MetricPlotCalcer.GetModel(),
            NCB::GetMonopolisticFreeCpuRam(),
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
    TVector<NCatboostOptions::TLossDescription> MetricLossDescriptions;
    TVector<THolder<IMetric>> Metrics;
    TMetricsPlotCalcer MetricPlotCalcer;
};

NJson::TJsonValue GetTrainingOptions(
    const NJson::TJsonValue& plainJsonParams,
    const NCB::TDataMetaInfo& trainDataMetaInfo,
    const TMaybe<NCB::TDataMetaInfo>& testDataMetaInfo
);

NJson::TJsonValue GetPlainJsonWithAllOptions(
    const TFullModel& model,
    bool hasCatFeatures,
    bool hasTextFeatures
);
