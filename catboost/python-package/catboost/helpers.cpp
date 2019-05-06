#include <Python.h>

#include "helpers.h"

#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/interrupt.h>
#include <catboost/libs/helpers/query_info_helper.h>

extern "C" PyObject* PyCatboostExceptionType;

void ProcessException() {
    try {
        throw;
    } catch (const TCatBoostException& exc) {
        PyErr_SetString(PyCatboostExceptionType, exc.what());
    } catch (const TInterruptException& exc) {
        PyErr_SetString(PyExc_KeyboardInterrupt, exc.what());
    } catch (const std::exception& exc) {
        PyErr_SetString(PyCatboostExceptionType, exc.what());
    }
}

void PyCheckInterrupted() {
    TGilGuard guard;
    if (PyErr_CheckSignals() == -1) {
        throw TInterruptException();
    }
}

void SetPythonInterruptHandler() {
    SetInterruptHandler(PyCheckInterrupted);
}

void ResetPythonInterruptHandler() {
    ResetInterruptHandler();
}

void ThrowCppExceptionWithMessage(const TString& message) {
    ythrow TCatBoostException() << message;
}

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
) {
    NPar::TLocalExecutor executor;
    executor.RunAdditionalThreads(threadCount - 1);

    TRestorableFastRng64 rand(0);

    auto metricLossDescriptions = CreateMetricLossDescriptions(metricsDescription);
    auto metrics = CreateMetrics(metricLossDescriptions, model.ObliviousTrees.ApproxDimension);
    TMetricsPlotCalcer plotCalcer = CreateMetricCalcer(
        model,
        begin,
        end,
        evalPeriod,
        /*processedIterationsStep=*/50,
        tmpDir,
        metrics,
        &executor
    );

    auto processedDataProvider = NCB::CreateModelCompatibleProcessedDataProvider(
        srcData,
        metricLossDescriptions,
        model,
        &rand,
        &executor
    );

    if (plotCalcer.HasAdditiveMetric()) {
        plotCalcer.ProceedDataSetForAdditiveMetrics(processedDataProvider);
    }
    if (plotCalcer.HasNonAdditiveMetric()) {
        while (!plotCalcer.AreAllIterationsProcessed()) {
            plotCalcer.ProceedDataSetForNonAdditiveMetrics(processedDataProvider);
            plotCalcer.FinishProceedDataSetForNonAdditiveMetrics();
        }
    }

    TVector<TVector<double>> metricsScore = plotCalcer.GetMetricsScore();

    plotCalcer.SaveResult(resultDir, /*metricsFile=*/"", /*saveMetrics*/ false, /*saveStats=*/true).ClearTempFiles();
    return metricsScore;
}

TVector<TString> GetMetricNames(const TFullModel& model, const TVector<TString>& metricsDescription) {
    auto metrics = CreateMetricsFromDescription(metricsDescription, model.ObliviousTrees.ApproxDimension);
    TVector<TString> metricNames;
    metricNames.reserve(metrics.ysize());
    for (auto& metric : metrics) {
        metricNames.push_back(metric->GetDescription());
    }
    return metricNames;
}

TVector<double> EvalMetricsForUtils(
    const TVector<float>& label,
    const TVector<TVector<double>>& approx,
    const TString& metricName,
    const TVector<float>& weight,
    const TVector<TGroupId>& groupId,
    int threadCount
) {
    NPar::TLocalExecutor executor;
    executor.RunAdditionalThreads(threadCount - 1);
    const int approxDimension = approx.ysize();
    const TVector<THolder<IMetric>> metrics = CreateMetricsFromDescription({metricName}, approxDimension);
    TVector<TQueryInfo> queriesInfo;

    // TODO(nikitxskv): Make GroupWeight, SubgroupId and Pairs support.
    UpdateQueriesInfo(groupId, /*groupWeight=*/{}, /*subgroupId=*/{}, /*beginDoc=*/0, groupId.ysize(), &queriesInfo);

    TVector<double> metricResults;
    metricResults.reserve(metrics.size());
    for (const auto& metric : metrics) {
        TMetricHolder metricResult;
        if (metric->GetErrorType() == EErrorType::PerObjectError) {
            const int begin = 0, end = label.ysize();
            metricResult = metric->Eval(approx, label, weight, queriesInfo, begin, end, executor);
        } else {
            Y_VERIFY(metric->GetErrorType() == EErrorType::QuerywiseError || metric->GetErrorType() == EErrorType::PairwiseError);
            CB_ENSURE(!queriesInfo.empty(), "You should provide group_id for groupwise metrics.");
            const int queryStartIndex = 0, queryEndIndex = queriesInfo.ysize();
            metricResult = metric->Eval(approx, label, weight, queriesInfo, queryStartIndex, queryEndIndex, executor);
        }
        metricResults.push_back(metric->GetFinalError(metricResult));
    }
    return metricResults;
}
