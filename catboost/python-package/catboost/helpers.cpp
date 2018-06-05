#include "helpers.h"

#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/interrupt.h>
#include <catboost/libs/data_types/groupid.h>

#include <Python.h>

extern "C" PyObject* PyCatboostExceptionType;

void ProcessException() {
    try {
        throw;
    } catch (const TCatboostException& exc) {
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

TVector<TVector<double>> EvalMetrics(
    const TFullModel& model,
    const TPool& pool,
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

    auto metrics = CreateMetricsFromDescription(metricsDescription, model.ObliviousTrees.ApproxDimension);
    TMetricsPlotCalcer plotCalcer = CreateMetricCalcer(
        model,
        begin,
        end,
        evalPeriod,
        /*processedIterationsStep=*/50,
        executor,
        tmpDir,
        metrics
    );

    if (plotCalcer.HasAdditiveMetric()) {
        plotCalcer.ProceedDataSetForAdditiveMetrics(pool, /*isProcessBoundaryGroups=*/false);
        plotCalcer.FinishProceedDataSetForAdditiveMetrics();
    }
    if (plotCalcer.HasNonAdditiveMetric()) {
        while (!plotCalcer.AreAllIterationsProcessed()) {
            plotCalcer.ProceedDataSetForNonAdditiveMetrics(pool);
            plotCalcer.FinishProceedDataSetForNonAdditiveMetrics();
        }
    }

    TVector<TVector<double>> metricsScore = plotCalcer.GetMetricsScore();

    plotCalcer.SaveResult(resultDir, /*metricsFile=*/"", /*saveMetrics*/ false, /*saveStats=*/true).ClearTempFiles();
    return metricsScore;
}

TVector<double> EvalMetricsForUtils(
    const TVector<float>& label,
    const TVector<TVector<double>>& approx,
    const TString& metricName,
    const TVector<float>& weight,
    const TVector<int>& groupIdParam,
    int threadCount
) {
    NPar::TLocalExecutor executor;
    executor.RunAdditionalThreads(threadCount - 1);
    const int approxDimension = approx.ysize();
    const TVector<TGroupId> groupId(groupIdParam.begin(), groupIdParam.end());
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
            const int queryStartIndex = 0, queryEndIndex = queriesInfo.ysize();
            metricResult = metric->Eval(approx, label, weight, queriesInfo, queryStartIndex, queryEndIndex, executor);
        }
        metricResults.push_back(metric->GetFinalError(metricResult));
    }
    return metricResults;
}
