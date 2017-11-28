#pragma once

#include <catboost/libs/algo/plot.h>

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

TVector<TVector<double>> EvalMetrics(
    const TFullModel& model,
    const TPool& pool,
    const TString& metricDescription,
    int begin,
    int end,
    int evalPeriod,
    int threadCount,
    const TString& tmpDir
);
