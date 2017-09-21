#pragma once

#include <util/generic/noncopyable.h>
#include <catboost/libs/data/pool.h>
#include <catboost/libs/model/model.h>

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

yvector<yvector<double>> CalcFstr(const TFullModel& model, const TPool& pool, int threadCount);
yvector<yvector<double>> CalcInteraction(const TFullModel& model, const TPool& pool);
yvector<yvector<double>> GetFeatureImportances(const TFullModel& model, const TPool& pool, const TString& type, int threadCount);
