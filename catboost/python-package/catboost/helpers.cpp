#include <Python.h>

#include "helpers.h"
#include "catboost/libs/helpers/exception.h"
#include "catboost/libs/algo/interrupt.h"
#include "catboost/libs/algo/features_layout.h"
#include "catboost/libs/algo/params.h"
#include "catboost/libs/algo/calc_fstr.h"
#include "catboost/libs/fstr/doc_fstr.h"

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

yvector<yvector<double>> CalcFstr(const TFullModel& model, const TPool& pool, int threadCount){
    yvector<double> regularEffect = CalcRegularFeatureEffect(model, pool, threadCount);
    yvector<yvector<double>> result;
    for (const auto& value : regularEffect){
        yvector<double> vec = {value};
        result.push_back(vec);
    }
    return result;
}

yvector<yvector<double>> CalcInteraction(const TFullModel& model, const TPool& pool){
    int featureCount = pool.Docs[0].Factors.ysize();
    TFeaturesLayout layout(featureCount, pool.CatFeatures, pool.FeatureId);

    yvector<TInternalFeatureInteraction> internalInteraction = CalcInternalFeatureInteraction(model);
    yvector<TFeatureInteraction> interaction = CalcFeatureInteraction(internalInteraction, layout);
    yvector<yvector<double>> result;
    for (const auto& value : interaction){
        int featureIdxFirst = layout.GetFeature(value.FirstFeature.Index, value.FirstFeature.Type);
        int featureIdxSecond = layout.GetFeature(value.SecondFeature.Index, value.SecondFeature.Type);
        yvector<double> vec = {static_cast<double>(featureIdxFirst), static_cast<double>(featureIdxSecond), value.Score};
        result.push_back(vec);
    }
    return result;
}

yvector<yvector<double>> GetFeatureImportances(const TFullModel& model, const TPool& pool, const TString& type, int threadCount){
    CB_ENSURE(!pool.Docs.empty(), "Pool should not be empty");
    EFstrType FstrType = FromString<EFstrType>(type);
    switch (FstrType) {
        case EFstrType::FeatureImportance:
            return CalcFstr(model, pool, threadCount);
        case EFstrType::Interaction:
            return CalcInteraction(model, pool);
        case EFstrType::Doc:
            return CalcFeatureImportancesForDocuments(model, pool, threadCount);
        default:
            Y_UNREACHABLE();
    }
}
