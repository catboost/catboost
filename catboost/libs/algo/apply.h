#pragma once

#include <catboost/libs/data_new/objects.h>
#include <catboost/libs/model/model.h>
#include <catboost/libs/model/formula_evaluator.h>
#include <catboost/libs/options/enums.h>

#include <library/threading/local_executor/local_executor.h>

#include <util/generic/ptr.h>
#include <util/generic/vector.h>

namespace NCB {
    template <class TTObjectsDataProvider>
    class TDataProviderTemplate;

    using TDataProvider = TDataProviderTemplate<TObjectsDataProvider>;
}

TVector<TVector<double>> ApplyModelMulti(
    const TFullModel& model,
    const NCB::TObjectsDataProvider& objectsData,
    const EPredictionType predictionType,
    int begin,
    int end,
    NPar::TLocalExecutor* executor = nullptr);

TVector<TVector<double>> ApplyModelMulti(
    const TFullModel& model,
    const NCB::TObjectsDataProvider& objectsData,
    bool verbose = false,
    const EPredictionType predictionType = EPredictionType::RawFormulaVal,
    int begin = 0,
    int end = 0,
    int threadCount = 1);

TVector<TVector<double>> ApplyModelMulti(
    const TFullModel& model,
    const NCB::TDataProvider& data,
    bool verbose = false,
    const EPredictionType predictionType = EPredictionType::RawFormulaVal,
    int begin = 0,
    int end = 0,
    int threadCount = 1);

TVector<double> ApplyModel(
    const TFullModel& model,
    const NCB::TObjectsDataProvider& objectsData,
    bool verbose = false,
    const EPredictionType predictionType = EPredictionType::RawFormulaVal,
    int begin = 0,
    int end = 0,
    int threadCount = 1);

/*
 * Tradeoff memory for speed
 * Don't use if you need to compute model only once and on all features
 */
class TModelCalcerOnPool {
public:
    TModelCalcerOnPool(
        const TFullModel& model,
        NCB::TObjectsDataProviderPtr objectsData,
        NPar::TLocalExecutor* executor);

    void ApplyModelMulti(
        const EPredictionType predictionType,
        int begin, /*= 0*/
        int end,
        TVector<double>* flatApproxBuffer,
        TVector<TVector<double>>* approx);

private:
    void InitForRawFeatures(
        const TFullModel& model,
        const NCB::TRawObjectsDataProvider& rawObjectsData,
        const THashMap<ui32, ui32> &columnReorderMap,
        const NPar::TLocalExecutor::TExecRangeParams& blockParams,
        NPar::TLocalExecutor* executor);
    void InitForQuantizedFeatures(
        const TFullModel& model,
        const NCB::TQuantizedForCPUObjectsDataProvider& quantizedObjectsData,
        const THashMap<ui32, ui32> &columnReorderMap,
        const NPar::TLocalExecutor::TExecRangeParams& blockParams,
        NPar::TLocalExecutor* executor);

private:
    const TFullModel* Model;
    NCB::TObjectsDataProviderPtr ObjectsData;
    NPar::TLocalExecutor* Executor;
    NPar::TLocalExecutor::TExecRangeParams BlockParams;
    TVector<THolder<TFeatureCachedTreeEvaluator>> ThreadCalcers;
};
