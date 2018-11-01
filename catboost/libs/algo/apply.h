#pragma once

#include "index_calcer.h"

#include <catboost/libs/data/pool.h>
#include <catboost/libs/model/model.h>
#include <catboost/libs/model/formula_evaluator.h>
#include <catboost/libs/model/model_pool_compatibility.h>

#include <util/generic/vector.h>

TVector<TVector<double>> ApplyModelMulti(
    const TFullModel& model,
    const TPool& pool,
    const EPredictionType predictionType,
    int begin,
    int end,
    NPar::TLocalExecutor* executor);


TVector<TVector<double>> ApplyModelMulti(
    const TFullModel& model,
    const TPool& pool,
    bool verbose = false,
    const EPredictionType predictionType = EPredictionType::RawFormulaVal,
    int begin = 0,
    int end = 0,
    int threadCount = 1);

TVector<double> ApplyModel(
    const TFullModel& model,
    const TPool& pool,
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
        const TPool& pool,
        NPar::TLocalExecutor* executor);

    void ApplyModelMulti(
        const EPredictionType predictionType,
        int begin, /*= 0*/
        int end,
        TVector<double>* flatApproxBuffer,
        TVector<TVector<double>>* approx);

private:
    const TFullModel* Model;
    const TPool* Pool;
    NPar::TLocalExecutor* Executor;
    NPar::TLocalExecutor::TExecRangeParams BlockParams;
    TVector<THolder<TFeatureCachedTreeEvaluator>> ThreadCalcers;
};
