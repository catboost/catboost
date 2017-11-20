#pragma once

#include "index_calcer.h"

#include <catboost/libs/data/pool.h>
#include <catboost/libs/model/model.h>
#include <catboost/libs/model/formula_evaluator.h>

#include <util/generic/vector.h>

TVector<TVector<double>> ApplyModelMulti(const TFullModel& model,
                                         const TPool& pool,
                                         const EPredictionType predictionType,
                                         int begin,
                                         int end,
                                         NPar::TLocalExecutor& executor);


TVector<TVector<double>> ApplyModelMulti(const TFullModel& model,
                                         const TPool& pool,
                                         bool verbose = false,
                                         const EPredictionType predictionType = EPredictionType::RawFormulaVal,
                                         int begin = 0,
                                         int end = 0,
                                         int threadCount = 1);

TVector<double> ApplyModel(const TFullModel& model,
                           const TPool& pool,
                           bool verbose = false,
                           const EPredictionType predictionType = EPredictionType::RawFormulaVal,
                           int begin = 0,
                           int end = 0,
                           int threadCount = 1);
