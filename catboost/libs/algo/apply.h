#pragma once

#include "index_calcer.h"

#include <catboost/libs/data/pool.h>
#include <catboost/libs/model/model.h>

#include <util/generic/vector.h>
#include <catboost/libs/model/formula_evaluator.h>

using TTreeFunction = std::function<void(const TAllFeatures& features,
                                         const TFullModel& model,
                                         int treeIdx,
                                         const TCommonContext& ctx,
                                         yvector<yvector<double>>* approxPtr)>;

void CalcApproxForTree(const TAllFeatures& features,
                       const TFullModel& model,
                       int treeIdx,
                       const TCommonContext& ctx,
                       yvector<yvector<double>>* resultPtr);

yvector<yvector<double>> MapFunctionToTrees(const TFullModel& model,
                                            const TAllFeatures& features,
                                            int begin,
                                            int end,
                                            const TTreeFunction& function,
                                            int resultDimension,
                                            TCommonContext* ctx);

yvector<yvector<double>> ApplyModelMulti(const NCatBoost::TFormulaEvaluator& calcer,
                                         const TPool& pool,
                                         const EPredictionType predictionType,
                                         int begin,
                                         int end,
                                         NPar::TLocalExecutor& executor);


yvector<yvector<double>> ApplyModelMulti(const NCatBoost::TFormulaEvaluator& calcer,
                                         const TPool& pool,
                                         bool verbose = false,
                                         const EPredictionType predictionType = EPredictionType::RawFormulaVal,
                                         int begin = 0,
                                         int end = 0,
                                         int threadCount = 1);

yvector<double> ApplyModel(const TFullModel& model,
                           const TPool& pool,
                           bool verbose = false,
                           const EPredictionType predictionType = EPredictionType::RawFormulaVal,
                           int begin = 0,
                           int end = 0,
                           int threadCount = 1);
