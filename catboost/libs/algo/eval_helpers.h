#pragma once

#include "params.h"

#include <catboost/libs/data/pool.h>

#include <library/threading/local_executor/local_executor.h>

#include <util/generic/vector.h>

yvector<yvector<double>> PrepareEval(const EPredictionType predictionType,
                                     const yvector<yvector<double>>& approx,
                                     NPar::TLocalExecutor* localExecutor);

yvector<yvector<double>> PrepareEval(const EPredictionType predictionType,
                                     const yvector<yvector<double>>& approx,
                                     int threadCount);

void OutputTestEval(const yvector<yvector<yvector<double>>>& testApprox,
                    const yvector<TDocInfo>& docIds,
                    const bool outputTarget,
                    TOFStream* outputStream);
