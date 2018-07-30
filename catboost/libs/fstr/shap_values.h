#pragma once

#include <catboost/libs/model/model.h>
#include <catboost/libs/data/pool.h>

#include <util/generic/vector.h>

// returned: shapValues[documentIdx][dimenesion][feature]
TVector<TVector<TVector<double>>> CalcShapValuesMulti(
    const TFullModel& model,
    const TPool& pool,
    NPar::TLocalExecutor* localExecutor,
    int logPeriod = 0
);

// returned: shapValues[documentIdx][feature]
TVector<TVector<double>> CalcShapValues(
    const TFullModel& model,
    const TPool& pool,
    NPar::TLocalExecutor* localExecutor,
    int logPeriod = 0
);

// outputs for each document in order for each dimension in order an array of feature contributions
void CalcAndOutputShapValues(
    const TFullModel& model,
    const TPool& pool,
    const TString& outputPath,
    int threadCount,
    int logPeriod = 0
);
