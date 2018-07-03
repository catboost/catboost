#pragma once

#include <catboost/libs/model/model.h>
#include <catboost/libs/data/pool.h>

#include <util/generic/vector.h>

/*In case of multiclass the returned value for each document in pool is
a vector of length (feature_count + 1) * approxDimension: shap values for each dimension in order.
The values are calculated for raw values.*/
TVector<TVector<double>> CalcShapValues(
    const TFullModel& model,
    const TPool& pool,
    int threadCount,
    int logPeriod = 0
);

/*In case of multiclass the returned value for each document in pool is
a vector of length (feature_count + 1) * approxDimension: shap values for each dimension in order.
The values are calculated for raw values.*/
void CalcAndOutputShapValues(
    const TFullModel& model,
    const TPool& pool,
    const TString& outputPath,
    int threadCount,
    int logPeriod = 0
);
