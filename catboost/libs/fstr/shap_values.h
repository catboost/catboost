#pragma once

#include <catboost/libs/model/model.h>
#include <catboost/libs/data/pool.h>

#include <util/generic/vector.h>

TVector<TVector<double>> CalcShapValues(const TFullModel& model,
                                        const TPool& pool,
                                        int threadCount,
                                        int dimension = 0);

void CalcAndOutputShapValues(const TFullModel& model,
                             const TPool& pool,
                             const TString& outputPath,
                             int threadCount,
                             int dimension = 0);
