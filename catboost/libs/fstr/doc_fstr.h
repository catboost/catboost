#pragma once

#include <catboost/libs/algo/calc_fstr.h>

yvector<yvector<double>> CalcFeatureImportancesForDocuments(const TFullModel& model,
                                                            const TPool& pool,
                                                            const int threadCount);
