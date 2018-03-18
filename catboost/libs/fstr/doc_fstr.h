#pragma once

#include "calc_fstr.h"

TVector<TVector<double>> CalcFeatureImportancesForDocuments(const TFullModel& model,
                                                            const TPool& pool,
                                                            const TPool& poolTrain,
                                                            bool computeLeafDocumentCountFromPoolTrain,
                                                            const int threadCount);
