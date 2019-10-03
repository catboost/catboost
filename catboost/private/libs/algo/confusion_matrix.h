#pragma once

#include <catboost/libs/model/model.h>
#include <catboost/private/libs/target/data_providers.h>

#include <util/generic/vector.h>

TVector<double> MakeConfusionMatrix(TConstArrayRef<TVector<double>> approxes, TConstArrayRef<float> labels, NPar::TLocalExecutor* localExecutor);
TVector<double> MakeConfusionMatrix(const TFullModel& model, const NCB::TDataProviderPtr dataset, int threadCount);
