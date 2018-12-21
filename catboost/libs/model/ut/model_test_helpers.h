#pragma once

#include <catboost/libs/data_new/data_provider.h>
#include <catboost/libs/model/model.h>

TFullModel TrainFloatCatboostModel(int iterations = 5, int seed = 123);

NCB::TDataProviderPtr GetAdultPool();
