#pragma once

#include <catboost/libs/data/pool.h>
#include <catboost/libs/model/model.h>

TFullModel TrainFloatCatboostModel(int iterations = 5, int seed = 123);

TPool GetAdultPool();
