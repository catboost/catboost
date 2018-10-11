#pragma once

#include <catboost/libs/data/pool.h>
#include <catboost/libs/model/model.h>

TFullModel TrainFloatCatboostModel();

TPool GetAdultPool();
