#pragma once

#include "model.h"

#include <catboost/libs/data/pool.h>

void CheckModelAndPoolCompatibility(const TFullModel& model, const TPool& pool, THashMap<int,int>* columnIndexesReorderMap);
