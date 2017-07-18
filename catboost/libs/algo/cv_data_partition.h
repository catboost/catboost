#pragma once

#include <catboost/libs/data/pool.h>

void BuildCvPools(
    int foldIdx,
    int foldCount,
    bool reverseCv,
    int seed,
    TPool* learnPool,
    TPool* testPool);
