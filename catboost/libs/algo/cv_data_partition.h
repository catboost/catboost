#pragma once

#include <catboost/libs/data/pool.h>

void BuildCvPools(
    int foldIdx,
    int foldCount,
    bool reverseCv,
    int seed,
    int threadCount,
    TPool* learnPool,
    TPool* testPool);
