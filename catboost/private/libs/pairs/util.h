#pragma once

#include <catboost/libs/data/objects_grouping.h>
#include <catboost/libs/data/weights.h>
#include <catboost/private/libs/data_types/pair.h>

#include <catboost/libs/helpers/restorable_rng.h>

#include <util/generic/array_ref.h>
#include <util/generic/vector.h>

namespace NCB {
    static const uint64_t MAX_PAIR_COUNT_ON_GPU = 1022 * 1023 / 2;  // cause 1023 is max group size for GPU
};

void GeneratePairLogitPairs(
    const NCB::TObjectsGrouping& objectsGrouping,
    const TMaybe<NCB::TSharedWeights<float>>& weights,
    TConstArrayRef<float> targetId,
    int maxPairCount,
    TRestorableFastRng64* rand,
    TVector<TPair>* result);
