#pragma once

#include <catboost/libs/data_types/pair.h>

namespace NCB {

    void ConstructConnectedComponents(
        ui32 docCount,
        const TConstArrayRef<TPair> pairs,
        TVector<ui32>* groupBounds,
        TVector<ui32>* permutationForGrouping,
        TVector<TPair>* pairsInPermutedDataset);
}
