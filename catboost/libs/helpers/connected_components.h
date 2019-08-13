#pragma once

#include <catboost/libs/data_new/objects_grouping.h>

namespace NCB {

    void ConstructConnectedComponents(
        ui32 docCount,
        const TConstArrayRef<TPair> pairs,
        TObjectsGrouping* objectsGrouping,
        TVector<ui32>* permutationForGrouping,
        TVector<TPair>* pairsInPermutedDataset);
}
