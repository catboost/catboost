#include "permutation.h"

#include "data_utils.h"

#include <numeric>

void NCatboostCuda::TDataPermutation::FillOrder(TVector<ui32>& order) const {
    if (Index != IdentityPermutationId()) {
        const auto seed = 1664525 * GetPermutationId() + 1013904223 + BlockSize;
        if (DataProvider->MetaInfo.HasGroupId) {
            QueryConsistentShuffle(seed, BlockSize, *DataProvider->ObjectsData->GetGroupIds(), &order);
        } else {
            Shuffle(seed, BlockSize, DataProvider->GetObjectCount(), &order);
        }
    } else {
        order.resize(DataProvider->GetObjectCount());
        std::iota(order.begin(), order.end(), 0);
    }
}

void NCatboostCuda::TDataPermutation::FillInversePermutation(TVector<ui32>& permutation) const {
    TVector<ui32> order;
    FillOrder(order);
    permutation.resize(order.size());
    for (ui32 i = 0; i < order.size(); ++i) {
        permutation[order[i]] = i;
    }
}
