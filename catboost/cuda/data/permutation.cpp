#include "permutation.h"

void NCatboostCuda::TDataPermutation::FillOrder(TVector<ui32>& order) const  {
    if (Index != IdentityPermutationId()) {
        const auto seed = 1664525 * GetPermutationId() + 1013904223 + BlockSize;
        if (DataProvider->HasQueries()) {
            QueryConsistentShuffle(seed, BlockSize, DataProvider->GetQueryIds(), &order);
        } else {
            Shuffle(seed, BlockSize, DataProvider->GetSampleCount(), &order);
        }
    } else {
        order.resize(DataProvider->GetSampleCount());
        std::iota(order.begin(), order.end(), 0);
    }
}

void NCatboostCuda::TDataPermutation::FillInversePermutation(TVector<ui32>& permutation) const  {
    TVector<ui32> order;
    FillOrder(order);
    permutation.resize(order.size());
    for (ui32 i = 0; i < order.size(); ++i) {
        permutation[order[i]] = i;
    }
}

