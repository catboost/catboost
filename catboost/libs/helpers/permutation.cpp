#include "permutation.h"

#include <util/random/shuffle.h>

template<typename TDataType>
static inline void ApplyPermutation(const TVector<ui64>& permutation, TVector<TDataType>* elements) {
    const ui64 elementCount = elements->size();
    if (elementCount == 0) {
        return;
    }
    TVector<ui64> toIndices(permutation);
    for (ui64 elementIdx = 0; elementIdx < elementCount; ++elementIdx) {
        while (toIndices[elementIdx] != elementIdx) {
            auto destinationIndex = toIndices[elementIdx];
            DoSwap((*elements)[elementIdx], (*elements)[destinationIndex]);
            DoSwap(toIndices[elementIdx], toIndices[destinationIndex]);
        }
    }
}

void ApplyPermutationToPairs(const TVector<ui64>& permutation, TVector<TPair>* pairs) {
    for (auto& pair : *pairs) {
        pair.WinnerId = permutation[pair.WinnerId];
        pair.LoserId = permutation[pair.LoserId];
    }
}

void ApplyPermutation(const TVector<ui64>& permutation, TPool* pool, NPar::TLocalExecutor* localExecutor) {
    Y_VERIFY(pool->Docs.GetDocCount() == 0 || permutation.size() == pool->Docs.GetDocCount());

    if (pool->Docs.GetDocCount() > 0) {
        NPar::TLocalExecutor::TExecRangeParams blockParams(0, pool->Docs.Factors.ysize());
        localExecutor->ExecRange([&] (int factorIdx) {
            ApplyPermutation(permutation, &pool->Docs.Factors[factorIdx]);
        }, blockParams, NPar::TLocalExecutor::WAIT_COMPLETE);

        for (int dim = 0; dim < pool->Docs.GetBaselineDimension(); ++dim) {
            ApplyPermutation(permutation, &pool->Docs.Baseline[dim]);
        }
        ApplyPermutation(permutation, &pool->Docs.Target);
        ApplyPermutation(permutation, &pool->Docs.Weight);
        ApplyPermutation(permutation, &pool->Docs.Id);
        ApplyPermutation(permutation, &pool->Docs.SubgroupId);
        ApplyPermutation(permutation, &pool->Docs.QueryId);
    }

    ApplyPermutationToPairs(permutation, &pool->Pairs);
}

TVector<ui64> CreateOrderByKey(const TVector<ui64>& key) {
    TVector<ui64> indices(key.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(
        indices.begin(),
        indices.end(),
        [&key](ui64 i1, ui64 i2) {
            return key[i1] < key[i2];
        }
    );

    return indices;
}
