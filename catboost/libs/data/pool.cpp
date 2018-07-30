#include "pool.h"

#include <catboost/libs/helpers/exception.h>

static TDocumentStorage SliceDocumentStorage(
    const TDocumentStorage& docs,
    const TVector<size_t>& rowIndices
) {
    for (size_t rowIndex : rowIndices) {
        CB_ENSURE(rowIndex < docs.GetDocCount(), "Pool doesn't have a row with index " << rowIndex);
    }

    TDocumentStorage slicedDocs;
    slicedDocs.Resize(rowIndices.size(), docs.GetEffectiveFactorCount());

    for (size_t newDocIdx = 0; newDocIdx < rowIndices.size(); ++newDocIdx) {
        size_t oldDocIdx = rowIndices[newDocIdx];
        slicedDocs.AssignDoc(newDocIdx, docs, oldDocIdx);
    }

    return slicedDocs;
}

static TVector<TPair> SlicePairs(const TPool& pool, const TVector<size_t>& rowIndices) {
    TVector<TPair> slicedPairs;
    if (!pool.Pairs.empty()) {
        TVector<TVector<size_t>> invertedIndices(pool.Docs.GetDocCount());
        for (size_t newDocIndex = 0; newDocIndex < rowIndices.size(); ++newDocIndex) {
            size_t oldDocIndex = rowIndices[newDocIndex];
            invertedIndices[oldDocIndex].push_back(newDocIndex);
        }
        for (const TPair& pair : pool.Pairs) {
            if (!invertedIndices[pair.LoserId].empty() && !invertedIndices[pair.WinnerId].empty()) {
                for (size_t newWinnerIndex : invertedIndices[pair.WinnerId]) {
                    for (size_t newLoserIndex : invertedIndices[pair.LoserId]) {
                        slicedPairs.emplace_back(newWinnerIndex, newLoserIndex, pair.Weight);
                    }
                }
            }
        }
    }
    return slicedPairs;
}

THolder<TPool> SlicePool(const TPool& pool, const TVector<size_t>& rowIndices) {
    THolder<TPool> slicedPool = new TPool();

    slicedPool->Docs = SliceDocumentStorage(pool.Docs, rowIndices);
    slicedPool->Pairs = SlicePairs(pool, rowIndices);

    slicedPool->CatFeatures = pool.CatFeatures;
    slicedPool->FeatureId = pool.FeatureId;
    slicedPool->CatFeaturesHashToString = pool.CatFeaturesHashToString;
    slicedPool->MetaInfo = pool.MetaInfo;

    return slicedPool;
}

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
        const int featureCount = pool->GetFactorCount();
        NPar::TLocalExecutor::TExecRangeParams blockParams(0, featureCount);
        if (!pool->Docs.Factors.empty()) {
            localExecutor->ExecRange([&] (int factorIdx) {
                ApplyPermutation(permutation, &pool->Docs.Factors[factorIdx]);
            }, blockParams, NPar::TLocalExecutor::WAIT_COMPLETE);
        } else {
            const int floatFeatureCount = pool->QuantizedFeatures.FloatHistograms.ysize();
            localExecutor->ExecRange([&] (int factorIdx) {
                if (factorIdx < floatFeatureCount) {
                    ApplyPermutation(permutation, &pool->QuantizedFeatures.FloatHistograms[factorIdx]);
                } else {
                    ApplyPermutation(permutation, &pool->QuantizedFeatures.CatFeaturesRemapped[factorIdx - floatFeatureCount]);
                }
            }, blockParams, NPar::TLocalExecutor::WAIT_COMPLETE);
        }

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
