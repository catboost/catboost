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
        TVector<std::function<void()>> permuters;
        permuters.emplace_back([&]() { ApplyPermutation(permutation, &pool->Docs.Target); });
        permuters.emplace_back([&]() { ApplyPermutation(permutation, &pool->Docs.Weight); });
        permuters.emplace_back([&]() { ApplyPermutation(permutation, &pool->Docs.Id); });
        permuters.emplace_back([&]() { ApplyPermutation(permutation, &pool->Docs.SubgroupId); });
        permuters.emplace_back([&]() { ApplyPermutation(permutation, &pool->Docs.QueryId); });
        for (auto& dimensionBaseline : pool->Docs.Baseline) {
            permuters.emplace_back([&]() { ApplyPermutation(permutation, &dimensionBaseline); });
        }
        if (pool->IsQuantized()) {
            for (auto& floatHistogram : pool->QuantizedFeatures.FloatHistograms) {
                permuters.emplace_back([&]() { ApplyPermutation(permutation, &floatHistogram); });
            }
            for (auto& catFeature : pool->QuantizedFeatures.CatFeaturesRemapped) {
                permuters.emplace_back([&]() { ApplyPermutation(permutation, &catFeature); });
            }
        } else {
            for (auto& factor : pool->Docs.Factors) {
                permuters.emplace_back([&]() { ApplyPermutation(permutation, &factor); });
            }
        }
        NPar::ParallelFor(*localExecutor, 0, permuters.size(), [&] (ui32 permuterIdx) {
            permuters[permuterIdx]();
        });
    }

    ApplyPermutationToPairs(permutation, &pool->Pairs);
}
