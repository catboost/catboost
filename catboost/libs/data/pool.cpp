#include "pool.h"

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
