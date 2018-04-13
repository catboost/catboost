#include "index_hash_calcer.h"

/// Compute reindexHash and reindex hash values in range [begin,end).
size_t ComputeReindexHash(ui64 topSize,
                          TDenseHash<ui64, ui32>* reindexHashPtr,
                          ui64* begin,
                          ui64* end) {
    auto& reindexHash = *reindexHashPtr;
    auto* hashArr = begin;
    size_t learnSize = end - begin;
    ui32 counter = 0;
    if (topSize > learnSize) {
        bool isInserted = false;
        for (size_t i = 0; i < learnSize; ++i) {
            auto& v = reindexHash.GetMutable(hashArr[i], &isInserted);
            if (isInserted) {
                v = counter++;
            }
            hashArr[i] = v;
        }
    } else {
        for (size_t i = 0; i < learnSize; ++i) {
            ++reindexHash.GetMutable(hashArr[i]);
        }

        if (reindexHash.Size() <= topSize) {
            for (auto& it : reindexHash) {
                it.Value() = counter;
                ++counter;
            }
            for (size_t i = 0; i < learnSize; ++i) {
                hashArr[i] = reindexHash.Get(hashArr[i]);
            }
        } else {
            // Limit reindexHash to topSize buckets
            using TFreqPair = std::pair<ui64, ui32>;
            TVector<TFreqPair> freqValList;

            freqValList.reserve(reindexHash.Size());
            for (const auto& it : reindexHash) {
                freqValList.emplace_back(it.Key(), it.Value());
            }
            std::nth_element(freqValList.begin(), freqValList.begin() + topSize, freqValList.end(),
                         [](const TFreqPair& a, const TFreqPair& b) {
                             return a.second > b.second;
                         });

            reindexHash.MakeEmpty();
            for (ui32 i = 0; i < topSize; ++i) {
                reindexHash.GetMutable(freqValList[i].first) = i;
            }
            for (ui64* hash = begin; hash != end; ++hash) {
               auto it = reindexHash.Find(*hash);
               if (it != reindexHash.end()) {
                   *hash = it.Value();
               } else {
                   *hash = reindexHash.Size() - 1;
               }
            }
        }
    }
    return reindexHash.Size();
}

/// Update reindexHash and reindex hash values in range [begin,end).
size_t UpdateReindexHash(TDenseHash<ui64, ui32>* reindexHashPtr, ui64* begin, ui64* end) {
    auto& reindexHash = *reindexHashPtr;
    ui32 counter = reindexHash.Size();
    for (ui64* hash = begin; hash != end; ++hash) {
        bool isInserted = false;
        auto& hashVal = reindexHash.GetMutable(*hash, &isInserted);
        if (isInserted) {
            hashVal = counter;
            *hash = counter;
            ++counter;
        } else {
            *hash = hashVal;
        }
    }
    return reindexHash.Size();
}
