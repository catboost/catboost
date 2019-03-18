#include "index_hash_calcer.h"

/// Compute reindexHash and reindex hash values in range [begin,end).
size_t ComputeReindexHash(ui64 topSize, TDenseHash<ui64, ui32>* reindexHashPtr, ui64* begin, ui64* end) {
    auto& reindexHash = *reindexHashPtr;
    auto* hashArr = begin;
    size_t learnSize = end - begin;
    ui32 counter = 0;
    if (topSize > learnSize) {
        for (size_t i = 0; i < learnSize; ++i) {
            auto p = reindexHash.emplace(hashArr[i], counter);
            if (p.second) {
                ++counter;
            }
            hashArr[i] = p.first->second;
        }
    } else {
        for (size_t i = 0; i < learnSize; ++i) {
            ++reindexHash[hashArr[i]];
        }

        if (reindexHash.Size() <= topSize) {
            for (auto& it : reindexHash) {
                it.second = counter;
                ++counter;
            }
            for (size_t i = 0; i < learnSize; ++i) {
                hashArr[i] = reindexHash.Value(hashArr[i], 0);
            }
        } else {
            // Limit reindexHash to topSize buckets
            using TFreqPair = std::pair<ui64, ui32>;
            TVector<TFreqPair> freqValList;

            freqValList.reserve(reindexHash.Size());
            for (const auto& it : reindexHash) {
                freqValList.emplace_back(it.first, it.second);
            }
            std::nth_element(freqValList.begin(), freqValList.begin() + topSize, freqValList.end(),
                         [](const TFreqPair& a, const TFreqPair& b) {
                             return a.second > b.second;
                         });

            reindexHash.MakeEmpty();
            for (ui32 i = 0; i < topSize; ++i) {
                reindexHash[freqValList[i].first] = i;
            }
            for (ui64* hash = begin; hash != end; ++hash) {
               if (auto* p = reindexHash.FindPtr(*hash)) {
                   *hash = *p;
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
        auto p = reindexHash.emplace(*hash, counter);
        if (p.second) {
            *hash = counter++;
        } else {
            *hash = p.first->second;
        }
    }
    return reindexHash.Size();
}
