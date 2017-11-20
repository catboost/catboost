#include "index_hash_calcer.h"

std::pair<size_t, size_t> ReindexHash(size_t learnSize,
                                      ui64 topSize,
                                      TVector<ui64>* hashVecPtr,
                                      TDenseHash<ui64, ui32>* reindexHashPtr) {
    auto& hashArr = *hashVecPtr;
    auto& reindexHash = *reindexHashPtr;

    bool earlyReindexing = topSize > learnSize;
    ui32 counter = 0;
    if (earlyReindexing) {
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
    }
    size_t learnNumLeaves = 0;
    size_t totalNumLeaves = 0;
    if (reindexHash.Size() <= topSize) {
        if (!earlyReindexing) {
            for (auto& it : reindexHash) {
                it.Value() = counter;
                ++counter;
            }
            for (size_t i = 0; i < learnSize; ++i) {
                hashArr[i] = reindexHash.Get(hashArr[i]);
            }
        }
        learnNumLeaves = reindexHash.Size();

        auto fullLen = hashArr.size();
        for (size_t i = learnSize; i < fullLen; ++i) {
            bool isInserted = false;
            auto& hashVal = reindexHash.GetMutable(hashArr[i], &isInserted);
            if (isInserted) {
                hashVal = counter;
                hashArr[i] = counter;
                ++counter;
            } else {
                hashArr[i] = hashVal;
            }
        }
        totalNumLeaves = reindexHash.Size();
    } else {
        using TFreqPair = std::pair<ui64, ui32>;
        TVector<TFreqPair> freqValList;

        freqValList.reserve(reindexHash.Size());
        for (const auto& it : reindexHash) {
            freqValList.emplace_back(it.Key(), it.Value());
        }
        reindexHash.MakeEmpty();
        std::nth_element(freqValList.begin(), freqValList.begin() + topSize, freqValList.end(),
                         [](const TFreqPair& a, const TFreqPair& b) {
                             return a.second > b.second;
                         });
        for (ui32 i = 0; i < topSize; ++i) {
            reindexHash.GetMutable(freqValList[i].first) = i;
        }
        for (auto& hash : hashArr) {
            auto it = reindexHash.Find(hash);
            if (it != reindexHash.end()) {
                hash = it.Value();
            } else {
                hash = topSize;
            }
        }
        learnNumLeaves = totalNumLeaves = topSize + 1;
    }
    return {learnNumLeaves, totalNumLeaves};
}
