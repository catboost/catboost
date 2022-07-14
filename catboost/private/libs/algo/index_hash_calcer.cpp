#include "index_hash_calcer.h"

#include "projection.h"

#include <util/generic/utility.h>
#include <util/generic/xrange.h>


using namespace NCB;

struct TCtrCalcerParams {
    const int IteratorBlockSize = 4096;

    TCtrCalcerParams(size_t sampleCount, ui64* hashArrayPtr, NPar::ILocalExecutor* localExecutor)
        : BlockExecutionParams(0, sampleCount)
        , HashArrayPtr(hashArrayPtr)
        , LocalExecutor(localExecutor)
    {
        BlockExecutionParams.SetBlockCount(LocalExecutor->GetThreadCount() + 1);
        // round per-thread block size to iteration block size
        BlockExecutionParams.SetBlockSize(
            Min<int>(
                sampleCount,
                CeilDiv<int>(BlockExecutionParams.GetBlockSize(), IteratorBlockSize) * IteratorBlockSize
            )
        );
    }

    TVector<THolder<IFeatureValuesHolder>> PermutedFeatureColumns;
    TVector<std::function<void(TArrayRef<ui64>, IDynamicBlockIteratorBase*)>> PerIteratorCallbacks;
    NPar::ILocalExecutor::TExecRangeParams BlockExecutionParams;
    ui64* HashArrayPtr;
    NPar::ILocalExecutor* LocalExecutor;

    void ThreadFunc(int threadId) {
        const int blockFirstId = BlockExecutionParams.FirstId + threadId * BlockExecutionParams.GetBlockSize();
        const int blockLastId = Min(BlockExecutionParams.LastId, blockFirstId + BlockExecutionParams.GetBlockSize());
        TVector<IDynamicBlockIteratorBasePtr> iterators;
        for (auto& column : PermutedFeatureColumns) {
            // TODO(kirillovs): this is redundant and only needed for current type-system
            if (auto catColumn = dynamic_cast<const IQuantizedCatValuesHolder*>(column.Get())) {
                iterators.emplace_back(catColumn->GetBlockIterator(blockFirstId));
            } else if (auto floatColumn = dynamic_cast<const IQuantizedFloatValuesHolder*>(column.Get())) {
                iterators.emplace_back(floatColumn->GetBlockIterator(blockFirstId));
            } else {
                CB_ENSURE_INTERNAL(false, "We only support quantized float and categorical columns here");
            }
        }
        Y_ASSERT(iterators.size() == PerIteratorCallbacks.size());
        for (int currentIdx = blockFirstId; currentIdx < blockLastId; currentIdx += IteratorBlockSize) {
            auto currentSubBlockSize = Min(IteratorBlockSize, blockLastId - currentIdx);
            TArrayRef currentHashView(HashArrayPtr + currentIdx, currentSubBlockSize);
            for (auto i : xrange(PermutedFeatureColumns.size())) {
                PerIteratorCallbacks[i](currentHashView, iterators[i].Get());
            }
        }
    }

    void Run() {
        LocalExecutor->ExecRange(
            [this] (int threadId) {
                this->ThreadFunc(threadId);
            },
            0, BlockExecutionParams.GetBlockCount(),
            NPar::TLocalExecutor::WAIT_COMPLETE
        );
    }
};

void CalcHashes(
    const TProjection& proj,
    const TQuantizedObjectsDataProvider& objectsDataProvider,
    const TFeaturesArraySubsetIndexing& featuresSubsetIndexing,
    const TPerfectHashedToHashedCatValuesMap* perfectHashedToHashedCatValuesMap,
    ui64* begin,
    ui64* end,
    NPar::ILocalExecutor* localExecutor) {

    const size_t sampleCount = end - begin;
    CB_ENSURE((size_t)featuresSubsetIndexing.Size() == sampleCount, "Unexpected range of samples");
    if (sampleCount == 0) {
        return;
    }

    TCloningParams cloningParams;
    cloningParams.SubsetIndexing = &featuresSubsetIndexing;
    TCtrCalcerParams ctrCalcerParams(sampleCount, begin, localExecutor);
    // TODO(kirillovs): this should be replaced with split-trueness iterator
    for (const int featureIdx : proj.CatFeatures) {
        ctrCalcerParams.PermutedFeatureColumns.emplace_back(
            (*objectsDataProvider.GetCatFeature(featureIdx))->CloneWithNewSubsetIndexing(
                cloningParams,
                localExecutor
            )
        );

        if (!perfectHashedToHashedCatValuesMap) {
            ctrCalcerParams.PerIteratorCallbacks.emplace_back(
                [] (TArrayRef<ui64> hashArr, IDynamicBlockIteratorBase* baseIterator) {
                    DispatchIteratorType(baseIterator, [hashArr] (auto iterator) {
                        auto block = iterator->Next(hashArr.size());
                        Y_ASSERT(block.size() == hashArr.size());
                        for (auto i : xrange(block.size())) {
                            hashArr[i] = CalcHash(hashArr[i], (ui64)block[i] + 1);
                        }
                    });
                }
            );
        } else {
            auto origValsView = MakeArrayRef(perfectHashedToHashedCatValuesMap->at(featureIdx));
            ctrCalcerParams.PerIteratorCallbacks.emplace_back(
                [origValsView] (TArrayRef<ui64> hashArr, IDynamicBlockIteratorBase* baseIterator) {
                    DispatchIteratorType(baseIterator, [hashArr, origValsView] (auto iterator) {
                        auto block = iterator->Next(hashArr.size());
                        Y_ASSERT(block.size() == hashArr.size());
                        for (auto i : xrange(block.size())) {
                            hashArr[i] = CalcHash(hashArr[i], (int)origValsView[block[i]]);
                        }
                    });
                }
            );
        }
    }

    for (const TBinFeature& feature : proj.BinFeatures) {
        ctrCalcerParams.PermutedFeatureColumns.emplace_back(
            (*objectsDataProvider.GetFloatFeature(feature.FloatFeature))->CloneWithNewSubsetIndexing(
                cloningParams,
                localExecutor
            )
        );
        ctrCalcerParams.PerIteratorCallbacks.emplace_back(
            [feature] (TArrayRef<ui64> hashArr, IDynamicBlockIteratorBase* baseIterator) {
                DispatchIteratorType(baseIterator, [hashArr, feature] (auto iterator) {
                    auto block = iterator->Next(hashArr.size());
                    Y_ASSERT(block.size() == hashArr.size());
                    for (auto i : xrange(block.size())) {
                        const bool isTrueFeature = IsTrueHistogram((ui16)block[i], (ui16)feature.SplitIdx);
                        hashArr[i] = CalcHash(hashArr[i], isTrueFeature);
                    }
                });
            }
        );
    }

    for (const TOneHotSplit& feature : proj.OneHotFeatures) {
        ctrCalcerParams.PermutedFeatureColumns.emplace_back(
            (*objectsDataProvider.GetCatFeature(feature.CatFeatureIdx))->CloneWithNewSubsetIndexing(
                cloningParams,
                localExecutor
            )
        );
        ctrCalcerParams.PerIteratorCallbacks.emplace_back(
            [feature] (TArrayRef<ui64> hashArr, IDynamicBlockIteratorBase* baseIterator) {
                DispatchIteratorType(baseIterator, [hashArr, feature] (auto iterator) {
                    auto block = iterator->Next(hashArr.size());
                    Y_ASSERT(block.size() == hashArr.size());
                    for (auto i : xrange(block.size())) {
                        const bool isTrueFeature = IsTrueOneHotFeature(block[i], (ui32)feature.Value);
                        hashArr[i] = CalcHash(hashArr[i], isTrueFeature);
                    }
                });
            }
        );
    }
    ctrCalcerParams.Run();
}


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
            std::nth_element(
                freqValList.begin(),
                freqValList.begin() + topSize,
                freqValList.end(),
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
