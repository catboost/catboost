#include "static_ctr_provider.h"

#include "ctr_helpers.h"
#include "model.h"

#include <util/generic/hash_set.h>
#include <util/generic/xrange.h>
#include <util/string/cast.h>


using namespace NCB;


void TStaticCtrProvider::CalcCtrs(const TConstArrayRef<TModelCtr> neededCtrs,
                                  const TConstArrayRef<ui8> binarizedFeatures,
                                  const TConstArrayRef<ui32> hashedCatFeatures,
                                  size_t docCount,
                                  TArrayRef<float> result) {
    if (neededCtrs.empty()) {
        return;
    }
    auto compressedModelCtrs = NCB::CompressModelCtrs(neededCtrs);
    size_t samplesCount = docCount;
    TVector<ui64> ctrHashes(samplesCount);
    TVector<ui64> buckets(samplesCount);
    size_t resultIdx = 0;
    float* resultPtr = result.data();
    TVector<int> transposedCatFeatureIndexes;
    TVector<TBinFeatureIndexValue> binarizedIndexes;
    for (size_t idx = 0; idx < compressedModelCtrs.size(); ++idx) {
        auto& proj = *compressedModelCtrs[idx].Projection;
        binarizedIndexes.clear();
        transposedCatFeatureIndexes.clear();
        for (const auto feature : proj.CatFeatures) {
            transposedCatFeatureIndexes.push_back(CatFeatureIndex.at(feature));
        }
        for (const auto feature : proj.BinFeatures ) {
            binarizedIndexes.push_back(FloatFeatureIndexes.at(feature));
        }
        for (const auto feature : proj.OneHotFeatures ) {
            binarizedIndexes.push_back(OneHotFeatureIndexes.at(feature));
        }
        CalcHashes(binarizedFeatures, hashedCatFeatures, transposedCatFeatureIndexes, binarizedIndexes, docCount, &ctrHashes);
        for (const auto& ctr: compressedModelCtrs[idx].ModelCtrs) {
            auto& learnCtr = CtrData.LearnCtrs.at(ctr->Base);
            auto hashIndexResolver = learnCtr.GetIndexHashViewer();
            const ECtrType ctrType = ctr->Base.CtrType;
            auto ptrBuckets = buckets.data();
            for (size_t docId = 0; docId < samplesCount; ++docId) {
                ptrBuckets[docId] = hashIndexResolver.GetIndex(ctrHashes[docId]);
            }
            if (ctrType == ECtrType::BinarizedTargetMeanValue || ctrType == ECtrType::FloatTargetMeanValue) {
                const auto emptyVal = ctr->Calc(0.f, 0.f);
                auto ctrMean = learnCtr.GetTypedArrayRefForBlobData<TCtrMeanHistory>();
                for (size_t doc = 0; doc < samplesCount; ++doc) {
                    if (ptrBuckets[doc] != NCatboost::TDenseIndexHashView::NotFoundIndex) {
                        const TCtrMeanHistory& ctrMeanHistory = ctrMean[ptrBuckets[doc]];
                        resultPtr[doc + resultIdx] = ctr->Calc(ctrMeanHistory.Sum, ctrMeanHistory.Count);
                    } else {
                        resultPtr[doc + resultIdx] = emptyVal;
                    }
                }
            } else if (ctrType == ECtrType::Counter || ctrType == ECtrType::FeatureFreq) {
                TConstArrayRef<int> ctrTotal = learnCtr.GetTypedArrayRefForBlobData<int>();
                const int denominator = learnCtr.CounterDenominator;
                auto emptyVal = ctr->Calc(0, denominator);
                for (size_t doc = 0; doc < samplesCount; ++doc) {
                    if (ptrBuckets[doc] != NCatboost::TDenseIndexHashView::NotFoundIndex) {
                        resultPtr[doc + resultIdx] = ctr->Calc(ctrTotal[ptrBuckets[doc]], denominator);
                    } else {
                        resultPtr[doc + resultIdx] = emptyVal;
                    }
                }
            } else if (ctrType == ECtrType::Buckets) {
                auto ctrIntArray = learnCtr.GetTypedArrayRefForBlobData<int>();
                const int targetClassesCount = learnCtr.TargetClassesCount;
                auto emptyVal = ctr->Calc(0, 0);
                for (size_t doc = 0; doc < samplesCount; ++doc) {
                    if (ptrBuckets[doc] != NCatboost::TDenseIndexHashView::NotFoundIndex) {
                        int goodCount = 0;
                        int totalCount = 0;
                        auto ctrHistory = MakeArrayRef(ctrIntArray.data() + ptrBuckets[doc] * targetClassesCount, targetClassesCount);
                        goodCount = ctrHistory[ctr->TargetBorderIdx];
                        for (int classId = 0; classId < targetClassesCount; ++classId) {
                            totalCount += ctrHistory[classId];
                        }
                        resultPtr[doc + resultIdx] = ctr->Calc(goodCount, totalCount);
                    } else {
                        resultPtr[doc + resultIdx] = emptyVal;
                    }
                }
            } else {
                auto ctrIntArray = learnCtr.GetTypedArrayRefForBlobData<int>();
                const int targetClassesCount = learnCtr.TargetClassesCount;

                auto emptyVal = ctr->Calc(0, 0);
                if (targetClassesCount > 2) {
                    for (size_t doc = 0; doc < samplesCount; ++doc) {
                        int goodCount = 0;
                        int totalCount = 0;
                        if (ptrBuckets[doc] != NCatboost::TDenseIndexHashView::NotFoundIndex) {
                            auto ctrHistory = MakeArrayRef(ctrIntArray.data() + ptrBuckets[doc] * targetClassesCount, targetClassesCount);
                            for (int classId = 0; classId < ctr->TargetBorderIdx + 1; ++classId) {
                                totalCount += ctrHistory[classId];
                            }
                            for (int classId = ctr->TargetBorderIdx + 1; classId < targetClassesCount; ++classId) {
                                goodCount += ctrHistory[classId];
                            }
                            totalCount += goodCount;
                        }
                        resultPtr[doc + resultIdx] = ctr->Calc(goodCount, totalCount);
                    }
                } else {
                    for (size_t doc = 0; doc < samplesCount; ++doc) {
                        if (ptrBuckets[doc] != NCatboost::TDenseIndexHashView::NotFoundIndex) {
                            const int* ctrHistory = &ctrIntArray[ptrBuckets[doc] * 2];
                            resultPtr[doc + resultIdx] = ctr->Calc(ctrHistory[1], ctrHistory[0] + ctrHistory[1]);
                        } else {
                            resultPtr[doc + resultIdx] = emptyVal;
                        }
                    }
                }
            }
            resultIdx += docCount;
        }
    }
}

bool TStaticCtrProvider::HasNeededCtrs(TConstArrayRef<TModelCtr> neededCtrs) const {
    for (const auto& ctr : neededCtrs) {
        if (!CtrData.LearnCtrs.contains(ctr.Base)) {
            return false;
        }
    }
    return true;
}

void TStaticCtrProvider::SetupBinFeatureIndexes(const TConstArrayRef<TFloatFeature> floatFeatures,
                                                const TConstArrayRef<TOneHotFeature> oheFeatures,
                                                const TConstArrayRef<TCatFeature> catFeatures) {
    ui32 currentIndex = 0;
    FloatFeatureIndexes.clear();
    for (const auto& floatFeature : floatFeatures) {
        if (!floatFeature.UsedInModel()) {
            continue;
        }
        for (size_t borderIdx = 0; borderIdx < floatFeature.Borders.size(); ++borderIdx) {
            TBinFeatureIndexValue featureIdx{currentIndex + (ui32)borderIdx / MAX_VALUES_PER_BIN, false, (ui8)((borderIdx % MAX_VALUES_PER_BIN)+ 1)};
            TFloatSplit split{floatFeature.Position.Index, floatFeature.Borders[borderIdx]};
            FloatFeatureIndexes[split] = featureIdx;
        }
        currentIndex += (floatFeature.Borders.size() + MAX_VALUES_PER_BIN - 1) / MAX_VALUES_PER_BIN;
    }
    OneHotFeatureIndexes.clear();
    for (const auto& oheFeature : oheFeatures) {
        for (size_t valueId = 0; valueId < oheFeature.Values.size(); ++valueId) {
            TBinFeatureIndexValue featureIdx{currentIndex + (ui32)valueId / MAX_VALUES_PER_BIN, true, (ui8)((valueId % MAX_VALUES_PER_BIN) + 1)};
            TOneHotSplit feature{oheFeature.CatFeatureIndex, oheFeature.Values[valueId]};
            OneHotFeatureIndexes[feature] = featureIdx;
        }
        currentIndex += (oheFeature.Values.size() + MAX_VALUES_PER_BIN - 1) / MAX_VALUES_PER_BIN;
    }
    CatFeatureIndex.clear();
    for (const auto& catFeature : catFeatures) {
        if (catFeature.UsedInModel()) {
            const int prevSize = CatFeatureIndex.ysize();
            CatFeatureIndex[catFeature.Position.Index] = prevSize;
        }
    }
}

TIntrusivePtr<ICtrProvider> TStaticCtrProvider::Clone() const {
    TIntrusivePtr<TStaticCtrProvider> result = new TStaticCtrProvider();
    result->CtrData = CtrData;
    return result;
}


/***
 * ATTENTION!
 * This function contains simple and mostly incorrect ctr values table merging approach.
 * In this version we just leave unique values as-is and are using averaging for intersection
 */
static void MergeBuckets(const TVector<const TCtrValueTable*>& tables, TCtrValueTable* target) {
    Y_ASSERT(!tables.empty());
    TVector<NCatboost::TDenseIndexHashView> indexViewers;
    THashSet<NCatboost::TBucket::THashType> uniqueHashes;
    for (const auto& table : tables) {
        indexViewers.emplace_back(table->GetIndexHashViewer());
        for (const auto bucket : indexViewers.back().GetBuckets()) {
            // make a copy because we can't pass a reference to an unaligned struct member to 'uniqueHashes' methods
            const auto bucketHash = bucket.Hash;
            if (bucketHash != NCatboost::TBucket::InvalidHashValue) {
                uniqueHashes.insert(bucketHash);
            }
        }
    }

    auto hashBuilder = target->GetIndexHashBuilder(uniqueHashes.size());
    switch (tables.back()->ModelCtrBase.CtrType)
    {
    case ECtrType::BinarizedTargetMeanValue:
    case ECtrType::FloatTargetMeanValue: {
            TVector<TConstArrayRef<TCtrMeanHistory>> meanHistories;
            for (const auto& table : tables) {
                meanHistories.emplace_back(table->GetTypedArrayRefForBlobData<TCtrMeanHistory>());
            }
            auto targetBuf = target->AllocateBlobAndGetArrayRef<TCtrMeanHistory>(uniqueHashes.size());
            for (auto hash : uniqueHashes) {
                TCtrMeanHistory value = {0.0f, 0};
                size_t count = 0;
                for (const auto viewerId : xrange(indexViewers.size())) {
                    auto index = indexViewers[viewerId].GetIndex(hash);
                    if (index != NCatboost::TDenseIndexHashView::NotFoundIndex) {
                        value.Add(meanHistories[viewerId][index]);
                        ++count;
                    }
                }
                value.Count /= count;
                value.Sum /= count;
                auto insertIndex = hashBuilder.AddIndex(hash);
                targetBuf[insertIndex] = value;
            }
        }
        break;
    case ECtrType::FeatureFreq:
    case ECtrType::Counter:
        {
            TVector<TConstArrayRef<int>> counters;
            for (const auto& table : tables) {
                counters.emplace_back(table->GetTypedArrayRefForBlobData<int>());
            }
            auto targetBuf = target->AllocateBlobAndGetArrayRef<int>(uniqueHashes.size());
            for (auto hash : uniqueHashes) {
                int value = 0;
                int count = 0;
                for (const auto viewerId : xrange(indexViewers.size())) {
                    auto index = indexViewers[viewerId].GetIndex(hash);
                    if (index != NCatboost::TDenseIndexHashView::NotFoundIndex) {
                        value += counters[viewerId][index];
                        ++count;
                    }
                }
                Y_ASSERT(count != 0);
                auto insertIndex = hashBuilder.AddIndex(hash);
                targetBuf[insertIndex] = value / count;
            }
            i64 denominatorSum = 0;
            for (const auto& table : tables) {
                denominatorSum += table->CounterDenominator;
            }
            target->CounterDenominator = denominatorSum / tables.size();
        }
        break;
    case ECtrType::Buckets:
    case ECtrType::Borders:
        {
            const auto targetClassesCount = tables.back()->TargetClassesCount;
            TVector<TConstArrayRef<int>> counters;
            for (const auto& table : tables) {
                counters.emplace_back(table->GetTypedArrayRefForBlobData<int>());
            }
            auto targetBuf = target->AllocateBlobAndGetArrayRef<int>(uniqueHashes.size() * tables.back()->TargetClassesCount);
            for (auto hash : uniqueHashes) {
                int count = 0;
                const auto insertIndex = hashBuilder.AddIndex(hash);
                TArrayRef<int> insertArrayView(&targetBuf[insertIndex * targetClassesCount], targetClassesCount);
                for (const auto viewerId : xrange(indexViewers.size())) {
                    auto index = indexViewers[viewerId].GetIndex(hash);
                    if (index != NCatboost::TDenseIndexHashView::NotFoundIndex) {
                        for (int targetClassId : xrange(targetClassesCount)) {
                            insertArrayView[targetClassId] += counters[viewerId][index * targetClassesCount + targetClassId];
                        }
                        ++count;
                    }
                }
                if (count > 1) {
                    for (auto& value : insertArrayView) {
                        value /= count;
                    }
                }
            }
            target->TargetClassesCount = targetClassesCount;
        }
        break;

    case ECtrType::CtrTypesCount:
        CB_ENSURE(false, "Unsupported CTR type");
    default:
        CB_ENSURE(false, "Unexpected CTR type");
    }
}

TIntrusivePtr<TStaticCtrProvider> MergeStaticCtrProvidersData(const TVector<const TStaticCtrProvider*>& providers, ECtrTableMergePolicy mergePolicy) {
    if (providers.empty()) {
        return TIntrusivePtr<TStaticCtrProvider>();
    }
    TIntrusivePtr<TStaticCtrProvider> result = new TStaticCtrProvider();
    if (providers.size() == 1) {
        result->CtrData = providers[0]->CtrData;
        return result;
    }
    THashMap<TModelCtrBase, TVector<const TCtrValueTable*>> valuesMap;
    THashMap<TModelCtrBaseMergeKey, TCtrTablesMergeStatus> ctrTablesIndices;
    for (auto idx : xrange(providers.size())) {
        const auto& provider = providers[idx];
        for (const auto& [ctrBase, ctrValueTables] : provider->CtrData.LearnCtrs) {
            if (mergePolicy == ECtrTableMergePolicy::KeepAllTables) {
                auto updatedBase = ctrBase;
                updatedBase.TargetBorderClassifierIdx = ctrTablesIndices[ctrBase].GetResultIndex(ctrBase.TargetBorderClassifierIdx);
                valuesMap[updatedBase].push_back(&ctrValueTables);
            } else {
                valuesMap[ctrBase].push_back(&ctrValueTables);
            }
        }
        if (mergePolicy == ECtrTableMergePolicy::KeepAllTables) {
            for (auto& [key, status] : ctrTablesIndices) {
                status.FinishModel();
            }
        }
    }
    for (const auto& [ctrBase, ctrValueTables] : valuesMap) {
        if (ctrValueTables.size() == 1) {
            result->CtrData.LearnCtrs[ctrBase] = *(ctrValueTables[0]);
            continue;
        }
        switch (mergePolicy) {
            case ECtrTableMergePolicy::KeepAllTables:
                CB_ENSURE_INTERNAL(false, "KeepAllTables policy encountered table duplicates when merging");
            case ECtrTableMergePolicy::FailIfCtrIntersects:
                throw TCatBoostException() << "FailIfCtrIntersects policy forbids model ctr tables intersection";
            case ECtrTableMergePolicy::LeaveMostDiversifiedTable: {
                size_t maxCtrTableSize = 0;
                const TCtrValueTable* maxTable = nullptr;
                for (const auto* valueTable : ctrValueTables) {
                    auto ctrSize = valueTable->GetIndexHashViewer().CountNonEmptyBuckets();
                    if (ctrSize > maxCtrTableSize) {
                        maxCtrTableSize = ctrSize;
                        maxTable = valueTable;
                    }
                }
                Y_ASSERT(maxTable != nullptr);
                result->CtrData.LearnCtrs[ctrBase] = *maxTable;
                break;
            }
            case ECtrTableMergePolicy::IntersectingCountersAverage: {
                auto& target = result->CtrData.LearnCtrs[ctrBase];
                target.ModelCtrBase = ctrBase;
                MergeBuckets(ctrValueTables, &target);
                break;
            }
            default:
                CB_ENSURE(false, "Unexpected CTR table merge policy");
        }
    }
    return result;
}
