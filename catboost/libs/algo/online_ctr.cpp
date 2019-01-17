#include "online_ctr.h"
#include "index_hash_calcer.h"
#include "fold.h"
#include "learn_context.h"
#include "score_calcer.h"
#include "tree_print.h"

#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/resource_constrained_executor.h>
#include <catboost/libs/model/model.h>

#include <util/generic/bitops.h>
#include <util/generic/utility.h>
#include <util/stream/format.h>
#include <util/system/mem_info.h>
#include <util/thread/singleton.h>

#include <numeric>


using namespace NCB;


bool HasFeaturesForCtrs(const NCB::TQuantizedFeaturesInfo& quantizedFeaturesInfo, ui32 oneHotMaxSize) {
    const auto& featuresLayout = *quantizedFeaturesInfo.GetFeaturesLayout();

    bool hasFeaturesForCtrs = false;
    featuresLayout.IterateOverAvailableFeatures<EFeatureType::Categorical>(
        [&] (TCatFeatureIdx catFeatureIdx) {
            if (quantizedFeaturesInfo.GetUniqueValuesCounts(catFeatureIdx).OnLearnOnly > oneHotMaxSize) {
                hasFeaturesForCtrs = true;
            }
        }
    );
    return hasFeaturesForCtrs;
}


struct TCtrCalcer {
    template <typename T>
    T* Alloc(size_t count) {
        static_assert(std::is_pod<T>::value, "expected POD type");
        const size_t neededSize = count * sizeof(T);
        if (neededSize > Storage.size()) {
            Storage.yresize(neededSize);
        }
        Fill(Storage.begin(), Storage.end(), 0);
        return (T*)Storage.data();
    }
    static inline TArrayRef<TCtrMeanHistory> GetCtrMeanHistoryArr(size_t maxCount) {
        return TArrayRef<TCtrMeanHistory>(FastTlsSingleton<TCtrCalcer>()->Alloc<TCtrMeanHistory>(maxCount), maxCount);
    }
    static inline TArrayRef<TCtrHistory> GetCtrHistoryArr(size_t maxCount) {
        return TArrayRef<TCtrHistory>(FastTlsSingleton<TCtrCalcer>()->Alloc<TCtrHistory>(maxCount), maxCount);
    }
    static inline int* GetCtrArrTotal(size_t maxCount) {
        return FastTlsSingleton<TCtrCalcer>()->Alloc<int>(maxCount);
    }

private:
    TVector<char> Storage;
};

struct TBucketsView {
    size_t MaxElem = 0;
    size_t BorderCount = 0;
    int* Data = 0;
    int* BucketData = 0;
    TBucketsView(size_t maxElem, size_t borderCount)
        : MaxElem(maxElem)
        , BorderCount(borderCount) {
        Data = FastTlsSingleton<TCtrCalcer>()->Alloc<int>(MaxElem * (BorderCount + 1));
        BucketData = Data + MaxElem;
    }
    inline int& GetTotal(size_t i) {
        Y_ASSERT(i < MaxElem);
        return *(Data + i);
    }
    inline TArrayRef<int> GetBorders(size_t i) {
        Y_ASSERT(i < MaxElem);
        return TArrayRef<int>(BucketData + BorderCount * i, BorderCount);
    }
};

void CalcNormalization(const TVector<float>& priors, TVector<float>* shift, TVector<float>* norm) {
    shift->yresize(priors.size());
    norm->yresize(priors.size());
    for (int i = 0; i < priors.ysize(); ++i) {
        float prior = priors[i];
        float left = Min(0.0f, prior);
        float right = Max(1.0f, prior);
        (*shift)[i] = -left;
        (*norm)[i] = (right - left);
    }
}



static void UpdateGoodCount(int curCount, ECtrType ctrType, int* goodCount) {
    if (ctrType == ECtrType::Buckets) {
        *goodCount = curCount;
    } else {
        *goodCount -= curCount;
    }
}

namespace {
    // Blocked 2-stage calculation director
    class TBlockedCalcer {
    public:
        explicit TBlockedCalcer(int blockSize)
            : BlockSize(blockSize) {
        }
        template <typename TCalc1, typename TCalc2>
        void Calc(TCalc1 calc1, TCalc2 calc2, int docOffset, int docCount) {
            for (int blockStart = 0; blockStart < docCount; blockStart += BlockSize) {
                const int nextBlockStart = Min<int>(docCount, blockStart + BlockSize);
                calc1(blockStart, nextBlockStart, docOffset);
                calc2(blockStart, nextBlockStart, docOffset);
            }
        }

    private:
        const int BlockSize;
    };
}

static void CalcOnlineCTRClasses(const TVector<size_t>& testOffsets,
                                 const TVector<ui64>& enumeratedCatFeatures,
                                 size_t leafCount,
                                 const TVector<int>& permutedTargetClass,
                                 int targetClassesCount, int targetBorderCount,
                                 const TVector<float>& priors,
                                 int ctrBorderCount,
                                 ECtrType ctrType,
                                 TArray2D<TVector<ui8>>* feature) {
    TVector<float> shift;
    TVector<float> norm;
    CalcNormalization(priors, &shift, &norm);

    const int blockSize = (1000 + targetBorderCount - 1) / targetBorderCount + 100; // ensure blocks have reasonable size
    TVector<int> totalCountByDoc(blockSize);
    TVector<TVector<int>> goodCountByBorderByDoc(targetBorderCount, TVector<int>(blockSize));
    TBucketsView bv(leafCount, targetClassesCount);

    auto calcGoodCounts = [&](int blockStart, int nextBlockStart, int docOffset) {
        for (int docId = blockStart; docId < nextBlockStart; ++docId) {
            const auto elemId = enumeratedCatFeatures[docOffset + docId];

            int goodCount = totalCountByDoc[docId - blockStart] = bv.GetTotal(elemId);
            auto bordersData = bv.GetBorders(elemId);
            for (int border = 0; border < targetBorderCount; ++border) {
                UpdateGoodCount(bordersData[border], ctrType, &goodCount);
                goodCountByBorderByDoc[border][docId - blockStart] = goodCount;
            }

            if (docOffset == 0) {
                ++bordersData[permutedTargetClass[docId]];
                ++bv.GetTotal(elemId);
            }
        }
    };

    auto calcCTRs = [&](int blockStart, int nextBlockStart, int docOffset) {
        for (int border = 0; border < targetBorderCount; ++border) {
            for (int prior = 0; prior < priors.ysize(); ++prior) {
                const float priorX = priors[prior];
                const float shiftX = shift[prior];
                const float normX = norm[prior];
                const int* goodCountData = goodCountByBorderByDoc[border].data();
                ui8* featureData = docOffset + (*feature)[border][prior].data();
                for (int docId = blockStart; docId < nextBlockStart; ++docId) {
                    featureData[docId] = CalcCTR(goodCountData[docId - blockStart], totalCountByDoc[docId - blockStart],
                                                 priorX, shiftX, normX, ctrBorderCount);
                }
            }
        }
    };

    TBlockedCalcer calcer(blockSize);
    const size_t learnSampleCount = testOffsets[0];
    calcer.Calc(calcGoodCounts, calcCTRs, 0, learnSampleCount);
    for (size_t docOffset = learnSampleCount, testIdx = 0; testIdx < testOffsets.size() - 1; ++testIdx) {
        const size_t testSampleCount = testOffsets[testIdx + 1] - testOffsets[testIdx];
        calcer.Calc(calcGoodCounts, calcCTRs, docOffset, testSampleCount);
        docOffset += testSampleCount;
    }
}

static void CalcOnlineCTRSimple(const TVector<size_t>& testOffsets,
                                const TVector<ui64>& enumeratedCatFeatures,
                                size_t leafCount,
                                const TVector<int>& permutedTargetClass,
                                const TVector<float>& priors,
                                int ctrBorderCount,
                                TArray2D<TVector<ui8>>* feature) {
    TVector<float> shift;
    TVector<float> norm;
    CalcNormalization(priors, &shift, &norm);

    const int blockSize = 1000;
    auto ctrArrSimple = TCtrCalcer::GetCtrHistoryArr(leafCount + blockSize);
    auto totalCount = reinterpret_cast<int*>(ctrArrSimple.data() + leafCount);
    auto goodCount = totalCount + blockSize;

    auto calcGoodCount = [&](int blockStart, int nextBlockStart, int docOffset) {
        for (int docId = blockStart; docId < nextBlockStart; ++docId) {
            TCtrHistory& elem = ctrArrSimple[enumeratedCatFeatures[docOffset + docId]];
            goodCount[docId - blockStart] = elem.N[1];
            totalCount[docId - blockStart] = elem.N[0] + elem.N[1];
            if (docOffset == 0) {
                ++elem.N[permutedTargetClass[docId]];
            }
        }
    };

    auto calcCTRs = [&](int blockStart, int nextBlockStart, int docOffset) {
        for (int prior = 0; prior < priors.ysize(); ++prior) {
            const float priorX = priors[prior];
            const float shiftX = shift[prior];
            const float normX = norm[prior];
            ui8* featureData = docOffset + (*feature)[0][prior].data();
            for (int docId = blockStart; docId < nextBlockStart; ++docId) {
                featureData[docId] = CalcCTR(goodCount[docId - blockStart], totalCount[docId - blockStart],
                                             priorX, shiftX, normX, ctrBorderCount);
            }
        }
    };

    TBlockedCalcer calcer(blockSize);
    const size_t learnSampleCount = testOffsets[0];
    calcer.Calc(calcGoodCount, calcCTRs, 0, learnSampleCount);
    for (size_t docOffset = learnSampleCount, testIdx = 0; testIdx < testOffsets.size() - 1; ++testIdx) {
        const size_t testSampleCount = testOffsets[testIdx + 1] - testOffsets[testIdx];
        calcer.Calc(calcGoodCount, calcCTRs, docOffset, testSampleCount);
        docOffset += testSampleCount;
    }
}

static void CalcOnlineCTRMean(const TVector<size_t>& testOffsets,
                              const TVector<ui64>& enumeratedCatFeatures,
                              size_t leafCount,
                              const TVector<int>& permutedTargetClass,
                              int targetBorderCount,
                              const TVector<float>& priors,
                              int ctrBorderCount,
                              TArray2D<TVector<ui8>>* feature) {
    TVector<float> shift;
    TVector<float> norm;
    CalcNormalization(priors, &shift, &norm);

    const int blockSize = 1000;
    TVector<float> sum(blockSize);
    TVector<int> count(blockSize);
    auto ctrArrMean = TCtrCalcer::GetCtrMeanHistoryArr(leafCount);

    auto calcCount = [&](int blockStart, int nextBlockStart, int docOffset) {
        for (int docId = blockStart; docId < nextBlockStart; ++docId) {
            TCtrMeanHistory& elem = ctrArrMean[enumeratedCatFeatures[docOffset + docId]];
            sum[docId - blockStart] = elem.Sum;
            count[docId - blockStart] = elem.Count;
            if (docOffset == 0) {
                elem.Add(static_cast<float>(permutedTargetClass[docId]) / targetBorderCount);
            }
        }
    };

    auto calcCTRs = [&](int blockStart, int nextBlockStart, int docOffset) {
        for (int prior = 0; prior < priors.ysize(); ++prior) {
            const float priorX = priors[prior];
            const float shiftX = shift[prior];
            const float normX = norm[prior];
            ui8* featureData = docOffset + (*feature)[0][prior].data();
            for (int docId = blockStart; docId < nextBlockStart; ++docId) {
                featureData[docId] = CalcCTR(sum[docId - blockStart], count[docId - blockStart],
                                             priorX, shiftX, normX, ctrBorderCount);
            }
        }
    };

    TBlockedCalcer calcer(blockSize);
    const size_t learnSampleCount = testOffsets[0];
    calcer.Calc(calcCount, calcCTRs, 0, learnSampleCount);
    for (size_t docOffset = learnSampleCount, testIdx = 0; testIdx < testOffsets.size() - 1; ++testIdx) {
        const size_t testSampleCount = testOffsets[testIdx + 1] - testOffsets[testIdx];
        calcer.Calc(calcCount, calcCTRs, docOffset, testSampleCount);
        docOffset += testSampleCount;
    }
}

static void CalcOnlineCTRCounter(const TVector<size_t>& testOffsets,
                                 const TVector<int>& counterCTRTotal,
                                 const TVector<ui64>& enumeratedCatFeatures,
                                 int denominator,
                                 const TVector<float>& priors,
                                 int ctrBorderCount,
                                 TArray2D<TVector<ui8>>* feature) {
    TVector<float> shift;
    TVector<float> norm;
    CalcNormalization(priors, &shift, &norm);

    const int blockSize = 1000;
    auto ctrTotal = TCtrCalcer::GetCtrArrTotal(blockSize);
    auto calcTotal = [&](int blockStart, int nextBlockStart, int docOffset) {
        for (int docId = blockStart; docId < nextBlockStart; ++docId) {
            const auto elemId = enumeratedCatFeatures[docOffset + docId];
            ctrTotal[docId - blockStart] = counterCTRTotal[elemId];
        }
    };

    auto calcCTRs = [&](int blockStart, int nextBlockStart, int docOffset) {
        for (int prior = 0; prior < priors.ysize(); ++prior) {
            const float priorX = priors[prior];
            const float shiftX = shift[prior];
            const float normX = norm[prior];
            ui8* featureData = docOffset + (*feature)[0][prior].data();
            for (int docId = blockStart; docId < nextBlockStart; ++docId) {
                featureData[docId] = CalcCTR(ctrTotal[docId - blockStart], denominator, priorX, shiftX, normX, ctrBorderCount);
            }
        }
    };

    TBlockedCalcer calcer(blockSize);
    const size_t learnSampleCount = testOffsets[0];
    calcer.Calc(calcTotal, calcCTRs, 0, learnSampleCount);
    for (size_t docOffset = learnSampleCount, testIdx = 0; testIdx < testOffsets.size() - 1; ++testIdx) {
        const size_t testSampleCount = testOffsets[testIdx + 1] - testOffsets[testIdx];
        calcer.Calc(calcTotal, calcCTRs, docOffset, testSampleCount);
        docOffset += testSampleCount;
    }
}

static inline void CountOnlineCTRTotal(const TVector<ui64>& hashArr, int sampleCount, TVector<int>* counterCTRTotal) {
    for (int sampleIdx = 0; sampleIdx < sampleCount; ++sampleIdx) {
        const auto elemId = hashArr[sampleIdx];
        ++(*counterCTRTotal)[elemId];
    }
}

void ComputeOnlineCTRs(const TTrainingForCPUDataProviders& data,
                       const TFold& fold,
                       const TProjection& proj,
                       const TLearnContext* ctx,
                       TOnlineCTR* dst) {
    const TCtrHelper& ctrHelper = ctx->CtrsHelper;
    const auto& ctrInfo = ctrHelper.GetCtrInfo(proj);
    dst->Feature.resize(ctrInfo.size());
    size_t learnSampleCount = data.Learn->GetObjectCount();
    const TVector<size_t>& testOffsets = data.CalcTestOffsets();
    size_t totalSampleCount = learnSampleCount + data.GetTestSampleCount();

    const auto& quantizedFeaturesInfo = *data.Learn->ObjectsData->GetQuantizedFeaturesInfo();

    using THashArr = TVector<ui64>;
    using TRehashHash = TDenseHash<ui64, ui32>;
    Y_STATIC_THREAD(THashArr) tlsHashArr;
    Y_STATIC_THREAD(TRehashHash) rehashHashTlsVal;
    TVector<ui64>& hashArr = tlsHashArr.Get();
    if (proj.IsSingleCatFeature()) {
        // Shortcut for simple ctrs
        Clear(&hashArr, totalSampleCount);
        TArrayRef<ui64> hashArrView = hashArr;
        if (learnSampleCount > 0) {
            SubsetWithAlternativeIndexing(
                data.Learn->ObjectsData->GetCatFeature((ui32)proj.CatFeatures[0]),
                &fold.LearnPermutationFeaturesSubset
            ).ForEach(
                [hashArrView] (ui32 i, ui32 featureValue) {
                    hashArrView[i] = (ui64)featureValue + 1;
                }
            );
        }
        for (size_t docOffset = learnSampleCount, testIdx = 0; docOffset < totalSampleCount && testIdx < data.Test.size(); ++testIdx) {
            const size_t testSampleCount = data.Test[testIdx]->GetObjectCount();
            (*data.Test[testIdx]->ObjectsData->GetCatFeature((ui32)proj.CatFeatures[0]))->GetArrayData()
                .ForEach(
                    [hashArrView, docOffset] (ui32 i, ui32 featureValue) {
                        hashArrView[docOffset + i] = (ui64)featureValue + 1;
                    }
                );

            docOffset += testSampleCount;
        }
        rehashHashTlsVal.Get().MakeEmpty(
            quantizedFeaturesInfo.GetUniqueValuesCounts(TCatFeatureIdx(proj.CatFeatures[0])).OnLearnOnly
        );
    } else {
        Clear(&hashArr, totalSampleCount);
        CalcHashes(
            proj,
            *data.Learn->ObjectsData,
            fold.LearnPermutationFeaturesSubset,
            nullptr,
            hashArr.begin(),
            hashArr.begin() + learnSampleCount);
        for (size_t docOffset = learnSampleCount, testIdx = 0; docOffset < totalSampleCount && testIdx < data.Test.size(); ++testIdx) {
            const size_t testSampleCount = data.Test[testIdx]->GetObjectCount();
            CalcHashes(
                proj,
                *data.Test[testIdx]->ObjectsData,
                data.Test[testIdx]->ObjectsData->GetFeaturesArraySubsetIndexing(),
                nullptr,
                hashArr.begin() + docOffset,
                hashArr.begin() + docOffset + testSampleCount);
            docOffset += testSampleCount;
        }
        size_t approxBucketsCount = 1;
        for (auto cf : proj.CatFeatures) {
            approxBucketsCount *= quantizedFeaturesInfo.GetUniqueValuesCounts(TCatFeatureIdx(cf)).OnLearnOnly;
            if (approxBucketsCount > learnSampleCount) {
                break;
            }
        }
        rehashHashTlsVal.Get().MakeEmpty(Min(learnSampleCount, approxBucketsCount));
    }
    ui64 topSize = ctx->Params.CatFeatureParams->CtrLeafCountLimit;
    if (proj.IsSingleCatFeature() && ctx->Params.CatFeatureParams->StoreAllSimpleCtrs) {
        topSize = Max<ui64>();
    }
    auto leafCount = ComputeReindexHash(topSize, rehashHashTlsVal.GetPtr(), hashArr.begin(), hashArr.begin() + learnSampleCount);
    dst->CounterUniqueValuesCount = dst->UniqueValuesCount = leafCount;

    for (size_t docOffset = learnSampleCount, testIdx = 0; docOffset < totalSampleCount && testIdx < data.Test.size(); ++testIdx) {
        const size_t testSampleCount = data.Test[testIdx]->GetObjectCount();
        leafCount = UpdateReindexHash(rehashHashTlsVal.GetPtr(), hashArr.begin() + docOffset, hashArr.begin() + docOffset + testSampleCount);
        docOffset += testSampleCount;
    }

    TVector<int> counterCTRTotal;
    int counterCTRDenominator = 0;
    if (AnyOf(ctrInfo.begin(), ctrInfo.begin() + dst->Feature.ysize(), [] (const auto& info) { return info.Type == ECtrType::Counter; })) {
        counterCTRTotal.resize(leafCount);
        int sampleCount = learnSampleCount;
        if (ctx->Params.CatFeatureParams->CounterCalcMethod == ECounterCalc::Full) {
            dst->CounterUniqueValuesCount = leafCount;
            sampleCount = hashArr.ysize();
        }
        CountOnlineCTRTotal(hashArr, sampleCount, &counterCTRTotal);
        counterCTRDenominator = *MaxElement(counterCTRTotal.begin(), counterCTRTotal.end());
    }

    for (int ctrIdx = 0; ctrIdx < dst->Feature.ysize(); ++ctrIdx) {
        const ECtrType ctrType = ctrInfo[ctrIdx].Type;
        const ui32 classifierId = ctrInfo[ctrIdx].TargetClassifierIdx;
        int targetClassesCount = fold.TargetClassesCount[classifierId];

        const ui32 targetBorderCount = GetTargetBorderCount(ctrInfo[ctrIdx], targetClassesCount);
        const ui32 ctrBorderCount = ctrInfo[ctrIdx].BorderCount;
        const auto& priors = ctrInfo[ctrIdx].Priors;
        dst->Feature[ctrIdx].SetSizes(priors.size(), targetBorderCount);

        for (ui32 border = 0; border < targetBorderCount; ++border) {
            for (int prior = 0; prior < priors.ysize(); ++prior) {
                Clear(&dst->Feature[ctrIdx][border][prior], totalSampleCount);
            }
        }

        if (ctrType == ECtrType::Borders && targetClassesCount == SIMPLE_CLASSES_COUNT) {
            CalcOnlineCTRSimple(
                testOffsets,
                hashArr,
                leafCount,
                fold.LearnTargetClass[classifierId],
                priors,
                ctrBorderCount,
                &dst->Feature[ctrIdx]);

        } else if (ctrType == ECtrType::BinarizedTargetMeanValue) {
            CalcOnlineCTRMean(
                testOffsets,
                hashArr,
                leafCount,
                fold.LearnTargetClass[classifierId],
                targetClassesCount - 1,
                priors,
                ctrBorderCount,
                &dst->Feature[ctrIdx]);

        } else if (ctrType == ECtrType::Buckets ||
                   (ctrType == ECtrType::Borders && targetClassesCount > SIMPLE_CLASSES_COUNT)) {
            CalcOnlineCTRClasses(
                testOffsets,
                hashArr,
                leafCount,
                fold.LearnTargetClass[classifierId],
                targetClassesCount,
                GetTargetBorderCount(ctrInfo[ctrIdx], targetClassesCount),
                priors,
                ctrBorderCount,
                ctrType,
                &dst->Feature[ctrIdx]);
        } else {
            Y_ASSERT(ctrType == ECtrType::Counter);
            CalcOnlineCTRCounter(
                testOffsets,
                counterCTRTotal,
                hashArr,
                counterCTRDenominator,
                priors,
                ctrBorderCount,
                &dst->Feature[ctrIdx]);
        }
    }
}

void CalcFinalCtrsImpl(
    const ECtrType ctrType,
    const ui64 ctrLeafCountLimit,
    const TVector<int>& targetClass,
    TConstArrayRef<float> targets,
    const ui32 totalSampleCount,
    int targetClassesCount,
    TVector<ui64>* hashArr,
    TCtrValueTable* result
) {
    Y_ASSERT(hashArr->size() == (size_t)totalSampleCount);

    size_t leafCount = 0;
    {
        TDenseHash<ui64, ui32> tmpHash;
        leafCount = ComputeReindexHash(ctrLeafCountLimit, &tmpHash, hashArr->begin(), hashArr->begin() + totalSampleCount);
        auto hashIndexBuilder = result->GetIndexHashBuilder(leafCount);
        for (const auto& kv : tmpHash) {
            hashIndexBuilder.SetIndex(kv.first, kv.second);
        }
    }

    TArrayRef<int> ctrIntArray;
    TArrayRef<TCtrMeanHistory> ctrMean;
    if (ctrType == ECtrType::BinarizedTargetMeanValue || ctrType == ECtrType::FloatTargetMeanValue) {
        ctrMean = result->AllocateBlobAndGetArrayRef<TCtrMeanHistory>(leafCount);
    } else if (ctrType == ECtrType::Counter || ctrType == ECtrType::FeatureFreq) {
        ctrIntArray = result->AllocateBlobAndGetArrayRef<int>(leafCount);
        result->CounterDenominator = 0;
    } else {
        result->TargetClassesCount = targetClassesCount;
        ctrIntArray = result->AllocateBlobAndGetArrayRef<int>(leafCount * targetClassesCount);
    }


    int targetBorderCount = targetClassesCount - 1;
    auto hashArrPtr = hashArr->data();
    for (ui32 z = 0; z < totalSampleCount; ++z) {
        const ui64 elemId = hashArrPtr[z];
        if (ctrType == ECtrType::BinarizedTargetMeanValue) {
            TCtrMeanHistory& elem = ctrMean[elemId];
            elem.Add(static_cast<float>(targetClass[z]) / targetBorderCount);
        } else if (ctrType == ECtrType::Counter || ctrType == ECtrType::FeatureFreq) {
            ++ctrIntArray[elemId];
        } else if (ctrType == ECtrType::FloatTargetMeanValue) {
            TCtrMeanHistory& elem = ctrMean[elemId];
            elem.Add(targets[z]);
        } else {
            TArrayRef<int> elem = MakeArrayRef(ctrIntArray.data() + targetClassesCount * elemId, targetClassesCount);
            ++elem[targetClass[z]];
        }
    }

    if (ctrType == ECtrType::Counter) {
        result->CounterDenominator = *MaxElement(ctrIntArray.begin(), ctrIntArray.end());
    }
    if (ctrType == ECtrType::FeatureFreq) {
        result->CounterDenominator = static_cast<int>(totalSampleCount);
    }
}


static void CalcFinalCtrs(
    const ECtrType ctrType,
    const TProjection& projection,
    const TDatasetDataForFinalCtrs& datasetDataForFinalCtrs,
    const NCB::TFeaturesArraySubsetIndexing& learnFeaturesSubsetIndexing,
    const NCB::TPerfectHashedToHashedCatValuesMap& perfectHashedToHashedCatValuesMap,
    int targetBorderClassifierIdx,
    ui64 ctrLeafCountLimit,
    bool storeAllSimpleCtr,
    ECounterCalc counterCalcMethod,
    TCtrValueTable* result
) {
    ui32 learnSampleCount = datasetDataForFinalCtrs.Data.Learn->GetObjectCount();
    ui32 totalSampleCount = learnSampleCount;
    if (ctrType == ECtrType::Counter && counterCalcMethod == ECounterCalc::Full) {
        totalSampleCount += datasetDataForFinalCtrs.Data.GetTestSampleCount();
    }
    TVector<ui64> hashArr(totalSampleCount);
    CalcHashes(
        projection,
        *datasetDataForFinalCtrs.Data.Learn->ObjectsData,
        learnFeaturesSubsetIndexing,
        &perfectHashedToHashedCatValuesMap,
        hashArr.begin(),
        hashArr.begin() + learnSampleCount
    );
    if (totalSampleCount > learnSampleCount) {
        ui64* testHashBegin = hashArr.begin() + learnSampleCount;
        for (const auto& testDataPtr : datasetDataForFinalCtrs.Data.Test) {
            ui64* testHashEnd = testHashBegin + testDataPtr->GetObjectCount();
            CalcHashes(
                projection,
                *testDataPtr->ObjectsData,
                testDataPtr->ObjectsData->GetFeaturesArraySubsetIndexing(),
                &perfectHashedToHashedCatValuesMap,
                testHashBegin,
                testHashEnd);
            testHashBegin = testHashEnd;
        }
    }

    if (projection.IsSingleCatFeature() && storeAllSimpleCtr) {
        ctrLeafCountLimit = Max<ui64>();
    }
    CalcFinalCtrsImpl(
        ctrType,
        ctrLeafCountLimit,
        NeedTargetClassifier(ctrType) ?
            (**datasetDataForFinalCtrs.LearnTargetClass)[targetBorderClassifierIdx] : TVector<int>(),
        *datasetDataForFinalCtrs.Targets,
        totalSampleCount,
        NeedTargetClassifier(ctrType) ?
            (**datasetDataForFinalCtrs.TargetClassesCount)[targetBorderClassifierIdx] : 0,
        &hashArr,
        result
    );
}

static ui64 EstimateCalcFinalCtrsCpuRamUsage(
    const ECtrType ctrType,
    const TTrainingForCPUDataProviders& data,
    int targetClassesCount,
    ui64 ctrLeafCountLimit,
    ECounterCalc counterCalcMethod
) {
    ui64 cpuRamUsageEstimate = 0;

    ui32 totalSampleCount = data.Learn->GetObjectCount();
    if (ctrType == ECtrType::Counter && counterCalcMethod == ECounterCalc::Full) {
        totalSampleCount += data.GetTestSampleCount();
    }
    // for hashArr in CalcFinalCtrs
    cpuRamUsageEstimate += sizeof(ui64)*totalSampleCount;

    ui64 reindexHashRamLimit =
        sizeof(TDenseHash<ui64,ui32>::value_type)*FastClp2(totalSampleCount*2);

    // data for temporary vector to calc top
    ui64 computeReindexHashTopRamLimit = (ctrLeafCountLimit < totalSampleCount) ?
        sizeof(std::pair<ui64, ui32>)*totalSampleCount : 0;

    // CalcFinalCtrsImpl stage 1
    ui64 computeReindexHashRamLimit = reindexHashRamLimit + computeReindexHashTopRamLimit;

    ui64 reindexHashAfterComputeSizeLimit = Min<ui64>(ctrLeafCountLimit, totalSampleCount);

    ui64 reindexHashAfterComputeRamLimit =
        sizeof(TDenseHash<ui64,ui32>::value_type)*FastClp2(reindexHashAfterComputeSizeLimit*2);


    ui64 indexBucketsRamLimit = (sizeof(NCatboost::TBucket) *
         NCatboost::TDenseIndexHashBuilder::GetProperBucketsCount(reindexHashAfterComputeSizeLimit)
    );

    // CalcFinalCtrsImplstage 2
    ui64 buildingHashIndexRamLimit = reindexHashAfterComputeRamLimit + indexBucketsRamLimit;

    ui64 ctrBlobRamLimit = 0;
    if (ctrType == ECtrType::BinarizedTargetMeanValue || ctrType == ECtrType::FloatTargetMeanValue) {
        ctrBlobRamLimit = sizeof(TCtrMeanHistory)*reindexHashAfterComputeSizeLimit;
    } else if (ctrType == ECtrType::Counter || ctrType == ECtrType::FeatureFreq) {
        ctrBlobRamLimit = sizeof(int)*reindexHashAfterComputeSizeLimit;
    } else {
        ctrBlobRamLimit = sizeof(int)*(reindexHashAfterComputeSizeLimit * targetClassesCount);
    }

    // CalcFinalCtrsImplstage 3
    ui64 fillingCtrBlobRamLimit = indexBucketsRamLimit + ctrBlobRamLimit;

    // max usage is max of CalcFinalCtrsImpl 3 stages
    cpuRamUsageEstimate += Max(computeReindexHashRamLimit, buildingHashIndexRamLimit, fillingCtrBlobRamLimit);

    return cpuRamUsageEstimate;
}

void CalcFinalCtrsAndSaveToModel(
    ui64 cpuRamLimit,
    NPar::TLocalExecutor& localExecutor,
    const THashMap<TFeatureCombination, TProjection>& featureCombinationToProjectionMap,
    const TDatasetDataForFinalCtrs& datasetDataForFinalCtrs,
    const NCB::TPerfectHashedToHashedCatValuesMap& perfectHashedToHashedCatValuesMap,
    ui64 ctrLeafCountLimit,
    bool storeAllSimpleCtrs,
    ECounterCalc counterCalcMethod,
    const TVector<TModelCtrBase>& usedCtrBases,
    std::function<void(TCtrValueTable&& table)>&& asyncCtrValueTableCallback
) {
    CATBOOST_DEBUG_LOG << "Started parallel calculation of " << usedCtrBases.size() << " unique ctrs" << Endl;

    TMaybe<TFeaturesArraySubsetIndexing> permutedLearnFeaturesSubsetIndexing;
    const TFeaturesArraySubsetIndexing* learnFeaturesSubsetIndexing = nullptr;
    if (datasetDataForFinalCtrs.LearnPermutation) {
        permutedLearnFeaturesSubsetIndexing = Compose(
            datasetDataForFinalCtrs.Data.Learn->ObjectsData->GetFeaturesArraySubsetIndexing(),
            **datasetDataForFinalCtrs.LearnPermutation
        );
        learnFeaturesSubsetIndexing = &*permutedLearnFeaturesSubsetIndexing;
    } else {
        learnFeaturesSubsetIndexing =
            &datasetDataForFinalCtrs.Data.Learn->ObjectsData->GetFeaturesArraySubsetIndexing();
    }


    ui64 cpuRamUsage = NMemInfo::GetMemInfo().RSS;

    if (cpuRamUsage > cpuRamLimit) {
        CATBOOST_WARNING_LOG << "CatBoost is using more CPU RAM ("
            << HumanReadableSize(cpuRamUsage, SF_BYTES)
            << ") than the limit (" << HumanReadableSize(cpuRamLimit, SF_BYTES) << ")\n";
    }

    {
        NCB::TResourceConstrainedExecutor finalCtrExecutor(
            localExecutor,
            "CPU RAM",
            cpuRamLimit - Min(cpuRamLimit, cpuRamUsage),
            true
        );

        const auto& layout = *datasetDataForFinalCtrs.Data.Learn->MetaInfo.FeaturesLayout;

        auto ctrTableGenerator = [&] (const TModelCtrBase& ctr) -> TCtrValueTable {
            TCtrValueTable resTable;
            CalcFinalCtrs(
                ctr.CtrType,
                featureCombinationToProjectionMap.at(ctr.Projection),
                datasetDataForFinalCtrs,
                *learnFeaturesSubsetIndexing,
                perfectHashedToHashedCatValuesMap,
                ctr.TargetBorderClassifierIdx,
                ctrLeafCountLimit,
                storeAllSimpleCtrs,
                counterCalcMethod,
                &resTable
            );
            resTable.ModelCtrBase = ctr;
            CATBOOST_DEBUG_LOG << "Finished CTR: " << ctr.CtrType << " "
                                << BuildDescription(layout, ctr.Projection) << Endl;
            return resTable;
        };

        for (const auto& ctr : usedCtrBases) {
            finalCtrExecutor.Add(
                {
                    EstimateCalcFinalCtrsCpuRamUsage(
                        ctr.CtrType,
                        datasetDataForFinalCtrs.Data,
                        NeedTargetClassifier(ctr.CtrType) ?
                            (**datasetDataForFinalCtrs.TargetClassesCount)[ctr.TargetBorderClassifierIdx]
                            : 0,
                        ctrLeafCountLimit,
                        counterCalcMethod
                    ),
                    [&asyncCtrValueTableCallback, &ctrTableGenerator, &ctr] () {
                        auto table = ctrTableGenerator(ctr);
                        asyncCtrValueTableCallback(std::move(table));
                    }
                }
            );
        }

        finalCtrExecutor.ExecTasks();
    }

    CATBOOST_DEBUG_LOG << "CTR calculation finished" << Endl;
}

