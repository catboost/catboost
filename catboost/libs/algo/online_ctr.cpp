#include "online_ctr.h"
#include "index_hash_calcer.h"
#include "fold.h"
#include "learn_context.h"
#include "score_calcer.h"

#include <catboost/libs/model/model.h>
#include <util/generic/utility.h>
#include <util/thread/singleton.h>


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

static void CalcOnlineCTRClasses(const TTrainData& data,
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
    const int docCount = enumeratedCatFeatures.ysize();
    for (int blockStart = 0; blockStart < docCount; blockStart += blockSize) {
        const int nextBlockStart = Min<int>(docCount, blockStart + blockSize);
        for (int docId = blockStart; docId < nextBlockStart; ++docId) {
            const auto elemId = enumeratedCatFeatures[docId];

            totalCountByDoc[docId - blockStart] = bv.GetTotal(elemId);
            int goodCount = totalCountByDoc[docId - blockStart];
            auto bordersData = bv.GetBorders(elemId);
            for (int border = 0; border < targetBorderCount; ++border) {
                UpdateGoodCount(bordersData[border], ctrType, &goodCount);
                goodCountByBorderByDoc[border][docId - blockStart] = goodCount;
            }

            if (docId < data.LearnSampleCount) {
                ++bordersData[permutedTargetClass[docId]];
                ++bv.GetTotal(elemId);
            }
        }

        for (int border = 0; border < targetBorderCount; ++border) {
            for (int prior = 0; prior < priors.ysize(); ++prior) {
                const float priorX = priors[prior];
                const float shiftX = shift[prior];
                const float normX = norm[prior];
                const int* goodCountData = goodCountByBorderByDoc[border].data();
                ui8* featureData = (*feature)[border][prior].data();
                for (int docId = blockStart; docId < nextBlockStart; ++docId) {
                    featureData[docId] = CalcCTR(goodCountData[docId - blockStart], totalCountByDoc[docId - blockStart], priorX,
                                                 shiftX, normX, ctrBorderCount);
                }
            }
        }
    }
}

static void CalcOnlineCTRSimple(const TTrainData& data,
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
    const auto docCount = enumeratedCatFeatures.ysize();
    for (int blockStart = 0; blockStart < docCount; blockStart += blockSize) {
        const int nextBlockStart = Min<int>(docCount, blockStart + blockSize);
        const int learnSampleCount = data.LearnSampleCount;
        for (int docId = blockStart; docId < nextBlockStart; ++docId) {
            TCtrHistory& elem = ctrArrSimple[enumeratedCatFeatures[docId]];
            goodCount[docId - blockStart] = elem.N[1];
            totalCount[docId - blockStart] = elem.N[0] + elem.N[1];
            if (docId < learnSampleCount) {
                ++elem.N[permutedTargetClass[docId]];
            }
        }
        for (int prior = 0; prior < priors.ysize(); ++prior) {
            const float priorX = priors[prior];
            const float shiftX = shift[prior];
            const float normX = norm[prior];
            ui8* featureData = (*feature)[0][prior].data();
            for (int docId = blockStart; docId < nextBlockStart; ++docId) {
                featureData[docId] = CalcCTR(goodCount[docId - blockStart], totalCount[docId - blockStart], priorX,
                                             shiftX, normX, ctrBorderCount);
            }
        }
    }
}

static void CalcOnlineCTRMean(const TTrainData& data,
                              const TVector<ui64>& enumeratedCatFeatures,
                              size_t leafCount,
                              const TVector<int>& permutedTargetClass,
                              int targetBorderCount,
                              const TVector<float>& priors,
                              int ctrBorderCount,
                              TArray2D<TVector<ui8>>* feature) {
    const auto docCount = enumeratedCatFeatures.ysize();
    auto ctrArrMean = TCtrCalcer::GetCtrMeanHistoryArr(leafCount);
    TVector<float> shift;
    TVector<float> norm;
    CalcNormalization(priors, &shift, &norm);

    const int blockSize = 1000;
    TVector<float> sum(blockSize);
    TVector<int> count(blockSize);
    for (int blockStart = 0; blockStart < docCount; blockStart += blockSize) {
        const int nextBlockStart = Min<int>(docCount, blockStart + blockSize);
        for (int docId = blockStart; docId < nextBlockStart; ++docId) {
            TCtrMeanHistory& elem = ctrArrMean[enumeratedCatFeatures[docId]];
            sum[docId - blockStart] = elem.Sum;
            count[docId - blockStart] = elem.Count;
            if (docId < data.LearnSampleCount) {
                elem.Add(static_cast<float>(permutedTargetClass[docId]) / targetBorderCount);
            }
        }

        for (int prior = 0; prior < priors.ysize(); ++prior) {
            const float priorX = priors[prior];
            const float shiftX = shift[prior];
            const float normX = norm[prior];
            ui8* featureData = (*feature)[0][prior].data();
            for (int docId = blockStart; docId < nextBlockStart; ++docId) {
                featureData[docId] = CalcCTR(sum[docId - blockStart], count[docId - blockStart], priorX,
                                             shiftX, normX, ctrBorderCount);
            }
        }
    }
}

static void CalcOnlineCTRCounter(const TVector<int>& counterCTRTotal,
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
    const int docCount = enumeratedCatFeatures.ysize();
    for (int blockStart = 0; blockStart < docCount; blockStart += blockSize) {
        const int blockEnd = Min(docCount, blockStart + blockSize);
        for (int docId = blockStart; docId < blockEnd; ++docId) {
            const auto elemId = enumeratedCatFeatures[docId];
            ctrTotal[docId - blockStart] = counterCTRTotal[elemId];
        }
        for (int prior = 0; prior < priors.ysize(); ++prior) {
            const float priorX = priors[prior];
            const float shiftX = shift[prior];
            const float normX = norm[prior];
            ui8* featureData = (*feature)[0][prior].data();
            for (int docId = blockStart; docId < blockEnd; ++docId) {
                featureData[docId] = CalcCTR(ctrTotal[docId - blockStart], denominator, priorX, shiftX, normX, ctrBorderCount);
            }
        }
    }
}

static inline void CountOnlineCTRTotal(const TVector<ui64>& hashArr, int sampleCount, TVector<int>* counterCTRTotal) {
    for (int sampleIdx = 0; sampleIdx < sampleCount; ++sampleIdx) {
        const auto elemId = hashArr[sampleIdx];
        ++(*counterCTRTotal)[elemId];
    }
}

void ComputeOnlineCTRs(const TTrainData& data,
                       const TFold& fold,
                       const TProjection& proj,
                       TLearnContext* ctx,
                       TOnlineCTR* dst) {

    const TCtrHelper& ctrHelper = ctx->CtrsHelper;
    const auto& ctrInfo = ctrHelper.GetCtrInfo(proj);
    dst->Feature.resize(ctrInfo.size());

    using THashArr = TVector<ui64>;
    using TRehashHash = TDenseHash<ui64, ui32>;
    Y_STATIC_THREAD(THashArr) tlsHashArr;
    Y_STATIC_THREAD(TRehashHash) rehashHashTlsVal;
    TVector<ui64>& hashArr = tlsHashArr.Get();
    CalcHashes(proj, data.AllFeatures, fold.EffectiveDocCount, fold.LearnPermutation, &hashArr);
    rehashHashTlsVal.Get().MakeEmpty(fold.LearnPermutation.size());
    ui64 topSize = ctx->Params.CatFeatureParams->CtrLeafCountLimit;
    if (proj.IsSingleCatFeature() && ctx->Params.CatFeatureParams->StoreAllSimpleCtrs) {
        topSize = Max<ui64>();
    }
    auto leafCount = ReindexHash(
        fold.LearnPermutation.size(),
        topSize,
        &hashArr,
        rehashHashTlsVal.GetPtr());
    dst->FeatureValueCount = leafCount.second;

    TVector<int> counterCTRTotal;
    int counterCTRDenominator = 0;
    if (AnyOf(ctrInfo.begin(), ctrInfo.begin() + dst->Feature.ysize(), [] (const auto& info) { return info.Type == ECtrType::Counter; })) {
        counterCTRTotal.resize(leafCount.second);
        const int sampleCount = ctx->Params.CatFeatureParams->CounterCalcMethod == ECounterCalc::Full ? hashArr.ysize() : data.LearnSampleCount;
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
                Clear(&dst->Feature[ctrIdx][border][prior], data.GetSampleCount());
            }
        }

        if (ctrType == ECtrType::Borders && targetClassesCount == SIMPLE_CLASSES_COUNT) {
            CalcOnlineCTRSimple(
                data,
                hashArr,
                leafCount.second,
                fold.LearnTargetClass[classifierId],
                priors,
                ctrBorderCount,
                &dst->Feature[ctrIdx]);

        } else if (ctrType == ECtrType::BinarizedTargetMeanValue) {
            CalcOnlineCTRMean(
                data,
                hashArr,
                leafCount.second,
                fold.LearnTargetClass[classifierId],
                targetClassesCount - 1,
                priors,
                ctrBorderCount,
                &dst->Feature[ctrIdx]);

        } else if (ctrType == ECtrType::Buckets ||
                   (ctrType == ECtrType::Borders && targetClassesCount > SIMPLE_CLASSES_COUNT)) {
            CalcOnlineCTRClasses(
                data,
                hashArr,
                leafCount.second,
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
                counterCTRTotal,
                hashArr,
                counterCTRDenominator,
                priors,
                ctrBorderCount,
                &dst->Feature[ctrIdx]);
        }
    }
}

void ComputeOnlineCTRs(const TTrainData& data,
                       const TProjection& proj,
                       TLearnContext* ctx,
                       TFold* fold) {
    TOnlineCTRHash& ctrs = fold->GetCtrs(proj);
    if (ctrs.has(proj)) {
        return;
    }

    ComputeOnlineCTRs(data,
                      *fold,
                      proj,
                      ctx,
                      &ctrs[proj]);
}

void CalcOnlineCTRsBatch(const TVector<TCalcOnlineCTRsBatchTask>& tasks, const TTrainData& data, TLearnContext* ctx) {
    auto calcer = [&](int i) {
        ComputeOnlineCTRs(data,
                          *tasks[i].Fold,
                          tasks[i].Projection,
                          ctx,
                          tasks[i].Ctr);
    };
    ctx->LocalExecutor.ExecRange(calcer, 0, tasks.size(), NPar::TLocalExecutor::WAIT_COMPLETE);
}

void CalcFinalCtrsImpl(
    const ECtrType ctrType,
    const ui64 ctrLeafCountLimit,
    const TVector<int>& permutedTargetClass,
    const TVector<float>& permutedTargets,
    const ui64 learnSampleCount,
    int targetClassesCount,
    TVector<ui64>* hashArr,
    TCtrValueTable* result
) {
    TDenseHash<ui64, ui32> tmpHash;
    auto leafCount = ReindexHash(
        learnSampleCount,
        ctrLeafCountLimit,
        hashArr,
        &tmpHash).first;
    auto hashIndexBuilder = result->GetIndexHashBuilder(leafCount);
    for (const auto& kv : tmpHash) {
        hashIndexBuilder.SetIndex(kv.Key(), kv.Value());
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

    Y_ASSERT(hashArr->size() == learnSampleCount);
    int targetBorderCount = targetClassesCount - 1;
    auto hashArrPtr = hashArr->data();
    for (ui32 z = 0; z < learnSampleCount; ++z) {
        const ui64 elemId = hashArrPtr[z];
        if (ctrType == ECtrType::BinarizedTargetMeanValue) {
            TCtrMeanHistory& elem = ctrMean[elemId];
            elem.Add(static_cast<float>(permutedTargetClass[z]) / targetBorderCount);
        } else if (ctrType == ECtrType::Counter || ctrType == ECtrType::FeatureFreq) {
            ++ctrIntArray[elemId];
        } else if (ctrType == ECtrType::FloatTargetMeanValue) {
            TCtrMeanHistory& elem = ctrMean[elemId];
            elem.Add(permutedTargets[z]);
        } else {
            TArrayRef<int> elem = MakeArrayRef(ctrIntArray.data() + targetClassesCount * elemId, targetClassesCount);
            ++elem[permutedTargetClass[z]];
        }
    }

    if (ctrType == ECtrType::Counter) {
        result->CounterDenominator = *MaxElement(ctrIntArray.begin(), ctrIntArray.end());
    }
    if (ctrType == ECtrType::FeatureFreq) {
        result->CounterDenominator = static_cast<int>(learnSampleCount);
    }
}

void CalcFinalCtrs(const ECtrType ctrType, const TProjection& projection, const TTrainData& data,
                   const TVector<int>& learnPermutation, const TVector<int>& permutedTargetClass,
                   int targetClassesCount, ui64 ctrLeafCountLimit, bool storeAllSimpleCtr,
                   ECounterCalc counterCalcMethod, TCtrValueTable* result) {
    ui64 sampleCount = data.LearnSampleCount;
    if (ctrType == ECtrType::Counter && counterCalcMethod == ECounterCalc::Full) {
        sampleCount = data.GetSampleCount();
    }
    TVector<ui64> hashArr;
    CalcHashes(projection, data.AllFeatures, sampleCount, learnPermutation, &hashArr);

    if (projection.IsSingleCatFeature() && storeAllSimpleCtr) {
        ctrLeafCountLimit = Max<ui64>();
    }
    CalcFinalCtrsImpl(
        ctrType,
        ctrLeafCountLimit,
        permutedTargetClass,
        TVector<float>(),
        sampleCount,
        targetClassesCount,
        &hashArr,
        result);
}

void CalcFinalCtrs(const ECtrType ctrType, const TFeatureCombination& projection, const TPool& pool, ui64 sampleCount,
                   const TVector<int>& permutedTargetClass, const TVector<float>& permutedTargets,
                   int targetClassesCount, ui64 ctrLeafCountLimit, bool storeAllSimpleCtr, TCtrValueTable* result) {
    TMap<int, int> floatFeatureIdxToFlatIdx;
    TMap<int, int> catFeatureIdxToFlatIdx;
    TSet<int> catFeatureSet(pool.CatFeatures.begin(), pool.CatFeatures.end());
    for (int i = 0; i < pool.Docs.GetFactorsCount(); ++i) {
        if (catFeatureSet.has(i)) {
            catFeatureIdxToFlatIdx[catFeatureIdxToFlatIdx.size()] = i;
        } else {
            floatFeatureIdxToFlatIdx[floatFeatureIdxToFlatIdx.size()] = i;
        }
    }
    TVector<ui64> hashArr;
    CalcHashes(
        projection,
        [&] (int floatFeatureIdx, size_t docId) -> float {
            return pool.Docs.Factors[floatFeatureIdxToFlatIdx[floatFeatureIdx]][docId];
        },
        [&] (int catFeatureIdx, size_t docId) -> int {
            return ConvertFloatCatFeatureToIntHash(pool.Docs.Factors[catFeatureIdxToFlatIdx[catFeatureIdx]][docId]);
        },
        sampleCount,
        &hashArr);

    if (projection.IsSingleCatFeature() && storeAllSimpleCtr) {
        ctrLeafCountLimit = Max<ui64>();
    }
    CalcFinalCtrsImpl(ctrType, ctrLeafCountLimit, permutedTargetClass, permutedTargets, sampleCount, targetClassesCount, &hashArr, result);
}
