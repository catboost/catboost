#include "online_ctr.h"

#include "bin_tracker.h"
#include "fold.h"
#include "index_hash_calcer.h"
#include "learn_context.h"
#include "score_calcer.h"
#include <catboost/libs/model/tensor_struct.h>

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
        memset(Storage.data(), 0, neededSize);
        return (T*)Storage.data();
    }
    static inline NArrayRef::TArrayRef<TCtrMeanHistory> GetCtrMeanHistoryArr(size_t maxCount) {
        return NArrayRef::TArrayRef<TCtrMeanHistory>(FastTlsSingleton<TCtrCalcer>()->Alloc<TCtrMeanHistory>(maxCount), maxCount);
    }
    static inline NArrayRef::TArrayRef<TCtrHistory> GetCtrHistoryArr(size_t maxCount) {
        return NArrayRef::TArrayRef<TCtrHistory>(FastTlsSingleton<TCtrCalcer>()->Alloc<TCtrHistory>(maxCount), maxCount);
    }
    static inline int* GetCtrArrTotal(size_t maxCount) {
        return FastTlsSingleton<TCtrCalcer>()->Alloc<int>(maxCount);
    }

private:
    yvector<char> Storage;
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
    inline NArrayRef::TArrayRef<int> GetBorders(size_t i) {
        Y_ASSERT(i < MaxElem);
        return NArrayRef::TArrayRef<int>(BucketData + BorderCount * i, BorderCount);
    }
};

void CalcNormalization(const yvector<float>& priors, yvector<float>* shift, yvector<float>* norm) {
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

int GetCtrBorderCount(int targetClassesCount, ECtrType ctrType) {
    if (ctrType == ECtrType::MeanValue || ctrType == ECtrType::Counter) {
        return 1;
    }
    return ctrType == ECtrType::Buckets ? targetClassesCount : targetClassesCount - 1;
}

static void UpdateGoodCount(int curCount, ECtrType ctrType, int* goodCount) {
    if (ctrType == ECtrType::Buckets) {
        *goodCount = curCount;
    } else {
        *goodCount -= curCount;
    }
}

static void CalcOnlineCTRClasses(const TTrainData& data,
                                 const yvector<ui64>& enumeratedCatFeatures,
                                 size_t leafCount,
                                 const yvector<int>& permutedTargetClass,
                                 int targetClassesCount,
                                 const yvector<float>& priors,
                                 int ctrBorderCount,
                                 ECtrType ctrType,
                                 TArray2D<yvector<ui8>>* feature) {
    const int targetBorderCount = GetCtrBorderCount(targetClassesCount, ctrType);
    const int docCount = enumeratedCatFeatures.ysize();

    TBucketsView bv(leafCount, targetClassesCount);
    yvector<float> shift;
    yvector<float> norm;
    CalcNormalization(priors, &shift, &norm);

    const int blockSize = (1000 + targetBorderCount - 1) / targetBorderCount + 100; // ensure blocks have reasonable size
    yvector<int> totalCountByDoc(blockSize);
    yvector<yvector<int>> goodCountByBorderByDoc(targetBorderCount, yvector<int>(blockSize));
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
                                const yvector<ui64>& enumeratedCatFeatures,
                                size_t leafCount,
                                const yvector<int>& permutedTargetClass,
                                const yvector<float>& priors,
                                int ctrBorderCount,
                                TArray2D<yvector<ui8>>* feature) {
    const auto docCount = enumeratedCatFeatures.ysize();
    auto ctrArrSimple = TCtrCalcer::GetCtrHistoryArr(leafCount);
    yvector<float> shift;
    yvector<float> norm;
    CalcNormalization(priors, &shift, &norm);

    const int blockSize = 1000;
    yvector<int> totalCount(blockSize);
    yvector<int> goodCount(blockSize);
    for (int blockStart = 0; blockStart < docCount; blockStart += blockSize) {
        const int nextBlockStart = Min<int>(docCount, blockStart + blockSize);
        for (int docId = blockStart; docId < nextBlockStart; ++docId) {
            TCtrHistory& elem = ctrArrSimple[enumeratedCatFeatures[docId]];
            goodCount[docId - blockStart] = elem.N[1];
            totalCount[docId - blockStart] = elem.N[0] + elem.N[1];
            if (docId < data.LearnSampleCount) {
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
                              const yvector<ui64>& enumeratedCatFeatures,
                              size_t leafCount,
                              const yvector<int>& permutedTargetClass,
                              int targetBorderCount,
                              const yvector<float>& priors,
                              int ctrBorderCount,
                              TArray2D<yvector<ui8>>* feature) {
    const auto docCount = enumeratedCatFeatures.ysize();
    auto ctrArrMean = TCtrCalcer::GetCtrMeanHistoryArr(leafCount);
    yvector<float> shift;
    yvector<float> norm;
    CalcNormalization(priors, &shift, &norm);

    const int blockSize = 1000;
    yvector<float> sum(blockSize);
    yvector<int> count(blockSize);
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

static void CalcOnlineCTRCounter(const TTrainData& data,
                                 const yvector<ui64>& enumeratedCatFeatures,
                                 size_t leafCount,
                                 const yvector<float>& priors,
                                 int ctrBorderCount,
                                 ECounterCalc counterCalc,
                                 TArray2D<yvector<ui8>>* feature) {
    const auto docCount = enumeratedCatFeatures.ysize();
    auto ctrArrTotal = TCtrCalcer::GetCtrArrTotal(leafCount);
    yvector<float> shift;
    yvector<float> norm;
    CalcNormalization(priors, &shift, &norm);
    Y_ASSERT(docCount >= data.LearnSampleCount);

    int denominator = 0;
    if (counterCalc == ECounterCalc::Universal) {
        auto CalcInlineCTR = [&](int firstPos, int lastPos) {
            for (int docId = firstPos; docId < lastPos; ++docId) {
                const auto bucketId = enumeratedCatFeatures[docId];
                ++ctrArrTotal[bucketId];
                denominator = Max(denominator, ctrArrTotal[bucketId]);
            }
            for (int prior = 0; prior < priors.ysize(); ++prior) {
                const float priorX = priors[prior];
                const float shiftX = shift[prior];
                const float normX = norm[prior];
                ui8* featureData = (*feature)[0][prior].data();
                for (int docId = firstPos; docId < lastPos; ++docId) {
                    const auto bucketId = enumeratedCatFeatures[docId];
                    featureData[docId] = CalcCTR(ctrArrTotal[bucketId], denominator, priorX,
                                                 shiftX, normX, ctrBorderCount);
                }
            }
        };
        CalcInlineCTR(0, data.LearnSampleCount);
        CalcInlineCTR(data.LearnSampleCount, docCount);
    } else {
        for (int docId = 0; docId < data.LearnSampleCount; ++docId) {
            const auto bucketId = enumeratedCatFeatures[docId];
            ++ctrArrTotal[bucketId];
            denominator = Max(denominator, ctrArrTotal[bucketId]);
        }

        const int blockSize = 1000;
        yvector<int> ctrTotal(blockSize);
        yvector<int> ctrDenominator(blockSize);
        for (int blockStart = 0; blockStart < docCount; blockStart += blockSize) {
            const int nextBlockStart = Min<int>(docCount, blockStart + blockSize);
            for (int docId = blockStart; docId < nextBlockStart; ++docId) {
                const auto elemId = enumeratedCatFeatures[docId];
                if (docId >= data.LearnSampleCount && counterCalc == ECounterCalc::Basic) {
                    ++ctrArrTotal[elemId];
                    denominator = Max(denominator, ctrArrTotal[elemId]);
                }
                ctrTotal[docId - blockStart] = ctrArrTotal[elemId];
                ctrDenominator[docId - blockStart] = denominator;
            }
            for (int prior = 0; prior < priors.ysize(); ++prior) {
                const float priorX = priors[prior];
                const float shiftX = shift[prior];
                const float normX = norm[prior];
                ui8* featureData = (*feature)[0][prior].data();
                for (int docId = blockStart; docId < nextBlockStart; ++docId) {
                    featureData[docId] = CalcCTR(ctrTotal[docId - blockStart], ctrDenominator[docId - blockStart], priorX,
                                                 shiftX, normX, ctrBorderCount);
                }
            }
        }
    }
}

void ComputeOnlineCTRs(const TTrainData& data,
                       const TFold& fold,
                       const TProjection& proj,
                       TLearnContext* ctx,
                       TOnlineCTR* dst) {
    dst->Feature.resize(ctx->Params.CtrParams.Ctrs.size());
    using THashArr = yvector<ui64>;
    using TRehashHash = TDenseHash<ui64, ui32>;
    Y_STATIC_THREAD(THashArr) tlsHashArr;
    Y_STATIC_THREAD(TRehashHash) rehashHashTlsVal;
    yvector<ui64>& hashArr = tlsHashArr.Get();

    CalcHashes(proj, data, fold, &hashArr);
    rehashHashTlsVal.Get().MakeEmpty(fold.LearnPermutation.size());
    ui64 topSize = ctx->Params.CtrLeafCountLimit;
    if (proj.IsSingleCatFeature() && ctx->Params.StoreAllSimpleCtr) {
        topSize = Max<ui64>();
    }
    auto leafCount = ReindexHash(
        fold.LearnPermutation.size(),
        topSize,
        &hashArr,
        rehashHashTlsVal.GetPtr());

    for (int ctrIdx = 0; ctrIdx < dst->Feature.ysize(); ++ctrIdx) {
        const yvector<float>& priors = ctx->Priors.GetPriors(proj, ctrIdx);
        int targetClassesCount = fold.TargetClassesCount[ctrIdx];
        ECtrType ctrType = ctx->Params.CtrParams.Ctrs[ctrIdx].CtrType;
        auto borderCount = GetCtrBorderCount(targetClassesCount, ctrType);
        dst->Feature[ctrIdx].SetSizes(priors.size(), borderCount);
        for (int border = 0; border < borderCount; ++border) {
            for (int prior = 0; prior < priors.ysize(); ++prior) {
                Clear(&dst->Feature[ctrIdx][border][prior], data.GetSampleCount());
            }
        }
        if (ctrType == ECtrType::Borders && targetClassesCount == SIMPLE_CLASSES_COUNT) {
            CalcOnlineCTRSimple(
                data,
                hashArr,
                leafCount.second,
                fold.LearnTargetClass[ctrIdx],
                priors,
                ctx->Params.CtrParams.CtrBorderCount,
                &dst->Feature[ctrIdx]);
        } else if (ctrType == ECtrType::MeanValue) {
            CalcOnlineCTRMean(
                data,
                hashArr,
                leafCount.second,
                fold.LearnTargetClass[ctrIdx],
                targetClassesCount - 1,
                priors,
                ctx->Params.CtrParams.CtrBorderCount,
                &dst->Feature[ctrIdx]);
        } else if (ctrType == ECtrType::Buckets ||
                   (ctrType == ECtrType::Borders && targetClassesCount > SIMPLE_CLASSES_COUNT)) {
            CalcOnlineCTRClasses(
                data,
                hashArr,
                leafCount.second,
                fold.LearnTargetClass[ctrIdx],
                targetClassesCount,
                priors,
                ctx->Params.CtrParams.CtrBorderCount,
                ctrType,
                &dst->Feature[ctrIdx]);
        } else {
            Y_ASSERT(ctrType == ECtrType::Counter);
            CalcOnlineCTRCounter(
                data,
                hashArr,
                leafCount.second,
                priors,
                ctx->Params.CtrParams.CtrBorderCount,
                ctx->Params.CounterCalcMethod,
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

void CalcOnlineCTRsBatch(const yvector<TCalcOnlineCTRsBatchTask>& tasks, const TTrainData& data, TLearnContext* ctx) {
    auto calcer = [&](int i) {
        ComputeOnlineCTRs(data,
                          *tasks[i].Fold,
                          tasks[i].Projection,
                          ctx,
                          tasks[i].Ctr);
    };
    ctx->LocalExecutor.ExecRange(calcer, 0, tasks.size(), NPar::TLocalExecutor::WAIT_COMPLETE);
}

void CalcFinalCtrs(const TModelCtr& ctr,
                   const TTrainData& data,
                   const yvector<int>& learnPermutation,
                   const yvector<int>& permutedTargetClass,
                   int targetClassesCount,
                   ui64 ctrLeafCountLimit,
                   bool storeAllSimpleCtr,
                   TCtrValueTable* result) {
    const ECtrType ctrType = ctr.CtrType;
    yvector<ui64> hashArr;
    CalcHashes(ctr.Projection, data.AllFeatures, data.LearnSampleCount, learnPermutation, &hashArr);

    ui64 topSize = ctrLeafCountLimit;
    if (ctr.Projection.IsSingleCatFeature() && storeAllSimpleCtr) {
        topSize = Max<ui64>();
    }

    auto leafCount = ReindexHash(
        data.LearnSampleCount,
        topSize,
        &hashArr,
        &result->Hash).first;

    if (ctrType == ECtrType::MeanValue) {
        result->CtrMean.resize(leafCount);
    } else if (ctrType == ECtrType::Counter || ctrType == ECtrType::FeatureFreq) {
        result->CtrTotal.resize(leafCount);
        result->CounterDenominator = 0;
    } else {
        result->Ctr.resize(leafCount, yvector<int>(targetClassesCount));
    }

    Y_ASSERT(hashArr.ysize() == data.LearnSampleCount);
    int targetBorderCount = targetClassesCount - 1;
    for (int z = 0; z < data.LearnSampleCount; ++z) {
        const ui64 elemId = hashArr[z];
        if (ctrType == ECtrType::MeanValue) {
            TCtrMeanHistory& elem = result->CtrMean[elemId];
            elem.Add(static_cast<float>(permutedTargetClass[z]) / targetBorderCount);
        } else if (ctrType == ECtrType::Counter || ctrType == ECtrType::FeatureFreq) {
            ++result->CtrTotal[elemId];
        } else {
            yvector<int>& elem = result->Ctr[elemId];
            ++elem[permutedTargetClass[z]];
        }
    }

    if (ctrType == ECtrType::Counter) {
        result->CounterDenominator = *MaxElement(result->CtrTotal.begin(), result->CtrTotal.end());
    }
    if (ctrType == ECtrType::FeatureFreq) {
        result->CounterDenominator = data.LearnSampleCount;
    }
}
