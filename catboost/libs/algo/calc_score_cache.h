#pragma once

#include "fold.h"
#include "rand_score.h"
#include "split.h"
#include "train_data.h"

#include <catboost/libs/params/params.h>

#include <util/memory/pool.h>
#include <util/system/guard.h>
#include <util/system/spinlock.h>

inline bool AreStatsFromPrevTreeUsed(const TFitParams& fitParams) {
    return fitParams.WeightSamplingFrequency == EWeightSamplingFrequency::PerTree;
}

struct TBucketStats {
    double SumWeightedDelta;
    double SumWeight;
    double SumDelta;
    double Count;

    void Add(const TBucketStats& other) {
        SumWeightedDelta += other.SumWeightedDelta;
        SumDelta += other.SumDelta;
        SumWeight += other.SumWeight;
        Count += other.Count;
    }

    void Remove(const TBucketStats& other) {
        SumWeightedDelta -= other.SumWeightedDelta;
        SumDelta -= other.SumDelta;
        SumWeight -= other.SumWeight;
        Count -= other.Count;
    }
};

static_assert(std::is_pod<TBucketStats>::value, "TBucketStats must be pod to avoid memory initialization in yresize");

inline static int CountNonCtrBuckets(const TVector<int>& splitCounts, const TVector<TVector<int>>& oneHotValues) {
    int nonCtrBucketCount = 0;
    for (int splitCount : splitCounts) {
        nonCtrBucketCount += splitCount + 1;
    }
    for (const auto& oneHotValue : oneHotValues) {
        nonCtrBucketCount += oneHotValue.ysize() + 1;
    }
    return nonCtrBucketCount;
}

struct TStatsFromPrevTree {
    TAdaptiveLock Lock;
    THashMap<TSplitCandidate, THolder<TVector<TBucketStats, TPoolAllocator>>> Stats;
    THolder<TMemoryPool> MemoryPool;
    inline void Create(int bucketCount, int depth, int approxDimension, int bodyTailCount) {
        const size_t initialSize = sizeof(TBucketStats) * bucketCount * (1U << depth) * approxDimension * bodyTailCount;
        Y_ASSERT(initialSize > 0);
        MemoryPool = new TMemoryPool(initialSize);
    }
    inline TVector<TBucketStats, TPoolAllocator>& GetStats(const TSplitCandidate& split, int statsCount, bool* areStatsDirty) {
        TVector<TBucketStats, TPoolAllocator>* splitStats;
        with_lock(Lock) {
            if (Stats.has(split) && Stats[split] != nullptr) {
                splitStats = Stats[split].Get();
                *areStatsDirty = false;
            } else {
                splitStats = new TVector<TBucketStats, TPoolAllocator>(MemoryPool.Get());
                splitStats->yresize(statsCount);
                Stats[split] = splitStats;
                *areStatsDirty = true;
            }
        }
        return *splitStats;
    }
};

// Reference implementation -- to be parallelized
struct TSmallestSplitSideFold {
    struct TBodyTail {
        TVector<TVector<double>> Derivatives;
        TVector<TVector<double>> WeightedDer;

        int BodyFinish = 0;
        int TailFinish = 0;
    };

    TVector<TIndexType> Indices;
    TVector<int> LearnPermutation;
    TVector<int> IndexInFold;
    TVector<float> LearnWeights;
    TVector<float> SampleWeights;
    TVector<TBodyTail> BodyTailArr; // [tail][dim][doc]
    ui32 SmallestSplitSideValue;
    int DocCount;

    inline void Create(const TFold& fold) {
        const int docCount = fold.LearnPermutation.ysize();
        Y_ASSERT(docCount > 0);
        Indices.yresize(docCount);
        LearnPermutation.yresize(docCount);
        IndexInFold.yresize(docCount);
        LearnWeights.yresize(docCount);
        SampleWeights.yresize(docCount);
        BodyTailArr.yresize(fold.BodyTailArr.ysize());
        const int approxDimension = fold.GetApproxDimension();
        Y_ASSERT(approxDimension > 0);
        for (int bodyTailIdx = 0; bodyTailIdx < BodyTailArr.ysize(); ++bodyTailIdx) {
            BodyTailArr[bodyTailIdx].Derivatives.yresize(approxDimension);
            BodyTailArr[bodyTailIdx].WeightedDer.yresize(approxDimension);
            for (int dimIdx = 0; dimIdx < approxDimension; ++dimIdx) {
                Y_ASSERT(fold.BodyTailArr[bodyTailIdx].BodyFinish > 0);
                BodyTailArr[bodyTailIdx].Derivatives[dimIdx].yresize(fold.BodyTailArr[bodyTailIdx].BodyFinish);
                Y_ASSERT(fold.BodyTailArr[bodyTailIdx].TailFinish > 0);
                BodyTailArr[bodyTailIdx].WeightedDer[dimIdx].yresize(fold.BodyTailArr[bodyTailIdx].TailFinish);
            }
        }
    }
    inline void SelectParametersForSmallestSplitSide(int curDepth, const TFold& fold, const TVector<TIndexType>& indices) {
        CB_ENSURE(curDepth > 0);
        int trueCount = 0;
        for (TIndexType docIdx : indices) {
            trueCount += docIdx >> (curDepth - 1);
        }
        SmallestSplitSideValue = trueCount * 2 < indices.ysize();
        SetDocCount(Min(trueCount, indices.ysize() - trueCount));
        const TIndexType splitWeight = 1 << (curDepth - 1);
        int ignored;
        SetSelectedElements(SmallestSplitSideValue, splitWeight, indices, indices, [=](const TIndexType* source, size_t j, TIndexType* destination) { *destination = source[j] | splitWeight; }, &Indices, &ignored);
        SetSelectedElements(SmallestSplitSideValue, splitWeight, indices, fold.LearnPermutation, CopyElement<int>, &LearnPermutation, &ignored);
        SetSelectedElements(SmallestSplitSideValue, splitWeight, indices, TVector<int>(), [](const int*, size_t j, int* destination) { *destination = j; }, &IndexInFold, &ignored);
        SetSelectedElements(SmallestSplitSideValue, splitWeight, indices, fold.LearnWeights, CopyElement<float>, &LearnWeights, &ignored);
        SetSelectedElements(SmallestSplitSideValue, splitWeight, indices, fold.SampleWeights, CopyElement<float>, &SampleWeights, &ignored);
        for (int bodyTailIdx = 0; bodyTailIdx < fold.BodyTailArr.ysize(); ++bodyTailIdx) {
            const auto& foldBodyTail = fold.BodyTailArr[bodyTailIdx];
            for (int dim = 0; dim < fold.GetApproxDimension(); ++dim) {
                SetSelectedElements(SmallestSplitSideValue, splitWeight, indices, foldBodyTail.Derivatives[dim], CopyElement<double>, &BodyTailArr[bodyTailIdx].Derivatives[dim], &BodyTailArr[bodyTailIdx].BodyFinish);
                SetSelectedElements(SmallestSplitSideValue, splitWeight, indices, foldBodyTail.WeightedDer[dim], CopyElement<double>, &BodyTailArr[bodyTailIdx].WeightedDer[dim], &BodyTailArr[bodyTailIdx].TailFinish);
            }
        }
    }
    int GetApproxDimension() const {
        return BodyTailArr[0].Derivatives.ysize();
    }
private:
    inline void SetDocCount(int docCount) {
        DocCount = docCount;
        for (auto& bodyTail : BodyTailArr) {
            bodyTail.BodyFinish = bodyTail.TailFinish = 0;
        }
    }
    template<typename TData>
    static inline void CopyElement(const TData* source, size_t j, TData* destination) {
        *destination = source[j];
    };
    template<typename TSetElementFunc, typename TVectorType>
    static inline void SetSelectedElements(bool splitValue, TIndexType splitWeight, const TVector<TIndexType>& indices, const TVectorType& source, const TSetElementFunc& SetElementFunc, TVectorType* destination, int* elementCount) {
        auto* __restrict destinationData = destination->data();
        const auto* sourceData = source.data();
        const TIndexType* indicesData = indices.data();
        const size_t count = destination->size();
        size_t endElementIdx = 0;
        if (splitValue) {
            for (size_t sourceIdx = 0; sourceIdx < count; ++sourceIdx) {
                SetElementFunc(sourceData, sourceIdx, destinationData + endElementIdx);
                endElementIdx += indicesData[sourceIdx] > splitWeight - 1; // clang generates a smaller loop body for strict comparisons
            }
        } else {
            for (size_t sourceIdx = 0; sourceIdx < count; ++sourceIdx) {
                SetElementFunc(sourceData, sourceIdx, destinationData + endElementIdx);
                endElementIdx += indicesData[sourceIdx] < splitWeight;
            }
        }
        *elementCount = endElementIdx;
    }
};
