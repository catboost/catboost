#pragma once

#include "fold.h"
#include "split.h"

#include <catboost/libs/options/restrictions.h>
#include <catboost/libs/options/oblivious_tree_options.h>

#include <util/memory/pool.h>
#include <util/system/atomic.h>
#include <util/system/spinlock.h>

bool AreStatsFromPrevTreeUsed(const NCatboostOptions::TObliviousTreeLearnerOptions& fitParams);

struct TBucketStats {
    double SumWeightedDelta;
    double SumWeight;
    double SumDelta;
    double Count;

    inline void Add(const TBucketStats& other) {
        SumWeightedDelta += other.SumWeightedDelta;
        SumDelta += other.SumDelta;
        SumWeight += other.SumWeight;
        Count += other.Count;
    }

    inline void Remove(const TBucketStats& other) {
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
        InitialSize = sizeof(TBucketStats) * bucketCount * (1U << depth) * approxDimension * bodyTailCount;
        Y_ASSERT(InitialSize > 0);
        MemoryPool = new TMemoryPool(InitialSize);
    }
    TVector<TBucketStats, TPoolAllocator>& GetStats(const TSplitCandidate& split, int statsCount, bool* areStatsDirty);
    void GarbageCollect();
private:
    size_t InitialSize;
};

struct TSmallestSplitSideFold {
    struct TBodyTail {
        TVector<TVector<double>> Derivatives;
        TVector<TVector<double>> WeightedDer;

        alignas(64) TAtomic BodyFinish = 0;
        alignas(64) TAtomic TailFinish = 0;
    };

    struct TVectorSlicing {
        int Total;
        struct TSlice {
            static const constexpr int InvalidOffset = -1;
            int Offset = InvalidOffset;
            static const constexpr int InvalidSize = -1;
            int Size = InvalidSize;
            TSlice Clip(int newSize) const {
                TSlice clippedSlice;
                clippedSlice.Offset = Offset;
                clippedSlice.Size = Min(Max(newSize - Offset, 0), Size);
                return clippedSlice;
            }
            template<typename TData>
            inline TArrayRef<const TData> GetConstRef(const TVector<TData>& vector) const {
                return MakeArrayRef(vector.data() + Offset, Size);
            }
            template<typename TData>
            inline TArrayRef<TData> GetRef(TVector<TData>& vector) const {
                return MakeArrayRef(vector.data() + Offset, Size);
            }
        };
        TVector<TSlice> Slices;
        void Create(const NPar::TLocalExecutor::TExecRangeParams& blockParams);
        void CreateForSmallestSide(const NPar::TLocalExecutor::TExecRangeParams& blockParams, const TIndexType* indices, int curDepth, NPar::TLocalExecutor* localExecutor);
        void Complement(const TVectorSlicing& slicing);
    };
    TVector<TIndexType> Indices;
    TVector<int> LearnPermutation;
    TVector<int> IndexInFold;
    TVector<float> LearnWeights;
    TVector<float> SampleWeights;
    TVector<TBodyTail> BodyTailArr; // [tail][dim][doc]
    bool SmallestSplitSideValue;
    int DocCount;

    void Create(const TFold& fold);
    void SelectParametersForSmallestSplitSide(int curDepth, const TFold& fold, const TVector<TIndexType>& indices, NPar::TLocalExecutor* localExecutor);
    int GetApproxDimension() const;
private:
    inline void SetDocCount(int docCount) {
        DocCount = docCount;
        for (auto& bodyTail : BodyTailArr) {
            bodyTail.BodyFinish = bodyTail.TailFinish = 0;
        }
    }
    template<typename TData>
    static inline TData GetElement(const TData* source, size_t j) {
        return source[j];
    }
    using TSlice = TVectorSlicing::TSlice;
    template<typename TSetElementsFunc, typename TGetIndexFunc>
    void SelectBlockFromFold(const TFold& fold, TSetElementsFunc SetElementsFunc, TGetIndexFunc GetIndexFunc, TSlice srcBlock, TSlice dstBlock);
};
