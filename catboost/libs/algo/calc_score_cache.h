#pragma once

#include "fold.h"
#include "split.h"

#include <catboost/libs/index_range/index_range.h>
#include <catboost/libs/helpers/restorable_rng.h>
#include <catboost/libs/options/restrictions.h>
#include <catboost/libs/options/oblivious_tree_options.h>

#include <util/generic/ptr.h>
#include <util/memory/pool.h>
#include <util/system/atomic.h>
#include <util/system/spinlock.h>

bool IsSamplingPerTree(const NCatboostOptions::TObliviousTreeLearnerOptions& fitParams);

template <typename TData, typename TAlloc>
static inline TData* GetDataPtr(TVector<TData, TAlloc>& data, size_t offset = 0) {
    return data.empty() ? nullptr : data.data() + offset;
}

template <typename TData, typename TAlloc>
static inline const TData* GetDataPtr(const TVector<TData, TAlloc>& data, size_t offset = 0) {
    return data.empty() ? nullptr : data.data() + offset;
}

static inline float GetBernoulliSampleRate(const NCatboostOptions::TOption<NCatboostOptions::TBootstrapConfig>& samplingConfig) {
    if (samplingConfig->GetBootstrapType() == EBootstrapType::Bernoulli) {
        return samplingConfig->GetTakenFraction();
    }
    return 1.0f;
}

static inline int GetMaxBodyTailCount(const TVector<TFold>& folds) {
    int maxBodyTailCount = 0;
    for (const auto& fold : folds) {
        maxBodyTailCount = Max(maxBodyTailCount, fold.BodyTailArr.ysize());
    }
    return maxBodyTailCount;
}

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
    SAVELOAD(SumWeightedDelta, SumWeight, SumDelta, Count);
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

struct TBucketStatsCache {
    THashMap<TSplitCandidate, THolder<TVector<TBucketStats, TPoolAllocator>>> Stats;
    inline void Create(const TVector<TFold>& folds, int bucketCount, int depth) {
        ApproxDimension = folds[0].GetApproxDimension();
        MaxBodyTailCount = GetMaxBodyTailCount(folds);
        InitialSize = sizeof(TBucketStats) * bucketCount * (1U << depth) * ApproxDimension * MaxBodyTailCount;
        Y_ASSERT(InitialSize > 0);
        MemoryPool = new TMemoryPool(InitialSize);
    }
    TVector<TBucketStats, TPoolAllocator>& GetStats(const TSplitCandidate& split, int statsCount, bool* areStatsDirty);
    void GarbageCollect();
    static TVector<TBucketStats> GetStatsInUse(int segmentCount,
        int segmentSize,
        int statsCount,
        const TVector<TBucketStats, TPoolAllocator>& cachedStats);
private:
    THolder<TMemoryPool> MemoryPool;
    TAdaptiveLock Lock;
    size_t InitialSize = 0;
    int MaxBodyTailCount = 0;
    int ApproxDimension = 0;
};

struct TCalcScoreFold {
    template <typename TDataType>
    class TUnsizedVector : public TVector<TDataType> {
        size_t size() = delete;
        int ysize() = delete;
    };

    struct TBodyTail {
        TUnsizedVector<TUnsizedVector<double>> WeightedDerivatives;
        TUnsizedVector<TUnsizedVector<double>> SampleWeightedDerivatives;
        TUnsizedVector<float> PairwiseWeights;
        TUnsizedVector<float> SamplePairwiseWeights;

        TAtomic BodyFinish = 0;
        TAtomic TailFinish = 0;
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
            template <typename TData>
            inline TArrayRef<const TData> GetConstRef(const TVector<TData>& data) const {
                return MakeArrayRef(GetDataPtr(data, Offset), Size);
            }
            template <typename TData>
            inline TArrayRef<TData> GetRef(TVector<TData>& data) const {
                const TSlice clippedSlice = Clip(data.ysize());
                return MakeArrayRef(GetDataPtr(data, clippedSlice.Offset), clippedSlice.Size);
            }
        };
        TUnsizedVector<TSlice> Slices;
        void Create(const NPar::TLocalExecutor::TExecRangeParams& docBlockParams);
        void CreateByControl(const NPar::TLocalExecutor::TExecRangeParams& docBlockParams, const TUnsizedVector<bool>& control, NPar::TLocalExecutor* localExecutor);

        void CreateByQueriesInfo(
            const TVector<TQueryInfo>& srcQueriesInfo,
            const NPar::TLocalExecutor::TExecRangeParams& queryBlockParams
        );
        void CreateByQueriesInfoAndControl(
            const TVector<TQueryInfo>& srcQueriesInfo,
            const NPar::TLocalExecutor::TExecRangeParams& queryBlockParams,
            const TUnsizedVector<bool>& control,
            bool isPairwiseScoring,
            NPar::TLocalExecutor* localExecutor,
            TVector<TQueryInfo>* dstQueriesInfo
        );
    };
    TUnsizedVector<TIndexType> Indices;
    TUnsizedVector<size_t> LearnPermutation;
    TUnsizedVector<size_t> IndexInFold;
    TUnsizedVector<float> LearnWeights;
    TUnsizedVector<float> SampleWeights;
    TVector<TQueryInfo> LearnQueriesInfo;
    TUnsizedVector<TBodyTail> BodyTailArr; // [tail][dim][doc]
    bool SmallestSplitSideValue;
    int PermutationBlockSize = FoldPermutationBlockSizeNotSet;

    void Create(const TVector<TFold>& folds, bool isPairwiseScoring, int defaultCalcStatsObjBlockSize, float sampleRate = 1.0f);
    void SelectSmallestSplitSide(int curDepth, const TCalcScoreFold& fold, NPar::TLocalExecutor* localExecutor);
    void Sample(const TFold& fold, const TVector<TIndexType>& indices, TRestorableFastRng64* rand, NPar::TLocalExecutor* localExecutor);
    void UpdateIndices(const TVector<TIndexType>& indices, NPar::TLocalExecutor* localExecutor);
    int GetDocCount() const;
    int GetBodyTailCount() const;
    int GetApproxDimension() const;
    const TVector<float>& GetLearnWeights() const { return LearnWeights; }

    bool HasQueryInfo() const;

    // for data with queries - query indices, object indices otherwise
    const NCB::IIndexRangesGenerator<int>& GetCalcStatsIndexRanges() const;

private:
    inline void ClearBodyTail() {
        for (auto& bodyTail : BodyTailArr) {
            bodyTail.BodyFinish = bodyTail.TailFinish = 0;
        }
    }
    using TSlice = TVectorSlicing::TSlice;
    template <typename TFoldType>
    void SelectBlockFromFold(const TFoldType& fold, TSlice srcBlock, TSlice dstBlock);
    void SetSmallestSideControl(int curDepth, int docCount, const TUnsizedVector<TIndexType>& indices, NPar::TLocalExecutor* localExecutor);
    void SetSampledControl(int docCount, TRestorableFastRng64* rand);

    void CreateBlocksAndUpdateQueriesInfoByControl(
        NPar::TLocalExecutor* localExecutor,
        int srcDocCount,
        const TVector<TQueryInfo>& srcQueriesInfo,
        int* blockCount,
        TVectorSlicing* srcBlocks,
        TVectorSlicing* dstBlocks,
        TVector<TQueryInfo>* dstQueriesInfo
    );

    void SetPermutationBlockSizeAndCalcStatsRanges(int permutationBlockSize);

    TUnsizedVector<bool> Control;
    int DocCount;
    int BodyTailCount;
    int ApproxDimension;
    float BernoulliSampleRate;
    bool HasPairwiseWeights;
    bool IsPairwiseScoring;
    int DefaultCalcStatsObjBlockSize;

    THolder<NCB::IIndexRangesGenerator<int>> CalcStatsIndexRanges;
};

struct TStats3D {
    TVector<TBucketStats> Stats; // [bodyTail & approxDim][leaf][bucket]
    int BucketCount = 0;
    int MaxLeafCount = 0;

    void Add(const TStats3D& stats3D);

    SAVELOAD(Stats, BucketCount, MaxLeafCount);
};
