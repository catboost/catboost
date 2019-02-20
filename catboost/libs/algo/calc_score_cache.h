#pragma once

#include <catboost/libs/helpers/dbg_output.h>

#include "fold.h"
#include "split.h"

#include <catboost/libs/data_new/columns.h>
#include <catboost/libs/index_range/index_range.h>
#include <catboost/libs/helpers/restorable_rng.h>
#include <catboost/libs/options/restrictions.h>
#include <catboost/libs/options/oblivious_tree_options.h>

#include <util/generic/array_ref.h>
#include <util/generic/ptr.h>
#include <util/memory/pool.h>
#include <util/system/info.h>
#include <util/system/atomic.h>
#include <util/system/spinlock.h>

bool IsSamplingPerTree(const NCatboostOptions::TObliviousTreeLearnerOptions& fitParams);


/* both TArrayRef and TVector variants are needed because of no automatic 2-hop casting
 * TUnsizedVector -> TVector -> TArrayRef
 */
template <typename TData>
static inline TData* GetDataPtr(TArrayRef<TData> data, size_t offset = 0) {
    return data.empty() ? nullptr : data.data() + offset;
}

template <typename TData>
static inline const TData* GetDataPtr(TConstArrayRef<TData> data, size_t offset = 0) {
    return data.empty() ? nullptr : data.data() + offset;
}

template <typename TData, typename TAlloc>
static inline TData* GetDataPtr(TVector<TData, TAlloc>& data, size_t offset = 0) {
    return GetDataPtr(TArrayRef<TData>(data), offset);
}

template <typename TData, typename TAlloc>
static inline const TData* GetDataPtr(const TVector<TData, TAlloc>& data, size_t offset = 0) {
    return GetDataPtr(TConstArrayRef<TData>(data), offset);
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

inline static int CountNonCtrBuckets(
    const NCB::TQuantizedFeaturesInfo& quantizedFeaturesInfo,
    ui32 oneHotMaxSize
) {
    int nonCtrBucketCount = 0;

    quantizedFeaturesInfo.GetFeaturesLayout()->IterateOverAvailableFeatures<EFeatureType::Float>(
        [&](NCB::TFloatFeatureIdx floatFeatureIdx) {
            nonCtrBucketCount += int(quantizedFeaturesInfo.GetBorders(floatFeatureIdx).size() + 1);
        }
    );

    quantizedFeaturesInfo.GetFeaturesLayout()->IterateOverAvailableFeatures<EFeatureType::Categorical>(
        [&](NCB::TCatFeatureIdx catFeatureIdx) {
            const auto uniqueValuesCounts = quantizedFeaturesInfo.GetUniqueValuesCounts(catFeatureIdx);
            if (uniqueValuesCounts.OnLearnOnly <= oneHotMaxSize) {
                nonCtrBucketCount += int(uniqueValuesCounts.OnLearnOnly + 1);
            }
        }
    );
    return nonCtrBucketCount;
}

struct TBucketStatsCache {
    THashMap<TSplitEnsemble, THolder<TVector<TBucketStats, TPoolAllocator>>> Stats;
    inline void Create(const TVector<TFold>& folds, int bucketCount, int depth) {
        ApproxDimension = folds[0].GetApproxDimension();
        MaxBodyTailCount = GetMaxBodyTailCount(folds);
        InitialSize = sizeof(TBucketStats) * bucketCount * (1U << depth) * ApproxDimension * MaxBodyTailCount;
        if (InitialSize == 0) {
            InitialSize = NSystemInfo::GetPageSize();
        }
        MemoryPool = new TMemoryPool(InitialSize);
    }
    TVector<TBucketStats, TPoolAllocator>& GetStats(const TSplitEnsemble& splitEnsemble, int statsCount, bool* areStatsDirty);
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
            inline TConstArrayRef<TData> GetConstRef(TConstArrayRef<TData> data) const {
                return MakeArrayRef(GetDataPtr(data, Offset), Size);
            }
            template <typename TData>
            inline TConstArrayRef<TData> GetConstRef(const TVector<TData>& data) const {
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

    /* indexing in features buckets arrays, always TIndexedSubset
     * initialized to some default value because TArraySubsetIndexing has no default constructor
     */
    NCB::TFeaturesArraySubsetIndexing LearnPermutationFeaturesSubset
        = NCB::TFeaturesArraySubsetIndexing(NCB::TIndexedSubset<ui32>(0));

    /* begin of subset of data in features buckets arrays, used only for permutation block index calculation
     * if (PermutationBlockSize != 1) && (PermutationBlockSize != learnSampleCount))
     */
    ui32 FeaturesSubsetBegin;

    TUnsizedVector<ui32> IndexInFold;
    TUnsizedVector<float> LearnWeights;
    TUnsizedVector<float> SampleWeights;
    TVector<TQueryInfo> LearnQueriesInfo;
    TUnsizedVector<TBodyTail> BodyTailArr; // [tail][dim][doc]
    bool SmallestSplitSideValue;
    int NonCtrDataPermutationBlockSize = FoldPermutationBlockSizeNotSet;
    int CtrDataPermutationBlockSize = FoldPermutationBlockSizeNotSet;


    void Create(const TVector<TFold>& folds, bool isPairwiseScoring, int defaultCalcStatsObjBlockSize, float sampleRate = 1.0f);
    void SelectSmallestSplitSide(int curDepth, const TCalcScoreFold& fold, NPar::TLocalExecutor* localExecutor);
    void Sample(const TFold& fold, ESamplingUnit samplingUnit, const TVector<TIndexType>& indices, TRestorableFastRng64* rand, NPar::TLocalExecutor* localExecutor);
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
    void SetSampledControl(int docCount, ESamplingUnit samplingUnit, const TVector<TQueryInfo>& queriesInfo, TRestorableFastRng64* rand);

    void CreateBlocksAndUpdateQueriesInfoByControl(
        NPar::TLocalExecutor* localExecutor,
        int srcDocCount,
        const TVector<TQueryInfo>& srcQueriesInfo,
        int* blockCount,
        TVectorSlicing* srcBlocks,
        TVectorSlicing* dstBlocks,
        TVector<TQueryInfo>* dstQueriesInfo
    );

    void SetPermutationBlockSizeAndCalcStatsRanges(int nonCtrDataPermutationBlockSize, int ctrDataPermutationBlockSize);

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

    TSplitEnsembleSpec SplitEnsembleSpec;

    void Add(const TStats3D& stats3D);

    SAVELOAD(Stats, BucketCount, MaxLeafCount, SplitEnsembleSpec);
};

