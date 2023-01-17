#pragma once

#include <catboost/libs/helpers/dbg_output.h>

#include "fold.h"
#include "split.h"

#include <catboost/libs/data/columns.h>
#include <catboost/private/libs/index_range/index_range.h>
#include <catboost/private/libs/options/restrictions.h>

#include <util/generic/array_ref.h>
#include <util/generic/ptr.h>
#include <util/memory/pool.h>
#include <library/cpp/deprecated/atomic/atomic.h>
#include <util/system/info.h>
#include <util/system/spinlock.h>


struct TRestorableFastRng64;

namespace NCatboostOptions {
    class TObliviousTreeLearnerOptions;
}


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

static inline float GetBernoulliSampleRate(
    const NCatboostOptions::TOption<NCatboostOptions::TBootstrapConfig>& samplingConfig
) {
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

public:
    SAVELOAD(SumWeightedDelta, SumWeight, SumDelta, Count);

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

static_assert(
    std::is_pod<TBucketStats>::value,
    "TBucketStats must be pod to avoid memory initialization in yresize"
);

inline static int CountNonCtrBuckets(
    const NCB::TFeaturesLayout& featuresLayout,
    const NCB::TQuantizedFeaturesInfo& quantizedFeaturesInfo,
    ui32 oneHotMaxSize
) {
    int nonCtrBucketCount = 0;

    featuresLayout.IterateOverAvailableFeatures<EFeatureType::Float>(
        [&](NCB::TFloatFeatureIdx floatFeatureIdx) {
            nonCtrBucketCount += int(quantizedFeaturesInfo.GetBorders(floatFeatureIdx).size() + 1);
        }
    );

    featuresLayout.IterateOverAvailableFeatures<EFeatureType::Categorical>(
        [&](NCB::TCatFeatureIdx catFeatureIdx) {
            const auto uniqueValuesCounts = quantizedFeaturesInfo.GetUniqueValuesCounts(catFeatureIdx);
            if (uniqueValuesCounts.OnLearnOnly <= oneHotMaxSize) {
                nonCtrBucketCount += int(uniqueValuesCounts.OnLearnOnly + 1);
            }
        }
    );
    return nonCtrBucketCount;
}

class TBucketStatsCache {
public:
    inline void Create(const TVector<TFold>& folds, int bucketCount, int depth) {
        Stats.clear();
        ApproxDimension = folds[0].GetApproxDimension();
        MaxBodyTailCount = GetMaxBodyTailCount(folds);
        InitialSize = sizeof(TBucketStats) * bucketCount * (1ULL << depth) * ApproxDimension * MaxBodyTailCount;
        if (InitialSize == 0) {
            InitialSize = NSystemInfo::GetPageSize();
        }
        MemoryPool = MakeHolder<TMemoryPool>(InitialSize);
    }
    TVector<TBucketStats, TPoolAllocator>& GetStats(
        const TSplitEnsemble& splitEnsemble,
        int statsCount,
        bool* areStatsDirty
    );
    void GarbageCollect();
    static TVector<TBucketStats> GetStatsInUse(
        int segmentCount,
        int segmentSize,
        int statsCount,
        const TVector<TBucketStats, TPoolAllocator>& cachedStats
    );

public:
    THashMap<TSplitEnsemble, THolder<TVector<TBucketStats, TPoolAllocator>>> Stats;

private:
    THolder<TMemoryPool> MemoryPool;
    TAdaptiveLock Lock;
    size_t InitialSize = 0;
    int MaxBodyTailCount = 0;
    int ApproxDimension = 0;
};

class TCalcScoreFold {
public:
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
        struct TSlice {
            static const constexpr int InvalidOffset = -1;
            static const constexpr int InvalidSize = -1;

            int Offset = InvalidOffset;
            int Size = InvalidSize;

        public:
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

    public:
        int Total;
        TUnsizedVector<TSlice> Slices;

    public:
        void Create(const NPar::ILocalExecutor::TExecRangeParams& docBlockParams);
        void CreateByControl(
            const NPar::ILocalExecutor::TExecRangeParams& docBlockParams,
            const TUnsizedVector<bool>& control,
            NPar::ILocalExecutor* localExecutor
        );
        void CreateByQueriesInfo(
            const TVector<TQueryInfo>& srcQueriesInfo,
            const NPar::ILocalExecutor::TExecRangeParams& queryBlockParams
        );
        void CreateByQueriesInfoAndControl(
            const TVector<TQueryInfo>& srcQueriesInfo,
            const NPar::ILocalExecutor::TExecRangeParams& queryBlockParams,
            const TUnsizedVector<bool>& control,
            bool isPairwiseScoring,
            NPar::ILocalExecutor* localExecutor,
            TVector<TQueryInfo>* dstQueriesInfo
        );
    };

public:
    void Create(
        const TVector<TFold>& folds,
        bool isPairwiseScoring,
        bool hasOfflineEstimatedFeatures,
        int defaultCalcStatsObjBlockSize,
        float sampleRate = 1.0f
    );
    void SelectSmallestSplitSide(
        int curDepth,
        const TCalcScoreFold& fold,
        NPar::ILocalExecutor* localExecutor
    );
    void Sample(
        const TFold& fold,
        ESamplingUnit samplingUnit,
        bool hasOfflineEstimatedFeatures,
        TConstArrayRef<TIndexType> indices,
        TRestorableFastRng64* rand,
        NPar::ILocalExecutor* localExecutor,
        bool performRandomChoice = true,
        bool shouldSortByLeaf = false,
        ui32 leavesCount = 0
    );
    void UpdateIndices(TConstArrayRef<TIndexType> indices, NPar::ILocalExecutor* localExecutor);
    // for lossguide
    void UpdateIndicesInLeafwiseSortedFoldForSingleLeaf(
        TIndexType leaf,
        TIndexType leftChildIdx,
        TIndexType rightChildIdx,
        const TVector<TIndexType>& indices,
        NPar::ILocalExecutor* localExecutor);
    // for depthwise
    void UpdateIndicesInLeafwiseSortedFold(
        const TVector<TIndexType>& leafs,
        const TVector<TIndexType>& childs,
        const TVector<TIndexType>& indices,
        NPar::ILocalExecutor* localExecutor);
    // for symmetric
    void UpdateIndicesInLeafwiseSortedFold(const TVector<TIndexType>& indices, NPar::ILocalExecutor* localExecutor);

    int GetDocCount() const;
    int GetBodyTailCount() const;
    int GetApproxDimension() const;
    const TVector<float>& GetLearnWeights() const { return LearnWeights; }

    bool HasQueryInfo() const;

    // for data with queries - query indices, object indices otherwise
    const NCB::IIndexRangesGenerator<int>& GetCalcStatsIndexRanges() const;

    TConstArrayRef<ui32> GetLearnPermutationOfflineEstimatedFeaturesSubset() const {
        return LearnPermutationOfflineEstimatedFeaturesSubset.Get<NCB::TIndexedSubset<ui32>>();
    }

private:
    using TSlice = TVectorSlicing::TSlice;

private:
    inline void ClearBodyTail() {
        for (auto& bodyTail : BodyTailArr) {
            bodyTail.BodyFinish = bodyTail.TailFinish = 0;
        }
    }

    template <typename TFoldType>
    void SelectBlockFromFold(const TFoldType& fold, TSlice srcBlock, TSlice dstBlock);
    void SetSmallestSideControl(
        int curDepth,
        int docCount,
        const TUnsizedVector<TIndexType>& indices,
        NPar::ILocalExecutor* localExecutor
    );
    void SetSampledControl(
        int docCount,
        ESamplingUnit samplingUnit,
        const TVector<TQueryInfo>& queriesInfo,
        TRestorableFastRng64* rand
    );
    void SetControlNoZeroWeighted(int docCount, const float* sampleWeights);

    void CreateBlocksAndUpdateQueriesInfoByControl(
        NPar::ILocalExecutor* localExecutor,
        int srcDocCount,
        const TVector<TQueryInfo>& srcQueriesInfo,
        int* blockCount,
        TVectorSlicing* srcBlocks,
        TVectorSlicing* dstBlocks,
        TVector<TQueryInfo>* dstQueriesInfo
    );

    void SetPermutationBlockSizeAndCalcStatsRanges(
        int mainDataPermutationBlockSize,
        int onlineDataPermutationBlockSize
    );

    void SortFoldByLeafIndex(ui32 leafCount, NPar::ILocalExecutor* localExecutor);

    struct TFoldPartitionOutput {
        void Create(int size, int dimension, bool hasOfflineEstimatedFeatures);

        struct TSlice {
            TArrayRef<float> SampleWeights;
            TArrayRef<ui32> IndexInFold;
            TArrayRef<ui32> LearnPermutationFeaturesSubset;
            TArrayRef<ui32> LearnPermutationOfflineEstimatedFeaturesSubset; // can be empty if unused
            TVector<TArrayRef<double>> SampleWeightedDerivatives;
        };

        TSlice GetSlice(NCB::TIndexRange<ui32> range);

        int Size;
        int Dimension;

        // don't process LearnPermutationOfflineEstimatedFeaturesSubset if false
        bool HasOfflineEstimatedFeatures;

        TUnsizedVector<float> SampleWeights;
        TUnsizedVector<ui32> IndexInFold;
        NCB::TIndexedSubset<ui32> LearnPermutationFeaturesSubset;
        NCB::TIndexedSubset<ui32> LearnPermutationOfflineEstimatedFeaturesSubset; // can be empty if unused
        TUnsizedVector<TUnsizedVector<double>> SampleWeightedDerivatives;
    };

    void UpdateIndicesInLeafwiseSortedFoldForSingleLeafImpl(
        TIndexType leaf,
        TIndexType leftChildIdx,
        TIndexType rightChildIdx,
        const TVector<TIndexType>& indices,
        NPar::ILocalExecutor* localExecutor,
        TFoldPartitionOutput::TSlice* out = nullptr);

public:
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

    /* indexing in offline estimated features buckets arrays (because it's different from main features subset),
     * always TIndexedSubset
     * initialized
     * initialized to some default value because TArraySubsetIndexing has no default constructor
     */
    NCB::TFeaturesArraySubsetIndexing LearnPermutationOfflineEstimatedFeaturesSubset
        = NCB::TFeaturesArraySubsetIndexing(NCB::TIndexedSubset<ui32>(0));

    TUnsizedVector<ui32> IndexInFold;
    TUnsizedVector<float> LearnWeights;
    TUnsizedVector<float> SampleWeights;
    TVector<TQueryInfo> LearnQueriesInfo;
    TUnsizedVector<TBodyTail> BodyTailArr; // [tail][dim][doc]
    bool SmallestSplitSideValue;
    int MainDataPermutationBlockSize = FoldPermutationBlockSizeNotSet;
    int OnlineDataPermutationBlockSize = FoldPermutationBlockSizeNotSet;
    ui32 LeavesCount;
    TVector<NCB::TIndexRange<ui32>> LeavesBounds;

private:
    TUnsizedVector<bool> Control;
    int DocCount;
    int BodyTailCount;
    int ApproxDimension;
    float BernoulliSampleRate;
    bool HasPairwiseWeights;
    bool IsPairwiseScoring;

    // don't process LearnPermutationOfflineEstimatedFeaturesSubset if false
    bool HasOfflineEstimatedFeatures;

    int DefaultCalcStatsObjBlockSize;

    THolder<NCB::IIndexRangesGenerator<int>> CalcStatsIndexRanges;
};


struct TStats3D {
    TVector<TBucketStats> Stats; // [bodyTail & approxDim][leaf][bucket]
    int BucketCount = 0;
    int MaxLeafCount = 0;

    TSplitEnsembleSpec SplitEnsembleSpec;

public:
    SAVELOAD(Stats, BucketCount, MaxLeafCount, SplitEnsembleSpec);

    void Add(const TStats3D& stats3D);
};

