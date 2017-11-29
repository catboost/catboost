#pragma once

#include "fold.h"
#include "rand_score.h"
#include "split.h"
#include "train_data.h"

#include <catboost/libs/options/restrictions.h>
#include <catboost/libs/options/oblivious_tree_options.h>

#include <util/memory/pool.h>
#include <util/system/atomic.h>
#include <util/system/guard.h>
#include <util/system/spinlock.h>

inline bool AreStatsFromPrevTreeUsed(const NCatboostOptions::TObliviousTreeLearnerOptions& fitParams) {
    return fitParams.WeightSamplingFrequency.Get() == EWeightSamplingFrequency::PerTree;
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
    inline void GarbageCollect() {
        if (MemoryPool->MemoryWaste() > InitialSize) { // limit memory overhead
            Stats.clear();
            MemoryPool->Clear();
        }
    }
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
        inline void Create(const NPar::TLocalExecutor::TExecRangeParams& blockParams) {
            Total = blockParams.LastId;
            Slices.yresize(blockParams.GetBlockCount());
            for (int sliceIdx = 0; sliceIdx < Slices.ysize(); ++sliceIdx) {
                Slices[sliceIdx].Offset = blockParams.GetBlockSize() * sliceIdx;
                Slices[sliceIdx].Size = Min(blockParams.GetBlockSize(), Total - Slices[sliceIdx].Offset);
            }
        }
        inline void CreateForSmallestSide(const NPar::TLocalExecutor::TExecRangeParams& blockParams, const TIndexType* indices, int curDepth, NPar::TLocalExecutor* localExecutor) {
            Y_ASSERT(curDepth > 0);
            Slices.yresize(blockParams.GetBlockCount());
            localExecutor->ExecRange([&](int sliceIdx) {
                int blockTrueCount = 0; // use a local var instead of Slices[sliceIdx].Size so that the compiler can use a register
                NPar::TLocalExecutor::BlockedLoopBody(blockParams, [=, &blockTrueCount](int doc) { blockTrueCount += indices[doc] >> (curDepth - 1); })(sliceIdx);
                Slices[sliceIdx].Size = blockTrueCount;
            }, 0, Slices.ysize(), NPar::TLocalExecutor::WAIT_COMPLETE);
            int offset = 0;
            for (auto& slice : Slices) {
                slice.Offset = offset;
                offset += slice.Size;
            }
            Total = offset;
        }
        inline void Complement(const TVectorSlicing& slicing) {
            Y_ASSERT(slicing.Total >= Total && slicing.Slices.ysize() == Slices.ysize());
            Total = slicing.Total - Total;
            for (int sliceIdx = 0; sliceIdx < Slices.ysize(); ++sliceIdx) {
                Slices[sliceIdx].Offset = slicing.Slices[sliceIdx].Offset - Slices[sliceIdx].Offset;
                Slices[sliceIdx].Size = slicing.Slices[sliceIdx].Size - Slices[sliceIdx].Size;
            }
        }
    };
    TVector<TIndexType> Indices;
    TVector<int> LearnPermutation;
    TVector<int> IndexInFold;
    TVector<float> LearnWeights;
    TVector<float> SampleWeights;
    TVector<TBodyTail> BodyTailArr; // [tail][dim][doc]
    bool SmallestSplitSideValue;
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
    inline void SelectParametersForSmallestSplitSide(int curDepth, const TFold& fold, const TVector<TIndexType>& indices, NPar::TLocalExecutor* localExecutor) {
        CB_ENSURE(curDepth > 0);

        NPar::TLocalExecutor::TExecRangeParams blockParams(0, indices.ysize());
        blockParams.SetBlockSize(2000);
        const int blockCount = blockParams.GetBlockCount();

        TVectorSlicing srcBlocks;
        srcBlocks.Create(blockParams);

        TVectorSlicing dstBlocks;
        dstBlocks.CreateForSmallestSide(blockParams, indices.data(), curDepth, localExecutor);
        SmallestSplitSideValue = true;
        if (dstBlocks.Total * 2 > indices.ysize()) {
            dstBlocks.Complement(srcBlocks);
            SmallestSplitSideValue = false;
        }

        SetDocCount(dstBlocks.Total);
        const TIndexType splitWeight = 1 << (curDepth - 1);
        localExecutor->ExecRange([&](int blockIdx) {
            const auto srcBlock = srcBlocks.Slices[blockIdx];
            const auto srcIndicesRef = srcBlock.GetConstRef(indices);
            const auto SetElements = [=] (auto srcRef, auto GetElementFunc, auto dstRef, int* dstCount) {
                const TIndexType* indicesData = srcIndicesRef.data();
                const auto* sourceData = srcRef.data();
                const size_t sourceCount = srcRef.size();
                auto* __restrict destinationData = dstRef.data();
                const size_t destinationCount = dstRef.size();
                size_t endElementIdx = 0;
                if (SmallestSplitSideValue) {
                    const TIndexType splitMask = splitWeight - 1; // see below -- clang generates a smaller loop body for strict comparisons
                    #pragma unroll(4)
                    for (size_t sourceIdx = 0; sourceIdx < sourceCount && endElementIdx < destinationCount; ++sourceIdx) {
                        destinationData[endElementIdx] = GetElementFunc(sourceData, sourceIdx);
                        endElementIdx += indicesData[sourceIdx] > splitMask;
                    }
                } else {
                    #pragma unroll(4)
                    for (size_t sourceIdx = 0; sourceIdx < sourceCount && endElementIdx < destinationCount; ++sourceIdx) {
                        destinationData[endElementIdx] = GetElementFunc(sourceData, sourceIdx);
                        endElementIdx += indicesData[sourceIdx] < splitWeight;
                    }
                }
                *dstCount = endElementIdx;
            };
            const auto dstBlock = dstBlocks.Slices[blockIdx];
            SelectBlockFromFold(fold, SetElements, [=] (const TIndexType*, size_t i) { return srcIndicesRef[i] | splitWeight; }, srcBlock, dstBlock);
        }, 0, blockCount, NPar::TLocalExecutor::WAIT_COMPLETE);
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
    static inline TData GetElement(const TData* source, size_t j) {
        return source[j];
    }
    using TSlice = TVectorSlicing::TSlice;
    template<typename TSetElementsFunc, typename TGetIndexFunc>
    inline void SelectBlockFromFold(const TFold& fold, TSetElementsFunc SetElementsFunc, TGetIndexFunc GetIndexFunc, TSlice srcBlock, TSlice dstBlock) {
        int ignored;
        SetElementsFunc(srcBlock.GetConstRef(TVector<TIndexType>()), GetIndexFunc, dstBlock.GetRef(Indices), &ignored);
        SetElementsFunc(srcBlock.GetConstRef(fold.LearnPermutation), GetElement<int>, dstBlock.GetRef(LearnPermutation), &ignored);
        SetElementsFunc(srcBlock.GetConstRef(TVector<int>()), [=](const int*, size_t j) { return srcBlock.Offset + j; }, dstBlock.GetRef(IndexInFold), &ignored);
        SetElementsFunc(srcBlock.GetConstRef(fold.LearnWeights), GetElement<float>, dstBlock.GetRef(LearnWeights), &ignored);
        SetElementsFunc(srcBlock.GetConstRef(fold.SampleWeights), GetElement<float>, dstBlock.GetRef(SampleWeights), &ignored);
        for (int bodyTailIdx = 0; bodyTailIdx < fold.BodyTailArr.ysize(); ++bodyTailIdx) {
            const auto& srcBodyTail = fold.BodyTailArr[bodyTailIdx];
            auto& dstBodyTail = BodyTailArr[bodyTailIdx];
            const auto srcBodyBlock = srcBlock.Clip(srcBodyTail.BodyFinish);
            const auto srcTailBlock = srcBlock.Clip(srcBodyTail.TailFinish);
            int bodyCount = 0;
            int tailCount = 0;
            for (int dim = 0; dim < fold.GetApproxDimension(); ++dim) {
                SetElementsFunc(srcBodyBlock.GetConstRef(srcBodyTail.Derivatives[dim]), GetElement<double>, dstBlock.GetRef(dstBodyTail.Derivatives[dim]), &bodyCount);
                SetElementsFunc(srcTailBlock.GetConstRef(srcBodyTail.WeightedDer[dim]), GetElement<double>, dstBlock.GetRef(dstBodyTail.WeightedDer[dim]), &tailCount);
            }
            AtomicAdd(dstBodyTail.BodyFinish, bodyCount); // these atomics may take up to 2-3% of iteration time
            AtomicAdd(dstBodyTail.TailFinish, tailCount);
        }
    }
};
