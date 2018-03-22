#include "calc_score_cache.h"
#include <util/system/guard.h>


bool IsSamplingPerTree(const NCatboostOptions::TObliviousTreeLearnerOptions& fitParams) {
    return fitParams.SamplingFrequency.Get() == ESamplingFrequency::PerTree;
}

TVector<TBucketStats, TPoolAllocator>& TBucketStatsCache::GetStats(const TSplitCandidate& split, int statsCount, bool* areStatsDirty) {
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

void TBucketStatsCache::GarbageCollect() {
    if (MemoryPool->MemoryWaste() > InitialSize) { // limit memory overhead
        Stats.clear();
        MemoryPool->Clear();
    }
}


void TCalcScoreFold::TVectorSlicing::Create(const NPar::TLocalExecutor::TExecRangeParams& blockParams) {
    Total = blockParams.LastId;
    Slices.yresize(blockParams.GetBlockCount());
    for (int sliceIdx = 0; sliceIdx < blockParams.GetBlockCount(); ++sliceIdx) {
        Slices[sliceIdx].Offset = blockParams.GetBlockSize() * sliceIdx;
        Slices[sliceIdx].Size = Min(blockParams.GetBlockSize(), Total - Slices[sliceIdx].Offset);
    }
}

void TCalcScoreFold::TVectorSlicing::CreateByControl(const NPar::TLocalExecutor::TExecRangeParams& blockParams, const TUnsizedVector<bool>& control, NPar::TLocalExecutor* localExecutor) {
    Slices.yresize(blockParams.GetBlockCount());
    const bool* controlData = GetDataPtr(control);
    TSlice* slicesData = GetDataPtr(Slices);
    localExecutor->ExecRange([=](int sliceIdx) {
        int sliceSize = 0; // use a local var instead of Slices[sliceIdx].Size so that the compiler can use a register
        NPar::TLocalExecutor::BlockedLoopBody(blockParams, [=, &sliceSize](int doc) { sliceSize += controlData[doc]; })(sliceIdx);
        slicesData[sliceIdx].Size = sliceSize;
    }, 0, blockParams.GetBlockCount(), NPar::TLocalExecutor::WAIT_COMPLETE);
    int offset = 0;
    for (auto& slice : Slices) {
        slice.Offset = offset;
        offset += slice.Size;
    }
    Total = offset;
}

static int GetMaxBodyFinish(const TVector<TFold>& folds, int bodyTailIdx) {
    int maxBodyFinish = 0;
    for (const auto& fold : folds) {
        if (bodyTailIdx < fold.BodyTailArr.ysize()) {
            maxBodyFinish = Max(maxBodyFinish, fold.BodyTailArr[bodyTailIdx].BodyFinish);
        }
    }
    return maxBodyFinish;
}

static int GetMaxTailFinish(const TVector<TFold>& folds, int bodyTailIdx) {
    int maxTailFinish = 0;
    for (const auto& fold : folds) {
        if (bodyTailIdx < fold.BodyTailArr.ysize()) {
            maxTailFinish = Max(maxTailFinish, fold.BodyTailArr[bodyTailIdx].TailFinish);
        }
    }
    return maxTailFinish;
}

void TCalcScoreFold::Create(const TVector<TFold>& folds, float sampleRate) {
    BernoulliSampleRate = sampleRate;
    Y_ASSERT(BernoulliSampleRate > 0.0f && BernoulliSampleRate <= 1.0f);
    DocCount = folds[0].LearnPermutation.ysize();
    Y_ASSERT(DocCount > 0);
    Indices.yresize(DocCount);
    LearnPermutation.yresize(DocCount);
    IndexInFold.yresize(DocCount);
    LearnWeights.yresize(DocCount);
    SampleWeights.yresize(DocCount);
    Control.yresize(DocCount);
    BodyTailCount = GetMaxBodyTailCount(folds);
    HasPairwiseWeights = !folds[0].BodyTailArr[0].PairwiseWeights.empty();
    Y_ASSERT(BodyTailCount > 0);
    BodyTailArr.yresize(BodyTailCount);
    ApproxDimension = folds[0].GetApproxDimension();
    Y_ASSERT(ApproxDimension > 0);
    for (int bodyTailIdx = 0; bodyTailIdx < BodyTailCount; ++bodyTailIdx) {
        BodyTailArr[bodyTailIdx].WeightedDerivatives.yresize(ApproxDimension);
        BodyTailArr[bodyTailIdx].SampleWeightedDerivatives.yresize(ApproxDimension);
        const int bodyFinish = GetMaxBodyFinish(folds, bodyTailIdx);
        Y_ASSERT(bodyFinish > 0);
        const int tailFinish = GetMaxTailFinish(folds, bodyTailIdx);
        Y_ASSERT(tailFinish > 0);
        if (HasPairwiseWeights) {
            BodyTailArr[bodyTailIdx].PairwiseWeights.yresize(tailFinish);
            BodyTailArr[bodyTailIdx].SamplePairwiseWeights.yresize(tailFinish);
        }
        for (int dimIdx = 0; dimIdx < ApproxDimension; ++dimIdx) {
            BodyTailArr[bodyTailIdx].WeightedDerivatives[dimIdx].yresize(bodyFinish);
            BodyTailArr[bodyTailIdx].SampleWeightedDerivatives[dimIdx].yresize(tailFinish);
        }
    }
}

template<typename TSrcRef, typename TGetElementFunc, typename TDstRef>
static inline void SetElements(TArrayRef<const bool> srcControlRef, TSrcRef srcRef, TGetElementFunc GetElementFunc, TDstRef dstRef, int* dstCount) {
    const auto* sourceData = srcRef.data();
    const size_t sourceCount = srcRef.size();
    auto* __restrict destinationData = dstRef.data();
    const size_t destinationCount = dstRef.size();
    if (sourceData != nullptr && srcControlRef.size() == destinationCount) {
        Copy(sourceData, sourceData + sourceCount, destinationData);
        *dstCount = sourceCount;
        return;
    }
    const bool* controlData = srcControlRef.data();
    size_t endElementIdx = 0;
#pragma unroll(4)
    for (size_t sourceIdx = 0; sourceIdx < sourceCount && endElementIdx < destinationCount; ++sourceIdx) {
        destinationData[endElementIdx] = GetElementFunc(sourceData, sourceIdx);
        endElementIdx += controlData[sourceIdx];
    }
    *dstCount = endElementIdx;
}

template<typename TData>
static inline TData GetElement(const TData* source, size_t j) {
    return source[j];
}

template<typename TFoldType>
void TCalcScoreFold::SelectBlockFromFold(const TFoldType& fold, TSlice srcBlock, TSlice dstBlock) {
    int ignored;
    const auto srcControlRef = srcBlock.GetConstRef(Control);
    SetElements(srcControlRef, srcBlock.GetConstRef(fold.LearnPermutation), GetElement<size_t>, dstBlock.GetRef(LearnPermutation), &ignored);
    SetElements(srcControlRef, srcBlock.GetConstRef(fold.LearnWeights), GetElement<float>, dstBlock.GetRef(LearnWeights), &ignored);
    SetElements(srcControlRef, srcBlock.GetConstRef(fold.SampleWeights), GetElement<float>, dstBlock.GetRef(SampleWeights), &ignored);
    for (int bodyTailIdx = 0; bodyTailIdx < BodyTailCount; ++bodyTailIdx) {
        const auto& srcBodyTail = fold.BodyTailArr[bodyTailIdx];
        auto& dstBodyTail = BodyTailArr[bodyTailIdx];
        const auto srcBodyBlock = srcBlock.Clip(srcBodyTail.BodyFinish);
        const auto srcTailBlock = srcBlock.Clip(srcBodyTail.TailFinish);
        int bodyCount = 0;
        int tailCount = 0;
        if (HasPairwiseWeights) {
            SetElements(srcControlRef, srcTailBlock.GetConstRef(srcBodyTail.PairwiseWeights), GetElement<float>, dstBlock.GetRef(dstBodyTail.PairwiseWeights), &tailCount);
            SetElements(srcControlRef, srcTailBlock.GetConstRef(srcBodyTail.SamplePairwiseWeights), GetElement<float>, dstBlock.GetRef(dstBodyTail.SamplePairwiseWeights), &tailCount);
        }
        for (int dim = 0; dim < ApproxDimension; ++dim) {
            SetElements(srcControlRef, srcBodyBlock.GetConstRef(srcBodyTail.WeightedDerivatives[dim]), GetElement<double>, dstBlock.GetRef(dstBodyTail.WeightedDerivatives[dim]), &bodyCount);
            SetElements(srcControlRef, srcTailBlock.GetConstRef(srcBodyTail.SampleWeightedDerivatives[dim]), GetElement<double>, dstBlock.GetRef(dstBodyTail.SampleWeightedDerivatives[dim]), &tailCount);
        }
        AtomicAdd(dstBodyTail.BodyFinish, bodyCount); // these atomics may take up to 2-3% of iteration time
        AtomicAdd(dstBodyTail.TailFinish, tailCount);
    }
}

void TCalcScoreFold::SelectSmallestSplitSide(int curDepth, const TCalcScoreFold& fold, NPar::TLocalExecutor* localExecutor) {
    NPar::TLocalExecutor::TExecRangeParams blockParams(0, fold.DocCount);
    blockParams.SetBlockSize(2000);
    const int blockCount = blockParams.GetBlockCount();

    TVectorSlicing srcBlocks;
    srcBlocks.Create(blockParams);

    TVectorSlicing dstBlocks;
    SetSmallestSideControl(curDepth, fold.DocCount, fold.Indices, localExecutor);
    dstBlocks.CreateByControl(blockParams, Control, localExecutor);

    DocCount = dstBlocks.Total;
    ClearBodyTail();
    localExecutor->ExecRange([&](int blockIdx) {
        int ignored;
        const auto srcBlock = srcBlocks.Slices[blockIdx];
        const auto srcControlRef = srcBlock.GetConstRef(Control);
        const auto srcIndicesRef = srcBlock.GetConstRef(fold.Indices);
        const auto dstBlock = dstBlocks.Slices[blockIdx];
        const TIndexType splitWeight = 1 << (curDepth - 1);
        SetElements(srcControlRef, srcBlock.GetConstRef(TVector<TIndexType>()), [=](const TIndexType*, size_t i) { return srcIndicesRef[i] | splitWeight; }, dstBlock.GetRef(Indices), &ignored);
        SetElements(srcControlRef, srcBlock.GetConstRef(fold.IndexInFold), GetElement<size_t>, dstBlock.GetRef(IndexInFold), &ignored);
        SelectBlockFromFold(fold, srcBlock, dstBlock);
    }, 0, blockCount, NPar::TLocalExecutor::WAIT_COMPLETE);
    PermutationBlockSize = FoldPermutationBlockSizeNotSet;
}

void TCalcScoreFold::Sample(const TFold& fold, const TVector<TIndexType>& indices, TRestorableFastRng64* rand, NPar::TLocalExecutor* localExecutor) {
    NPar::TLocalExecutor::TExecRangeParams blockParams(0, indices.ysize());
    blockParams.SetBlockSize(2000);
    const int blockCount = blockParams.GetBlockCount();
    TVectorSlicing srcBlocks;
    srcBlocks.Create(blockParams);

    TVectorSlicing dstBlocks;
    SetSampledControl(indices.ysize(), rand);
    dstBlocks.CreateByControl(blockParams, Control, localExecutor);

    DocCount = dstBlocks.Total;
    ClearBodyTail();
    BodyTailCount = fold.BodyTailArr.ysize();
    localExecutor->ExecRange([&](int blockIdx) {
        const auto srcBlock = srcBlocks.Slices[blockIdx];
        const auto srcControlRef = srcBlock.GetConstRef(Control);
        const auto dstBlock = dstBlocks.Slices[blockIdx];
        int ignored;
        SetElements(srcControlRef, srcBlock.GetConstRef(indices), GetElement<TIndexType>, dstBlock.GetRef(Indices), &ignored);
        SetElements(srcControlRef, srcBlock.GetConstRef(TVector<size_t>()), [=](const size_t*, size_t j) { return srcBlock.Offset + j; }, dstBlock.GetRef(IndexInFold), &ignored);
        SelectBlockFromFold(fold, srcBlock, dstBlock);
    }, 0, blockCount, NPar::TLocalExecutor::WAIT_COMPLETE);
    PermutationBlockSize = BernoulliSampleRate == 1.0f ? fold.PermutationBlockSize : FoldPermutationBlockSizeNotSet;
}

void TCalcScoreFold::UpdateIndices(const TVector<TIndexType>& indices, NPar::TLocalExecutor* localExecutor) {
    NPar::TLocalExecutor::TExecRangeParams blockParams(0, indices.ysize());
    blockParams.SetBlockSize(2000);
    const int blockCount = blockParams.GetBlockCount();
    TVectorSlicing srcBlocks;
    srcBlocks.Create(blockParams);

    TVectorSlicing dstBlocks;
    if (BernoulliSampleRate < 1.0f) {
        dstBlocks.CreateByControl(blockParams, Control, localExecutor);
    } else {
        dstBlocks = srcBlocks;
    }

    DocCount = dstBlocks.Total;
    localExecutor->ExecRange([&](int blockIdx) {
        const auto srcBlock = srcBlocks.Slices[blockIdx];
        const auto dstBlock = dstBlocks.Slices[blockIdx];
        int ignored;
        const auto srcControlRef = srcBlock.GetConstRef(Control);
        SetElements(srcControlRef, srcBlock.GetConstRef(indices), GetElement<TIndexType>, dstBlock.GetRef(Indices), &ignored);
    }, 0, blockCount, NPar::TLocalExecutor::WAIT_COMPLETE);
}

int TCalcScoreFold::GetApproxDimension() const {
    return ApproxDimension;
}

int TCalcScoreFold::GetDocCount() const {
    return DocCount;
}

int TCalcScoreFold::GetBodyTailCount() const {
    return BodyTailCount;
}

void TCalcScoreFold::SetSmallestSideControl(int curDepth, int docCount, const TUnsizedVector<TIndexType>& indices, NPar::TLocalExecutor* localExecutor) {
    Y_ASSERT(curDepth > 0);

    NPar::TLocalExecutor::TExecRangeParams blockParams(0, docCount);
    blockParams.SetBlockSize(4000);
    const int blockCount = blockParams.GetBlockCount();

    TVector<int> blockSize(blockCount, 0);
    const TIndexType* indicesData = GetDataPtr(indices);
    localExecutor->ExecRange([=, &blockSize](int blockIdx) {
        int size = 0;
        NPar::TLocalExecutor::BlockedLoopBody(blockParams, [=, &size](int docIdx) {
            size += indicesData[docIdx] >> (curDepth - 1);
        })(blockIdx);
        blockSize[blockIdx] = size;
    }, 0, blockCount, NPar::TLocalExecutor::WAIT_COMPLETE);

    int trueCount = 0;
    for (int size : blockSize) {
        trueCount += size;
    }
    const TIndexType splitWeight = 1 << (curDepth - 1);
    bool* controlData = GetDataPtr(Control);
    if (trueCount * 2 > docCount) {
        SmallestSplitSideValue = false;
        localExecutor->ExecRange([=](int docIdx) {
            controlData[docIdx] = indicesData[docIdx] < splitWeight;
        }, blockParams, NPar::TLocalExecutor::WAIT_COMPLETE);
    } else {
        SmallestSplitSideValue = true;
        localExecutor->ExecRange([=](int docIdx) {
            controlData[docIdx] = indicesData[docIdx] > splitWeight - 1;
        }, blockParams, NPar::TLocalExecutor::WAIT_COMPLETE);
    }
}

void TCalcScoreFold::SetSampledControl(int docCount, TRestorableFastRng64* rand) {
    if (BernoulliSampleRate == 1.0f) {
        Fill(Control.begin(), Control.end(), true);
        return;
    }
    for (int docIdx = 0; docIdx < docCount; ++docIdx) {
        Control[docIdx] = rand->GenRandReal1() < BernoulliSampleRate;
    }
}
