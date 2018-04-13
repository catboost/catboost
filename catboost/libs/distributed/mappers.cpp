#include "mappers.h"

#include <catboost/libs/algo/approx_calcer.h>
#include <catboost/libs/algo/error_functions.h>
#include <catboost/libs/helpers/exception.h>

namespace NCatboostDistributed {
void TPlainFoldBuilder::DoMap(NPar::IUserContext* ctx, int hostId, TInput* /*unused*/, TOutput* /*unused*/) const {
    NPar::TCtxPtr<TTrainData> trainData(ctx, SHARED_ID_TRAIN_DATA, hostId);
    auto& localData = TLocalTensorSearchData::GetRef();
    auto& plainFold = localData.PlainFold;
    localData.Rand = new TRestorableFastRng64(trainData->RandomSeed + hostId);
    plainFold = BuildPlainFold(trainData->TrainData,
        trainData->TargetClassifiers,
        /*shuffle*/ false,
        trainData->TrainData.GetSampleCount(),
        trainData->ApproxDimension,
        trainData->StoreExpApprox,
        trainData->HasPairwiseWeights,
        *localData.Rand);
    Y_ASSERT(plainFold.BodyTailArr.ysize() == 1);
    localData.SampledDocs.Create({plainFold}, GetBernoulliSampleRate(localData.Params.ObliviousTreeOptions->BootstrapConfig));
    localData.SmallestSplitSideDocs.Create({plainFold});
    localData.PrevTreeLevelStats.Create({plainFold},
        CountNonCtrBuckets(trainData->SplitCounts, trainData->TrainData.AllFeatures.OneHotValues),
        localData.Params.ObliviousTreeOptions->MaxDepth);
    localData.Indices.yresize(plainFold.LearnPermutation.ysize());
}

void TTensorSearchStarter::DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* /*unused*/, TOutput* /*unused*/) const {
    auto& localData = TLocalTensorSearchData::GetRef();
    localData.Depth = 0;
    Fill(localData.Indices.begin(), localData.Indices.end(), 0);
    localData.PrevTreeLevelStats.GarbageCollect();
}

void TBootstrapMaker::DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* /*unused*/, TOutput* /*unused*/) const {
    auto& localData = TLocalTensorSearchData::GetRef();
    Bootstrap(localData.Params,
        localData.Indices,
        &localData.PlainFold,
        &localData.SampledDocs,
        &NPar::LocalExecutor(),
        localData.Rand.Get());
}

void TScoreCalcer::DoMap(NPar::IUserContext* ctx, int hostId, TInput* candidateList, TOutput* bucketStats) const {
    const TCandidateList& candList = candidateList->Data;
    bucketStats->Data.yresize(candList.ysize());
    NPar::TCtxPtr<TTrainData> trainData(ctx, SHARED_ID_TRAIN_DATA, hostId);
    auto& localData = TLocalTensorSearchData::GetRef();
    NPar::LocalExecutor().ExecRange([&](int id) {
        const auto& candidate = candList[id];
        auto& allScores = bucketStats->Data[id];
        allScores.yresize(candidate.Candidates.ysize());
        NPar::LocalExecutor().ExecRange([&](int oneCandidate) {
            if (candidate.Candidates[oneCandidate].SplitCandidate.Type == ESplitType::OnlineCtr) {
                const auto& proj = candidate.Candidates[oneCandidate].SplitCandidate.Ctr.Projection;
                // so far distributed training works only for floating-point features
                // this assert fails because we do call ComputeOnlineCRTs like we do in single node training
                Y_ASSERT(!localData.PlainFold.GetCtrRef(proj).Feature.empty());
            }
            allScores[oneCandidate] = CalcStats3D(trainData->TrainData.AllFeatures,
                                        trainData->SplitCounts,
                                        localData.PlainFold.GetAllCtrs(),
                                        localData.SampledDocs,
                                        localData.SmallestSplitSideDocs,
                                        localData.Params,
                                        candidate.Candidates[oneCandidate].SplitCandidate,
                                        localData.Depth,
                                        &localData.PrevTreeLevelStats);
        }, NPar::TLocalExecutor::TExecRangeParams(0, candidate.Candidates.ysize())
         , NPar::TLocalExecutor::WAIT_COMPLETE);
    }, 0, candList.ysize(), NPar::TLocalExecutor::WAIT_COMPLETE);
}

void TRemoteBinCalcer::DoMap(NPar::IUserContext* ctx, int hostId, TInput* candidate, TOutput* bucketStats) const { // subcandidates -> TStats4D
    NPar::TCtxPtr<TTrainData> trainData(ctx, SHARED_ID_TRAIN_DATA, hostId);
    auto& localData = TLocalTensorSearchData::GetRef();
    bucketStats->yresize(candidate->Candidates.ysize());
    for (int subcandidateIdx = 0; subcandidateIdx < candidate->Candidates.ysize(); ++subcandidateIdx) {
        if (candidate->Candidates[subcandidateIdx].SplitCandidate.Type == ESplitType::OnlineCtr) {
            const auto& proj = candidate->Candidates[subcandidateIdx].SplitCandidate.Ctr.Projection;
            // so far distributed training works only for floating-point features
            // this assert fails because we do call ComputeOnlineCRTs like we do in single node training
            Y_ASSERT(!localData.PlainFold.GetCtrRef(proj).Feature.empty());
        }
        (*bucketStats)[subcandidateIdx] = CalcStats3D(trainData->TrainData.AllFeatures,
                                        trainData->SplitCounts,
                                        localData.PlainFold.GetAllCtrs(),
                                        localData.SampledDocs,
                                        localData.SmallestSplitSideDocs,
                                        localData.Params,
                                        candidate->Candidates[subcandidateIdx].SplitCandidate,
                                        localData.Depth,
                                        &localData.PrevTreeLevelStats);
    }
}

void TRemoteBinCalcer::DoReduce(TVector<TOutput>* bucketStatsFromAllWorkers, TOutput* bucketStats) const { // vector<TStats4D> -> TStats4D
    const int workerCount = bucketStatsFromAllWorkers->ysize();
    *bucketStats = (*bucketStatsFromAllWorkers)[0];
    const int leafCount = 1U << TLocalTensorSearchData::GetRef().Depth;
    for (int workerIdx = 1; workerIdx < workerCount; ++workerIdx) {
        const int subcandidateCount = bucketStats->ysize();
        for (int subcandidateIdx = 0; subcandidateIdx < subcandidateCount; ++subcandidateIdx) {
            const auto& stats = (*bucketStatsFromAllWorkers)[workerIdx][subcandidateIdx];
            const int splitStatsCount = stats.BucketCount * stats.MaxLeafCount;
            for (int statsIdx = 0; statsIdx * splitStatsCount < stats.Stats.ysize(); ++statsIdx) { // bodytail + dim
                TBucketStats* firstStatsData = GetDataPtr((*bucketStats)[subcandidateIdx].Stats) + statsIdx * splitStatsCount;
                const TBucketStats* statsData = GetDataPtr(stats.Stats) + statsIdx * splitStatsCount;
                for (int bucketIdx = 0; bucketIdx < stats.BucketCount * leafCount; ++bucketIdx) { // bucket, leaf
                    firstStatsData[bucketIdx].Add(statsData[bucketIdx]);
                }
            }
        }
    }
}

void TRemoteScoreCalcer::DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* bucketStats, TOutput* scores) const { // TStats4D -> TVector<TVector<double>> [subcandidate][bucket]
    const int depth = TLocalTensorSearchData::GetRef().Depth;
    const auto& fitParams = TLocalTensorSearchData::GetRef().Params;
    scores->yresize(bucketStats->ysize());
    const int subcandidateCount = bucketStats->ysize();
    for (int subcandidateIdx = 0; subcandidateIdx < subcandidateCount; ++subcandidateIdx) {
        (*scores)[subcandidateIdx] = GetScores(GetScoreBins((*bucketStats)[subcandidateIdx], ESplitType::FloatFeature, depth, fitParams));
    }
}

void TLeafIndexSetter::DoMap(NPar::IUserContext* ctx, int hostId, TInput* bestSplitCandidate, TOutput* /*unused*/) const {
    const TSplit bestSplit(bestSplitCandidate->Data.SplitCandidate, bestSplitCandidate->Data.BestBinBorderId);
    Y_ASSERT(bestSplit.Type != ESplitType::OnlineCtr);
    auto& localData = TLocalTensorSearchData::GetRef();
    NPar::TCtxPtr<TTrainData> trainData(ctx, SHARED_ID_TRAIN_DATA, hostId);
    SetPermutedIndices(bestSplit,
        trainData->TrainData.AllFeatures,
        localData.Depth + 1,
        localData.PlainFold,
        &localData.Indices,
        &NPar::LocalExecutor());
    if (IsSamplingPerTree(localData.Params.ObliviousTreeOptions)) {
        localData.SampledDocs.UpdateIndices(localData.Indices, &NPar::LocalExecutor());
        localData.SmallestSplitSideDocs.SelectSmallestSplitSide(localData.Depth + 1, localData.SampledDocs, &NPar::LocalExecutor());
    }
}

void TEmptyLeafFinder::DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* /*unused*/, TOutput* isLeafEmpty) const {
    auto& localData = TLocalTensorSearchData::GetRef();
    isLeafEmpty->Data = GetIsLeafEmpty(localData.Depth + 1, localData.Indices);
    ++localData.Depth; // tree level completed
}

template<typename TError>
void TBucketSimpleUpdater<TError>::DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* /*unused*/, TOutput* sums) const {
    auto& localData = TLocalTensorSearchData::GetRef();
    const int approxDimension = localData.PlainFold.GetApproxDimension();
    Y_ASSERT(approxDimension == 1);
    const auto error = BuildError<TError>(localData.Params, localData.Objective); // esp: would move to LocalTensorSearchData if not TError
    const auto estimationMethod = localData.Params.ObliviousTreeOptions->LeavesEstimationMethod;
    const int scratchSize = error.GetErrorType() == EErrorType::PerObjectError ? APPROX_BLOCK_SIZE * CB_THREAD_LIMIT
        : localData.PlainFold.BodyTailArr[0].BodyFinish; // plain boosting ==> not approx on full history
    TVector<TDers> weightedDers;
    weightedDers.yresize(scratchSize);

    UpdateBucketsSimple(localData.Indices,
        localData.PlainFold,
        localData.PlainFold.BodyTailArr[0],
        localData.PlainFold.BodyTailArr[0].Approx[0],
        localData.ApproxDeltas[0],
        error,
        localData.PlainFold.BodyTailArr[0].BodyFinish,
        localData.PlainFold.BodyTailArr[0].BodyQueryFinish,
        localData.GradientIteration,
        estimationMethod,
        localData.Params,
        localData.Rand->GenRand(),
        &NPar::LocalExecutor(),
        &localData.Buckets,
        &weightedDers);
    sums->Data = localData.Buckets;
}
template void TBucketSimpleUpdater<TCrossEntropyError>::DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* /*unused*/, TOutput* sums) const;
template void TBucketSimpleUpdater<TRMSEError>::DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* /*unused*/, TOutput* sums) const;
template void TBucketSimpleUpdater<TQuantileError>::DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* /*unused*/, TOutput* sums) const;
template void TBucketSimpleUpdater<TLogLinQuantileError>::DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* /*unused*/, TOutput* sums) const;
template void TBucketSimpleUpdater<TMAPError>::DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* /*unused*/, TOutput* sums) const;
template void TBucketSimpleUpdater<TPoissonError>::DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* /*unused*/, TOutput* sums) const;
template void TBucketSimpleUpdater<TMultiClassError>::DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* /*unused*/, TOutput* sums) const;
template void TBucketSimpleUpdater<TMultiClassOneVsAllError>::DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* /*unused*/, TOutput* sums) const;
template void TBucketSimpleUpdater<TPairLogitError>::DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* /*unused*/, TOutput* sums) const;
template void TBucketSimpleUpdater<TQueryRmseError>::DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* /*unused*/, TOutput* sums) const;
template void TBucketSimpleUpdater<TQuerySoftMaxError>::DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* /*unused*/, TOutput* sums) const;
template void TBucketSimpleUpdater<TCustomError>::DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* /*unused*/, TOutput* sums) const;
template void TBucketSimpleUpdater<TUserDefinedPerObjectError>::DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* /*unused*/, TOutput* sums) const;
template void TBucketSimpleUpdater<TUserDefinedQuerywiseError>::DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* /*unused*/, TOutput* sums) const;

void TCalcApproxStarter::DoMap(NPar::IUserContext* ctx, int hostId, TInput* splitTree, TOutput* /*unused*/) const {
    auto& localData = TLocalTensorSearchData::GetRef();
    Y_ASSERT(localData.PlainFold.GetApproxDimension() == 1);
    NPar::TCtxPtr<TTrainData> trainData(ctx, SHARED_ID_TRAIN_DATA, hostId);
    localData.Indices = BuildIndices(localData.PlainFold,
        splitTree->Data,
        trainData->TrainData,
        /*testData*/ nullptr,
        &NPar::LocalExecutor());
    if (localData.ApproxDeltas.empty()) {
        localData.ApproxDeltas.resize(1); // 1D so far
        localData.ApproxDeltas[0].yresize(localData.PlainFold.BodyTailArr[0].TailFinish);
    }
    Fill(localData.ApproxDeltas[0].begin(), localData.ApproxDeltas[0].end(), GetNeutralApprox(trainData->StoreExpApprox));
    if (localData.Buckets.empty()) {
        localData.Buckets.yresize(splitTree->Data.GetLeafCount());
    }
    Fill(localData.Buckets.begin(), localData.Buckets.end(), TSum(localData.Params.ObliviousTreeOptions->LeavesEstimationIterations));
    if (localData.LeafValues.empty()) {
        localData.LeafValues.yresize(splitTree->Data.GetLeafCount());
    }
    localData.GradientIteration = 0;
}

void TDeltaSimpleUpdater::DoMap(NPar::IUserContext* ctx, int hostId, TInput* sums, TOutput* /*unused*/) const {
    auto& localData = TLocalTensorSearchData::GetRef();
    CalcMixedModelSimple(sums->Data, localData.GradientIteration, localData.Params, &localData.LeafValues);
    NPar::TCtxPtr<TTrainData> trainData(ctx, SHARED_ID_TRAIN_DATA, hostId);
    if (trainData->StoreExpApprox) {
        UpdateApproxDeltas</*StoreExpApprox*/ true>(localData.Indices,
            localData.PlainFold.BodyTailArr[0].TailFinish,
            &NPar::LocalExecutor(),
            &localData.LeafValues,
            &localData.ApproxDeltas[0]);
    } else {
        UpdateApproxDeltas</*StoreExpApprox*/ false>(localData.Indices,
            localData.PlainFold.BodyTailArr[0].TailFinish,
            &NPar::LocalExecutor(),
            &localData.LeafValues,
            &localData.ApproxDeltas[0]);
    }
    ++localData.GradientIteration; // gradient iteration completed
}

void TApproxSimpleUpdater::DoMap(NPar::IUserContext* ctx, int hostId, TInput* /*unused*/, TOutput* /*unused*/) const {
    auto& localData = TLocalTensorSearchData::GetRef();
    NPar::TCtxPtr<TTrainData> trainData(ctx, SHARED_ID_TRAIN_DATA, hostId);
    if (trainData->StoreExpApprox) {
        UpdateBodyTailApprox</*StoreExpApprox*/ true>({localData.ApproxDeltas},
            localData.Params.BoostingOptions->LearningRate,
            &NPar::LocalExecutor(),
            &localData.PlainFold);
    } else {
        UpdateBodyTailApprox</*StoreExpApprox*/ false>({localData.ApproxDeltas},
            localData.Params.BoostingOptions->LearningRate,
            &NPar::LocalExecutor(),
            &localData.PlainFold);
    }
}

template<typename TError>
void TDerivativeSetter<TError>::DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* /*unused*/, TOutput* /*unused*/) const {
    auto& localData = TLocalTensorSearchData::GetRef();
    Y_ASSERT(localData.PlainFold.BodyTailArr.ysize() == 1);
    CalcWeightedDerivatives(BuildError<TError>(localData.Params, localData.Objective),
        /*bodyTailIdx*/ 0,
        localData.Params,
        localData.Rand->GenRand(),
        &localData.PlainFold,
        &NPar::LocalExecutor());
}
template void TDerivativeSetter<TCrossEntropyError>::DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* /*unused*/, TOutput* /*unused*/) const;
template void TDerivativeSetter<TRMSEError>::DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* /*unused*/, TOutput* /*unused*/) const;
template void TDerivativeSetter<TQuantileError>::DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* /*unused*/, TOutput* /*unused*/) const;
template void TDerivativeSetter<TLogLinQuantileError>::DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* /*unused*/, TOutput* /*unused*/) const;
template void TDerivativeSetter<TMAPError>::DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* /*unused*/, TOutput* /*unused*/) const;
template void TDerivativeSetter<TPoissonError>::DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* /*unused*/, TOutput* /*unused*/) const;
template void TDerivativeSetter<TMultiClassError>::DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* /*unused*/, TOutput* /*unused*/) const;
template void TDerivativeSetter<TMultiClassOneVsAllError>::DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* /*unused*/, TOutput* /*unused*/) const;
template void TDerivativeSetter<TPairLogitError>::DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* /*unused*/, TOutput* /*unused*/) const;
template void TDerivativeSetter<TQueryRmseError>::DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* /*unused*/, TOutput* /*unused*/) const;
template void TDerivativeSetter<TQuerySoftMaxError>::DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* /*unused*/, TOutput* /*unused*/) const;
template void TDerivativeSetter<TCustomError>::DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* /*unused*/, TOutput* /*unused*/) const;
template void TDerivativeSetter<TUserDefinedPerObjectError>::DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* /*unused*/, TOutput* /*unused*/) const;
template void TDerivativeSetter<TUserDefinedQuerywiseError>::DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* /*unused*/, TOutput* /*unused*/) const;
} // NCatboostDistributed

using namespace NCatboostDistributed;

REGISTER_SAVELOAD_NM_CLASS(0xd66d481, NCatboostDistributed, TTrainData);
REGISTER_SAVELOAD_NM_CLASS(0xd66d482, NCatboostDistributed, TPlainFoldBuilder);
REGISTER_SAVELOAD_NM_CLASS(0xd66d483, NCatboostDistributed, TTensorSearchStarter);
REGISTER_SAVELOAD_NM_CLASS(0xd66d484, NCatboostDistributed, TBootstrapMaker);
REGISTER_SAVELOAD_NM_CLASS(0xd66d485, NCatboostDistributed, TScoreCalcer);
REGISTER_SAVELOAD_NM_CLASS(0xd66d585, NCatboostDistributed, TRemoteBinCalcer);
REGISTER_SAVELOAD_NM_CLASS(0xd66d685, NCatboostDistributed, TRemoteScoreCalcer);
REGISTER_SAVELOAD_NM_CLASS(0xd66d486, NCatboostDistributed, TLeafIndexSetter);
REGISTER_SAVELOAD_NM_CLASS(0xd66d487, NCatboostDistributed, TEmptyLeafFinder);
REGISTER_SAVELOAD_NM_CLASS(0xd66d488, NCatboostDistributed, TCalcApproxStarter);
REGISTER_SAVELOAD_NM_CLASS(0xd66d489, NCatboostDistributed, TDeltaSimpleUpdater);
REGISTER_SAVELOAD_NM_CLASS(0xd66d48a, NCatboostDistributed, TApproxSimpleUpdater);
REGISTER_SAVELOAD_TEMPL1_NM_CLASS(0xd66d48b, NCatboostDistributed, TEnvelope, TCandidateList);
REGISTER_SAVELOAD_TEMPL1_NM_CLASS(0xd66d48c, NCatboostDistributed, TEnvelope, TStats5D);
REGISTER_SAVELOAD_TEMPL1_NM_CLASS(0xd66d48d, NCatboostDistributed, TEnvelope, TIsLeafEmpty);
REGISTER_SAVELOAD_TEMPL1_NM_CLASS(0xd66d48e, NCatboostDistributed, TEnvelope, TCandidateInfo);
REGISTER_SAVELOAD_TEMPL1_NM_CLASS(0xd66d48f, NCatboostDistributed, TEnvelope, TSplitTree);
REGISTER_SAVELOAD_TEMPL1_NM_CLASS(0xd66d490, NCatboostDistributed, TEnvelope, TSums);
REGISTER_SAVELOAD_TEMPL1_NM_CLASS(0xd66d491, NCatboostDistributed, TBucketSimpleUpdater, TCrossEntropyError);
REGISTER_SAVELOAD_TEMPL1_NM_CLASS(0xd66d493, NCatboostDistributed, TBucketSimpleUpdater, TRMSEError);
REGISTER_SAVELOAD_TEMPL1_NM_CLASS(0xd66d494, NCatboostDistributed, TBucketSimpleUpdater, TQuantileError);
REGISTER_SAVELOAD_TEMPL1_NM_CLASS(0xd66d495, NCatboostDistributed, TBucketSimpleUpdater, TLogLinQuantileError);
REGISTER_SAVELOAD_TEMPL1_NM_CLASS(0xd66d496, NCatboostDistributed, TBucketSimpleUpdater, TMAPError);
REGISTER_SAVELOAD_TEMPL1_NM_CLASS(0xd66d497, NCatboostDistributed, TBucketSimpleUpdater, TPoissonError);
REGISTER_SAVELOAD_TEMPL1_NM_CLASS(0xd66d498, NCatboostDistributed, TBucketSimpleUpdater, TMultiClassError);
REGISTER_SAVELOAD_TEMPL1_NM_CLASS(0xd66d499, NCatboostDistributed, TBucketSimpleUpdater, TMultiClassOneVsAllError);
REGISTER_SAVELOAD_TEMPL1_NM_CLASS(0xd66d49a, NCatboostDistributed, TBucketSimpleUpdater, TPairLogitError);
REGISTER_SAVELOAD_TEMPL1_NM_CLASS(0xd66d49b, NCatboostDistributed, TBucketSimpleUpdater, TQueryRmseError);
REGISTER_SAVELOAD_TEMPL1_NM_CLASS(0xd66d49c, NCatboostDistributed, TBucketSimpleUpdater, TQuerySoftMaxError);
REGISTER_SAVELOAD_TEMPL1_NM_CLASS(0xd66d49d, NCatboostDistributed, TBucketSimpleUpdater, TCustomError);
REGISTER_SAVELOAD_TEMPL1_NM_CLASS(0xd66d49e, NCatboostDistributed, TBucketSimpleUpdater, TUserDefinedPerObjectError);
REGISTER_SAVELOAD_TEMPL1_NM_CLASS(0xd66d49f, NCatboostDistributed, TBucketSimpleUpdater, TUserDefinedQuerywiseError);
REGISTER_SAVELOAD_TEMPL1_NM_CLASS(0xd66d4a0, NCatboostDistributed, TDerivativeSetter, TCrossEntropyError);
REGISTER_SAVELOAD_TEMPL1_NM_CLASS(0xd66d4a2, NCatboostDistributed, TDerivativeSetter, TRMSEError);
REGISTER_SAVELOAD_TEMPL1_NM_CLASS(0xd66d4a3, NCatboostDistributed, TDerivativeSetter, TQuantileError);
REGISTER_SAVELOAD_TEMPL1_NM_CLASS(0xd66d4a4, NCatboostDistributed, TDerivativeSetter, TLogLinQuantileError);
REGISTER_SAVELOAD_TEMPL1_NM_CLASS(0xd66d4a5, NCatboostDistributed, TDerivativeSetter, TMAPError);
REGISTER_SAVELOAD_TEMPL1_NM_CLASS(0xd66d4a6, NCatboostDistributed, TDerivativeSetter, TPoissonError);
REGISTER_SAVELOAD_TEMPL1_NM_CLASS(0xd66d4a7, NCatboostDistributed, TDerivativeSetter, TMultiClassError);
REGISTER_SAVELOAD_TEMPL1_NM_CLASS(0xd66d4a8, NCatboostDistributed, TDerivativeSetter, TMultiClassOneVsAllError);
REGISTER_SAVELOAD_TEMPL1_NM_CLASS(0xd66d4a9, NCatboostDistributed, TDerivativeSetter, TPairLogitError);
REGISTER_SAVELOAD_TEMPL1_NM_CLASS(0xd66d4aa, NCatboostDistributed, TDerivativeSetter, TQueryRmseError);
REGISTER_SAVELOAD_TEMPL1_NM_CLASS(0xd66d4ab, NCatboostDistributed, TDerivativeSetter, TQuerySoftMaxError);
REGISTER_SAVELOAD_TEMPL1_NM_CLASS(0xd66d4ac, NCatboostDistributed, TDerivativeSetter, TCustomError);
REGISTER_SAVELOAD_TEMPL1_NM_CLASS(0xd66d4ad, NCatboostDistributed, TDerivativeSetter, TUserDefinedPerObjectError);
REGISTER_SAVELOAD_TEMPL1_NM_CLASS(0xd66d4ae, NCatboostDistributed, TDerivativeSetter, TUserDefinedQuerywiseError);
