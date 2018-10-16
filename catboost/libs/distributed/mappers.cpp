#include "mappers.h"

#include <catboost/libs/algo/approx_calcer.h>
#include <catboost/libs/algo/error_functions.h>
#include <catboost/libs/algo/score_calcer.h>
#include <catboost/libs/helpers/exception.h>

#include <utility>

namespace NCatboostDistributed {
void TPlainFoldBuilder::DoMap(NPar::IUserContext* ctx, int hostId, TInput* /*unused*/, TOutput* /*unused*/) const {
    NPar::TCtxPtr<TTrainData> trainData(ctx, SHARED_ID_TRAIN_DATA, hostId);
    auto& localData = TLocalTensorSearchData::GetRef();
    auto& plainFold = localData.PlainFold;
    localData.Rand = new TRestorableFastRng64(trainData->RandomSeed + hostId);

    NJson::TJsonValue jsonParams;
    const bool jsonParamsOK = ReadJsonTree(trainData->StringParams, &jsonParams);
    Y_ASSERT(jsonParamsOK);
    localData.Params.Load(jsonParams);
    localData.StoreExpApprox = IsStoreExpApprox(localData.Params.LossFunctionDescription->GetLossFunction());
    plainFold = TFold::BuildPlainFold(trainData->TrainData,
        trainData->TargetClassifiers,
        /*shuffle*/ false,
        trainData->TrainData.GetSampleCount(),
        trainData->ApproxDimension,
        localData.StoreExpApprox,
        IsPairwiseError(localData.Params.LossFunctionDescription->GetLossFunction()),
        *localData.Rand);
    Y_ASSERT(plainFold.BodyTailArr.ysize() == 1);
    const bool isPairwiseScoring = IsPairwiseScoring(localData.Params.LossFunctionDescription->GetLossFunction());
    const int defaultCalcStatsObjBlockSize = static_cast<int>(localData.Params.ObliviousTreeOptions->DevScoreCalcObjBlockSize);
    localData.SampledDocs.Create({plainFold}, isPairwiseScoring, defaultCalcStatsObjBlockSize, GetBernoulliSampleRate(localData.Params.ObliviousTreeOptions->BootstrapConfig));
    localData.SmallestSplitSideDocs.Create({plainFold}, isPairwiseScoring, defaultCalcStatsObjBlockSize);
    localData.PrevTreeLevelStats.Create({plainFold},
        CountNonCtrBuckets(trainData->SplitCounts, trainData->TrainData.AllFeatures),
        localData.Params.ObliviousTreeOptions->MaxDepth);
    localData.Indices.yresize(plainFold.LearnPermutation.ysize());
    localData.AllDocCount = trainData->AllDocCount;
    localData.SumAllWeights = trainData->SumAllWeights;
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

template <typename TMapFunc, typename TInputType, typename TOutputType>
static void MapVector(const TMapFunc mapFunc,
    const TVector<TInputType>& inputs,
    TVector<TOutputType>* mappedInputs
) {
    mappedInputs->yresize(inputs.ysize());
    NPar::ParallelFor(0, inputs.ysize(), [&] (int inputIdx) {
        mapFunc(inputs[inputIdx], &(*mappedInputs)[inputIdx]);
    });
}

template <typename TMapFunc, typename TStatsType>
static void MapCandidateList(const TMapFunc mapFunc,
    const TCandidateList& candidates,
    TVector<TVector<TStatsType>>* candidateStats
) {
    const auto mapSubcandidate = [&] (const TCandidateInfo& subcandidate, TStatsType* subcandidateStats) {
        mapFunc(subcandidate, subcandidateStats);
    };
    const auto mapCandidate = [&] (const TCandidatesInfoList& candidate, TVector<TStatsType>* candidateStats) {
        MapVector(mapSubcandidate, candidate.Candidates, candidateStats);
    };
    MapVector(mapCandidate, candidates, candidateStats );
}

static void CalcStats3D(const NPar::TCtxPtr<TTrainData>& trainData,
    const TCandidateInfo& candidate,
    TStats3D* stats3D
) {
    auto& localData = TLocalTensorSearchData::GetRef();
    CalcStatsAndScores(trainData->TrainData.AllFeatures,
        trainData->SplitCounts,
        localData.PlainFold.GetAllCtrs(),
        localData.SampledDocs,
        localData.SmallestSplitSideDocs,
        /*initialFold*/nullptr,
        /*pairs*/{},
        localData.Params,
        candidate.SplitCandidate,
        localData.Depth,
        &NPar::LocalExecutor(),
        &localData.PrevTreeLevelStats,
        stats3D,
        /*pairwiseStats*/nullptr,
        /*scoreBins*/nullptr);
}

static void CalcPairwiseStats(const NPar::TCtxPtr<TTrainData>& trainData,
    const TFlatPairsInfo& pairs,
    const TCandidateInfo& candidate,
    TPairwiseStats* pairwiseStats
) {
    auto& localData = TLocalTensorSearchData::GetRef();
    CalcStatsAndScores(trainData->TrainData.AllFeatures,
        trainData->SplitCounts,
        localData.PlainFold.GetAllCtrs(),
        localData.SampledDocs,
        localData.SmallestSplitSideDocs,
        /*initialFold*/nullptr,
        pairs,
        localData.Params,
        candidate.SplitCandidate,
        localData.Depth,
        &NPar::LocalExecutor(),
        &localData.PrevTreeLevelStats,
        /*stats3D*/nullptr,
        pairwiseStats,
        /*scoreBins*/nullptr);
}

void TScoreCalcer::DoMap(NPar::IUserContext* ctx, int hostId, TInput* candidateList, TOutput* bucketStats) const {
    NPar::TCtxPtr<TTrainData> trainData(ctx, SHARED_ID_TRAIN_DATA, hostId);
    auto calcStats3D = [&](const TCandidateInfo& candidate, TStats3D* stats3D) {
        CalcStats3D(trainData, candidate, stats3D);
    };
    MapCandidateList(calcStats3D, candidateList->Data, &bucketStats->Data);
}

void TPairwiseScoreCalcer::DoMap(NPar::IUserContext* ctx, int hostId, TInput* candidateList, TOutput* bucketStats) const {
    NPar::TCtxPtr<TTrainData> trainData(ctx, SHARED_ID_TRAIN_DATA, hostId);
    auto& localData = TLocalTensorSearchData::GetRef();
    const auto pairs = UnpackPairsFromQueries(localData.PlainFold.LearnQueriesInfo);
    auto calcPairwiseStats = [&](const TCandidateInfo& candidate, TPairwiseStats* pairwiseStats) {
        CalcPairwiseStats(trainData, pairs, candidate, pairwiseStats);
    };
    MapCandidateList(calcPairwiseStats, candidateList->Data, &bucketStats->Data);
}

void TRemotePairwiseBinCalcer::DoMap(NPar::IUserContext* ctx, int hostId, TInput* candidate, TOutput* bucketStats) const { // buckets -> workerPairwiseStats
    NPar::TCtxPtr<TTrainData> trainData(ctx, SHARED_ID_TRAIN_DATA, hostId);
    auto& localData = TLocalTensorSearchData::GetRef();
    const auto pairs = UnpackPairsFromQueries(localData.PlainFold.LearnQueriesInfo);
    auto calcPairwiseStats = [&](const TCandidateInfo& candidate, TPairwiseStats* pairwiseStats) {
        CalcPairwiseStats(trainData, pairs, candidate, pairwiseStats);
    };
    MapVector(calcPairwiseStats, candidate->Candidates, bucketStats);
}

void TRemotePairwiseBinCalcer::DoReduce(TVector<TOutput>* statsFromAllWorkers, TOutput* stats) const { // workerPairwiseStats -> pairwiseStats
    const int workerCount = statsFromAllWorkers->ysize();
    const int bucketCount = (*statsFromAllWorkers)[0].ysize();
    stats->yresize(bucketCount);
    NPar::ParallelFor(0, bucketCount, [&] (int bucketIdx) {
        (*stats)[bucketIdx] = (*statsFromAllWorkers)[0][bucketIdx];
        for (int workerIdx : xrange(1, workerCount)) {
            (*stats)[bucketIdx].Add((*statsFromAllWorkers)[workerIdx][bucketIdx]);
        }
    });
}

void TRemotePairwiseScoreCalcer::DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* bucketStats, TOutput* scores) const { // TStats4D -> TVector<TVector<double>> [subcandidate][bucket]
    const auto& localData = TLocalTensorSearchData::GetRef();
    const int bucketCount = (*bucketStats)[0].DerSums[0].ysize();
    const auto getScores = [&] (const TPairwiseStats& candidatePairwiseStats, TVector<double>* candidateScores) {
        TVector<TScoreBin> scoreBins;
        CalculatePairwiseScore(
            candidatePairwiseStats,
            bucketCount,
            ESplitType::FloatFeature,
            localData.Params.ObliviousTreeOptions->L2Reg,
            localData.Params.ObliviousTreeOptions->PairwiseNonDiagReg,
            &scoreBins
        );
        *candidateScores = GetScores(scoreBins);
    };
    MapVector(getScores, *bucketStats, scores);
}

void TRemoteBinCalcer::DoMap(NPar::IUserContext* ctx, int hostId, TInput* candidate, TOutput* bucketStats) const { // subcandidates -> TStats4D
    NPar::TCtxPtr<TTrainData> trainData(ctx, SHARED_ID_TRAIN_DATA, hostId);
    auto calcStats3D = [&](const TCandidateInfo& candidate, TStats3D* stats3D) {
        CalcStats3D(trainData, candidate, stats3D);
    };
    MapVector(calcStats3D, candidate->Candidates, bucketStats);
}

void TRemoteBinCalcer::DoReduce(TVector<TOutput>* statsFromAllWorkers, TOutput* stats) const { // vector<TStats4D> -> TStats4D
    const int workerCount = statsFromAllWorkers->ysize();
    const int bucketCount = (*statsFromAllWorkers)[0].ysize();
    stats->yresize(bucketCount);
    NPar::ParallelFor(0, bucketCount, [&] (int bucketIdx) {
        (*stats)[bucketIdx] = (*statsFromAllWorkers)[0][bucketIdx];
        for (int workerIdx = 1; workerIdx < workerCount; ++workerIdx) {
            (*stats)[bucketIdx].Add((*statsFromAllWorkers)[workerIdx][bucketIdx]);
        }
    });
}

void TRemoteScoreCalcer::DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* bucketStats, TOutput* scores) const { // TStats4D -> TVector<TVector<double>> [subcandidate][bucket]
    const auto& localData = TLocalTensorSearchData::GetRef();
    const auto getScores = [&] (const TStats3D& candidateStats3D, TVector<double>* candidateScores) {
        *candidateScores = GetScores(GetScoreBins(candidateStats3D, ESplitType::FloatFeature, localData.Depth, localData.SumAllWeights, localData.AllDocCount, localData.Params));
    };
    MapVector(getScores, *bucketStats, scores);
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
        if (!IsPairwiseScoring(localData.Params.LossFunctionDescription->GetLossFunction())) {
            localData.SmallestSplitSideDocs.SelectSmallestSplitSide(localData.Depth + 1, localData.SampledDocs, &NPar::LocalExecutor());
        }
    }
}

void TEmptyLeafFinder::DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* /*unused*/, TOutput* isLeafEmpty) const {
    auto& localData = TLocalTensorSearchData::GetRef();
    isLeafEmpty->Data = GetIsLeafEmpty(localData.Depth + 1, localData.Indices);
    ++localData.Depth; // tree level completed
}

template <typename TError>
void TBucketSimpleUpdater<TError>::DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* /*unused*/, TOutput* sums) const {
    auto& localData = TLocalTensorSearchData::GetRef();
    const int approxDimension = localData.PlainFold.GetApproxDimension();
    Y_ASSERT(approxDimension == 1);
    const auto error = BuildError<TError>(localData.Params, /*custom objective*/ Nothing());
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
        &localData.PairwiseBuckets,
        &weightedDers);
    sums->Data = std::make_pair(localData.Buckets, localData.PairwiseBuckets);
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
template void TBucketSimpleUpdater<TLqError>::DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* /*unused*/, TOutput* sums) const;

template <> void TBucketSimpleUpdater<TCustomError>::DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* /*unused*/, TOutput* /*sums*/) const {
    CB_ENSURE(false, "Custom objective not supported in distributed training");
}
template void TBucketSimpleUpdater<TUserDefinedPerObjectError>::DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* /*unused*/, TOutput* sums) const;
template void TBucketSimpleUpdater<TUserDefinedQuerywiseError>::DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* /*unused*/, TOutput* sums) const;

void TCalcApproxStarter::DoMap(NPar::IUserContext* ctx, int hostId, TInput* splitTree, TOutput* /*unused*/) const {
    auto& localData = TLocalTensorSearchData::GetRef();
    NPar::TCtxPtr<TTrainData> trainData(ctx, SHARED_ID_TRAIN_DATA, hostId);
    localData.Indices = BuildIndices(localData.PlainFold,
        splitTree->Data,
        trainData->TrainData,
        /*testDataPtrs*/ {},
        &NPar::LocalExecutor());
    const int approxDimension = localData.PlainFold.GetApproxDimension();
    if (localData.ApproxDeltas.empty()) {
        localData.ApproxDeltas.resize(approxDimension); // 1D or nD
        for (auto& dimensionDelta : localData.ApproxDeltas) {
            dimensionDelta.yresize(localData.PlainFold.BodyTailArr[0].TailFinish);
        }
    }
    for (auto& dimensionDelta : localData.ApproxDeltas) {
        Fill(dimensionDelta.begin(), dimensionDelta.end(), GetNeutralApprox(localData.StoreExpApprox));
    }
    localData.Buckets.resize(splitTree->Data.GetLeafCount());
    Fill(localData.Buckets.begin(), localData.Buckets.end(), TSum(localData.Params.ObliviousTreeOptions->LeavesEstimationIterations));
    localData.MultiBuckets.resize(splitTree->Data.GetLeafCount());
    Fill(localData.MultiBuckets.begin(), localData.MultiBuckets.end(),
        TSumMulti(localData.Params.ObliviousTreeOptions->LeavesEstimationIterations, approxDimension, trainData->HessianType)
    );
    localData.PairwiseBuckets.SetSizes(splitTree->Data.GetLeafCount(), splitTree->Data.GetLeafCount());
    localData.PairwiseBuckets.FillZero();
    localData.LeafValues.yresize(approxDimension);
    for (auto& dimensionLeafValues : localData.LeafValues) {
        dimensionLeafValues.yresize(splitTree->Data.GetLeafCount());
    }
    localData.GradientIteration = 0;
}

void TDeltaSimpleUpdater::DoMap(NPar::IUserContext* /*unused*/, int /*unused*/, TInput* sums, TOutput* /*unused*/) const {
    auto& localData = TLocalTensorSearchData::GetRef();
    CalcMixedModelSimple(/*individual*/ sums->Data.first, /*pairwise*/ sums->Data.second, localData.GradientIteration, localData.Params, localData.SumAllWeights, localData.AllDocCount, &localData.LeafValues[0]);
    if (localData.StoreExpApprox) {
        UpdateApproxDeltas</*StoreExpApprox*/ true>(localData.Indices,
            localData.PlainFold.BodyTailArr[0].TailFinish,
            &NPar::LocalExecutor(),
            &localData.LeafValues[0],
            &localData.ApproxDeltas[0]);
    } else {
        UpdateApproxDeltas</*StoreExpApprox*/ false>(localData.Indices,
            localData.PlainFold.BodyTailArr[0].TailFinish,
            &NPar::LocalExecutor(),
            &localData.LeafValues[0],
            &localData.ApproxDeltas[0]);
    }
    ++localData.GradientIteration; // gradient iteration completed
}

void TApproxUpdater::DoMap(NPar::IUserContext* /*unused*/, int /*unused*/, TInput* /*unused*/, TOutput* /*unused*/) const {
    auto& localData = TLocalTensorSearchData::GetRef();
    if (localData.StoreExpApprox) {
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

template <typename TError>
void TDerivativeSetter<TError>::DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* /*unused*/, TOutput* /*unused*/) const {
    auto& localData = TLocalTensorSearchData::GetRef();
    Y_ASSERT(localData.PlainFold.BodyTailArr.ysize() == 1);
    CalcWeightedDerivatives(BuildError<TError>(localData.Params, /*custom objective*/ Nothing()),
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
template void TDerivativeSetter<TLqError>::DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* /*unused*/, TOutput* /*unused*/) const;

template <> void TDerivativeSetter<TCustomError>::DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* /*unused*/, TOutput* /*unused*/) const {
    CB_ENSURE(false, "Custom objective not supported in distributed training");
}
template void TDerivativeSetter<TUserDefinedPerObjectError>::DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* /*unused*/, TOutput* /*unused*/) const;
template void TDerivativeSetter<TUserDefinedQuerywiseError>::DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* /*unused*/, TOutput* /*unused*/) const;

template <typename TError>
void TBucketMultiUpdater<TError>::DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* /*unused*/, TOutput* sums) const {
    auto& localData = TLocalTensorSearchData::GetRef();
    const int approxDimension = localData.PlainFold.GetApproxDimension();
    Y_ASSERT(approxDimension > 1);
    const auto error = BuildError<TError>(localData.Params, /*custom objective*/ Nothing());
    const auto estimationMethod = localData.Params.ObliviousTreeOptions->LeavesEstimationMethod;

    if (estimationMethod == ELeavesEstimation::Newton) {
        UpdateBucketsMulti(AddSampleToBucketNewtonMulti<TError>,
            localData.Indices,
            localData.PlainFold.LearnTarget,
            localData.PlainFold.GetLearnWeights(),
            localData.PlainFold.BodyTailArr[0].Approx,
            localData.ApproxDeltas,
            error,
            localData.PlainFold.BodyTailArr[0].BodyFinish,
            localData.GradientIteration,
            &localData.MultiBuckets);
    } else {
        Y_ASSERT(estimationMethod == ELeavesEstimation::Gradient);
        UpdateBucketsMulti(AddSampleToBucketGradientMulti<TError>,
            localData.Indices,
            localData.PlainFold.LearnTarget,
            localData.PlainFold.GetLearnWeights(),
            localData.PlainFold.BodyTailArr[0].Approx,
            localData.ApproxDeltas,
            error,
            localData.PlainFold.BodyTailArr[0].BodyFinish,
            localData.GradientIteration,
            &localData.MultiBuckets);
    }
    sums->Data = std::make_pair(localData.MultiBuckets, TUnusedInitializedParam());
}
template void TBucketMultiUpdater<TCrossEntropyError>::DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* /*unused*/, TOutput* sums) const;
template void TBucketMultiUpdater<TRMSEError>::DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* /*unused*/, TOutput* sums) const;
template void TBucketMultiUpdater<TQuantileError>::DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* /*unused*/, TOutput* sums) const;
template void TBucketMultiUpdater<TLogLinQuantileError>::DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* /*unused*/, TOutput* sums) const;
template void TBucketMultiUpdater<TMAPError>::DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* /*unused*/, TOutput* sums) const;
template void TBucketMultiUpdater<TPoissonError>::DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* /*unused*/, TOutput* sums) const;
template void TBucketMultiUpdater<TMultiClassError>::DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* /*unused*/, TOutput* sums) const;
template void TBucketMultiUpdater<TMultiClassOneVsAllError>::DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* /*unused*/, TOutput* sums) const;
template void TBucketMultiUpdater<TPairLogitError>::DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* /*unused*/, TOutput* sums) const;
template void TBucketMultiUpdater<TQueryRmseError>::DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* /*unused*/, TOutput* sums) const;
template void TBucketMultiUpdater<TQuerySoftMaxError>::DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* /*unused*/, TOutput* sums) const;
template void TBucketMultiUpdater<TLqError>::DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* /*unused*/, TOutput* sums) const;

template <> void TBucketMultiUpdater<TCustomError>::DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* /*unused*/, TOutput* /*sums*/) const {
    CB_ENSURE(false, "Custom objective not supported in distributed training");
}
template void TBucketMultiUpdater<TUserDefinedPerObjectError>::DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* /*unused*/, TOutput* sums) const;
template void TBucketMultiUpdater<TUserDefinedQuerywiseError>::DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* /*unused*/, TOutput* sums) const;


void TDeltaMultiUpdater::DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* sums, TOutput* /*unused*/) const {
    auto& localData = TLocalTensorSearchData::GetRef();
    const auto estimationMethod = localData.Params.ObliviousTreeOptions->LeavesEstimationMethod;
    const float l2Regularizer = localData.Params.ObliviousTreeOptions->L2Reg;

    if (estimationMethod == ELeavesEstimation::Newton) {
        CalcMixedModelMulti(CalcModelNewtonMulti,
            sums->Data.first,
            localData.GradientIteration,
            l2Regularizer,
            localData.SumAllWeights,
            localData.AllDocCount,
            &localData.LeafValues);
    } else {
        Y_ASSERT(estimationMethod == ELeavesEstimation::Gradient);
        CalcMixedModelMulti(CalcModelGradientMulti,
            sums->Data.first,
            localData.GradientIteration,
            l2Regularizer,
            localData.SumAllWeights,
            localData.AllDocCount,
            &localData.LeafValues);
    }
    if (localData.StoreExpApprox) {
        UpdateApproxDeltasMulti</*StoreExpApprox*/ true>(localData.Indices,
            localData.PlainFold.BodyTailArr[0].BodyFinish,
            &localData.LeafValues,
            &localData.ApproxDeltas);
    } else {
        UpdateApproxDeltasMulti</*StoreExpApprox*/ false>(localData.Indices,
            localData.PlainFold.BodyTailArr[0].BodyFinish,
            &localData.LeafValues,
            &localData.ApproxDeltas);
    }
    ++localData.GradientIteration; // gradient iteration completed
}

void TErrorCalcer::DoMap(NPar::IUserContext* ctx, int hostId, TInput* /*unused*/, TOutput* additiveStats) const {
    const auto& localData = TLocalTensorSearchData::GetRef();
    const auto errors = CreateMetrics(
        localData.Params.LossFunctionDescription,
        localData.Params.MetricOptions,
        /*evalMetricDescriptor*/Nothing(),
        /*approxDimension*/localData.ApproxDeltas.ysize()
    );
    const auto skipMetricOnTrain = GetSkipMetricOnTrain(errors);
    NPar::TCtxPtr<TTrainData> trainData(ctx, SHARED_ID_TRAIN_DATA, hostId);
    for (int errorIdx = 0; errorIdx < errors.ysize(); ++errorIdx) {
        if (!skipMetricOnTrain[errorIdx] && errors[errorIdx]->IsAdditiveMetric()) {
            const TString metricDescription = errors[errorIdx]->GetDescription();
            (*additiveStats)[metricDescription] = EvalErrors(
                localData.PlainFold.BodyTailArr[0].Approx,
                trainData->TrainData.Target,
                trainData->TrainData.Weights,
                trainData->TrainData.QueryInfo,
                errors[errorIdx],
                &NPar::LocalExecutor()
            );
        }
    }
}

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
REGISTER_SAVELOAD_NM_CLASS(0xd66d48a, NCatboostDistributed, TApproxUpdater);
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
REGISTER_SAVELOAD_TEMPL1_NM_CLASS(0xd66d50f, NCatboostDistributed, TBucketSimpleUpdater, TLqError);
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
REGISTER_SAVELOAD_TEMPL1_NM_CLASS(0xd66d4af, NCatboostDistributed, TDerivativeSetter, TLqError);

REGISTER_SAVELOAD_NM_CLASS(0xd66d4b2, NCatboostDistributed, TDeltaMultiUpdater);
REGISTER_SAVELOAD_TEMPL1_NM_CLASS(0xd66d4b1, NCatboostDistributed, TBucketMultiUpdater, TCrossEntropyError);
REGISTER_SAVELOAD_TEMPL1_NM_CLASS(0xd66d4b3, NCatboostDistributed, TBucketMultiUpdater, TRMSEError);
REGISTER_SAVELOAD_TEMPL1_NM_CLASS(0xd66d4b4, NCatboostDistributed, TBucketMultiUpdater, TQuantileError);
REGISTER_SAVELOAD_TEMPL1_NM_CLASS(0xd66d4b5, NCatboostDistributed, TBucketMultiUpdater, TLogLinQuantileError);
REGISTER_SAVELOAD_TEMPL1_NM_CLASS(0xd66d4b6, NCatboostDistributed, TBucketMultiUpdater, TMAPError);
REGISTER_SAVELOAD_TEMPL1_NM_CLASS(0xd66d4b7, NCatboostDistributed, TBucketMultiUpdater, TPoissonError);
REGISTER_SAVELOAD_TEMPL1_NM_CLASS(0xd66d4b8, NCatboostDistributed, TBucketMultiUpdater, TMultiClassError);
REGISTER_SAVELOAD_TEMPL1_NM_CLASS(0xd66d4b9, NCatboostDistributed, TBucketMultiUpdater, TMultiClassOneVsAllError);
REGISTER_SAVELOAD_TEMPL1_NM_CLASS(0xd66d4ba, NCatboostDistributed, TBucketMultiUpdater, TPairLogitError);
REGISTER_SAVELOAD_TEMPL1_NM_CLASS(0xd66d4bb, NCatboostDistributed, TBucketMultiUpdater, TQueryRmseError);
REGISTER_SAVELOAD_TEMPL1_NM_CLASS(0xd66d4bc, NCatboostDistributed, TBucketMultiUpdater, TQuerySoftMaxError);
REGISTER_SAVELOAD_TEMPL1_NM_CLASS(0xd66d4bd, NCatboostDistributed, TBucketMultiUpdater, TCustomError);
REGISTER_SAVELOAD_TEMPL1_NM_CLASS(0xd66d4be, NCatboostDistributed, TBucketMultiUpdater, TUserDefinedPerObjectError);
REGISTER_SAVELOAD_TEMPL1_NM_CLASS(0xd66d4bf, NCatboostDistributed, TBucketMultiUpdater, TUserDefinedQuerywiseError);
REGISTER_SAVELOAD_TEMPL1_NM_CLASS(0xd66d4c1, NCatboostDistributed, TBucketMultiUpdater, TLqError);

REGISTER_SAVELOAD_NM_CLASS(0xd66d4d1, NCatboostDistributed, TPairwiseScoreCalcer);
REGISTER_SAVELOAD_NM_CLASS(0xd66d4d2, NCatboostDistributed, TRemotePairwiseBinCalcer);
REGISTER_SAVELOAD_NM_CLASS(0xd66d4d3, NCatboostDistributed, TRemotePairwiseScoreCalcer);
REGISTER_SAVELOAD_NM_CLASS(0xd66d4d4, NCatboostDistributed, TErrorCalcer);
