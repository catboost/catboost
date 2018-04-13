#include "master.h"
#include "mappers.h"

#include <catboost/libs/algo/error_functions.h>
#include <catboost/libs/algo/index_calcer.h>
#include <catboost/libs/helpers/data_split.h>

#include <library/par/par_settings.h>

using namespace NCatboostDistributed;

template<typename TData>
static TVector<TData> GetWorkerPart(const TVector<TData>& column, const std::pair<size_t, size_t>& part) {
    const size_t columnSize = column.size();
    if (part.first >= columnSize) {
        return TVector<TData>();
    }
    const auto& columnBegin = column.begin();
    return TVector<TData>(columnBegin + part.first, columnBegin + Min(part.second, columnSize));
}

template<typename TData>
static TVector<TVector<TData>> GetWorkerPart(const TVector<TVector<TData>>& masterTable, const std::pair<size_t, size_t>& part) {
    TVector<TVector<TData>> workerPart;
    workerPart.reserve(masterTable.ysize());
    for (const auto& masterColumn : masterTable) {
        workerPart.emplace_back(GetWorkerPart(masterColumn, part));
    }
    return workerPart;
}

static TAllFeatures GetWorkerPart(const TAllFeatures& allFeatures, const std::pair<size_t, size_t>& part) {
    TAllFeatures workerPart;
    workerPart.FloatHistograms = GetWorkerPart(allFeatures.FloatHistograms, part);
    workerPart.CatFeaturesRemapped = GetWorkerPart(allFeatures.CatFeaturesRemapped, part);
    workerPart.OneHotValues = GetWorkerPart(allFeatures.OneHotValues, part);
    workerPart.IsOneHot = allFeatures.IsOneHot;
    return workerPart;
}

static ::TDataset GetWorkerPart(const ::TDataset& trainData, const std::pair<size_t, size_t>& part) {
    ::TDataset workerPart;
    workerPart.AllFeatures = GetWorkerPart(trainData.AllFeatures, part);
    workerPart.Baseline = GetWorkerPart(trainData.Baseline, part);
    workerPart.Target = GetWorkerPart(trainData.Target, part);
    workerPart.Weights = GetWorkerPart(trainData.Weights, part);
    workerPart.QueryId = GetWorkerPart(trainData.QueryId, part);
    workerPart.QueryInfo = GetWorkerPart(trainData.QueryInfo, part);
    workerPart.Pairs = GetWorkerPart(trainData.Pairs, part);
    return workerPart;
}

template<typename TMapper>
static TVector<typename TMapper::TOutput> ApplyMapper(int workerCount, TObj<NPar::IEnvironment> environment, const typename TMapper::TInput& value = typename TMapper::TInput()) {
    NPar::TJobDescription job;
    TVector<typename TMapper::TInput> mapperInput(1);
    mapperInput[0] = value;
    NPar::Map(&job, new TMapper(), &mapperInput);
    job.SeparateResults(workerCount);
    NPar::TJobExecutor exec(&job, environment);
    TVector<typename TMapper::TOutput> mapperOutput;
    exec.GetResultVec(&mapperOutput);
    return mapperOutput;
}

void InitializeMaster(TLearnContext* ctx) {
    Y_ASSERT(ctx->Params.SystemOptions->IsMaster());
    const auto& systemOptions = ctx->Params.SystemOptions;
    const ui32 unusedNodePort = NCatboostOptions::TSystemOptions::GetUnusedNodePort();
    NPar::TParNetworkSettings::GetRef().RequesterType = NPar::TParNetworkSettings::ERequesterType::NEH; // avoid Netliba
    ctx->RootEnvironment = NPar::RunMaster(systemOptions->NodePort,
        systemOptions->NumThreads,
        systemOptions->FileWithHosts->c_str(),
        unusedNodePort,
        unusedNodePort);
    const int workerCount = ctx->RootEnvironment->GetSlaveCount();
    const auto& workerMapping = ctx->RootEnvironment->MakeHostIdMapping(workerCount);
    ctx->SharedTrainData = ctx->RootEnvironment->CreateEnvironment(SHARED_ID_TRAIN_DATA, workerMapping);
}

void FinalizeMaster(TLearnContext* ctx) {
    Y_ASSERT(ctx->Params.SystemOptions->IsMaster());
    if (ctx->RootEnvironment != nullptr) {
        ctx->RootEnvironment->Stop();
    }
}

void MapBuildPlainFold(const ::TDataset& trainData, TLearnContext* ctx) {
    Y_ASSERT(ctx->Params.SystemOptions->IsMaster());
    const auto& plainFold = ctx->LearnProgress.Folds[0];
    Y_ASSERT(plainFold.PermutationBlockSize == plainFold.LearnPermutation.ysize());
    const ui64 randomSeed = ctx->Rand.GenRand();
    const auto& splitCounts = CountSplits(ctx->LearnProgress.FloatFeatures);
    const auto& targetClassifiers = ctx->CtrsHelper.GetTargetClassifiers();
    const int workerCount = ctx->RootEnvironment->GetSlaveCount();
    TVector<std::pair<size_t, size_t>> workerParts;
    if (trainData.QueryId.empty()) {
        workerParts = Split(trainData.GetSampleCount(), workerCount);
    } else {
        workerParts = Split(trainData.GetSampleCount(), trainData.QueryId, workerCount);
    }
    for (int workerIdx = 0; workerIdx < workerCount; ++workerIdx) {
        ctx->SharedTrainData->SetContextData(workerIdx,
            new NCatboostDistributed::TTrainData(GetWorkerPart(trainData, workerParts[workerIdx]),
                targetClassifiers,
                splitCounts,
                randomSeed,
                ctx->LearnProgress.ApproxDimension,
                IsStoreExpApprox(ctx->Params.LossFunctionDescription->GetLossFunction()),
                IsPairwiseError(ctx->Params.LossFunctionDescription->GetLossFunction())),
            NPar::DELETE_RAW_DATA); // only workers
    }
    ApplyMapper<TPlainFoldBuilder>(workerCount, ctx->SharedTrainData);
}

void MapTensorSearchStart(TLearnContext* ctx) {
    Y_ASSERT(ctx->Params.SystemOptions->IsMaster());
    ApplyMapper<TTensorSearchStarter>(ctx->RootEnvironment->GetSlaveCount(), ctx->SharedTrainData);
}

void MapBootstrap(TLearnContext* ctx) {
    Y_ASSERT(ctx->Params.SystemOptions->IsMaster());
    ApplyMapper<TBootstrapMaker>(ctx->RootEnvironment->GetSlaveCount(), ctx->SharedTrainData);
}

void MapCalcScore(double scoreStDev, int depth, TCandidateList* candidateList, TLearnContext* ctx) {
    Y_ASSERT(ctx->Params.SystemOptions->IsMaster());
    const int workerCount = ctx->RootEnvironment->GetSlaveCount();
    TVector<TScoreCalcer::TOutput> allStatsFromAllWorkers = ApplyMapper<TScoreCalcer>(workerCount, ctx->SharedTrainData, TEnvelope<TCandidateList>(*candidateList));
    // reduce aross workers
    const int leafCount = 1U << depth;
    const int candidateCount = candidateList->ysize();
    for (int workerIdx = 1; workerIdx < workerCount; ++workerIdx) {
        ctx->LocalExecutor.ExecRange([&] (int candidateIdx) {
            const int subcandidateCount = (*candidateList)[candidateIdx].Candidates.ysize();
            for (int subcandidateIdx = 0; subcandidateIdx < subcandidateCount; ++subcandidateIdx) {
                auto& firstStats = allStatsFromAllWorkers[0].Data[candidateIdx][subcandidateIdx];
                const auto& stats = allStatsFromAllWorkers[workerIdx].Data[candidateIdx][subcandidateIdx];
                const int splitStatsCount = firstStats.BucketCount * firstStats.MaxLeafCount;
                for (int statsIdx = 0; statsIdx * splitStatsCount < firstStats.Stats.ysize(); ++statsIdx) {
                    TBucketStats* firstStatsData = GetDataPtr(firstStats.Stats) + statsIdx * splitStatsCount;
                    const TBucketStats* statsData = GetDataPtr(stats.Stats) + statsIdx * splitStatsCount;
                    for (int bucketIdx = 0; bucketIdx < firstStats.BucketCount * leafCount; ++bucketIdx) {
                        firstStatsData[bucketIdx].Add(statsData[bucketIdx]);
                    }
                }
            }
        }, NPar::TLocalExecutor::TExecRangeParams(0, candidateCount), NPar::TLocalExecutor::WAIT_COMPLETE);
    }
    // set best split for each candidate
    const ui64 randSeed = ctx->Rand.GenRand();
    ctx->LocalExecutor.ExecRange([&] (int candidateIdx) {
        const auto& allStats = allStatsFromAllWorkers[0].Data[candidateIdx];
        auto& candidate = (*candidateList)[candidateIdx];
        const int subcandidateCount = candidate.Candidates.ysize();
        TVector<TVector<double>> allScores(subcandidateCount);
        for (int subcandidateIdx = 0; subcandidateIdx < subcandidateCount; ++subcandidateIdx) {
            const auto& splitInfo = candidate.Candidates[subcandidateIdx];
            allScores[subcandidateIdx] = GetScores(GetScoreBins(allStats[subcandidateIdx], splitInfo.SplitCandidate.Type, depth, ctx->Params));
        }
        SetBestScore(randSeed + candidateIdx, allScores, scoreStDev, &candidate.Candidates);
    }, 0, candidateCount, NPar::TLocalExecutor::WAIT_COMPLETE);
}

void MapRemoteCalcScore(double scoreStDev, int /*depth*/, TCandidateList* candidateList, TLearnContext* ctx) {
    Y_ASSERT(ctx->Params.SystemOptions->IsMaster());
    NPar::TJobDescription job;
    NPar::Map(&job, new TRemoteBinCalcer(), candidateList); // candidateList[i] -map-> {TStats4D[i][worker]} -reduce-> TStats4D[i]
    NPar::RemoteMap(&job, new TRemoteScoreCalcer); // TStats4D[i] -remote_map-> Scores[i]
    NPar::TJobExecutor exec(&job, ctx->SharedTrainData);
    TVector<typename TRemoteScoreCalcer::TOutput> allScores; // [candidate][subcandidate][bucket]
    exec.GetRemoteMapResults(&allScores);
    // set best split for each candidate
    const int candidateCount = candidateList->ysize();
    Y_ASSERT(candidateCount == allScores.ysize());
    const ui64 randSeed = ctx->Rand.GenRand();
    ctx->LocalExecutor.ExecRange([&] (int candidateIdx) {
        SetBestScore(randSeed + candidateIdx, allScores[candidateIdx], scoreStDev, &(*candidateList)[candidateIdx].Candidates);
    }, 0, candidateCount, NPar::TLocalExecutor::WAIT_COMPLETE);
}

void MapSetIndices(const TCandidateInfo& bestSplitCandidate, TLearnContext* ctx) {
    Y_ASSERT(ctx->Params.SystemOptions->IsMaster());
    const int workerCount = ctx->RootEnvironment->GetSlaveCount();
    ApplyMapper<TLeafIndexSetter>(workerCount, ctx->SharedTrainData, TEnvelope<TCandidateInfo>(bestSplitCandidate));
}

int MapGetRedundantSplitIdx(TLearnContext* ctx) {
    Y_ASSERT(ctx->Params.SystemOptions->IsMaster());
    const int workerCount = ctx->RootEnvironment->GetSlaveCount();
    TVector<TEmptyLeafFinder::TOutput> isLeafEmptyFromAllWorkers = ApplyMapper<TEmptyLeafFinder>(workerCount, ctx->SharedTrainData); // poll workers
    for (int workerIdx = 1; workerIdx < workerCount; ++workerIdx) {
        for (int leafIdx = 0; leafIdx < isLeafEmptyFromAllWorkers[0].Data.ysize(); ++leafIdx) {
            isLeafEmptyFromAllWorkers[0].Data[leafIdx] |= isLeafEmptyFromAllWorkers[workerIdx].Data[leafIdx];
        }
    }
    return GetRedundantSplitIdx(isLeafEmptyFromAllWorkers[0].Data);
}

template<typename TError>
void MapSetApproxes(const TSplitTree& splitTree, TLearnContext* ctx) {
    Y_ASSERT(ctx->Params.SystemOptions->IsMaster());
    const int workerCount = ctx->RootEnvironment->GetSlaveCount();
    ApplyMapper<TCalcApproxStarter>(workerCount, ctx->SharedTrainData, TEnvelope<TSplitTree>(splitTree));
    const int gradientIterations = ctx->Params.ObliviousTreeOptions->LeavesEstimationIterations;
    for (int it = 0; it < gradientIterations; ++it) {
        TVector<typename TBucketSimpleUpdater<TError>::TOutput> bucketsFromAllWorkers = ApplyMapper<TBucketSimpleUpdater<TError>>(workerCount, ctx->SharedTrainData);
        // reduce across workers
        for (int workerIdx = 1; workerIdx < workerCount; ++workerIdx) {
            for (int leafIdx = 0; leafIdx < bucketsFromAllWorkers[0].Data.ysize(); ++leafIdx) {
                if (ctx->Params.ObliviousTreeOptions->LeavesEstimationMethod == ELeavesEstimation::Gradient) {
                    bucketsFromAllWorkers[0].Data[leafIdx].AddDerWeight(
                        bucketsFromAllWorkers[workerIdx].Data[leafIdx].SumDerHistory[it],
                        bucketsFromAllWorkers[workerIdx].Data[leafIdx].SumWeights,
                        it);
                } else {
                    Y_ASSERT(ctx->Params.ObliviousTreeOptions->LeavesEstimationMethod == ELeavesEstimation::Newton);
                    bucketsFromAllWorkers[0].Data[leafIdx].AddDerDer2(
                        bucketsFromAllWorkers[workerIdx].Data[leafIdx].SumDerHistory[it],
                        bucketsFromAllWorkers[workerIdx].Data[leafIdx].SumDer2History[it],
                        it);
                }
            }
        }
        // calc model and update approx deltas on workers
        ApplyMapper<TDeltaSimpleUpdater>(workerCount, ctx->SharedTrainData, TEnvelope<TSums>(bucketsFromAllWorkers[0].Data));
    }
    ApplyMapper<TApproxSimpleUpdater>(workerCount, ctx->SharedTrainData);
}

template void MapSetApproxes<TCrossEntropyError>(const TSplitTree& splitTree, TLearnContext* ctx);
template void MapSetApproxes<TRMSEError>(const TSplitTree& splitTree, TLearnContext* ctx);
template void MapSetApproxes<TQuantileError>(const TSplitTree& splitTree, TLearnContext* ctx);
template void MapSetApproxes<TLogLinQuantileError>(const TSplitTree& splitTree, TLearnContext* ctx);
template void MapSetApproxes<TMAPError>(const TSplitTree& splitTree, TLearnContext* ctx);
template void MapSetApproxes<TPoissonError>(const TSplitTree& splitTree, TLearnContext* ctx);
template void MapSetApproxes<TMultiClassError>(const TSplitTree& splitTree, TLearnContext* ctx);
template void MapSetApproxes<TMultiClassOneVsAllError>(const TSplitTree& splitTree, TLearnContext* ctx);
template void MapSetApproxes<TPairLogitError>(const TSplitTree& splitTree, TLearnContext* ctx);
template void MapSetApproxes<TQueryRmseError>(const TSplitTree& splitTree, TLearnContext* ctx);
template void MapSetApproxes<TQuerySoftMaxError>(const TSplitTree& splitTree, TLearnContext* ctx);
template void MapSetApproxes<TCustomError>(const TSplitTree& splitTree, TLearnContext* ctx);
template void MapSetApproxes<TUserDefinedPerObjectError>(const TSplitTree& splitTree, TLearnContext* ctx);
template void MapSetApproxes<TUserDefinedQuerywiseError>(const TSplitTree& splitTree, TLearnContext* ctx);

template<typename TError>
void MapSetDerivatives(TLearnContext* ctx) {
    Y_ASSERT(ctx->Params.SystemOptions->IsMaster());
    ApplyMapper<TDerivativeSetter<TError>>(ctx->RootEnvironment->GetSlaveCount(), ctx->SharedTrainData);
}

template void MapSetDerivatives<TCrossEntropyError>(TLearnContext* ctx);
template void MapSetDerivatives<TRMSEError>(TLearnContext* ctx);
template void MapSetDerivatives<TQuantileError>(TLearnContext* ctx);
template void MapSetDerivatives<TLogLinQuantileError>(TLearnContext* ctx);
template void MapSetDerivatives<TMAPError>(TLearnContext* ctx);
template void MapSetDerivatives<TPoissonError>(TLearnContext* ctx);
template void MapSetDerivatives<TMultiClassError>(TLearnContext* ctx);
template void MapSetDerivatives<TMultiClassOneVsAllError>(TLearnContext* ctx);
template void MapSetDerivatives<TPairLogitError>(TLearnContext* ctx);
template void MapSetDerivatives<TQueryRmseError>(TLearnContext* ctx);
template void MapSetDerivatives<TQuerySoftMaxError>(TLearnContext* ctx);
template void MapSetDerivatives<TCustomError>(TLearnContext* ctx);
template void MapSetDerivatives<TUserDefinedPerObjectError>(TLearnContext* ctx);
template void MapSetDerivatives<TUserDefinedQuerywiseError>(TLearnContext* ctx);

