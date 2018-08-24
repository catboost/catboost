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
    workerPart.OneHotValues = allFeatures.OneHotValues;
    workerPart.IsOneHot = allFeatures.IsOneHot;
    return workerPart;
}

using TPartPairMap = THashMap<std::pair<size_t, size_t>, TVector<TPair>>;

static ::TDataset GetWorkerPart(const ::TDataset& trainData, const TPartPairMap& partPairMap, const std::pair<size_t, size_t>& part) {
    ::TDataset workerPart;
    workerPart.AllFeatures = GetWorkerPart(trainData.AllFeatures, part);
    workerPart.Baseline = GetWorkerPart(trainData.Baseline, part);
    workerPart.Target = GetWorkerPart(trainData.Target, part);
    workerPart.Weights = GetWorkerPart(trainData.Weights, part);
    workerPart.QueryId = GetWorkerPart(trainData.QueryId, part);
    workerPart.SubgroupId = GetWorkerPart(trainData.SubgroupId, part);
    if (!trainData.Pairs.empty()) {
        Y_ASSERT(partPairMap.has(part));
        workerPart.Pairs = partPairMap.at(part);
    }
    workerPart.HasGroupWeight = trainData.HasGroupWeight;
    UpdateQueryInfo(&workerPart);
    return workerPart;
}

static TPartPairMap GetPairsForParts(const TVector<TPair>& pairs, const TVector<std::pair<size_t, size_t>>& parts) {
    TPartPairMap pairsForParts;
    const auto IsElement = [](size_t value, const std::pair<size_t, size_t>& range) { return range.first <= value && value < range.second; };
    for (const auto& pair : pairs) {
        const auto winnerPart = FindIf(parts, [pair, &IsElement](const auto& part) { return IsElement(pair.WinnerId, part); } );
        Y_ASSERT(winnerPart != parts.end() && IsElement(pair.LoserId, *winnerPart));
        const auto partStart = winnerPart->first;
        pairsForParts[*winnerPart].emplace_back(pair.WinnerId - partStart, pair.LoserId - partStart, pair.Weight);
    }
    return pairsForParts;
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
    const int workerCount = ctx->RootEnvironment->GetSlaveCount();
    TVector<std::pair<size_t, size_t>> workerParts;
    TPartPairMap pairsForParts;
    if (trainData.QueryId.empty()) {
        workerParts = Split(trainData.GetSampleCount(), workerCount);
    } else {
        workerParts = Split(trainData.GetSampleCount(), trainData.QueryId, workerCount);
        pairsForParts = GetPairsForParts(trainData.Pairs, workerParts);
    }
    const ui64 randomSeed = ctx->Rand.GenRand();
    const auto& splitCounts = CountSplits(ctx->LearnProgress.FloatFeatures);
    const auto& targetClassifiers = ctx->CtrsHelper.GetTargetClassifiers();
    NJson::TJsonValue jsonParams;
    ctx->Params.Save(&jsonParams);
    const TString stringParams = ToString(jsonParams);
    for (int workerIdx = 0; workerIdx < workerCount; ++workerIdx) {
        ctx->SharedTrainData->SetContextData(workerIdx,
            new NCatboostDistributed::TTrainData(
                GetWorkerPart(trainData, pairsForParts, workerParts[workerIdx]),
                targetClassifiers,
                splitCounts,
                randomSeed,
                ctx->LearnProgress.ApproxDimension,
                stringParams,
                plainFold.GetLearnSampleCount(),
                plainFold.GetSumWeight(),
                ctx->LearnProgress.HessianType),
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

void MapPairwiseCalcScore(double scoreStDev, TCandidateList* candidateList, TLearnContext* ctx) {
    Y_ASSERT(ctx->Params.SystemOptions->IsMaster());
    const int workerCount = ctx->RootEnvironment->GetSlaveCount();
    TVector<TPairwiseScoreCalcer::TOutput> allStatsFromAllWorkers = ApplyMapper<TPairwiseScoreCalcer>(workerCount, ctx->SharedTrainData, TEnvelope<TCandidateList>(*candidateList));
    const int candidateCount = candidateList->ysize();
    const ui64 randSeed = ctx->Rand.GenRand();
    const float l2Reg = ctx->Params.ObliviousTreeOptions->L2Reg;
    const float pairwiseBucketWeightPriorReg = ctx->Params.ObliviousTreeOptions->PairwiseNonDiagReg;
    const auto splitCount = CountSplits(ctx->LearnProgress.FloatFeatures);
    // set best split for each candidate
    ctx->LocalExecutor.ExecRange([&] (int candidateIdx) {
        auto& subCandidates = (*candidateList)[candidateIdx].Candidates;
        const int subcandidateCount = subCandidates.ysize();
        TVector<TVector<double>> allScores(subcandidateCount);
        for (int subcandidateIdx = 0; subcandidateIdx < subcandidateCount; ++subcandidateIdx) {
            // reduce across workers
            auto& reducedStats = allStatsFromAllWorkers[0].Data[candidateIdx][subcandidateIdx];
            for (int workerIdx = 1; workerIdx < workerCount; ++workerIdx) {
                const auto& stats = allStatsFromAllWorkers[workerIdx].Data[candidateIdx][subcandidateIdx];
                reducedStats.Add(stats);
            }
            const auto& splitInfo = subCandidates[subcandidateIdx];
            const int bucketCount = GetSplitCount(splitCount, /*oneHotValues*/ {}, splitInfo.SplitCandidate) + 1;
            TVector<TScoreBin> scoreBins(bucketCount);
            CalculatePairwiseScore(reducedStats,
                bucketCount,
                splitInfo.SplitCandidate.Type,
                l2Reg,
                pairwiseBucketWeightPriorReg,
                &scoreBins);
            allScores[subcandidateIdx] = GetScores(scoreBins);
        }
        SetBestScore(randSeed + candidateIdx, allScores, scoreStDev, &subCandidates);
    }, 0, candidateCount, NPar::TLocalExecutor::WAIT_COMPLETE);
}

// TODO(espetrov): Remove unused code.
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

    const auto& plainFold = ctx->LearnProgress.Folds[0];
    ctx->LocalExecutor.ExecRange([&] (int candidateIdx) {
        const auto& allStats = allStatsFromAllWorkers[0].Data[candidateIdx];
        auto& candidate = (*candidateList)[candidateIdx];
        const int subcandidateCount = candidate.Candidates.ysize();
        TVector<TVector<double>> allScores(subcandidateCount);
        for (int subcandidateIdx = 0; subcandidateIdx < subcandidateCount; ++subcandidateIdx) {
            const auto& splitInfo = candidate.Candidates[subcandidateIdx];
            allScores[subcandidateIdx] = GetScores(GetScoreBins(allStats[subcandidateIdx], splitInfo.SplitCandidate.Type, depth, plainFold.GetSumWeight(), plainFold.GetLearnSampleCount(), ctx->Params));
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
            isLeafEmptyFromAllWorkers[0].Data[leafIdx] &= isLeafEmptyFromAllWorkers[workerIdx].Data[leafIdx];
        }
    }
    return GetRedundantSplitIdx(isLeafEmptyFromAllWorkers[0].Data);
}
