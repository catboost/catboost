#include "master.h"
#include "mappers.h"

#include <catboost/libs/algo/error_functions.h>
#include <catboost/libs/algo/index_calcer.h>

#include <library/par/par_settings.h>

using namespace NCatboostDistributed;
using namespace NCB;


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

void MapBuildPlainFold(NCB::TTrainingForCPUDataProviderPtr trainData, TLearnContext* ctx) {
    Y_ASSERT(ctx->Params.SystemOptions->IsMaster());
    const auto& plainFold = ctx->LearnProgress.Folds[0];
    Y_ASSERT(plainFold.PermutationBlockSize == plainFold.GetLearnSampleCount());
    const int workerCount = ctx->RootEnvironment->GetSlaveCount();
    TVector<TArraySubsetIndexing<ui32>> workerParts = Split(*trainData->ObjectsGrouping, (ui32)workerCount);

    const ui64 randomSeed = ctx->Rand.GenRand();
    const auto& splitCounts = CountSplits(ctx->LearnProgress.FloatFeatures);
    const auto& targetClassifiers = ctx->CtrsHelper.GetTargetClassifiers();
    NJson::TJsonValue jsonParams;
    ctx->Params.Save(&jsonParams);
    const auto& metricOptions = ctx->Params.MetricOptions;
    if (metricOptions->EvalMetric.NotSet()) { // workaround for NotSet + Save + Load = DefaultValue
        if (ctx->Params.LossFunctionDescription->GetLossFunction() != metricOptions->EvalMetric->GetLossFunction()) { // skip only if default metric differs from loss function
            const auto& evalMetric = metricOptions->EvalMetric;
            jsonParams[metricOptions.GetName()][evalMetric.GetName()][evalMetric->LossParams.GetName()].InsertValue("hints", "skip_train~true");
        }
    }
    const TString stringParams = ToString(jsonParams);
    for (int workerIdx = 0; workerIdx < workerCount; ++workerIdx) {
        ctx->SharedTrainData->SetContextData(workerIdx,
            new NCatboostDistributed::TTrainData(
                trainData->GetSubset(
                    NCB::GetSubset(trainData->ObjectsGrouping, std::move(workerParts[workerIdx]), EObjectsOrder::Ordered),
                    ctx->LocalExecutor
                ),
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

void MapRestoreApproxFromTreeStruct(TLearnContext* ctx) {
    Y_ASSERT(ctx->Params.SystemOptions->IsMaster());
    ApplyMapper<TApproxReconstructor>(
        ctx->RootEnvironment->GetSlaveCount(),
        ctx->SharedTrainData,
        MakeEnvelope(std::make_pair(ctx->LearnProgress.TreeStruct, ctx->LearnProgress.LeafValues)));
}

void MapTensorSearchStart(TLearnContext* ctx) {
    Y_ASSERT(ctx->Params.SystemOptions->IsMaster());
    ApplyMapper<TTensorSearchStarter>(ctx->RootEnvironment->GetSlaveCount(), ctx->SharedTrainData);
}

void MapBootstrap(TLearnContext* ctx) {
    Y_ASSERT(ctx->Params.SystemOptions->IsMaster());
    ApplyMapper<TBootstrapMaker>(ctx->RootEnvironment->GetSlaveCount(), ctx->SharedTrainData);
}

template <typename TScoreCalcMapper, typename TGetScore>
void MapGenericCalcScore(TGetScore getScore, double scoreStDev, TCandidateList* candidateList, TLearnContext* ctx) {
    Y_ASSERT(ctx->Params.SystemOptions->IsMaster());
    const int workerCount = ctx->RootEnvironment->GetSlaveCount();
    auto allStatsFromAllWorkers = ApplyMapper<TScoreCalcMapper>(workerCount, ctx->SharedTrainData, MakeEnvelope(*candidateList));
    const int candidateCount = candidateList->ysize();
    const ui64 randSeed = ctx->Rand.GenRand();
    // set best split for each candidate
    NPar::ParallelFor(*ctx->LocalExecutor, 0, candidateCount, [&] (int candidateIdx) {
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
            allScores[subcandidateIdx] = getScore(reducedStats, splitInfo);
        }
        SetBestScore(randSeed + candidateIdx, allScores, scoreStDev, &subCandidates);
    });
}

// TODO(espetrov): Remove unused code.
void MapCalcScore(double scoreStDev, int depth, TCandidateList* candidateList, TLearnContext* ctx) {
    const auto& plainFold = ctx->LearnProgress.Folds[0];
    const auto getScore = [&] (const TStats3D& stats3D, const TCandidateInfo& splitInfo) {
        return GetScores(GetScoreBins(stats3D,
            splitInfo.SplitCandidate.Type,
            depth,
            plainFold.GetSumWeight(),
            plainFold.GetLearnSampleCount(),
            ctx->Params));
    };
    MapGenericCalcScore<TScoreCalcer>(getScore, scoreStDev, candidateList, ctx);
}

template <typename TBinCalcMapper, typename TScoreCalcMapper>
void MapGenericRemoteCalcScore(double scoreStDev, TCandidateList* candidateList, TLearnContext* ctx) {
    Y_ASSERT(ctx->Params.SystemOptions->IsMaster());
    NPar::TJobDescription job;
    NPar::Map(&job, new TBinCalcMapper(), candidateList);
    NPar::RemoteMap(&job, new TScoreCalcMapper);
    NPar::TJobExecutor exec(&job, ctx->SharedTrainData);
    TVector<typename TScoreCalcMapper::TOutput> allScores;
    exec.GetRemoteMapResults(&allScores);
    // set best split for each candidate
    const int candidateCount = candidateList->ysize();
    Y_ASSERT(candidateCount == allScores.ysize());
    const ui64 randSeed = ctx->Rand.GenRand();
    ctx->LocalExecutor->ExecRange([&] (int candidateIdx) {
        SetBestScore(randSeed + candidateIdx, allScores[candidateIdx], scoreStDev, &(*candidateList)[candidateIdx].Candidates);
    }, 0, candidateCount, NPar::TLocalExecutor::WAIT_COMPLETE);
}

void MapRemotePairwiseCalcScore(double scoreStDev, TCandidateList* candidateList, TLearnContext* ctx) {
    MapGenericRemoteCalcScore<TRemotePairwiseBinCalcer, TRemotePairwiseScoreCalcer>(scoreStDev, candidateList, ctx);
}

void MapRemoteCalcScore(double scoreStDev, int /*depth*/, TCandidateList* candidateList, TLearnContext* ctx) {
    MapGenericRemoteCalcScore<TRemoteBinCalcer, TRemoteScoreCalcer>(scoreStDev, candidateList, ctx);
}

void MapSetIndices(const TCandidateInfo& bestSplitCandidate, TLearnContext* ctx) {
    Y_ASSERT(ctx->Params.SystemOptions->IsMaster());
    const int workerCount = ctx->RootEnvironment->GetSlaveCount();
    ApplyMapper<TLeafIndexSetter>(workerCount, ctx->SharedTrainData, MakeEnvelope(bestSplitCandidate));
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

void MapCalcErrors(TLearnContext* ctx) {
    Y_ASSERT(ctx->Params.SystemOptions->IsMaster());
    const size_t workerCount = ctx->RootEnvironment->GetSlaveCount();
    auto additiveStatsFromAllWorkers = ApplyMapper<TErrorCalcer>(workerCount, ctx->SharedTrainData); // poll workers
    Y_ASSERT(additiveStatsFromAllWorkers.size() == workerCount);

    auto& additiveStats = additiveStatsFromAllWorkers[0];
    for (size_t workerIdx : xrange<size_t>(1, workerCount)) {
        const auto& workerAdditiveStats = additiveStatsFromAllWorkers[workerIdx];
        for (auto& [description, stats] : additiveStats) {
            Y_ASSERT(workerAdditiveStats.contains(description));
            stats.Add(workerAdditiveStats.at(description));
        }
    }

    const auto metrics = CreateMetrics(
        ctx->Params.LossFunctionDescription,
        ctx->Params.MetricOptions,
        ctx->EvalMetricDescriptor,
        ctx->LearnProgress.ApproxDimension
    );
    const auto skipMetricOnTrain = GetSkipMetricOnTrain(metrics);
    Y_VERIFY(Accumulate(skipMetricOnTrain.begin(), skipMetricOnTrain.end(), 0) + additiveStats.size() == metrics.size());
    for (int metricIdx = 0; metricIdx < metrics.ysize(); ++metricIdx) {
        if (!skipMetricOnTrain[metricIdx] && metrics[metricIdx]->IsAdditiveMetric()) {
            const auto description = metrics[metricIdx]->GetDescription();
            ctx->LearnProgress.MetricsAndTimeHistory.AddLearnError(*metrics[metricIdx].Get(), metrics[metricIdx]->GetFinalError(additiveStats[description]));
        }
    }
}
