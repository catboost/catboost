#include "master.h"
#include "mappers.h"

#include <catboost/libs/algo/error_functions.h>
#include <catboost/libs/algo/index_calcer.h>
#include <catboost/libs/algo/score_bin.h>
#include <catboost/libs/algo/score_calcer.h>

#include <library/par/par_settings.h>

#include <util/system/yassert.h>


using namespace NCatboostDistributed;
using namespace NCB;


void InitializeMaster(TLearnContext* ctx) {
    Y_ASSERT(ctx->Params.SystemOptions->IsMaster());
    const auto& systemOptions = ctx->Params.SystemOptions;
    const ui32 unusedNodePort = NCatboostOptions::TSystemOptions::GetUnusedNodePort();

    // avoid Netliba
    NPar::TParNetworkSettings::GetRef().RequesterType = NPar::TParNetworkSettings::ERequesterType::NEH;
    ctx->RootEnvironment = NPar::RunMaster(
        systemOptions->NodePort,
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
    const auto& targetClassifiers = ctx->CtrsHelper.GetTargetClassifiers();
    NJson::TJsonValue jsonParams;
    ctx->Params.Save(&jsonParams);
    const auto& metricOptions = ctx->Params.MetricOptions;
    if (metricOptions->EvalMetric.NotSet()) { // workaround for NotSet + Save + Load = DefaultValue
        if (ctx->Params.LossFunctionDescription->GetLossFunction() !=
            metricOptions->EvalMetric->GetLossFunction()) {
            // skip only if default metric differs from loss function

            const auto& evalMetric = metricOptions->EvalMetric;
            jsonParams[metricOptions.GetName()][evalMetric.GetName()][evalMetric->LossParams.GetName()]
                .InsertValue("hints", "skip_train~true");
        }
    }
    const TString stringParams = ToString(jsonParams);
    for (int workerIdx = 0; workerIdx < workerCount; ++workerIdx) {
        ctx->SharedTrainData->SetContextData(
            workerIdx,
            new NCatboostDistributed::TTrainData(
                trainData->GetSubset(
                    NCB::GetSubset(
                        trainData->ObjectsGrouping,
                        std::move(workerParts[workerIdx]),
                        EObjectsOrder::Ordered),
                    ctx->LocalExecutor),
                targetClassifiers,
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
void MapGenericCalcScore(
    TGetScore getScore,
    double scoreStDev,
    TConstArrayRef<TBinaryFeaturesPack> perPackMasks,
    TCandidateList* candidateList,
    TLearnContext* ctx) {

    Y_ASSERT(ctx->Params.SystemOptions->IsMaster());
    const int workerCount = ctx->RootEnvironment->GetSlaveCount();
    auto allStatsFromAllWorkers = ApplyMapper<TScoreCalcMapper>(
        workerCount,
        ctx->SharedTrainData,
        MakeEnvelope(*candidateList));
    const int candidateCount = candidateList->ysize();
    const ui64 randSeed = ctx->Rand.GenRand();
    // set best split for each candidate
    NPar::ParallelFor(
        *ctx->LocalExecutor,
        0,
        candidateCount,
        [&] (int candidateIdx) {
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
            SetBestScore(randSeed + candidateIdx, allScores, scoreStDev, perPackMasks, &subCandidates);
        });
}

// TODO(espetrov): Remove unused code.
void MapCalcScore(
    double scoreStDev,
    int depth,
    TConstArrayRef<TBinaryFeaturesPack> perPackMasks,
    TCandidateList* candidateList,
    TLearnContext* ctx) {

    const auto& plainFold = ctx->LearnProgress.Folds[0];
    const auto getScore = [&] (const TStats3D& stats3D, const TCandidateInfo& splitInfo) {
        Y_UNUSED(splitInfo);

        return GetScores(
            GetScoreBins(
                stats3D,
                depth,
                plainFold.GetSumWeight(),
                plainFold.GetLearnSampleCount(),
                ctx->Params));
    };
    MapGenericCalcScore<TScoreCalcer>(getScore, scoreStDev, perPackMasks, candidateList, ctx);
}

template <typename TBinCalcMapper, typename TScoreCalcMapper>
void MapGenericRemoteCalcScore(
    double scoreStDev,
    TConstArrayRef<TBinaryFeaturesPack> perPackMasks,
    TCandidateList* candidateList,
    TLearnContext* ctx) {

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
    ctx->LocalExecutor->ExecRange(
        [&] (int candidateIdx) {
            auto& candidates = (*candidateList)[candidateIdx].Candidates;
            Y_VERIFY(candidates.size() > 0);

            SetBestScore(
                randSeed + candidateIdx,
                allScores[candidateIdx],
                scoreStDev,
                perPackMasks,
                &candidates);
        },
        0,
        candidateCount,
        NPar::TLocalExecutor::WAIT_COMPLETE);
}

void MapRemotePairwiseCalcScore(
    double scoreStDev,
    TConstArrayRef<TBinaryFeaturesPack> perPackMasks,
    TCandidateList* candidateList,
    TLearnContext* ctx) {

    MapGenericRemoteCalcScore<TRemotePairwiseBinCalcer, TRemotePairwiseScoreCalcer>(
        scoreStDev,
        perPackMasks,
        candidateList,
        ctx);
}

void MapRemoteCalcScore(
    double scoreStDev,
    TConstArrayRef<TBinaryFeaturesPack> perPackMasks,
    TCandidateList* candidateList,
    TLearnContext* ctx) {

    MapGenericRemoteCalcScore<TRemoteBinCalcer, TRemoteScoreCalcer>(
        scoreStDev,
        perPackMasks,
        candidateList,
        ctx);
}

void MapSetIndices(const TSplit& bestSplit, TLearnContext* ctx) {
    Y_ASSERT(ctx->Params.SystemOptions->IsMaster());
    const int workerCount = ctx->RootEnvironment->GetSlaveCount();
    ApplyMapper<TLeafIndexSetter>(workerCount, ctx->SharedTrainData, MakeEnvelope(bestSplit));
}

int MapGetRedundantSplitIdx(TLearnContext* ctx) {
    Y_ASSERT(ctx->Params.SystemOptions->IsMaster());
    const int workerCount = ctx->RootEnvironment->GetSlaveCount();
    TVector<TEmptyLeafFinder::TOutput> isLeafEmptyFromAllWorkers
        = ApplyMapper<TEmptyLeafFinder>(workerCount, ctx->SharedTrainData); // poll workers
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

    // poll workers
    auto additiveStatsFromAllWorkers = ApplyMapper<TErrorCalcer>(workerCount, ctx->SharedTrainData);
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
    Y_VERIFY(
        Accumulate(skipMetricOnTrain.begin(), skipMetricOnTrain.end(), 0) + additiveStats.size() ==
            metrics.size());
    for (int metricIdx = 0; metricIdx < metrics.ysize(); ++metricIdx) {
        if (!skipMetricOnTrain[metricIdx] && metrics[metricIdx]->IsAdditiveMetric()) {
            const auto description = metrics[metricIdx]->GetDescription();
            ctx->LearnProgress.MetricsAndTimeHistory.AddLearnError(
                *metrics[metricIdx].Get(),
                metrics[metricIdx]->GetFinalError(additiveStats[description]));
        }
    }
}

template <typename TApproxDefs>
void MapSetApproxes(
    const IDerCalcer& error,
    const TSplitTree& splitTree,
    TConstArrayRef<NCB::TTrainingForCPUDataProviderPtr> testData,
    TVector<TVector<double>>* averageLeafValues,
    TVector<double>* sumLeafWeights,
    TLearnContext* ctx) {

    using namespace NCatboostDistributed;
    using TSum = typename TApproxDefs::TSumType;
    using TPairwiseBuckets = typename TApproxDefs::TPairwiseBuckets;
    using TBucketUpdater = typename TApproxDefs::TBucketUpdater;
    using TDeltaUpdater = typename TApproxDefs::TDeltaUpdater;

    Y_ASSERT(ctx->Params.SystemOptions->IsMaster());
    const int workerCount = ctx->RootEnvironment->GetSlaveCount();
    ApplyMapper<TCalcApproxStarter>(workerCount, ctx->SharedTrainData, MakeEnvelope(splitTree));
    const int gradientIterations = ctx->Params.ObliviousTreeOptions->LeavesEstimationIterations;
    const int approxDimension = ctx->LearnProgress.ApproxDimension;
    const int leafCount = splitTree.GetLeafCount();
    TVector<TSum> buckets(leafCount, TSum(approxDimension, error.GetHessianType()));
    averageLeafValues->resize(approxDimension, TVector<double>(leafCount));
    for (int it = 0; it < gradientIterations; ++it) {
        for (auto& bucket : buckets) {
            bucket.SetZeroDers();
        }

        TPairwiseBuckets pairwiseBuckets;
        TApproxDefs::SetPairwiseBucketsSize(leafCount, &pairwiseBuckets);
        const auto bucketsFromAllWorkers = ApplyMapper<TBucketUpdater>(workerCount, ctx->SharedTrainData);
        // reduce across workers
        for (int workerIdx = 0; workerIdx < workerCount; ++workerIdx) {
            const auto& workerBuckets = bucketsFromAllWorkers[workerIdx].Data.first;
            for (int leafIdx = 0; leafIdx < leafCount; ++leafIdx) {
                if (ctx->Params.ObliviousTreeOptions->LeavesEstimationMethod == ELeavesEstimation::Gradient) {
                    buckets[leafIdx].AddDerWeight(
                        workerBuckets[leafIdx].SumDer,
                        workerBuckets[leafIdx].SumWeights,
                        it);
                } else {
                    Y_ASSERT(
                        ctx->Params.ObliviousTreeOptions->LeavesEstimationMethod == ELeavesEstimation::Newton);
                    buckets[leafIdx].AddDerDer2(workerBuckets[leafIdx].SumDer, workerBuckets[leafIdx].SumDer2);
                }
            }
            TApproxDefs::AddPairwiseBuckets(bucketsFromAllWorkers[workerIdx].Data.second, &pairwiseBuckets);
        }
        const auto leafValues = TApproxDefs::CalcLeafValues(buckets, pairwiseBuckets, *ctx);
        AddElementwise(leafValues, averageLeafValues);
        // calc model and update approx deltas on workers
        ApplyMapper<TDeltaUpdater>(workerCount, ctx->SharedTrainData, leafValues);
    }

    // [workerIdx][dimIdx][leafIdx]
    const auto leafWeightsFromAllWorkers = ApplyMapper<TLeafWeightsGetter>(workerCount, ctx->SharedTrainData);
    sumLeafWeights->resize(leafCount);
    for (const auto& workerLeafWeights : leafWeightsFromAllWorkers) {
        AddElementwise(workerLeafWeights, sumLeafWeights);
    }

    NormalizeLeafValues(
        UsesPairsForCalculation(ctx->Params.LossFunctionDescription->GetLossFunction()),
        ctx->Params.BoostingOptions->LearningRate,
        *sumLeafWeights,
        averageLeafValues);

    // update learn approx and average approx
    ApplyMapper<TApproxUpdater>(workerCount, ctx->SharedTrainData, *averageLeafValues);
    // update test
    const auto indices = BuildIndices(
        /*unused fold*/{ },
        splitTree, /*learnData*/
        { },
        testData,
        ctx->LocalExecutor);
    UpdateAvrgApprox(
        error.GetIsExpApprox(), /*learnSampleCount*/
        0,
        indices,
        *averageLeafValues,
        testData,
        &ctx->LearnProgress,
        ctx->LocalExecutor);
}

struct TSetApproxesSimpleDefs {
    using TSumType = TSum;
    using TPairwiseBuckets = TArray2D<double>;
    using TBucketUpdater = NCatboostDistributed::TBucketSimpleUpdater;
    using TDeltaUpdater = NCatboostDistributed::TDeltaSimpleUpdater;

public:
    static void SetPairwiseBucketsSize(size_t leafCount, TPairwiseBuckets* pairwiseBuckets) {
        pairwiseBuckets->SetSizes(leafCount, leafCount);
        pairwiseBuckets->FillZero();
    }
    static void AddPairwiseBuckets(const TPairwiseBuckets& increment, TPairwiseBuckets* total) {
        Y_ASSERT(increment.GetXSize() == total->GetXSize() && increment.GetYSize() == total->GetYSize());
        for (size_t winnerIdx = 0; winnerIdx < increment.GetYSize(); ++winnerIdx) {
            for (size_t loserIdx = 0; loserIdx < increment.GetXSize(); ++loserIdx) {
                (*total)[winnerIdx][loserIdx] += increment[winnerIdx][loserIdx];
            }
        }
    }
    static TVector<TVector<double>> CalcLeafValues(const TVector<TSumType>& buckets,
        const TPairwiseBuckets& pairwiseBuckets,
        const TLearnContext& ctx) {

        const size_t leafCount = buckets.size();
        TVector<TVector<double>> leafValues(/*dimensionCount*/ 1, TVector<double>(leafCount));
        const size_t allDocCount = ctx.LearnProgress.Folds[0].GetLearnSampleCount();
        const double sumAllWeights = ctx.LearnProgress.Folds[0].GetSumWeight();
        CalcLeafDeltasSimple(buckets, pairwiseBuckets, ctx.Params, sumAllWeights, allDocCount, &leafValues[0]);
        return leafValues;
    }
};

struct TSetApproxesMultiDefs {
    using TSumType = TSumMulti;
    using TPairwiseBuckets = NCatboostDistributed::TUnusedInitializedParam;
    using TBucketUpdater = NCatboostDistributed::TBucketMultiUpdater;
    using TDeltaUpdater = NCatboostDistributed::TDeltaMultiUpdater;

public:
    static void SetPairwiseBucketsSize(size_t /*leafCount*/, TPairwiseBuckets* /*pairwiseBuckets*/) {}
    static void AddPairwiseBuckets(const TPairwiseBuckets& /*increment*/, TPairwiseBuckets* /*total*/) {}
    static TVector<TVector<double>> CalcLeafValues(const TVector<TSumType>& buckets,
        const TPairwiseBuckets& /*pairwiseBuckets*/,
        const TLearnContext& ctx) {

        const int dimensionCount = ctx.LearnProgress.ApproxDimension;
        const size_t leafCount = buckets.size();
        TVector<TVector<double>> leafValues(dimensionCount, TVector<double>(leafCount));
        const auto estimationMethod = ctx.Params.ObliviousTreeOptions->LeavesEstimationMethod;
        const float l2Regularizer = ctx.Params.ObliviousTreeOptions->L2Reg;
        const size_t allDocCount = ctx.LearnProgress.Folds[0].GetLearnSampleCount();
        const double sumAllWeights = ctx.LearnProgress.Folds[0].GetSumWeight();
        if (estimationMethod == ELeavesEstimation::Newton) {
        CalcMixedModelMulti(
            CalcDeltaNewtonMulti,
            buckets,
            l2Regularizer,
            sumAllWeights,
            allDocCount,
            &leafValues);
        } else {
            Y_ASSERT(estimationMethod == ELeavesEstimation::Gradient);
            CalcMixedModelMulti(
                CalcDeltaGradientMulti,
                buckets,
                l2Regularizer,
                sumAllWeights,
                allDocCount,
                &leafValues);
        }
        return leafValues;
    }
};

void MapSetApproxesSimple(
    const IDerCalcer& error,
    const TSplitTree& splitTree,
    TConstArrayRef<NCB::TTrainingForCPUDataProviderPtr> testData,
    TVector<TVector<double>>* averageLeafValues,
    TVector<double>* sumLeafWeights,
    TLearnContext* ctx) {

    MapSetApproxes<TSetApproxesSimpleDefs>(error, splitTree, testData, averageLeafValues, sumLeafWeights, ctx);
}

void MapSetApproxesMulti(
    const IDerCalcer& error,
    const TSplitTree& splitTree,
    TConstArrayRef<NCB::TTrainingForCPUDataProviderPtr> testData,
    TVector<TVector<double>>* averageLeafValues,
    TVector<double>* sumLeafWeights,
    TLearnContext* ctx) {

    MapSetApproxes<TSetApproxesMultiDefs>(error, splitTree, testData, averageLeafValues, sumLeafWeights, ctx);
}

void MapSetDerivatives(TLearnContext* ctx) {
    using namespace NCatboostDistributed;
    Y_ASSERT(ctx->Params.SystemOptions->IsMaster());
    ApplyMapper<TDerivativeSetter>(ctx->RootEnvironment->GetSlaveCount(), ctx->SharedTrainData);
}
