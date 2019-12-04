#include "master.h"
#include "mappers.h"

#include <catboost/libs/data/load_data.h>
#include <catboost/libs/helpers/quantile.h>
#include <catboost/private/libs/algo/approx_updater_helpers.h>
#include <catboost/private/libs/algo/data.h>
#include <catboost/private/libs/algo/index_calcer.h>
#include <catboost/private/libs/algo/score_calcers.h>
#include <catboost/private/libs/algo/scoring.h>
#include <catboost/private/libs/algo_helpers/approx_calcer_multi_helpers.h>
#include <catboost/private/libs/algo_helpers/error_functions.h>

#include <library/par/par_settings.h>

#include <util/system/yassert.h>


using namespace NCatboostDistributed;
using namespace NCB;

struct TMasterEnvironment {
    TObj<NPar::IRootEnvironment> RootEnvironment = nullptr;
    TObj<NPar::IEnvironment> SharedTrainData = nullptr;

    Y_DECLARE_SINGLETON_FRIEND();

    inline static TMasterEnvironment& GetRef() {
        return *Singleton<TMasterEnvironment>();
    }
};

void InitializeMaster(const NCatboostOptions::TSystemOptions& systemOptions) {
    Y_ASSERT(systemOptions.IsMaster());
    const ui32 unusedNodePort = NCatboostOptions::TSystemOptions::GetUnusedNodePort();

    // avoid Netliba
    NPar::TParNetworkSettings::GetRef().RequesterType = NPar::TParNetworkSettings::ERequesterType::NEH;
    TMasterEnvironment::GetRef().RootEnvironment = NPar::RunMaster(
        systemOptions.NodePort,
        systemOptions.NumThreads,
        systemOptions.FileWithHosts->c_str(),
        unusedNodePort,
        unusedNodePort);
    const int workerCount = TMasterEnvironment::GetRef().RootEnvironment->GetSlaveCount();
    const auto& workerMapping = TMasterEnvironment::GetRef().RootEnvironment->MakeHostIdMapping(workerCount);
    TMasterEnvironment::GetRef().SharedTrainData = TMasterEnvironment::GetRef().RootEnvironment->CreateEnvironment(SHARED_ID_TRAIN_DATA, workerMapping);
}

void FinalizeMaster(TLearnContext* ctx) {
    Y_ASSERT(ctx->Params.SystemOptions->IsMaster());
    if (TMasterEnvironment::GetRef().RootEnvironment != nullptr) {
        TMasterEnvironment::GetRef().RootEnvironment->Stop();
    }
}

void SetTrainDataFromQuantizedPool(
    const NCatboostOptions::TPoolLoadParams& poolLoadOptions,
    const NCatboostOptions::TCatBoostOptions& catBoostOptions,
    const NCB::TObjectsGrouping& objectsGrouping,
    const NCB::TFeaturesLayout& featuresLayout,
    TRestorableFastRng64* rand
) {
    const int workerCount = TMasterEnvironment::GetRef().RootEnvironment->GetSlaveCount();
    for (int workerIdx : xrange(workerCount)) {
        TMasterEnvironment::GetRef().SharedTrainData->DeleteContextRawData(workerIdx);
    }
    NJson::TJsonValue trainParams;
    catBoostOptions.Save(&trainParams);
    const auto objectsOrder = catBoostOptions.DataProcessingOptions->HasTimeFlag.Get() ?
        EObjectsOrder::Ordered : EObjectsOrder::Undefined;
    ApplyMapper<TDatasetLoader>(
        workerCount,
        TMasterEnvironment::GetRef().SharedTrainData,
        TDatasetLoaderParams{
            poolLoadOptions,
            ToString(trainParams),
            objectsOrder,
            objectsGrouping,
            featuresLayout,
            rand->GenRand()
        }
    );
}

void SetTrainDataFromMaster(
    NCB::TTrainingForCPUDataProviderPtr trainData,
    ui64 cpuUsedRamLimit,
    NPar::TLocalExecutor* localExecutor
) {
    const int workerCount = TMasterEnvironment::GetRef().RootEnvironment->GetSlaveCount();
    auto workerParts = Split(*trainData->ObjectsGrouping, (ui32)workerCount);
    for (int workerIdx = 0; workerIdx < workerCount; ++workerIdx) {
        TMasterEnvironment::GetRef().SharedTrainData->SetContextData(
            workerIdx,
            new NCatboostDistributed::TTrainData(
                trainData->GetSubset(
                    NCB::GetSubset(
                        trainData->ObjectsGrouping,
                        std::move(workerParts[workerIdx]),
                        EObjectsOrder::Ordered),
                    cpuUsedRamLimit,
                    localExecutor)),
            NPar::DELETE_RAW_DATA); // only workers
    }
}

void MapBuildPlainFold(TLearnContext* ctx) {
    Y_ASSERT(ctx->Params.SystemOptions->IsMaster());

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
    const auto& plainFold = ctx->LearnProgress->Folds[0];
    Y_ASSERT(plainFold.PermutationBlockSize == plainFold.GetLearnSampleCount() ||
        plainFold.PermutationBlockSize == 1);
    const int workerCount = TMasterEnvironment::GetRef().RootEnvironment->GetSlaveCount();
    ApplyMapper<TPlainFoldBuilder>(
        workerCount,
        TMasterEnvironment::GetRef().SharedTrainData,
        TPlainFoldBuilderParams({
            ctx->CtrsHelper.GetTargetClassifiers(),
            ctx->LearnProgress->Rand.GenRand(),
            ctx->LearnProgress->ApproxDimension,
            ToString(jsonParams),
            plainFold.GetLearnSampleCount(),
            plainFold.GetSumWeight(),
            ctx->LearnProgress->HessianType
        })
    );
}

void MapRestoreApproxFromTreeStruct(TLearnContext* ctx) {
    Y_ASSERT(ctx->Params.SystemOptions->IsMaster());
    ApplyMapper<TApproxReconstructor>(
        TMasterEnvironment::GetRef().RootEnvironment->GetSlaveCount(),
        TMasterEnvironment::GetRef().SharedTrainData,
        std::make_pair(ctx->LearnProgress->TreeStruct, ctx->LearnProgress->LeafValues));
}

void MapTensorSearchStart(TLearnContext* ctx) {
    Y_ASSERT(ctx->Params.SystemOptions->IsMaster());
    ApplyMapper<TTensorSearchStarter>(TMasterEnvironment::GetRef().RootEnvironment->GetSlaveCount(), TMasterEnvironment::GetRef().SharedTrainData);
}

void MapBootstrap(TLearnContext* ctx) {
    Y_ASSERT(ctx->Params.SystemOptions->IsMaster());
    ApplyMapper<TBootstrapMaker>(TMasterEnvironment::GetRef().RootEnvironment->GetSlaveCount(), TMasterEnvironment::GetRef().SharedTrainData);
}

double MapCalcDerivativesStDevFromZero(ui32 learnSampleCount, TLearnContext* ctx) {
    Y_ASSERT(ctx->Params.SystemOptions->IsMaster());
    const TVector<double> sumsFromWorkers = ApplyMapper<TDerivativesStDevFromZeroCalcer>(
        TMasterEnvironment::GetRef().RootEnvironment->GetSlaveCount(),
        TMasterEnvironment::GetRef().SharedTrainData);
    const double sum2 = Accumulate(sumsFromWorkers, 0.0);
    return sqrt(sum2 / learnSampleCount);
}

template <typename TScoreCalcMapper, typename TGetScore>
void MapGenericCalcScore(
    TGetScore getScore,
    double scoreStDev,
    TCandidatesContext* candidatesContext,
    TLearnContext* ctx) {

    Y_ASSERT(ctx->Params.SystemOptions->IsMaster());

    auto& candidateList = candidatesContext->CandidateList;

    const int workerCount = TMasterEnvironment::GetRef().RootEnvironment->GetSlaveCount();
    auto allStatsFromAllWorkers = ApplyMapper<TScoreCalcMapper>(
        workerCount,
        TMasterEnvironment::GetRef().SharedTrainData,
        candidateList);
    const int candidateCount = candidateList.ysize();
    const ui64 randSeed = ctx->LearnProgress->Rand.GenRand();
    // set best split for each candidate
    NPar::ParallelFor(
        *ctx->LocalExecutor,
        0,
        candidateCount,
        [&] (int candidateIdx) {
            auto& subCandidates = candidateList[candidateIdx].Candidates;
            const int subcandidateCount = subCandidates.ysize();
            TVector<TVector<double>> allScores(subcandidateCount);
            for (int subcandidateIdx = 0; subcandidateIdx < subcandidateCount; ++subcandidateIdx) {
                // reduce across workers
                auto& reducedStats = allStatsFromAllWorkers[0][candidateIdx][subcandidateIdx];
                for (int workerIdx = 1; workerIdx < workerCount; ++workerIdx) {
                    const auto& stats = allStatsFromAllWorkers[workerIdx][candidateIdx][subcandidateIdx];
                    reducedStats.Add(stats);
                }
                const auto& splitInfo = subCandidates[subcandidateIdx];
                allScores[subcandidateIdx] = getScore(reducedStats, splitInfo);
            }
            SetBestScore(randSeed + candidateIdx, allScores, scoreStDev, *candidatesContext, &subCandidates);
        });
}

// TODO(espetrov): Remove unused code.
void MapCalcScore(
    double scoreStDev,
    int depth,
    TCandidatesContext* candidatesContext,
    TLearnContext* ctx) {

    const auto& plainFold = ctx->LearnProgress->Folds[0];
    const auto getScore = [&] (const TStats3D& stats3D, const TCandidateInfo& splitInfo) {
        Y_UNUSED(splitInfo);

        return GetScores(stats3D,
                         depth,
                         plainFold.GetSumWeight(),
                         plainFold.GetLearnSampleCount(),
                         ctx->Params);
    };
    MapGenericCalcScore<TScoreCalcer>(getScore, scoreStDev, candidatesContext, ctx);
}

template <typename TBinCalcMapper, typename TScoreCalcMapper>
void MapGenericRemoteCalcScore(
    double scoreStDev,
    TCandidatesContext* candidatesContext,
    TLearnContext* ctx) {

    Y_ASSERT(ctx->Params.SystemOptions->IsMaster());

    auto& candidateList = candidatesContext->CandidateList;

    NPar::TJobDescription job;
    NPar::Map(&job, new TBinCalcMapper(), &candidateList);
    NPar::RemoteMap(&job, new TScoreCalcMapper);
    NPar::TJobExecutor exec(&job, TMasterEnvironment::GetRef().SharedTrainData);
    TVector<typename TScoreCalcMapper::TOutput> allScores;
    exec.GetRemoteMapResults(&allScores);
    // set best split for each candidate
    const int candidateCount = candidateList.ysize();
    Y_ASSERT(candidateCount == allScores.ysize());
    const ui64 randSeed = ctx->LearnProgress->Rand.GenRand();
    ctx->LocalExecutor->ExecRange(
        [&] (int candidateIdx) {
            auto& candidates = candidateList[candidateIdx].Candidates;
            Y_VERIFY(candidates.size() > 0);

            SetBestScore(
                randSeed + candidateIdx,
                allScores[candidateIdx],
                scoreStDev,
                *candidatesContext,
                &candidates);
        },
        0,
        candidateCount,
        NPar::TLocalExecutor::WAIT_COMPLETE);
}

void MapRemotePairwiseCalcScore(
    double scoreStDev,
    TCandidatesContext* candidatesContext,
    TLearnContext* ctx) {

    MapGenericRemoteCalcScore<TRemotePairwiseBinCalcer, TRemotePairwiseScoreCalcer>(
        scoreStDev,
        candidatesContext,
        ctx);
}

void MapRemoteCalcScore(
    double scoreStDev,
    TCandidatesContext* candidatesContext,
    TLearnContext* ctx) {

    MapGenericRemoteCalcScore<TRemoteBinCalcer, TRemoteScoreCalcer>(
        scoreStDev,
        candidatesContext,
        ctx);
}

void MapSetIndices(const TSplit& bestSplit, TLearnContext* ctx) {
    Y_ASSERT(ctx->Params.SystemOptions->IsMaster());
    const int workerCount = TMasterEnvironment::GetRef().RootEnvironment->GetSlaveCount();
    ApplyMapper<TLeafIndexSetter>(workerCount, TMasterEnvironment::GetRef().SharedTrainData, bestSplit);
}

int MapGetRedundantSplitIdx(TLearnContext* ctx) {
    Y_ASSERT(ctx->Params.SystemOptions->IsMaster());
    const int workerCount = TMasterEnvironment::GetRef().RootEnvironment->GetSlaveCount();
    TVector<TEmptyLeafFinder::TOutput> isLeafEmptyFromAllWorkers
        = ApplyMapper<TEmptyLeafFinder>(workerCount, TMasterEnvironment::GetRef().SharedTrainData); // poll workers
    for (int workerIdx = 1; workerIdx < workerCount; ++workerIdx) {
        for (int leafIdx = 0; leafIdx < isLeafEmptyFromAllWorkers[0].ysize(); ++leafIdx) {
            isLeafEmptyFromAllWorkers[0][leafIdx] &= isLeafEmptyFromAllWorkers[workerIdx][leafIdx];
        }
    }
    return GetRedundantSplitIdx(isLeafEmptyFromAllWorkers[0]);
}

void MapCalcErrors(TLearnContext* ctx) {
    Y_ASSERT(ctx->Params.SystemOptions->IsMaster());
    const size_t workerCount = TMasterEnvironment::GetRef().RootEnvironment->GetSlaveCount();

    // poll workers
    auto additiveStatsFromAllWorkers = ApplyMapper<TErrorCalcer>(workerCount, TMasterEnvironment::GetRef().SharedTrainData);
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
        ctx->Params.MetricOptions,
        ctx->EvalMetricDescriptor,
        ctx->LearnProgress->ApproxDimension,
        ctx->GetHasWeights()
    );
    const auto skipMetricOnTrain = GetSkipMetricOnTrain(metrics);
    Y_VERIFY(
        Accumulate(skipMetricOnTrain.begin(), skipMetricOnTrain.end(), 0) + additiveStats.size() ==
            metrics.size());
    for (int metricIdx = 0; metricIdx < metrics.ysize(); ++metricIdx) {
        if (!skipMetricOnTrain[metricIdx] && metrics[metricIdx]->IsAdditiveMetric()) {
            const auto description = metrics[metricIdx]->GetDescription();
            ctx->LearnProgress->MetricsAndTimeHistory.AddLearnError(
                *metrics[metricIdx].Get(),
                metrics[metricIdx]->GetFinalError(additiveStats[description]));
        }
    }
}

template <typename TApproxDefs>
void MapSetApproxes(
    const IDerCalcer& error,
    const TVariant<TSplitTree, TNonSymmetricTreeStructure>& splitTree,
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
    const int workerCount = TMasterEnvironment::GetRef().RootEnvironment->GetSlaveCount();
    ApplyMapper<TCalcApproxStarter>(workerCount, TMasterEnvironment::GetRef().SharedTrainData, splitTree);
    const int gradientIterations = ctx->Params.ObliviousTreeOptions->LeavesEstimationIterations;
    const int approxDimension = ctx->LearnProgress->ApproxDimension;
    const int leafCount = GetLeafCount(splitTree);
    const auto lossFunction = ctx->Params.LossFunctionDescription;
    const auto estimationMethod = ctx->Params.ObliviousTreeOptions->LeavesEstimationMethod;
    if (estimationMethod == ELeavesEstimation::Exact) {
        averageLeafValues->resize(approxDimension, TVector<double>(leafCount));
        double alpha;
        double delta;
        if (lossFunction->GetLossFunction() == ELossFunction::Quantile) {
            const auto& quantileError = dynamic_cast<const TQuantileError&>(error);
            alpha = quantileError.Alpha;
            delta = quantileError.Delta;
        } else {
            alpha = 0.5;
            delta = DBL_EPSILON;
        }

        const auto quantileLeafDeltasCalcer = ApplyMapper<TQuantileLeafDeltasCalcer>(workerCount, TMasterEnvironment::GetRef().SharedTrainData);

        TVector<TVector<double>> leafValues(approxDimension, TVector<double>(leafCount));

        TVector<TVector<std::pair<float, float>>> leafSamples(leafCount);
        TVector<std::pair<float, float>> tmp;
        for (int workerIdx = 0; workerIdx < workerCount; ++workerIdx) {
            const auto& workerSamples = quantileLeafDeltasCalcer[workerIdx];
            Y_ASSERT(leafCount == (int) workerSamples.size());
            for (int i = 0; i < leafCount; i++) {
                tmp.resize(leafSamples[i].size() + workerSamples[i].size());
                std::merge(leafSamples[i].begin(), leafSamples[i].end(), workerSamples[i].begin(), workerSamples[i].end(), tmp.begin());
                leafSamples[i].resize(leafSamples[i].size() + workerSamples[i].size());
                std::copy(tmp.begin(), tmp.end(), leafSamples[i].begin());
            }
        }

        NPar::ParallelFor(0, leafCount, [&](int i) {
            TVector<float> sample;
            TVector<float> weights;
            for (auto p: leafSamples[i]) {
                sample.push_back(p.first);
                weights.push_back(p.second);
            }
            leafValues[0][i] = CalcSampleQuantileSorted(sample, weights, alpha, delta);
        });

        AddElementwise(leafValues, averageLeafValues);
        ApplyMapper<TDeltaUpdater>(workerCount, TMasterEnvironment::GetRef().SharedTrainData, leafValues);
    } else {
        TVector<TSum> buckets(leafCount, TSum(approxDimension, error.GetHessianType()));
        averageLeafValues->resize(approxDimension, TVector<double>(leafCount));
        for (int it = 0; it < gradientIterations; ++it) {
            for (auto &bucket : buckets) {
                bucket.SetZeroDers();
            }

            TPairwiseBuckets pairwiseBuckets;
            TApproxDefs::SetPairwiseBucketsSize(leafCount, &pairwiseBuckets);
            const auto bucketsFromAllWorkers = ApplyMapper<TBucketUpdater>(workerCount, TMasterEnvironment::GetRef().SharedTrainData);
            // reduce across workers
            for (int workerIdx = 0; workerIdx < workerCount; ++workerIdx) {
                const auto &workerBuckets = bucketsFromAllWorkers[workerIdx].first;
                for (int leafIdx = 0; leafIdx < leafCount; ++leafIdx) {
                    if (ctx->Params.ObliviousTreeOptions->LeavesEstimationMethod == ELeavesEstimation::Gradient) {
                        buckets[leafIdx].AddDerWeight(
                                workerBuckets[leafIdx].SumDer,
                                workerBuckets[leafIdx].SumWeights,
                                /* updateWeight */ it == 0);
                    } else {
                        Y_ASSERT(
                                ctx->Params.ObliviousTreeOptions->LeavesEstimationMethod == ELeavesEstimation::Newton);
                        buckets[leafIdx].AddDerDer2(workerBuckets[leafIdx].SumDer, workerBuckets[leafIdx].SumDer2);
                    }
                }
                TApproxDefs::AddPairwiseBuckets(bucketsFromAllWorkers[workerIdx].second, &pairwiseBuckets);
            }
            const auto leafValues = TApproxDefs::CalcLeafValues(buckets, pairwiseBuckets, *ctx);
            AddElementwise(leafValues, averageLeafValues);
            // calc model and update approx deltas on workers
            ApplyMapper<TDeltaUpdater>(workerCount, TMasterEnvironment::GetRef().SharedTrainData, leafValues);
        }
    }

    // [workerIdx][dimIdx][leafIdx]
    const auto leafWeightsFromAllWorkers = ApplyMapper<TLeafWeightsGetter>(workerCount, TMasterEnvironment::GetRef().SharedTrainData);
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
    ApplyMapper<TApproxUpdater>(workerCount, TMasterEnvironment::GetRef().SharedTrainData, *averageLeafValues);
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
        ctx->LearnProgress.Get(),
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
        const size_t allDocCount = ctx.LearnProgress->Folds[0].GetLearnSampleCount();
        const double sumAllWeights = ctx.LearnProgress->Folds[0].GetSumWeight();
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

        const int dimensionCount = ctx.LearnProgress->ApproxDimension;
        const size_t leafCount = buckets.size();
        TVector<TVector<double>> leafValues(dimensionCount, TVector<double>(leafCount));
        const auto estimationMethod = ctx.Params.ObliviousTreeOptions->LeavesEstimationMethod;
        const float l2Regularizer = ctx.Params.ObliviousTreeOptions->L2Reg;
        const size_t allDocCount = ctx.LearnProgress->Folds[0].GetLearnSampleCount();
        const double sumAllWeights = ctx.LearnProgress->Folds[0].GetSumWeight();
        CalcLeafDeltasMulti(
            buckets,
            estimationMethod,
            l2Regularizer,
            sumAllWeights,
            allDocCount,
            &leafValues);
        return leafValues;
    }
};

void MapSetApproxesSimple(
    const IDerCalcer& error,
    const TVariant<TSplitTree, TNonSymmetricTreeStructure>& splitTree,
    TConstArrayRef<NCB::TTrainingForCPUDataProviderPtr> testData,
    TVector<TVector<double>>* averageLeafValues,
    TVector<double>* sumLeafWeights,
    TLearnContext* ctx) {

    MapSetApproxes<TSetApproxesSimpleDefs>(error, splitTree, testData, averageLeafValues, sumLeafWeights, ctx);
}

void MapSetApproxesMulti(
    const IDerCalcer& error,
    const TVariant<TSplitTree, TNonSymmetricTreeStructure>& splitTree,
    TConstArrayRef<NCB::TTrainingForCPUDataProviderPtr> testData,
    TVector<TVector<double>>* averageLeafValues,
    TVector<double>* sumLeafWeights,
    TLearnContext* ctx) {

    CB_ENSURE(!error.GetIsExpApprox(), "Multi-class does not support exponentiated approxes");
    MapSetApproxes<TSetApproxesMultiDefs>(error, splitTree, testData, averageLeafValues, sumLeafWeights, ctx);
}

void MapSetDerivatives(TLearnContext* ctx) {
    using namespace NCatboostDistributed;
    Y_ASSERT(ctx->Params.SystemOptions->IsMaster());
    ApplyMapper<TDerivativeSetter>(TMasterEnvironment::GetRef().RootEnvironment->GetSlaveCount(), TMasterEnvironment::GetRef().SharedTrainData);
}
