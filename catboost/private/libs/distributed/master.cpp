#include "master.h"
#include "mappers.h"

#include <catboost/libs/data/load_data.h>
#include <catboost/libs/helpers/quantile.h>
#include <catboost/private/libs/algo/approx_calcer_helpers.h>
#include <catboost/private/libs/algo/approx_calcer/gradient_walker.h>
#include <catboost/private/libs/algo/approx_updater_helpers.h>
#include <catboost/private/libs/algo/data.h>
#include <catboost/private/libs/algo/index_calcer.h>
#include <catboost/private/libs/algo/score_calcers.h>
#include <catboost/private/libs/algo/scoring.h>
#include <catboost/private/libs/algo_helpers/approx_calcer_multi_helpers.h>
#include <catboost/private/libs/algo_helpers/error_functions.h>
#include <catboost/private/libs/options/json_helper.h>

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
            WriteTJsonValue(trainParams),
            objectsOrder,
            objectsGrouping,
            featuresLayout,
            rand->GenRand()
        }
    );
}

void SetTrainDataFromMaster(
    const TTrainingDataProviders& trainData,
    ui64 cpuUsedRamLimit,
    NPar::TLocalExecutor* localExecutor
) {
    const int workerCount = TMasterEnvironment::GetRef().RootEnvironment->GetSlaveCount();
    auto workerParts = Split(*trainData.Learn->ObjectsGrouping, (ui32)workerCount);
    for (int workerIdx = 0; workerIdx < workerCount; ++workerIdx) {
        const TObjectsGroupingSubset objectsGroupingSubset = NCB::GetSubset(
            trainData.Learn->ObjectsGrouping,
            std::move(workerParts[workerIdx]),
            EObjectsOrder::Ordered);

        NCB::TTrainingDataProviders workerTrainData;
        workerTrainData.Learn = trainData.Learn->GetSubset(
            objectsGroupingSubset,
            cpuUsedRamLimit,
            localExecutor);
        workerTrainData.FeatureEstimators = trainData.FeatureEstimators;
        if (trainData.EstimatedObjectsData.Learn) {
            workerTrainData.EstimatedObjectsData.Learn =
                dynamic_cast<TQuantizedForCPUObjectsDataProvider*>(
                    trainData.EstimatedObjectsData.Learn->GetSubset(
                        objectsGroupingSubset,
                        cpuUsedRamLimit,
                        localExecutor).Get());
        }
        workerTrainData.EstimatedObjectsData.FeatureEstimators
            = trainData.EstimatedObjectsData.FeatureEstimators;
        workerTrainData.EstimatedObjectsData.QuantizedEstimatedFeaturesInfo
            = trainData.EstimatedObjectsData.QuantizedEstimatedFeaturesInfo;

        TMasterEnvironment::GetRef().SharedTrainData->SetContextData(
            workerIdx,
            new NCatboostDistributed::TTrainData(std::move(workerTrainData)),
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
            WriteTJsonValue(jsonParams),
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
    TVector<TCandidatesContext>* candidatesContexts,
    TLearnContext* ctx) {

    Y_ASSERT(ctx->Params.SystemOptions->IsMaster());

    // Flatten candidateLists from all contexts to ensure even parallelization
    TCandidateList allCandidatesList;
    for (auto& candidatesContext : *candidatesContexts) {
        allCandidatesList.insert(
            allCandidatesList.end(),
            candidatesContext.CandidateList.begin(),
            candidatesContext.CandidateList.end());
    }

    NPar::TJobDescription job;
    NPar::Map(&job, new TBinCalcMapper(), &allCandidatesList);
    NPar::RemoteMap(&job, new TScoreCalcMapper);
    NPar::TJobExecutor exec(&job, TMasterEnvironment::GetRef().SharedTrainData);
    TVector<typename TScoreCalcMapper::TOutput> allScores;
    exec.GetRemoteMapResults(&allScores);
    // set best split for each candidate
    Y_ASSERT(allCandidatesList.size() == allScores.size());
    const ui64 randSeed = ctx->LearnProgress->Rand.GenRand();

    size_t allScoresOffset = 0;
    for (auto& candidatesContext : *candidatesContexts) {
        auto& candidateList = candidatesContext.CandidateList;
        ctx->LocalExecutor->ExecRange(
            [&] (int candidateIdx) {
                auto& candidates = candidateList[candidateIdx].Candidates;
                Y_VERIFY(candidates.size() > 0);

                SetBestScore(
                    randSeed + candidateIdx,
                    allScores[allScoresOffset + candidateIdx],
                    scoreStDev,
                    candidatesContext,
                    &candidates);
            },
            0,
            candidateList.ysize(),
            NPar::TLocalExecutor::WAIT_COMPLETE);
        allScoresOffset += candidateList.size();
    }
}

void MapRemotePairwiseCalcScore(
    double scoreStDev,
    TVector<TCandidatesContext>* candidatesContexts,
    TLearnContext* ctx) {

    MapGenericRemoteCalcScore<TRemotePairwiseBinCalcer, TRemotePairwiseScoreCalcer>(
        scoreStDev,
        candidatesContexts,
        ctx);
}

void MapRemoteCalcScore(
    double scoreStDev,
    TVector<TCandidatesContext>* candidatesContexts,
    TLearnContext* ctx) {

    MapGenericRemoteCalcScore<TRemoteBinCalcer, TRemoteScoreCalcer>(
        scoreStDev,
        candidatesContexts,
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

static THashMap<TString, TMetricHolder> CalcAdditiveStats(bool useAveragingFold) {
    const size_t workerCount = TMasterEnvironment::GetRef().RootEnvironment->GetSlaveCount();

    // poll workers
    auto additiveStatsFromAllWorkers = ApplyMapper<TErrorCalcer>(
        workerCount,
        TMasterEnvironment::GetRef().SharedTrainData,
        useAveragingFold);
    Y_ASSERT(additiveStatsFromAllWorkers.size() == workerCount);

    auto& additiveStats = additiveStatsFromAllWorkers[0];
    for (size_t workerIdx : xrange<size_t>(1, workerCount)) {
        const auto& workerAdditiveStats = additiveStatsFromAllWorkers[workerIdx];
        for (auto& [description, stats] : additiveStats) {
            Y_ASSERT(workerAdditiveStats.contains(description));
            stats.Add(workerAdditiveStats.at(description));
        }
    }
    return additiveStats;
}

void MapCalcErrors(TLearnContext* ctx) {
    Y_ASSERT(ctx->Params.SystemOptions->IsMaster());

    const auto& additiveStats = CalcAdditiveStats(/*useAveragingFold*/false);

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
                metrics[metricIdx]->GetFinalError(additiveStats.at(description)));
        }
    }
}

template <typename TDeltaUpdater>
static void UpdateLeavesExact(
    const IDerCalcer& error,
    int leafCount,
    TVector<TVector<double>>* averageLeafValues,
    TLearnContext* ctx
) {
    const int approxDimension = ctx->LearnProgress->ApproxDimension;
    const auto lossFunction = ctx->Params.LossFunctionDescription;

    Y_ASSERT(EqualToOneOf(lossFunction->GetLossFunction(), ELossFunction::Quantile, ELossFunction::MAE, ELossFunction::MAPE));
    averageLeafValues->resize(approxDimension, TVector<double>(leafCount));
    double alpha = 0.5;
    double delta = 0.0;
    if (const auto quantileError = dynamic_cast<const TQuantileError*>(&error)) {
        alpha = quantileError->Alpha;
        delta = quantileError->Delta;
    }

    TVector<TUnusedInitializedParam> emptyInput(1);

    TVector<TVector<TMinMax<double>>> searchIntervals;
    NPar::RunMapReduce(TMasterEnvironment::GetRef().SharedTrainData.Get(), new TQuantileExactApproxStarter(), &emptyInput, &searchIntervals);

    TVector<double> totalLeafWeights;
    NPar::RunMapReduce(TMasterEnvironment::GetRef().SharedTrainData.Get(), new TLeafWeightsGetter(), &emptyInput, &totalLeafWeights);

    TVector<double> neededLeftWeigths(leafCount);
    for (auto leaf : xrange(leafCount)) {
        neededLeftWeigths[leaf] = alpha * totalLeafWeights[leaf];
    }

    constexpr int BINARY_SEARCH_ITERATIONS = 100;
    TVector<TVector<double>> pivots(approxDimension, TVector<double>(leafCount));
    TVector<TVector<double>> leftWeights(approxDimension, TVector<double>(leafCount));
    for (auto iter : xrange(BINARY_SEARCH_ITERATIONS)) {
        Y_UNUSED(iter);
        // calc new pivots
        for (auto dimension : xrange(approxDimension)) {
            for (auto leaf : xrange(leafCount)) {
                const double leftValue = searchIntervals[dimension][leaf].Min;
                const double rightValue = searchIntervals[dimension][leaf].Max;
                pivots[dimension][leaf] = (leftValue + rightValue) / 2;
            }
        }
        TVector<TVector<TVector<double>>> splitCmdInput(1, pivots);
        NPar::RunMapReduce(TMasterEnvironment::GetRef().SharedTrainData.Get(), new TQuantileArraySplitter(), &splitCmdInput, &leftWeights);
        for (auto dimension : xrange(approxDimension)) {
            for (auto leaf : xrange(leafCount)) {
                if (leftWeights[dimension][leaf] < neededLeftWeigths[leaf]) {
                    searchIntervals[dimension][leaf].Min = pivots[dimension][leaf];
                } else {
                    searchIntervals[dimension][leaf].Max = pivots[dimension][leaf];
                }
            }
        }
    }

    TVector<TVector<double>> leafValues(approxDimension, TVector<double>(leafCount));
    for (auto dimension : xrange(approxDimension)) {
        for (auto leaf : xrange(leafCount)) {
            if (totalLeafWeights[leaf] > 0) {
                leafValues[dimension][leaf] = searchIntervals[dimension][leaf].Max;
            }
        }
    }

    // specific adjust according to delta parameter of Quantile loss
    if (delta > 0) {
        TVector<TVector<TVector<double>>> cmdInput(1, leafValues);
        TVector<TVector<double>> equalSumWeights;
        NPar::RunMapReduce(TMasterEnvironment::GetRef().SharedTrainData.Get(), new TQuantileEqualWeightsCalcer(), &cmdInput, &equalSumWeights);
        for (auto dimension : xrange(approxDimension)) {
            for (auto leaf : xrange(leafCount)) {
                if (totalLeafWeights[leaf] > 0) {
                    if (leftWeights[dimension][leaf] + alpha * equalSumWeights[dimension][leaf] >= neededLeftWeigths[leaf] - DBL_EPSILON) {
                        leafValues[dimension][leaf] -= delta;
                    } else {
                        leafValues[dimension][leaf] += delta;
                    }
                }
            }
        }
    }

    AddElementwise(leafValues, averageLeafValues);
    const int workerCount = TMasterEnvironment::GetRef().RootEnvironment->GetSlaveCount();
    ApplyMapper<TDeltaUpdater>(workerCount, TMasterEnvironment::GetRef().SharedTrainData, leafValues);
}

template <typename TApproxDefs>
void MapSetApproxes(
    const IDerCalcer& error,
    const TVariant<TSplitTree, TNonSymmetricTreeStructure>& splitTree,
    const NCB::TTrainingDataProviders data, // only test part is used
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
        UpdateLeavesExact<TDeltaUpdater>(error, leafCount, averageLeafValues, ctx);
    } else {
        TVector<TSum> buckets(leafCount, TSum(approxDimension, error.GetHessianType()));
        const auto leafUpdaterFunc = [&] (
            bool recalcLeafWeights,
            const TVector<TVector<double>>& /*approxesPlaceholder*/,
            TVector<TVector<double>>* leafValues
        ) {
            for (auto &bucket : buckets) {
                bucket.SetZeroDers();
            }
            TPairwiseBuckets pairwiseBuckets;
            TApproxDefs::SetPairwiseBucketsSize(leafCount, &pairwiseBuckets);
            const auto bucketsFromAllWorkers = ApplyMapper<TBucketUpdater>(workerCount, TMasterEnvironment::GetRef().SharedTrainData);
            // reduce across workers
            for (const auto& workerBuckets : bucketsFromAllWorkers) {
                const auto& singleBuckets = workerBuckets.first;
                for (int leafIdx = 0; leafIdx < leafCount; ++leafIdx) {
                    if (ctx->Params.ObliviousTreeOptions->LeavesEstimationMethod == ELeavesEstimation::Gradient) {
                        buckets[leafIdx].AddDerWeight(
                            singleBuckets[leafIdx].SumDer,
                            singleBuckets[leafIdx].SumWeights,
                            recalcLeafWeights);
                    } else {
                        Y_ASSERT(
                            ctx->Params.ObliviousTreeOptions->LeavesEstimationMethod == ELeavesEstimation::Newton);
                        buckets[leafIdx].AddDerDer2(singleBuckets[leafIdx].SumDer, singleBuckets[leafIdx].SumDer2);
                    }
                }
                TApproxDefs::AddPairwiseBuckets(workerBuckets.second, &pairwiseBuckets);
            }
            *leafValues = TApproxDefs::CalcLeafValues(buckets, pairwiseBuckets, *ctx);
        };

        const auto approxUpdaterFunc = [&] (
            const TVector<TVector<double>>& leafValues,
            TVector<TVector<double>>* /*approxesPlaceholder*/
        ) {
            ApplyMapper<TDeltaUpdater>(workerCount, TMasterEnvironment::GetRef().SharedTrainData, leafValues);
        };

        bool haveBacktrackingObjective;
        double minimizationSign;
        TVector<THolder<IMetric>> lossFunction;
        CreateBacktrackingObjective(*ctx, &haveBacktrackingObjective, &minimizationSign, &lossFunction);
        const auto lossCalcerFunc = [&] (const TVector<TVector<double>>& /*approxPlaceholder*/) {
            CB_ENSURE_INTERNAL(
                haveBacktrackingObjective,
                "Trivial gradient walker should not calculate loss function");
            const auto& additiveStats = CalcAdditiveStats(/*useAveragingFold*/true);
            const auto objectiveDescription = lossFunction[0]->GetDescription();
            return minimizationSign * lossFunction[0]->GetFinalError(additiveStats.at(objectiveDescription));
        };

        TVector<TVector<double>> approxesPlaceholder;
        const auto approxCopyFunc = [&](const TVector<TVector<double>>& /*src*/, TVector<TVector<double>>* dst) {
            const bool isRestore = dst == &approxesPlaceholder;
            ApplyMapper<TArmijoStartPointBackupper>(workerCount, TMasterEnvironment::GetRef().SharedTrainData, isRestore);
        };

        averageLeafValues->resize(approxDimension, TVector<double>(leafCount));
        GradientWalker</*IsLeafwise*/ false>(
            /*isTrivialWalker*/ !haveBacktrackingObjective,
            gradientIterations,
            leafCount,
            ctx->LearnProgress->ApproxDimension,
            leafUpdaterFunc,
            approxUpdaterFunc,
            lossCalcerFunc,
            approxCopyFunc,
            &approxesPlaceholder,
            averageLeafValues);
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
        splitTree,
        data,
        EBuildIndicesDataParts::TestOnly,
        ctx->LocalExecutor);
    UpdateAvrgApprox(
        error.GetIsExpApprox(),
        /*learnSampleCount*/ 0,
        indices,
        *averageLeafValues,
        data.Test,
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
    const NCB::TTrainingDataProviders data, // only test part is used
    TVector<TVector<double>>* averageLeafValues,
    TVector<double>* sumLeafWeights,
    TLearnContext* ctx) {

    MapSetApproxes<TSetApproxesSimpleDefs>(error, splitTree, data, averageLeafValues, sumLeafWeights, ctx);
}

void MapSetApproxesMulti(
    const IDerCalcer& error,
    const TVariant<TSplitTree, TNonSymmetricTreeStructure>& splitTree,
    const NCB::TTrainingDataProviders data, // only test part is used
    TVector<TVector<double>>* averageLeafValues,
    TVector<double>* sumLeafWeights,
    TLearnContext* ctx) {

    CB_ENSURE(!error.GetIsExpApprox(), "Multi-class does not support exponentiated approxes");
    MapSetApproxes<TSetApproxesMultiDefs>(error, splitTree, data, averageLeafValues, sumLeafWeights, ctx);
}

void MapSetDerivatives(TLearnContext* ctx) {
    using namespace NCatboostDistributed;
    Y_ASSERT(ctx->Params.SystemOptions->IsMaster());
    ApplyMapper<TDerivativeSetter>(TMasterEnvironment::GetRef().RootEnvironment->GetSlaveCount(), TMasterEnvironment::GetRef().SharedTrainData);
}
