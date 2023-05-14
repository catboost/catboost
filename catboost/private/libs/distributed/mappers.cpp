#include "mappers.h"

#include <catboost/libs/helpers/matrix.h>

#include <catboost/private/libs/algo/approx_calcer.h>
#include <catboost/private/libs/algo/approx_delta_calcer_multi.h>
#include <catboost/private/libs/algo/approx_updater_helpers.h>
#include <catboost/private/libs/algo/data.h>
#include <catboost/private/libs/algo/helpers.h>
#include <catboost/private/libs/algo/learn_context.h>
#include <catboost/private/libs/algo/online_ctr.h>
#include <catboost/private/libs/algo/preprocess.h>
#include <catboost/private/libs/algo/scoring.h>
#include <catboost/private/libs/algo/split.h>
#include <catboost/private/libs/algo_helpers/approx_calcer_helpers.h>
#include <catboost/private/libs/algo_helpers/approx_calcer_multi_helpers.h>
#include <catboost/private/libs/algo_helpers/error_functions.h>
#include <catboost/libs/data/load_data.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/parallel_tasks.h>
#include <catboost/libs/helpers/query_info_helper.h>
#include <catboost/libs/helpers/vector_helpers.h>
#include <catboost/private/libs/index_range/index_range.h>

#include <util/generic/ymath.h>

#include <limits>
#include <utility>


namespace NCatboostDistributed {

    static const NCB::TTrainingDataProviders& GetTrainData(NPar::TCtxPtr<TTrainData> trainData) {
        if (trainData != nullptr) {
            return trainData->TrainData;
        } else {
            return TLocalTensorSearchData::GetRef().TrainData;
        }
    }

    static const NCB::TQuantizedObjectsDataProvider& GetLearnObjectsData(
        const NCB::TTrainingDataProviders& trainingData,
        bool estimated
    ) {
        return *(estimated ? trainingData.EstimatedObjectsData.Learn : trainingData.Learn->ObjectsData);
    }


    static NJson::TJsonValue GetJson(const TString& string) {
        NJson::TJsonValue json;
        const bool isJson = ReadJsonTree(string, &json);
        Y_ASSERT(isJson);
        return json;
    }

    static TVector<NCB::TIndexRange<ui32>> WorkaroundSplit(
        const NCB::TObjectsGrouping& objectsGrouping,
        ui32 workerCount
    ) {
        const ui32 groupCount = objectsGrouping.GetGroupCount();
        const ui32 objectCount = objectsGrouping.GetObjectCount();
        TVector<NCB::TIndexRange<ui32>> result;
        if (groupCount == objectCount) {
            CB_ENSURE(objectCount >= workerCount, "Pool must contain at least " << workerCount << " objects");
            for (ui32 workerIdx : xrange(workerCount)) {
                const ui32 workerStart = CeilDiv(objectCount, workerCount) * workerIdx;
                const ui32 workerEnd = Min(workerStart + CeilDiv(objectCount, workerCount), objectCount);
                result.emplace_back(workerStart, workerEnd);
            }
        } else {
            CB_ENSURE(groupCount >= workerCount, "Pool must contain at least " << workerCount << " groups");
            ui32 previousGroupEnd = 0;
            for (const auto& groupIdx : xrange(groupCount)) {
                CB_ENSURE(objectsGrouping.GetGroup(groupIdx).Begin == previousGroupEnd, "Groups must follow each other without gaps");
                previousGroupEnd = objectsGrouping.GetGroup(groupIdx).End;
            }
            for (ui32 workerIdx : xrange(workerCount)) {
                const ui32 workerStartGroup = CeilDiv(groupCount, workerCount) * workerIdx;
                const ui32 workerEndGroup = Min(workerStartGroup + CeilDiv(groupCount, workerCount), groupCount);
                if (workerEndGroup == groupCount) {
                    result.emplace_back(
                        objectsGrouping.GetGroup(workerStartGroup).Begin,
                        objectCount
                    );
                } else {
                    result.emplace_back(
                        objectsGrouping.GetGroup(workerStartGroup).Begin,
                        objectsGrouping.GetGroup(workerEndGroup).Begin
                    );
                }
            }
        }
        return result;
    }

    static NCB::TDatasetSubset GetSubsetForWorker(
       int workerCount,
       int hostId,
       const NCB::TObjectsGrouping& objectsGrouping
    ) {
        const auto workerParts = WorkaroundSplit(objectsGrouping, workerCount);
        const ui32 loadStart = workerParts[hostId].Begin;
        const ui32 loadEnd = workerParts[hostId].End;
        return NCB::TDatasetSubset::MakeRange(loadStart, loadEnd);
    }

    void TDatasetsLoader::DoMap(
        NPar::IUserContext* ctx,
        int hostId,
        TInput* params,
        TOutput* /*unused*/
    ) const {
        auto& localData = TLocalTensorSearchData::GetRef();
        if (localData.Rand == nullptr) {
            localData.Rand = MakeHolder<TRestorableFastRng64>(params->RandomSeed + hostId);
        }

        const int workerCount = ctx->GetHostIdCount();
        CATBOOST_DEBUG_LOG << "Worker count " << workerCount << Endl;

        TVector<NCB::TDatasetSubset> testDatasetSubsets;
        for (const auto& testObjectsGrouping : params->TestObjectsGroupings) {
            testDatasetSubsets.push_back(GetSubsetForWorker(workerCount, hostId, testObjectsGrouping));
        }

        NCatboostOptions::TCatBoostOptions catBoostOptions(ETaskType::CPU);
        catBoostOptions.Load(GetJson(params->TrainOptions));
        catBoostOptions.SystemOptions->FileWithHosts->clear();

        const auto poolLoadOptions = params->PoolLoadOptions;
        TProfileInfo profile;
        CATBOOST_DEBUG_LOG << "Load quantized pool section for worker " << hostId << "..." << Endl;
        auto pools = NCB::ReadTrainDatasets(
            /*taskType*/ETaskType::CPU,
            poolLoadOptions,
            params->ObjectsOrder,
            /*readTest*/true,
            GetSubsetForWorker(workerCount, hostId, params->LearnObjectsGrouping),
            testDatasetSubsets,
            catBoostOptions.DataProcessingOptions->ForceUnitAutoPairWeights,
            &localData.ClassLabelsFromDataset,
            &NPar::LocalExecutor(),
            &profile
        );

        auto quantizedFeaturesInfo = MakeIntrusive<NCB::TQuantizedFeaturesInfo>(
            params->FeaturesLayout,
            catBoostOptions.DataProcessingOptions.Get().IgnoredFeatures.Get(),
            catBoostOptions.DataProcessingOptions->FloatFeaturesBinarization.Get(),
            catBoostOptions.DataProcessingOptions->PerFloatFeatureQuantization.Get()
        );

        CATBOOST_DEBUG_LOG << "Create train data for worker " << hostId << "..." << Endl;
        localData.TrainData = GetTrainingData(
            std::move(pools),
            /*trainDataCanBeEmpty*/ true,
            /*borders*/ Nothing(), // borders are already loaded to quantizedFeaturesInfo
            /*ensureConsecutiveIfDenseLearnFeaturesDataForCpu*/ true,
            /*allowWriteFiles*/ false,
            /*tmpDir*/ TString(), // does not matter, because allowWritingFiles == false
            quantizedFeaturesInfo,
            &catBoostOptions,
            &params->LabelConverter,
            &NPar::LocalExecutor(),
            localData.Rand.Get()
        );

        CATBOOST_DEBUG_LOG << "Done for worker " << hostId << Endl;
    }

    void TPlainFoldBuilder::DoMap(
        NPar::IUserContext* ctx,
        int hostId,
        TInput* params,
        TOutput* /*unused*/
    ) const {
        NPar::TCtxPtr<TTrainData> trainData(ctx, SHARED_ID_TRAIN_DATA, hostId);
        auto& localData = TLocalTensorSearchData::GetRef();
        if (localData.Rand == nullptr) { // may be set by TDatasetsLoader
            localData.Rand = MakeHolder<TRestorableFastRng64>(params->RandomSeed + hostId);
        }

        auto trainParamsJson = GetJson(params->TrainParams);
        UpdateUndefinedClassLabels(localData.ClassLabelsFromDataset, &trainParamsJson);
        localData.Params.Load(trainParamsJson);

        const auto& trainParams = localData.Params;

        const NCB::TTrainingDataProviders& trainingDataProviders = GetTrainData(trainData);

        const TFoldsCreationParams foldsCreationParams(
            trainParams,
            *trainingDataProviders.Learn->ObjectsData,
            /*startingApprox*/ Nothing(),
            /*isForWorkerLocalData*/ true);

        localData.Progress = MakeHolder<TLearnProgress>(
            /*isForWorkerLocalData*/ true,
            /*isSingleHost*/ false,
            trainingDataProviders,
            params->ApproxDimension,
            TLabelConverter(), // unused in case of localData
            params->RandomSeed + hostId,
            /*initRand*/ localData.Rand.Get(),
            foldsCreationParams,
            /*datasetsCanContainBaseline*/ true,
            params->TargetClassifiers,
            /*featuresCheckSum*/ 0, // unused in case of localData
            /*foldCreationParamsCheckSum*/ 0,
            /*estimatedFeaturesQuantizationOptions*/
                trainParams.DataProcessingOptions->FloatFeaturesBinarization.Get(),
            localData.PrecomputedSingleOnlineCtrDataForSingleFold,
            trainParams.ObliviousTreeOptions.Get(),
            /*initModel*/ Nothing(),
            /*initModelApplyCompatiblePools*/ NCB::TDataProviders(),
            &NPar::LocalExecutor());

        localData.HessianType = params->HessianType;

        localData.StoreExpApprox = foldsCreationParams.StoreExpApproxes;

        const bool isPairwiseScoring = IsPairwiseScoring(
            trainParams.LossFunctionDescription->GetLossFunction());
        const int defaultCalcStatsObjBlockSize =
            static_cast<int>(trainParams.ObliviousTreeOptions->DevScoreCalcObjBlockSize);
        const bool hasOfflineEstimatedFeatures =
            !GetTrainData(trainData).EstimatedObjectsData.QuantizedEstimatedFeaturesInfo.Layout.empty();

        const auto learnObjectCount = trainingDataProviders.Learn->GetObjectCount();
        if (learnObjectCount) {
            Y_ASSERT(localData.Progress->AveragingFold.BodyTailArr.ysize() == 1);

            localData.UseTreeLevelCaching = NeedToUseTreeLevelCaching(
                trainParams,
                /*maxBodyTailCount=*/1,
                localData.Progress->AveragingFold.GetApproxDimension());

            auto& plainFold = localData.Progress->AveragingFold;
            localData.SampledDocs.Create(
                { plainFold },
                isPairwiseScoring,
                hasOfflineEstimatedFeatures,
                defaultCalcStatsObjBlockSize,
                GetBernoulliSampleRate(trainParams.ObliviousTreeOptions->BootstrapConfig));
            if (localData.UseTreeLevelCaching) {
                localData.SmallestSplitSideDocs.Create(
                    { plainFold },
                    isPairwiseScoring,
                    hasOfflineEstimatedFeatures,
                    defaultCalcStatsObjBlockSize);
                localData.PrevTreeLevelStats.Create(
                    { plainFold },
                    CountNonCtrBuckets(
                        *(GetTrainData(trainData).Learn->ObjectsData->GetFeaturesLayout()),
                        *(GetTrainData(trainData).Learn->ObjectsData->GetQuantizedFeaturesInfo()),
                        trainParams.CatFeatureParams->OneHotMaxSize.Get()),
                    trainParams.ObliviousTreeOptions->MaxDepth);
            }
        }
        localData.Indices.yresize(learnObjectCount + trainingDataProviders.GetTestSampleCount());
        localData.AllDocCount = params->AllDocCount;
        localData.SumAllWeights = params->SumAllWeights;
    }

    void TApproxReconstructor::DoMap(
        NPar::IUserContext* ctx,
        int hostId,
        TInput* params,
        TOutput* /*unused*/
    ) const {
        NPar::TCtxPtr<TTrainData> trainData(ctx, SHARED_ID_TRAIN_DATA, hostId);

        auto& localData = TLocalTensorSearchData::GetRef();
        Y_ASSERT(IsPlainMode(localData.Params.BoostingOptions->BoostingType));

        const auto& forest = params->TreeStruct;
        const auto& leafValues = params->LeafValues;
        Y_ASSERT(forest.size() == leafValues.size());

        const auto& trainingDataProviders = GetTrainData(trainData);

        auto maybeBaseline = trainingDataProviders.Learn->TargetData->GetBaseline();
        if (maybeBaseline) {
            AssignRank2<float>(*maybeBaseline, &localData.Progress->AvrgApprox);
        }
        for (auto testIdx : xrange(trainingDataProviders.Test.size())) {
            auto maybeBaseline = trainingDataProviders.Test[testIdx]->TargetData->GetBaseline();
            if (maybeBaseline) {
                AssignRank2<float>(*maybeBaseline, &localData.Progress->TestApprox[testIdx]);
            }
        }

        const ui32 learnSampleCount = GetTrainData(trainData).Learn->GetObjectCount();
        const bool storeExpApprox = IsStoreExpApprox(
            localData.Params.LossFunctionDescription->GetLossFunction());
        const auto& avrgFold = localData.Progress->AveragingFold;
        for (size_t treeIdx : xrange(forest.size())) {
            const auto leafIndices = BuildIndices(
                avrgFold,
                forest[treeIdx],
                GetTrainData(trainData),
                EBuildIndicesDataParts::All,
                &NPar::LocalExecutor());
            UpdateAvrgApprox(
                storeExpApprox,
                learnSampleCount,
                leafIndices,
                leafValues[treeIdx],
                GetTrainData(trainData).Test,
                localData.Progress.Get(),
                &NPar::LocalExecutor());
            if (params->BestIteration && (treeIdx == SafeIntegerCast<size_t>(*params->BestIteration))) {
                localData.Progress->BestTestApprox = localData.Progress->TestApprox.back();
            }
        }
    }

    void TTensorSearchStarter::DoMap(
        NPar::IUserContext* /*ctx*/,
        int /*hostId*/,
        TInput* /*unused*/,
        TOutput* /*unused*/
    ) const {
        auto& localData = TLocalTensorSearchData::GetRef();
        localData.Depth = 0;
        Fill(localData.Indices.begin(), localData.Indices.end(), 0);
        if (localData.Progress->AveragingFold.GetLearnSampleCount() && localData.UseTreeLevelCaching) {
            localData.PrevTreeLevelStats.GarbageCollect();
        }
    }

    void TBootstrapMaker::DoMap(
        NPar::IUserContext* ctx,
        int hostId,
        TInput* /*unused*/,
        TOutput* /*unused*/
    ) const {
        auto& localData = TLocalTensorSearchData::GetRef();

        NPar::TCtxPtr<TTrainData> trainData(ctx, SHARED_ID_TRAIN_DATA, hostId);

        const auto learnObjectCount = GetTrainData(trainData).Learn->GetObjectCount();

        if (learnObjectCount) {
            Bootstrap(
                localData.Params,
                !localData.Progress->EstimatedFeaturesContext.OfflineEstimatedFeaturesLayout.empty(),
                TConstArrayRef<TIndexType>(
                    localData.Indices.data(),
                    learnObjectCount),
                localData.Progress->LeafValues,
                &localData.Progress->AveragingFold,
                &localData.SampledDocs,
                &NPar::LocalExecutor(),
                &localData.Progress->Rand);
            localData.FlatPairs = UnpackPairsFromQueries(localData.Progress->AveragingFold.LearnQueriesInfo);
        }
    }

    void TDerivativesStDevFromZeroCalcer::DoMap(
        NPar::IUserContext* /*ctx*/,
        int /*hostId*/,
        TInput* /*unused*/,
        TOutput* outSum2
    ) const {
        auto& localData = TLocalTensorSearchData::GetRef();

        double sum2 = 0;
        if (localData.Progress->AveragingFold.GetLearnSampleCount()) {
            const auto& bodyTailArr = localData.Progress->AveragingFold.BodyTailArr;
            Y_ASSERT(bodyTailArr.size() == 1);
            const auto& weightedDerivatives = bodyTailArr.front().WeightedDerivatives;
            Y_ASSERT(weightedDerivatives.size() > 0);


            for (const auto& perDimensionWeightedDerivatives : weightedDerivatives) {
                sum2 += NCB::L2NormSquared<double>(perDimensionWeightedDerivatives, &NPar::LocalExecutor());
            }
        }
        *outSum2 = sum2;
    }

    template <typename TMapFunc, typename TInputType, typename TOutputType>
    static void MapVector(const TMapFunc mapFunc,
        const TVector<TInputType>& inputs,
        TVector<TOutputType>* mappedInputs
    ) {
        mappedInputs->yresize(inputs.ysize());
        NPar::ParallelFor(
            0,
            inputs.ysize(),
            [&] (int inputIdx) {
                mapFunc(inputs[inputIdx], &(*mappedInputs)[inputIdx]);
            });
    }

    template <typename TMapFunc, typename TStatsType>
    static void MapCandidateList(
        const TMapFunc mapFunc,
        const TCandidateList& candidates,
        TVector<TVector<TStatsType>>* candidateStats
    ) {
        const auto mapSubcandidate = [&] (const TCandidateInfo& subcandidate, TStatsType* subcandidateStats) {
            mapFunc(subcandidate, subcandidateStats);
        };
        const auto mapCandidate = [&] (
            const TCandidatesInfoList& candidate,
            TVector<TStatsType>* candidateStats
        ) {
            MapVector(mapSubcandidate, candidate.Candidates, candidateStats);
        };
        MapVector(mapCandidate, candidates, candidateStats );
    }

    static void CalcStats3D(
        const NPar::TCtxPtr<TTrainData>& trainData,
        const TCandidateInfo& candidate,
        TStats3D* stats3D
    ) {
        auto& localData = TLocalTensorSearchData::GetRef();
        const NCB::TTrainingDataProviders& trainingData = GetTrainData(trainData);
        CalcStatsAndScores(
            GetLearnObjectsData(trainingData, candidate.SplitEnsemble.IsEstimated),
            localData.Progress->AveragingFold.GetAllCtrs(),
            localData.SampledDocs,
            localData.SmallestSplitSideDocs,
            /*initialFold*/nullptr,
            /*pairs*/{},
            localData.Params,
            candidate,
            localData.Depth,
            localData.UseTreeLevelCaching,
            /*currTreeMonotonicConstraints*/{},
            /*monotonicConstraints*/{},
            &NPar::LocalExecutor(),
            &localData.PrevTreeLevelStats,
            stats3D,
            /*pairwiseStats*/nullptr,
            /*scoreCalcer*/nullptr);
    }

    static void CalcPairwiseStats(const NPar::TCtxPtr<TTrainData>& trainData,
        const TFlatPairsInfo& pairs,
        const TCandidateInfo& candidate,
        TPairwiseStats* pairwiseStats
    ) {
        auto& localData = TLocalTensorSearchData::GetRef();
        const NCB::TTrainingDataProviders& trainingData = GetTrainData(trainData);
        CalcStatsAndScores(
            GetLearnObjectsData(trainingData, candidate.SplitEnsemble.IsEstimated),
            localData.Progress->AveragingFold.GetAllCtrs(),
            localData.SampledDocs,
            localData.SmallestSplitSideDocs,
            /*initialFold*/nullptr,
            pairs,
            localData.Params,
            candidate,
            localData.Depth,
            localData.UseTreeLevelCaching,
            /*currTreeMonotonicConstraints*/{},
            /*monotonicConstraints*/{},
            &NPar::LocalExecutor(),
            &localData.PrevTreeLevelStats,
            /*stats3D*/nullptr,
            pairwiseStats,
            /*scoreCalcer*/nullptr);
    }

    void TScoreCalcer::DoMap(
        NPar::IUserContext* ctx,
        int hostId,
        TInput* candidateList,
        TOutput* bucketStats
    ) const {
        NPar::TCtxPtr<TTrainData> trainData(ctx, SHARED_ID_TRAIN_DATA, hostId);
        if (GetTrainData(trainData).Learn->GetObjectCount()) {
            auto calcStats3D = [&](const TCandidateInfo& candidate, TStats3D* stats3D) {
                CalcStats3D(trainData, candidate, stats3D);
            };
            MapCandidateList(calcStats3D, *candidateList, bucketStats);
        }
    }

    void TPairwiseScoreCalcer::DoMap(
        NPar::IUserContext* ctx,
        int hostId,
        TInput* candidateList,
        TOutput* bucketStats
    ) const {
        NPar::TCtxPtr<TTrainData> trainData(ctx, SHARED_ID_TRAIN_DATA, hostId);
        if (GetTrainData(trainData).Learn->GetObjectCount()) {
            auto& localData = TLocalTensorSearchData::GetRef();
            auto calcPairwiseStats = [&](const TCandidateInfo& candidate, TPairwiseStats* pairwiseStats) {
                CalcPairwiseStats(trainData, localData.FlatPairs, candidate, pairwiseStats);
            };
            MapCandidateList(calcPairwiseStats, *candidateList, bucketStats);
        }
    }

    // buckets -> workerPairwiseStats
    void TRemotePairwiseBinCalcer::DoMap(
        NPar::IUserContext* ctx,
        int hostId,
        TInput* candidate,
        TOutput* bucketStats
    ) const {
        NPar::TCtxPtr<TTrainData> trainData(ctx, SHARED_ID_TRAIN_DATA, hostId);
        if (GetTrainData(trainData).Learn->GetObjectCount()) {
            auto& localData = TLocalTensorSearchData::GetRef();
            auto calcPairwiseStats = [&](const TCandidateInfo& candidate, TPairwiseStats* pairwiseStats) {
                CalcPairwiseStats(trainData, localData.FlatPairs, candidate, pairwiseStats);
            };
            MapVector(calcPairwiseStats, candidate->Candidates, bucketStats);
        }
    }

    // workerPairwiseStats -> pairwiseStats
    void TRemotePairwiseBinCalcer::DoReduce(TVector<TOutput>* statsFromAllWorkers, TOutput* stats) const {
        // some workers may return empty result because they don't have any learn subset
        const TVector<size_t> validWorkers = GetNonEmptyElementsIndices(*statsFromAllWorkers);
        const auto validWorkersSize = validWorkers.size();
        CB_ENSURE_INTERNAL(validWorkersSize, "No workers returned bin stats");

        const int bucketCount = (*statsFromAllWorkers)[validWorkers[0]].ysize();
        stats->yresize(bucketCount);
        NPar::ParallelFor(
            0,
            bucketCount,
            [&, validWorkersSize] (int bucketIdx) {
                (*stats)[bucketIdx] = (*statsFromAllWorkers)[validWorkers[0]][bucketIdx];
                for (size_t validWorkerIdx = 1; validWorkerIdx < validWorkersSize; ++validWorkerIdx) {
                    (*stats)[bucketIdx].Add((*statsFromAllWorkers)[validWorkers[validWorkerIdx]][bucketIdx]);
                }
            });
    }

    // TStats4D -> TVector<TVector<double>> [subcandidate][bucket]
    void TRemotePairwiseScoreCalcer::DoMap(
        NPar::IUserContext* /*ctx*/,
        int /*hostId*/,
        TInput* bucketStats,
        TOutput* scores
    ) const {
        const auto& localData = TLocalTensorSearchData::GetRef();
        const int bucketCount = (*bucketStats)[0].DerSums[0].ysize();
        const auto getScores =
            [&] (const TPairwiseStats& candidatePairwiseStats, TVector<double>* candidateScores) {
                ::TPairwiseScoreCalcer scoreCalcer;
                CalculatePairwiseScore(
                    candidatePairwiseStats,
                    bucketCount,
                    localData.Params.ObliviousTreeOptions->L2Reg,
                    localData.Params.ObliviousTreeOptions->PairwiseNonDiagReg,
                    localData.Params.CatFeatureParams->OneHotMaxSize,
                    &scoreCalcer);
                *candidateScores = scoreCalcer.GetScores();
            };
        MapVector(getScores, *bucketStats, scores);
    }

    // subcandidates -> TStats4D
    void TRemoteBinCalcer::DoMap(
        NPar::IUserContext* ctx,
        int hostId,
        TInput* candidatesInfoList,
        TOutput* bucketStats
    ) const {
        NPar::TCtxPtr<TTrainData> trainData(ctx, SHARED_ID_TRAIN_DATA, hostId);
        if (GetTrainData(trainData).Learn->GetObjectCount()) {
            auto calcStats3D = [&](const TCandidateInfo& candidate, TStats3D* stats3D) {
                CalcStats3D(trainData, candidate, stats3D);
            };
            MapVector(calcStats3D, candidatesInfoList->Candidates, bucketStats);
        }
    }

    // vector<TStats4D> -> TStats4D
    void TRemoteBinCalcer::DoReduce(TVector<TOutput>* statsFromAllWorkers, TOutput* stats) const {
        // some workers may return empty result because they don't have any learn subset
        const TVector<size_t> validWorkers = GetNonEmptyElementsIndices(*statsFromAllWorkers);
        const auto validWorkersSize = validWorkers.size();
        CB_ENSURE_INTERNAL(validWorkersSize, "No workers returned bin stats");

        const int bucketCount = (*statsFromAllWorkers)[validWorkers[0]].ysize();
        stats->yresize(bucketCount);
        NPar::ParallelFor(
            0,
            bucketCount,
            [&, validWorkersSize] (int bucketIdx) {
                (*stats)[bucketIdx] = (*statsFromAllWorkers)[validWorkers[0]][bucketIdx];
                for (size_t validWorkerIdx = 1; validWorkerIdx < validWorkersSize; ++validWorkerIdx) {
                    (*stats)[bucketIdx].Add((*statsFromAllWorkers)[validWorkers[validWorkerIdx]][bucketIdx]);
                }
            });
    }

    // TStats4D -> TVector<TVector<double>> [subcandidate][bucket]
    void TRemoteScoreCalcer::DoMap(
        NPar::IUserContext* /*ctx*/,
        int /*hostId*/,
        TInput* bucketStats,
        TOutput* scores
    ) const {
        const auto& localData = TLocalTensorSearchData::GetRef();
        const auto getScores =
            [&] (const TStats3D& candidateStats3D, TVector<double>* candidateScores) {
                *candidateScores = GetScores(candidateStats3D,
                                             localData.Depth,
                                             localData.SumAllWeights,
                                             localData.AllDocCount,
                                             localData.Params);
            };
        MapVector(getScores, *bucketStats, scores);
    }

    void TLeafIndexSetter::DoMap(
        NPar::IUserContext* ctx,
        int hostId,
        TInput* bestSplit,
        TOutput* /*unused*/
    ) const {
        auto& localData = TLocalTensorSearchData::GetRef();
        NPar::TCtxPtr<TTrainData> trainData(ctx, SHARED_ID_TRAIN_DATA, hostId);

        TArrayRef<TIndexType> learnIndices(
            localData.Indices.data(),
            GetTrainData(trainData).Learn->GetObjectCount());

        if (!learnIndices.empty()) {
            SetPermutedIndices(
                *bestSplit,
                GetTrainData(trainData),
                localData.Depth + 1,
                localData.Progress->AveragingFold,
                learnIndices,
                &NPar::LocalExecutor());
            if (IsSamplingPerTree(localData.Params.ObliviousTreeOptions)) {
                localData.SampledDocs.UpdateIndices(learnIndices, &NPar::LocalExecutor());
                if (localData.UseTreeLevelCaching) {
                    localData.SmallestSplitSideDocs.SelectSmallestSplitSide(
                        localData.Depth + 1,
                        localData.SampledDocs,
                        &NPar::LocalExecutor());
                }
            }
        }
    }

    void TEmptyLeafFinder::DoMap(
        NPar::IUserContext* ctx,
        int hostId,
        TInput* /*unused*/,
        TOutput* isLeafEmpty
    ) const {
        auto& localData = TLocalTensorSearchData::GetRef();
        NPar::TCtxPtr<TTrainData> trainData(ctx, SHARED_ID_TRAIN_DATA, hostId);

        TArrayRef<TIndexType> learnIndices(
            localData.Indices.data(),
            GetTrainData(trainData).Learn->GetObjectCount());

        if (!learnIndices.empty()) {
            *isLeafEmpty = GetIsLeafEmpty(localData.Depth + 1, learnIndices, &NPar::LocalExecutor());
        }
        ++localData.Depth; // tree level completed
    }

    void TBucketSimpleUpdater::DoMap(
        NPar::IUserContext* /*ctx*/,
        int /*hostId*/,
        TInput* /*unused*/,
        TOutput* sums
    ) const {
        auto& localData = TLocalTensorSearchData::GetRef();
        if (localData.Progress->AveragingFold.GetLearnSampleCount()) {
            const int approxDimension = localData.Progress->ApproxDimension;
            Y_ASSERT(approxDimension == 1);
            const auto error = BuildError(localData.Params, /*custom objective*/Nothing());
            const auto estimationMethod = localData.Params.ObliviousTreeOptions->LeavesEstimationMethod;
            const int scratchSize =
                error->GetErrorType() == EErrorType::PerObjectError ?
                    APPROX_BLOCK_SIZE * CB_THREAD_LIMIT :
                    // plain boosting ==> not approx on full history
                    localData.Progress->AveragingFold.BodyTailArr[0].BodyFinish;
            TVector<TDers> weightedDers;
            weightedDers.yresize(scratchSize);

            CalcLeafDersSimple(
                localData.Indices,
                localData.Progress->AveragingFold,
                localData.Progress->AveragingFold.BodyTailArr[0],
                localData.Progress->AveragingFold.BodyTailArr[0].Approx[0],
                localData.ApproxDeltas[0],
                *error,
                localData.Progress->AveragingFold.BodyTailArr[0].BodyFinish,
                localData.Progress->AveragingFold.BodyTailArr[0].BodyQueryFinish,
                localData.GradientIteration == 0,
                estimationMethod,
                localData.Params,
                localData.Progress->Rand.GenRand(),
                &NPar::LocalExecutor(),
                &localData.Buckets,
                &localData.PairwiseBuckets,
                &weightedDers);
            *sums = std::make_pair(localData.Buckets, localData.PairwiseBuckets);
        }
    }

    void TCalcApproxStarter::DoMap(
        NPar::IUserContext* ctx,
        int hostId,
        TInput* splitTree,
        TOutput* /*unused*/
    ) const {
        auto& localData = TLocalTensorSearchData::GetRef();
        const auto& error = BuildError(localData.Params, /*custom objective*/Nothing());
        NPar::TCtxPtr<TTrainData> trainData(ctx, SHARED_ID_TRAIN_DATA, hostId);
        localData.Indices = BuildIndices(
            localData.Progress->AveragingFold,
            *splitTree,
            GetTrainData(trainData),
            EBuildIndicesDataParts::All,
            &NPar::LocalExecutor());
        if (localData.Progress->AveragingFold.GetLearnSampleCount()) {
            const int approxDimension = localData.Progress->ApproxDimension;
            if (localData.ApproxDeltas.empty()) {
                localData.ApproxDeltas.resize(approxDimension); // 1D or nD
                for (auto& dimensionDelta : localData.ApproxDeltas) {
                    dimensionDelta.yresize(localData.Progress->AveragingFold.BodyTailArr[0].TailFinish);
                }
            }
            for (auto& dimensionDelta : localData.ApproxDeltas) {
                Fill(dimensionDelta.begin(), dimensionDelta.end(), GetNeutralApprox(localData.StoreExpApprox));
            }
            localData.Buckets.resize(GetLeafCount(*splitTree));
            Fill(localData.Buckets.begin(), localData.Buckets.end(), TSum());
            localData.MultiBuckets.resize(GetLeafCount(*splitTree));
            Fill(
                localData.MultiBuckets.begin(),
                localData.MultiBuckets.end(),
                TSumMulti(approxDimension, error->GetHessianType()));
            localData.PairwiseBuckets.SetSizes(GetLeafCount(*splitTree), GetLeafCount(*splitTree));
            localData.PairwiseBuckets.FillZero();
        }
        localData.GradientIteration = 0;
    }

    void TDeltaSimpleUpdater::DoMap(
        NPar::IUserContext* /*unused*/,
        int /*unused*/,
        TInput* leafValues,
        TOutput* /*unused*/
    ) const {
        auto& localData = TLocalTensorSearchData::GetRef();
        if (localData.Progress->AveragingFold.GetLearnSampleCount()) {
            UpdateApproxDeltas(
                localData.StoreExpApprox,
                localData.Indices,
                localData.Progress->AveragingFold.BodyTailArr[0].TailFinish,
                &NPar::LocalExecutor(),
                &(*leafValues)[0],
                &localData.ApproxDeltas[0]);
        }
        ++localData.GradientIteration; // gradient iteration completed
    }

    void TApproxUpdater::DoMap(
        NPar::IUserContext* ctx,
        int hostId,
        TInput* averageLeafValues,
        TOutput* /*unused*/
    ) const {
        auto& localData = TLocalTensorSearchData::GetRef();

        NPar::TCtxPtr<TTrainData> trainData(ctx, SHARED_ID_TRAIN_DATA, hostId);

        TConstArrayRef<NCB::TTrainingDataProviderPtr> testData = GetTrainData(trainData).Test;

        const TVector<size_t> testOffsets = CalcTestOffsets(
            GetTrainData(trainData).Learn->GetObjectCount(),
            testData);

        NPar::LocalExecutor().ExecRange(
            [&](int setIdx) {
                if (setIdx == 0) { // learn data set
                    if (localData.Progress->AveragingFold.GetLearnSampleCount()) {
                        if (localData.StoreExpApprox) {
                            UpdateBodyTailApprox</*StoreExpApprox*/true>(
                                { localData.ApproxDeltas },
                                localData.Params.BoostingOptions->LearningRate,
                                &NPar::LocalExecutor(),
                                &localData.Progress->AveragingFold);
                        } else {
                            UpdateBodyTailApprox</*StoreExpApprox*/false>(
                                { localData.ApproxDeltas },
                                localData.Params.BoostingOptions->LearningRate,
                                &NPar::LocalExecutor(),
                                &localData.Progress->AveragingFold);
                        }
                        TConstArrayRef<ui32> learnPermutationRef(localData.Progress->AveragingFold.GetLearnPermutationArray());
                        TConstArrayRef<TIndexType> indicesRef(localData.Indices);
                        const auto updateAvrgApprox =
                            [=](TConstArrayRef<double> delta, TArrayRef<double> approx, size_t idx) {
                                approx[learnPermutationRef[idx]] += delta[indicesRef[idx]];
                            };
                        UpdateApprox(
                            updateAvrgApprox,
                            *averageLeafValues,
                            &localData.Progress->AvrgApprox,
                            &NPar::LocalExecutor());
                    }
                } else { // test data set
                    const int testIdx = setIdx - 1;
                    const size_t testSampleCount = testData[testIdx]->GetObjectCount();
                    if (testSampleCount) {
                        TConstArrayRef<TIndexType> indicesRef(
                            localData.Indices.data() + testOffsets[testIdx],
                            testSampleCount);
                        const auto updateTestApprox = [=](
                            TConstArrayRef<double> delta,
                            TArrayRef<double> approx,
                            size_t idx
                        ) {
                            approx[idx] += delta[indicesRef[idx]];
                        };
                        Y_ASSERT(localData.Progress->TestApprox[testIdx][0].size() == testSampleCount);
                        UpdateApprox(
                            updateTestApprox,
                            *averageLeafValues,
                            &localData.Progress->TestApprox[testIdx],
                            &NPar::LocalExecutor());
                    }
                }
            },
            0,
            1 + SafeIntegerCast<int>(testData.size()),
            NPar::TLocalExecutor::WAIT_COMPLETE);
    }

    void TDerivativeSetter::DoMap(
        NPar::IUserContext* /*ctx*/,
        int /*hostId*/,
        TInput* /*unused*/,
        TOutput* /*unused*/
    ) const {
        auto& localData = TLocalTensorSearchData::GetRef();
        if (localData.Progress->AveragingFold.GetLearnSampleCount()) {
            Y_ASSERT(localData.Progress->AveragingFold.BodyTailArr.ysize() == 1);
            const auto error = BuildError(localData.Params, /*custom objective*/Nothing());
            CalcWeightedDerivatives(
                *error,
                /*bodyTailIdx*/0,
                localData.Params,
                localData.Progress->Rand.GenRand(),
                &localData.Progress->AveragingFold,
                &NPar::LocalExecutor());
        }
    }

    void TBestApproxSetter::DoMap(
        NPar::IUserContext* /*ctx*/,
        int /*hostId*/,
        TInput* /*unused*/,
        TOutput* /*unused*/
    ) const {
        auto& localData = TLocalTensorSearchData::GetRef();
        localData.Progress->BestTestApprox = localData.Progress->TestApprox.back();
    }

    void TApproxGetter::DoMap(
        NPar::IUserContext* /*ctx*/,
        int /*hostId*/,
        TInput* approxGetterParams,
        TOutput* approxesResult
    ) const {
        auto& localData = TLocalTensorSearchData::GetRef();
        if (approxGetterParams->ReturnLearnApprox) {
            approxesResult->LearnApprox = localData.Progress->AvrgApprox;
        }
        if (approxGetterParams->ReturnTestApprox) {
            approxesResult->TestApprox = localData.Progress->TestApprox;
        }
        if (approxGetterParams->ReturnBestTestApprox) {
            approxesResult->BestTestApprox = localData.Progress->BestTestApprox;
        }
    }

    void TBucketMultiUpdater::DoMap(
        NPar::IUserContext* /*ctx*/,
        int /*hostId*/,
        TInput* /*unused*/,
        TOutput* sums
    ) const {
        auto& localData = TLocalTensorSearchData::GetRef();
        if (localData.Progress->AveragingFold.GetLearnSampleCount()) {
            const auto error = BuildError(localData.Params, /*custom objective*/Nothing());
            const auto estimationMethod = localData.Params.ObliviousTreeOptions->LeavesEstimationMethod;

            CalcLeafDersMulti(
                localData.Indices,
                To2DConstArrayRef<float>(localData.Progress->AveragingFold.LearnTarget),
                localData.Progress->AveragingFold.GetLearnWeights(),
                localData.Progress->AveragingFold.BodyTailArr[0].Approx,
                localData.ApproxDeltas,
                *error,
                localData.Progress->AveragingFold.BodyTailArr[0].BodyFinish,
                /*isUpdateWeight*/localData.GradientIteration == 0,
                estimationMethod,
                &NPar::LocalExecutor(),
                &localData.MultiBuckets);
            *sums = std::make_pair(localData.MultiBuckets, TUnusedInitializedParam());
        }
    }

    void TDeltaMultiUpdater::DoMap(
        NPar::IUserContext* /*ctx*/,
        int /*hostId*/,
        TInput* leafValues,
        TOutput* /*unused*/
    ) const {
        auto& localData = TLocalTensorSearchData::GetRef();
        if (localData.Progress->AveragingFold.GetLearnSampleCount()) {
            UpdateApproxDeltasMulti(
                localData.Indices,
                localData.Progress->AveragingFold.BodyTailArr[0].BodyFinish,
                MakeConstArrayRef(*leafValues),
                &localData.ApproxDeltas,
                &NPar::LocalExecutor());
        }
        ++localData.GradientIteration; // gradient iteration completed
    }

    void TErrorCalcer::DoMap(
        NPar::IUserContext* ctx,
        int hostId,
        TInput* errorCalcerParams,
        TOutput* additiveStats
    ) const {
        const auto& localData = TLocalTensorSearchData::GetRef();
        NPar::TCtxPtr<TTrainData> trainData(ctx, SHARED_ID_TRAIN_DATA, hostId);

        const auto& trainingDataProviders = GetTrainData(trainData);
        const NCB::TTrainingDataProvider& learnData = *(trainingDataProviders.Learn);
        const TVector<NCB::TTrainingDataProviderPtr>& testData = trainingDataProviders.Test;

        if (errorCalcerParams->CalcOnlyBacktrackingObjective) {
            TVector<THolder<IMetric>> metrics;
            bool haveBacktrackingObjective;
            double minimizationSign;
            CreateBacktrackingObjective(
                localData.Params.MetricOptions->ObjectiveMetric,
                /*customMetric*/ Nothing(),
                localData.Params.ObliviousTreeOptions,
                localData.Progress->ApproxDimension,
                &haveBacktrackingObjective,
                &minimizationSign,
                &metrics);

            Y_ASSERT(metrics.size() == 1);
            additiveStats->resize(1);

            const IMetric* metric = metrics[0].Get();

            TVector<TMetricHolder> calculatedErrors;
            if (learnData.GetObjectCount()) {
                calculatedErrors = EvalErrorsWithCaching(
                    localData.Progress->AveragingFold.BodyTailArr[0].Approx,
                    localData.ApproxDeltas,
                    localData.StoreExpApprox,
                    learnData.TargetData->GetTarget().GetOrElse(TConstArrayRef<TConstArrayRef<float>>()),
                    GetWeights(*learnData.TargetData),
                    learnData.TargetData->GetGroupInfo().GetOrElse(TConstArrayRef<TQueryInfo>()),
                    MakeArrayRef(&metric, 1),
                    &NPar::LocalExecutor());
            } else {
                const TVector<TVector<double>> emptyApprox(localData.Progress->ApproxDimension);
                const TVector<TConstArrayRef<float>> emptyTargetData(localData.Progress->ApproxDimension);
                calculatedErrors = EvalErrorsWithCaching(
                    /*approx*/ emptyApprox,
                    /*approxDeltas*/ emptyApprox,
                    localData.StoreExpApprox,
                    emptyTargetData,
                    /*weight*/ TConstArrayRef<float>(),
                    /*groupInfo*/ TConstArrayRef<TQueryInfo>(),
                    MakeArrayRef(&metric, 1),
                    &NPar::LocalExecutor());
            }
            (*additiveStats)[0][metrics[0]->GetDescription()] = calculatedErrors[0];
        } else {
            additiveStats->resize(1 + testData.size());

            const auto metrics = CreateMetrics(
                localData.Params.MetricOptions,
                /*evalMetricDescriptor*/Nothing(),
                localData.Progress->ApproxDimension,
                learnData.MetaInfo.HasWeights);

            auto onLearn = [&] (TConstArrayRef<const IMetric*> trainMetrics) {
                auto calculatedErrors = EvalErrorsWithCaching(
                    localData.Progress->AvrgApprox,
                    /*approxDelta*/TVector<TVector<double>>{},
                    /*isExpApprox*/false,
                    learnData.TargetData->GetTarget().GetOrElse(TConstArrayRef<TConstArrayRef<float>>()),
                    GetWeights(*learnData.TargetData),
                    learnData.TargetData->GetGroupInfo().GetOrElse(TConstArrayRef<TQueryInfo>()),
                    trainMetrics,
                    &NPar::LocalExecutor());

                for (auto i : xrange(trainMetrics.size())) {
                    const TString metricDescription = trainMetrics[i]->GetDescription();
                    (*additiveStats)[0][metricDescription] = calculatedErrors[i];
                }
            };
            auto onTest = [&] (
                size_t testIdx,
                TConstArrayRef<const IMetric*> testMetrics,
                TMaybe<int> /*filteredTrackerIdx*/
            ) {
                const auto &targetData = testData[testIdx]->TargetData;

                auto maybeTarget = targetData->GetTarget();
                auto weights = GetWeights(*targetData);
                auto groupInfo = targetData->GetGroupInfo().GetOrElse(TConstArrayRef<TQueryInfo>());

                auto calculatedErrors = EvalErrorsWithCaching(
                    localData.Progress->TestApprox[testIdx],
                    /*approxDelta*/TVector<TVector<double>>{},
                    /*isExpApprox*/false,
                    maybeTarget.GetOrElse(TConstArrayRef<TConstArrayRef<float>>()),
                    weights,
                    groupInfo,
                    testMetrics,
                    &NPar::LocalExecutor());

                for (auto i : xrange(testMetrics.size())) {
                    const TString metricDescription = testMetrics[i]->GetDescription();
                    (*additiveStats)[1 + testIdx][metricDescription] = calculatedErrors[i];
                }
            };

            IterateOverMetrics(
                trainingDataProviders,
                metrics,
                errorCalcerParams->CalcAllMetrics,
                errorCalcerParams->CalcErrorTrackerMetric,
                /*calcAdditiveMetrics*/ true,
                /*calcNonAdditiveMetrics*/ false,
                onLearn,
                onTest
            );
        }
    }

    void TLeafWeightsGetter::DoMap(
        NPar::IUserContext* ctx,
        int hostId,
        TInput* /*unused*/,
        TOutput* leafWeights
    ) const {
        auto& localData = TLocalTensorSearchData::GetRef();
        NPar::TCtxPtr<TTrainData> trainData(ctx, SHARED_ID_TRAIN_DATA, hostId);
        if (localData.Progress->AveragingFold.GetLearnSampleCount()) {
            const size_t leafCount = localData.Buckets.size();
            *leafWeights = SumLeafWeights(
                leafCount,
                localData.Indices,
                localData.Progress->AveragingFold.GetLearnPermutationArray(),
                GetWeights(*GetTrainData(trainData).Learn->TargetData),
                &NPar::LocalExecutor());
        }
    }

    void TLeafWeightsGetter::DoReduce(TVector<TOutput>* inLeafWeightsFromWorkers, TOutput* outTotalLeafWeights) const {
        const auto& leafWeightsFromWorkers = *inLeafWeightsFromWorkers;
        // some workers may return empty result because they don't have any learn subset
        const TVector<size_t> validWorkers = GetNonEmptyElementsIndices(leafWeightsFromWorkers);
        const auto validWorkersSize = validWorkers.size();
        CB_ENSURE_INTERNAL(validWorkersSize, "No workers returned leaf weight stats");

        TOutput totalLeafWeights = leafWeightsFromWorkers[validWorkers[0]];
        for (auto i : xrange<size_t>(1, validWorkersSize)) {
            AddElementwise(leafWeightsFromWorkers[validWorkers[i]], &totalLeafWeights);
        }
        *outTotalLeafWeights = std::move(totalLeafWeights);
    }

    /*
     * Prepares data structures for exact quantile approx calculation.
     * Returns Min and Max values of target for binary search of quantile.
     */
    void TQuantileExactApproxStarter::DoMap(
        NPar::IUserContext* /*ctx*/,
        int /*hostId*/,
        TInput* /*input*/,
        TOutput* outMinMaxDiffs
    ) const {
        auto& localData = TLocalTensorSearchData::GetRef();

        if (localData.Progress->AveragingFold.GetLearnSampleCount()) {
            const int leafCount = localData.Buckets.size();
            const int approxDimension = localData.Progress->AvrgApprox.size();
            const auto& target = localData.Progress->AveragingFold.LearnTarget;
            const auto& weights = localData.Progress->AveragingFold.GetLearnWeights();
            const auto& avrgFoldIndexing = localData.Progress->AveragingFold.LearnPermutation.Get()->GetObjectsIndexing();

            localData.ExactDiff.yresize(approxDimension);
            localData.SplitBounds.yresize(approxDimension);
            localData.LastPivot.yresize(approxDimension);
            localData.LastPartitionPoint.yresize(approxDimension);
            localData.CollectedLeftSumWeight.yresize(approxDimension);
            localData.LastSplitLeftSumWeight.yresize(approxDimension);

            TOutput minMaxDiffs(approxDimension);
            for (auto dimension : xrange(approxDimension)) {
                localData.ExactDiff[dimension].yresize(leafCount);
                for (auto leaf : xrange(leafCount)) {
                    localData.ExactDiff[dimension][leaf].clear();
                }
                minMaxDiffs[dimension].assign(leafCount, {std::numeric_limits<double>::max(), std::numeric_limits<double>::lowest()});
                avrgFoldIndexing.ForEach([&](int idx, int srcIdx) {
                    const double diff = target[dimension][srcIdx] - localData.Progress->AvrgApprox[dimension][srcIdx];
                    const double weight = weights.empty() ? 1.0 : weights[srcIdx];
                    const TIndexType leaf = localData.Indices[idx];
                    localData.ExactDiff[dimension][leaf].emplace_back(diff, weight);
                    minMaxDiffs[dimension][leaf].Min = Min(minMaxDiffs[dimension][leaf].Min, diff);
                    minMaxDiffs[dimension][leaf].Max = Max(minMaxDiffs[dimension][leaf].Max, diff);
                });

                localData.SplitBounds[dimension].yresize(leafCount);
                localData.LastPartitionPoint[dimension].yresize(leafCount);
                localData.LastPivot[dimension].yresize(leafCount);
                localData.CollectedLeftSumWeight[dimension].yresize(leafCount);
                localData.LastSplitLeftSumWeight[dimension].yresize(leafCount);
                for (auto leaf : xrange(leafCount)) {
                    const int objectCount = localData.ExactDiff[dimension][leaf].size();
                    localData.SplitBounds[dimension][leaf] = {0, objectCount};
                    localData.LastPivot[dimension][leaf] = std::numeric_limits<double>::max();
                    localData.LastPartitionPoint[dimension][leaf] = objectCount;
                    localData.CollectedLeftSumWeight[dimension][leaf]= 0;
                    localData.LastSplitLeftSumWeight[dimension][leaf] = 0;
                }
            }

            *outMinMaxDiffs = std::move(minMaxDiffs);
        }
    }

    void TQuantileExactApproxStarter::DoReduce(TVector<TOutput>* inMinMaxDiffsFromWorkers,
                                               TOutput* reducedMinMaxDiffs) const {
        const auto& minMaxDiffsFromWorkers = *inMinMaxDiffsFromWorkers;
        // some workers may return empty result because they don't have any learn subset
        const TVector<size_t> validWorkers = GetNonEmptyElementsIndices(minMaxDiffsFromWorkers);
        const auto validWorkersSize = validWorkers.size();
        CB_ENSURE_INTERNAL(validWorkersSize, "No workers returned bin stats");

        TOutput result = minMaxDiffsFromWorkers[validWorkers[0]];
        for (auto worker : xrange<size_t>(1, validWorkersSize)) {
            const auto& resultFromWorker = minMaxDiffsFromWorkers[validWorkers[worker]];
            for (auto dimension : xrange(result.size())) {
                for (auto leaf : xrange(result[dimension].size())) {
                    result[dimension][leaf].Min = Min(result[dimension][leaf].Min, resultFromWorker[dimension][leaf].Min);
                    result[dimension][leaf].Max = Max(result[dimension][leaf].Max, resultFromWorker[dimension][leaf].Max);
                }
            }
        }
        *reducedMinMaxDiffs = std::move(result);
    }

    /*
     * Split exact diff arrays by pivots
     * Returns weights of left parts after splitting
     */
    void TQuantileArraySplitter::DoMap(
        NPar::IUserContext* /*ctx*/,
        int /*hostId*/,
        TInput* pivots,
        TOutput* outLeftRightWeights
    ) const {
        auto& localData = TLocalTensorSearchData::GetRef();
        if (localData.Progress->AveragingFold.GetLearnSampleCount()) {
            const auto leafCount = localData.Buckets.size();
            const auto& pivotsRef = *pivots;
            const auto approxDimension = pivotsRef.size();
            TOutput leftWeights(approxDimension);
            for (auto dimension : xrange(approxDimension)) {
                leftWeights[dimension].assign(leafCount, 0);
                for (auto leaf : xrange(leafCount)) {
                    auto& lastPivot = localData.LastPivot[dimension][leaf];
                    auto& lastPartitionPoint = localData.LastPartitionPoint[dimension][leaf];
                    auto& splitBounds = localData.SplitBounds[dimension][leaf];
                    auto& exactDiff = localData.ExactDiff[dimension][leaf];
                    auto& collectedLeftSumWeight = localData.CollectedLeftSumWeight[dimension][leaf];
                    auto& lastLeftSumWeight = localData.LastSplitLeftSumWeight[dimension][leaf];

                    const double curPivot = pivotsRef[dimension][leaf];
                    if (curPivot < lastPivot) {
                        splitBounds.Max = lastPartitionPoint;
                    } else {
                        splitBounds.Min = lastPartitionPoint;
                        collectedLeftSumWeight += lastLeftSumWeight;
                    }
                    lastPivot = curPivot;
                    const auto partitionBegin = exactDiff.begin() + splitBounds.Min;
                    const auto partitionEnd = exactDiff.begin() + splitBounds.Max;
                    const auto partitionIt = std::partition(
                        partitionBegin,
                        partitionEnd,
                        [curPivot](const std::pair<double, double>& diff) {
                            return diff.first <= curPivot;
                        });
                    int curPartitionPoint = partitionIt - exactDiff.begin();
                    lastLeftSumWeight = Accumulate(partitionBegin, partitionIt, 0.0, [](double totalWeight, const std::pair<double, double>& diff) {
                        return totalWeight + diff.second;
                    });
                    leftWeights[dimension][leaf] = collectedLeftSumWeight + lastLeftSumWeight;
                    lastPartitionPoint = curPartitionPoint;
                }
            }
            *outLeftRightWeights = std::move(leftWeights);
        }
    }

    void TQuantileArraySplitter::DoReduce(
        TVector<TOutput>* inLeftWeightsFromWorkers,
        TOutput* outTotalLeftWeights
    ) const {
        const auto& leftWeightsFromWorkers = *inLeftWeightsFromWorkers;
        // some workers may return empty result because they don't have any learn subset
        const TVector<size_t> validWorkers = GetNonEmptyElementsIndices(leftWeightsFromWorkers);
        const auto validWorkersSize = validWorkers.size();
        CB_ENSURE_INTERNAL(validWorkersSize, "No workers returned left weights stats");

        TOutput totalLeftWeights = leftWeightsFromWorkers[validWorkers[0]];
        for (auto worker : xrange<size_t>(1, validWorkersSize)) {
            for (auto dimension : xrange(totalLeftWeights.size())) {
                AddElementwise(leftWeightsFromWorkers[validWorkers[worker]][dimension], &totalLeftWeights[dimension]);
            }
        }
        *outTotalLeftWeights = std::move(totalLeftWeights);
    }

    /*
     * Calculates weight of elements equal to pivots
     * Needed to adjusting quantile value according to 'delta' parameter of Quantile metric
     */
    void TQuantileEqualWeightsCalcer::DoMap(
        NPar::IUserContext* /*ctx*/,
        int /*hostId*/,
        TInput* inPivots,
        TOutput* outEqualSumWeights
    ) const {
        auto& localData = TLocalTensorSearchData::GetRef();
        if (localData.Progress->AveragingFold.GetLearnSampleCount()) {
            const auto& pivots = *inPivots;
            const auto approxDimension = pivots.size();
            const auto leafCount = pivots[0].size();
            TOutput equalSumWeights(approxDimension, TVector<double>(leafCount, 0.0));
            for (auto dimension : xrange(approxDimension)) {
                for (auto leaf : xrange(leafCount)) {
                    const auto pivot = pivots[dimension][leaf];
                    const auto& exactDiff = localData.ExactDiff[dimension][leaf];
                    double sumWeights = 0;
                    for (const auto& [diff, weight] : exactDiff) {
                        if (diff == pivot) {
                            sumWeights += weight;
                        }
                    }
                    equalSumWeights[dimension][leaf] = sumWeights;
                }
            }
            *outEqualSumWeights = std::move(equalSumWeights);
        }
    }

    void TQuantileEqualWeightsCalcer::DoReduce(TVector<TOutput>* inEqualSumWeightsFromWorkers, TOutput* outTotalEqualSumWeights) const {
        const auto& equalSumWeightsFromWorkers = *inEqualSumWeightsFromWorkers;
        // some workers may return empty result because they don't have any learn subset
        const TVector<size_t> validWorkers = GetNonEmptyElementsIndices(equalSumWeightsFromWorkers);
        const auto validWorkersSize = validWorkers.size();
        CB_ENSURE_INTERNAL(validWorkersSize, "No workers returned equal sum weights stats");

        TOutput totalEqualSumWeights = equalSumWeightsFromWorkers[validWorkers[0]];
        for (auto worker : xrange<size_t>(1, validWorkersSize)) {
            for (auto dimension : xrange(totalEqualSumWeights.size())) {
                AddElementwise(equalSumWeightsFromWorkers[validWorkers[worker]][dimension], &totalEqualSumWeights[dimension]);
            }
        }
        *outTotalEqualSumWeights = std::move(totalEqualSumWeights);
    }

    void TArmijoStartPointBackupper::DoMap(NPar::IUserContext* /*ctx*/, int /*hostId*/, TInput* isRestore, TOutput* /*unused*/) const {
        auto& localData = TLocalTensorSearchData::GetRef();
        if (localData.Progress->AveragingFold.GetLearnSampleCount()) {
            if (*isRestore) {
                CB_ENSURE_INTERNAL(!localData.BacktrackingStart.empty(), "Need saved backtracking start point to restore from");
                localData.ApproxDeltas = localData.BacktrackingStart;
            } else {
                localData.BacktrackingStart = localData.ApproxDeltas;
            }
        }
    };

} // NCatboostDistributed

using namespace NCatboostDistributed;

REGISTER_SAVELOAD_NM_CLASS(0xd66d481, NCatboostDistributed, TTrainData);
REGISTER_SAVELOAD_NM_CLASS(0xd66d482, NCatboostDistributed, TPlainFoldBuilder);
REGISTER_SAVELOAD_NM_CLASS(0xd66d483, NCatboostDistributed, TTensorSearchStarter);
REGISTER_SAVELOAD_NM_CLASS(0xd66d484, NCatboostDistributed, TBootstrapMaker);
REGISTER_SAVELOAD_NM_CLASS(0xd66d584, NCatboostDistributed, TDerivativesStDevFromZeroCalcer);
REGISTER_SAVELOAD_NM_CLASS(0xd66d485, NCatboostDistributed, TScoreCalcer);
REGISTER_SAVELOAD_NM_CLASS(0xd66d585, NCatboostDistributed, TRemoteBinCalcer);
REGISTER_SAVELOAD_NM_CLASS(0xd66d685, NCatboostDistributed, TRemoteScoreCalcer);
REGISTER_SAVELOAD_NM_CLASS(0xd66d486, NCatboostDistributed, TLeafIndexSetter);
REGISTER_SAVELOAD_NM_CLASS(0xd66d487, NCatboostDistributed, TEmptyLeafFinder);
REGISTER_SAVELOAD_NM_CLASS(0xd66d488, NCatboostDistributed, TCalcApproxStarter);
REGISTER_SAVELOAD_NM_CLASS(0xd66d489, NCatboostDistributed, TDeltaSimpleUpdater);
REGISTER_SAVELOAD_NM_CLASS(0xd66d48a, NCatboostDistributed, TApproxUpdater);
REGISTER_SAVELOAD_NM_CLASS(0xd66d50f, NCatboostDistributed, TBucketSimpleUpdater);
REGISTER_SAVELOAD_NM_CLASS(0xd66d4af, NCatboostDistributed, TDerivativeSetter);
REGISTER_SAVELOAD_NM_CLASS(0xd66d4b2, NCatboostDistributed, TDeltaMultiUpdater);
REGISTER_SAVELOAD_NM_CLASS(0xd66d4c1, NCatboostDistributed, TBucketMultiUpdater);

REGISTER_SAVELOAD_NM_CLASS(0xd66d4c2, NCatboostDistributed, TBestApproxSetter);
REGISTER_SAVELOAD_NM_CLASS(0xd66d4c3, NCatboostDistributed, TApproxGetter);

REGISTER_SAVELOAD_NM_CLASS(0xd66d4d1, NCatboostDistributed, TPairwiseScoreCalcer);
REGISTER_SAVELOAD_NM_CLASS(0xd66d4d2, NCatboostDistributed, TRemotePairwiseBinCalcer);
REGISTER_SAVELOAD_NM_CLASS(0xd66d4d3, NCatboostDistributed, TRemotePairwiseScoreCalcer);
REGISTER_SAVELOAD_NM_CLASS(0xd66d4d4, NCatboostDistributed, TErrorCalcer);

REGISTER_SAVELOAD_NM_CLASS(0xd66d4d6, NCatboostDistributed, TApproxReconstructor);
REGISTER_SAVELOAD_NM_CLASS(0xd66d4e0, NCatboostDistributed, TLeafWeightsGetter);

REGISTER_SAVELOAD_NM_CLASS(0xd66d4e1, NCatboostDistributed, TDatasetsLoader);
REGISTER_SAVELOAD_NM_CLASS(0xd66d4e3, NCatboostDistributed, TQuantileExactApproxStarter);
REGISTER_SAVELOAD_NM_CLASS(0xd66d4e4, NCatboostDistributed, TQuantileArraySplitter);
REGISTER_SAVELOAD_NM_CLASS(0xd66d4e5, NCatboostDistributed, TQuantileEqualWeightsCalcer);

REGISTER_SAVELOAD_NM_CLASS(0xd66d4e6, NCatboostDistributed, TArmijoStartPointBackupper);
