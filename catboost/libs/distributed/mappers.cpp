#include "mappers.h"

#include <catboost/libs/algo/approx_calcer.h>
#include <catboost/libs/algo/approx_calcer_multi.h>
#include <catboost/libs/algo/data.h>
#include <catboost/libs/algo/error_functions.h>
#include <catboost/libs/algo/score_calcer.h>
#include <catboost/libs/algo/learn_context.h>
#include <catboost/libs/algo/online_ctr.h>
#include <catboost/libs/algo/preprocess.h>
#include <catboost/libs/data_new/load_data.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/query_info_helper.h>
#include <catboost/libs/helpers/vector_helpers.h>
#include <catboost/libs/index_range/index_range.h>

#include <util/generic/ymath.h>

#include <utility>


namespace NCatboostDistributed {

    static NCB::TTrainingForCPUDataProviderPtr GetTrainData(NPar::TCtxPtr<TTrainData> trainData) {
        if (trainData != nullptr) {
            return trainData->TrainData;
        } else {
            return TLocalTensorSearchData::GetRef().TrainData;
        }
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

    void TDatasetLoader::DoMap(
        NPar::IUserContext* ctx,
        int hostId,
        TInput* params,
        TOutput* /*unused*/
    ) const {
        auto& localData = TLocalTensorSearchData::GetRef();
        if (localData.Rand == nullptr) {
            localData.Rand = new TRestorableFastRng64(params->Data.RandomSeed + hostId);
        }

        const int workerCount = ctx->GetHostIdCount();
        CATBOOST_DEBUG_LOG << "Worker count " << workerCount << Endl;
        const auto workerParts = WorkaroundSplit(params->Data.ObjectsGrouping, workerCount);
        const ui32 loadStart = workerParts[hostId].Begin;
        const ui32 loadEnd = workerParts[hostId].End;

        const auto poolLoadOptions = params->Data.PoolLoadOptions;
        TProfileInfo profile;
        CATBOOST_DEBUG_LOG << "Load quantized pool section for worker " << hostId << "..." << Endl;
        auto pools = NCB::ReadTrainDatasets(
            poolLoadOptions,
            params->Data.ObjectsOrder,
            /*readTest*/false,
            NCB::TDatasetSubset::MakeRange(loadStart, loadEnd),
            &localData.ClassNamesFromDataset,
            &NPar::LocalExecutor(),
            &profile
        );

        NCatboostOptions::TCatBoostOptions catBoostOptions(ETaskType::CPU);
        catBoostOptions.Load(GetJson(params->Data.TrainOptions));
        TLabelConverter labelConverter;
        auto quantizedFeaturesInfo = MakeIntrusive<NCB::TQuantizedFeaturesInfo>(
            params->Data.FeaturesLayout,
            catBoostOptions.DataProcessingOptions.Get().IgnoredFeatures.Get(),
            catBoostOptions.DataProcessingOptions->FloatFeaturesBinarization.Get(),
            catBoostOptions.DataProcessingOptions->PerFloatFeatureBinarization.Get(),
            /*allowNansInTestOnly*/true,
            /*allowWriteFiles*/false
        );

        CATBOOST_DEBUG_LOG << "Create train data for worker " << hostId << "..." << Endl;
        localData.TrainData = MakeIntrusive<NCB::TTrainingForCPUDataProvider>(
            GetTrainingData(
                std::move(pools),
                /*borders*/ Nothing(), // borders are already loaded to quantizedFeaturesInfo
                /*ensureConsecutiveLearnFeaturesDataForCpu*/ true,
                /*allowWriteFiles*/ false,
                quantizedFeaturesInfo,
                &catBoostOptions,
                &labelConverter,
                &NPar::LocalExecutor(),
                localData.Rand.Get()
            ).Learn->Cast<NCB::TQuantizedForCPUObjectsDataProvider>()
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
        if (localData.Rand == nullptr) { // may be set by TDatasetLoader
            localData.Rand = new TRestorableFastRng64(params->Data.RandomSeed + hostId);
        }

        auto trainParamsJson = GetJson(params->Data.TrainParams);
        UpdateUndefinedClassNames(localData.ClassNamesFromDataset, &trainParamsJson);
        localData.Params.Load(trainParamsJson);

        const auto& trainParams = localData.Params;

        NCB::TTrainingForCPUDataProviders trainingDataProviders;
        trainingDataProviders.Learn = GetTrainData(trainData);

        const TFoldsCreationParams foldsCreationParams(
            trainParams,
            *trainingDataProviders.Learn->ObjectsData,
            /*isForWorkerLocalData*/ true);

        localData.Progress = MakeHolder<TLearnProgress>(
            /*isForWorkerLocalData*/ true,
            /*isSingleHost*/ false,
            trainingDataProviders,
            params->Data.ApproxDimension,
            TLabelConverter(), // unused in case of localData
            params->Data.RandomSeed + hostId,
            /*initRand*/ localData.Rand.Get(),
            foldsCreationParams,
            /*datasetsCanContainBaseline*/ true,
            params->Data.TargetClassifiers,
            /*featuresCheckSum*/ 0, // unused in case of localData
            /*foldCreationParamsCheckSum*/ 0,
            /*initModel*/ Nothing(),
            /*initModelApplyCompatiblePools*/ NCB::TDataProviders(),
            &NPar::LocalExecutor());
        Y_ASSERT(localData.Progress->AveragingFold.BodyTailArr.ysize() == 1);

        localData.HessianType = params->Data.HessianType;

        localData.StoreExpApprox = foldsCreationParams.StoreExpApproxes;

        localData.UseTreeLevelCaching = NeedToUseTreeLevelCaching(
            trainParams,
            /*maxBodyTailCount=*/1,
            localData.Progress->AveragingFold.GetApproxDimension());

        const bool isPairwiseScoring = IsPairwiseScoring(
            trainParams.LossFunctionDescription->GetLossFunction());
        const int defaultCalcStatsObjBlockSize =
            static_cast<int>(trainParams.ObliviousTreeOptions->DevScoreCalcObjBlockSize);
        auto& plainFold = localData.Progress->AveragingFold;
        localData.SampledDocs.Create(
            { plainFold },
            isPairwiseScoring,
            defaultCalcStatsObjBlockSize,
            GetBernoulliSampleRate(trainParams.ObliviousTreeOptions->BootstrapConfig));
        if (localData.UseTreeLevelCaching) {
            localData.SmallestSplitSideDocs.Create(
                { plainFold },
                isPairwiseScoring,
                defaultCalcStatsObjBlockSize);
            localData.PrevTreeLevelStats.Create(
                { plainFold },
                CountNonCtrBuckets(
                    *(GetTrainData(trainData)->ObjectsData->GetQuantizedFeaturesInfo()),
                    trainParams.CatFeatureParams->OneHotMaxSize.Get()),
                trainParams.ObliviousTreeOptions->MaxDepth);
        }
        localData.Indices.yresize(plainFold.GetLearnSampleCount());
        localData.AllDocCount = params->Data.AllDocCount;
        localData.SumAllWeights = params->Data.SumAllWeights;
    }

    void TApproxReconstructor::DoMap(
        NPar::IUserContext* ctx,
        int hostId,
        TInput* valuedForest,
        TOutput* /*unused*/
    ) const {
        NPar::TCtxPtr<TTrainData> trainData(ctx, SHARED_ID_TRAIN_DATA, hostId);
        Y_ASSERT(!GetTrainData(trainData)->MetaInfo.FeaturesLayout->GetCatFeatureCount());

        auto& localData = TLocalTensorSearchData::GetRef();
        Y_ASSERT(IsPlainMode(localData.Params.BoostingOptions->BoostingType));

        const auto& forest = valuedForest->Data.first;
        const auto& leafValues = valuedForest->Data.second;
        Y_ASSERT(forest.size() == leafValues.size());

        auto maybeBaseline = GetTrainData(trainData)->TargetData->GetBaseline();
        if (maybeBaseline) {
            AssignRank2<float>(*maybeBaseline, &localData.Progress->AvrgApprox);
        }

        const ui32 learnSampleCount = GetTrainData(trainData)->GetObjectCount();
        const bool storeExpApprox = IsStoreExpApprox(
            localData.Params.LossFunctionDescription->GetLossFunction());
        const auto& avrgFold = localData.Progress->AveragingFold;
        for (size_t treeIdx : xrange(forest.size())) {
            const auto leafIndices = BuildIndices(
                avrgFold,
                forest[treeIdx],
                GetTrainData(trainData), /*testData*/
                { },
                &NPar::LocalExecutor());
            UpdateAvrgApprox(
                storeExpApprox,
                learnSampleCount,
                leafIndices,
                leafValues[treeIdx], /*testData*/
                { },
                localData.Progress.Get(),
                &NPar::LocalExecutor());
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
        if (localData.UseTreeLevelCaching) {
            localData.PrevTreeLevelStats.GarbageCollect();
        }
    }

    void TBootstrapMaker::DoMap(
        NPar::IUserContext* /*ctx*/,
        int /*hostId*/,
        TInput* /*unused*/,
        TOutput* /*unused*/
    ) const {
        auto& localData = TLocalTensorSearchData::GetRef();
        Bootstrap(
            localData.Params,
            localData.Indices,
            &localData.Progress->AveragingFold,
            &localData.SampledDocs,
            &NPar::LocalExecutor(),
            &localData.Progress->Rand);
        localData.FlatPairs = UnpackPairsFromQueries(localData.Progress->AveragingFold.LearnQueriesInfo);
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
        CalcStatsAndScores(
            *GetTrainData(trainData)->ObjectsData,
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
            /*scoreBins*/nullptr);
    }

    static void CalcPairwiseStats(const NPar::TCtxPtr<TTrainData>& trainData,
        const TFlatPairsInfo& pairs,
        const TCandidateInfo& candidate,
        TPairwiseStats* pairwiseStats
    ) {
        auto& localData = TLocalTensorSearchData::GetRef();
        CalcStatsAndScores(
            *GetTrainData(trainData)->ObjectsData,
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
            /*scoreBins*/nullptr);
    }

    void TScoreCalcer::DoMap(
        NPar::IUserContext* ctx,
        int hostId,
        TInput* candidateList,
        TOutput* bucketStats
    ) const {
        NPar::TCtxPtr<TTrainData> trainData(ctx, SHARED_ID_TRAIN_DATA, hostId);
        auto calcStats3D = [&](const TCandidateInfo& candidate, TStats3D* stats3D) {
            CalcStats3D(trainData, candidate, stats3D);
        };
        MapCandidateList(calcStats3D, candidateList->Data, &bucketStats->Data);
    }

    void TPairwiseScoreCalcer::DoMap(
        NPar::IUserContext* ctx,
        int hostId,
        TInput* candidateList,
        TOutput* bucketStats
    ) const {
        NPar::TCtxPtr<TTrainData> trainData(ctx, SHARED_ID_TRAIN_DATA, hostId);
        auto& localData = TLocalTensorSearchData::GetRef();
        auto calcPairwiseStats = [&](const TCandidateInfo& candidate, TPairwiseStats* pairwiseStats) {
            CalcPairwiseStats(trainData, localData.FlatPairs, candidate, pairwiseStats);
        };
        MapCandidateList(calcPairwiseStats, candidateList->Data, &bucketStats->Data);
    }

    // buckets -> workerPairwiseStats
    void TRemotePairwiseBinCalcer::DoMap(
        NPar::IUserContext* ctx,
        int hostId,
        TInput* candidate,
        TOutput* bucketStats
    ) const {
        NPar::TCtxPtr<TTrainData> trainData(ctx, SHARED_ID_TRAIN_DATA, hostId);
        auto& localData = TLocalTensorSearchData::GetRef();
        auto calcPairwiseStats = [&](const TCandidateInfo& candidate, TPairwiseStats* pairwiseStats) {
            CalcPairwiseStats(trainData, localData.FlatPairs, candidate, pairwiseStats);
        };
        MapVector(calcPairwiseStats, candidate->Candidates, bucketStats);
    }

    // workerPairwiseStats -> pairwiseStats
    void TRemotePairwiseBinCalcer::DoReduce(TVector<TOutput>* statsFromAllWorkers, TOutput* stats) const {
        const int workerCount = statsFromAllWorkers->ysize();
        const int bucketCount = (*statsFromAllWorkers)[0].ysize();
        stats->yresize(bucketCount);
        NPar::ParallelFor(
            0,
            bucketCount,
            [&] (int bucketIdx) {
                (*stats)[bucketIdx] = (*statsFromAllWorkers)[0][bucketIdx];
                for (int workerIdx : xrange(1, workerCount)) {
                    (*stats)[bucketIdx].Add((*statsFromAllWorkers)[workerIdx][bucketIdx]);
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
                TVector<TScoreBin> scoreBins;
                CalculatePairwiseScore(
                    candidatePairwiseStats,
                    bucketCount,
                    localData.Params.ObliviousTreeOptions->L2Reg,
                    localData.Params.ObliviousTreeOptions->PairwiseNonDiagReg,
                    localData.Params.CatFeatureParams->OneHotMaxSize,
                    &scoreBins);
                *candidateScores = GetScores(scoreBins);
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
        auto calcStats3D = [&](const TCandidateInfo& candidate, TStats3D* stats3D) {
            CalcStats3D(trainData, candidate, stats3D);
        };
        MapVector(calcStats3D, candidatesInfoList->Candidates, bucketStats);
    }

    // vector<TStats4D> -> TStats4D
    void TRemoteBinCalcer::DoReduce(TVector<TOutput>* statsFromAllWorkers, TOutput* stats) const {
        const int workerCount = statsFromAllWorkers->ysize();
        const int bucketCount = (*statsFromAllWorkers)[0].ysize();
        stats->yresize(bucketCount);
        NPar::ParallelFor(
            0,
            bucketCount,
            [&] (int bucketIdx) {
                (*stats)[bucketIdx] = (*statsFromAllWorkers)[0][bucketIdx];
                for (int workerIdx = 1; workerIdx < workerCount; ++workerIdx) {
                    (*stats)[bucketIdx].Add((*statsFromAllWorkers)[workerIdx][bucketIdx]);
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
                *candidateScores = GetScores(
                    GetScoreBins(
                        candidateStats3D,
                        localData.Depth,
                        localData.SumAllWeights,
                        localData.AllDocCount,
                        localData.Params));
            };
        MapVector(getScores, *bucketStats, scores);
    }

    void TLeafIndexSetter::DoMap(
        NPar::IUserContext* ctx,
        int hostId,
        TInput* bestSplit,
        TOutput* /*unused*/
    ) const {
        Y_ASSERT(bestSplit->Data.Type != ESplitType::OnlineCtr);
        auto& localData = TLocalTensorSearchData::GetRef();
        NPar::TCtxPtr<TTrainData> trainData(ctx, SHARED_ID_TRAIN_DATA, hostId);
        SetPermutedIndices(
            bestSplit->Data,
            *GetTrainData(trainData)->ObjectsData,
            localData.Depth + 1,
            localData.Progress->AveragingFold,
            &localData.Indices,
            &NPar::LocalExecutor());
        if (IsSamplingPerTree(localData.Params.ObliviousTreeOptions)) {
            localData.SampledDocs.UpdateIndices(localData.Indices, &NPar::LocalExecutor());
            if (localData.UseTreeLevelCaching) {
                localData.SmallestSplitSideDocs.SelectSmallestSplitSide(
                    localData.Depth + 1,
                    localData.SampledDocs,
                    &NPar::LocalExecutor());
            }
        }
    }

    void TEmptyLeafFinder::DoMap(
        NPar::IUserContext* /*ctx*/,
        int /*hostId*/,
        TInput* /*unused*/,
        TOutput* isLeafEmpty
    ) const {
        auto& localData = TLocalTensorSearchData::GetRef();
        isLeafEmpty->Data = GetIsLeafEmpty(localData.Depth + 1, localData.Indices);
        ++localData.Depth; // tree level completed
    }

    void TBucketSimpleUpdater::DoMap(
        NPar::IUserContext* /*ctx*/,
        int /*hostId*/,
        TInput* /*unused*/,
        TOutput* sums
    ) const {
        auto& localData = TLocalTensorSearchData::GetRef();
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

        for (auto& bucket : localData.Buckets) {
            bucket.SetZeroDers();
        }
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
        sums->Data = std::make_pair(localData.Buckets, localData.PairwiseBuckets);
    }

    void TCalcApproxStarter::DoMap(
        NPar::IUserContext* ctx,
        int hostId,
        TInput* splitTree,
        TOutput* /*unused*/
    ) const {
        auto& localData = TLocalTensorSearchData::GetRef();
        NPar::TCtxPtr<TTrainData> trainData(ctx, SHARED_ID_TRAIN_DATA, hostId);
        localData.Indices = BuildIndices(
            localData.Progress->AveragingFold,
            splitTree->Data,
            GetTrainData(trainData),
            /*testDataPtrs*/{ },
            &NPar::LocalExecutor());
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
        localData.Buckets.resize(splitTree->Data.GetLeafCount());
        Fill(localData.Buckets.begin(), localData.Buckets.end(), TSum());
        localData.MultiBuckets.resize(splitTree->Data.GetLeafCount());
        Fill(
            localData.MultiBuckets.begin(),
            localData.MultiBuckets.end(),
            TSumMulti(approxDimension, localData.HessianType));
        localData.PairwiseBuckets.SetSizes(splitTree->Data.GetLeafCount(), splitTree->Data.GetLeafCount());
        localData.PairwiseBuckets.FillZero();
        localData.GradientIteration = 0;
    }

    void TDeltaSimpleUpdater::DoMap(
        NPar::IUserContext* /*unused*/,
        int /*unused*/,
        TInput* leafValues,
        TOutput* /*unused*/
    ) const {
        auto& localData = TLocalTensorSearchData::GetRef();
        UpdateApproxDeltas(
            localData.StoreExpApprox,
            localData.Indices,
            localData.Progress->AveragingFold.BodyTailArr[0].TailFinish,
            &NPar::LocalExecutor(),
            &(*leafValues)[0],
            &localData.ApproxDeltas[0]);
        ++localData.GradientIteration; // gradient iteration completed
    }

    void TApproxUpdater::DoMap(
        NPar::IUserContext* /*unused*/,
        int /*unused*/,
        TInput* averageLeafValues,
        TOutput* /*unused*/
    ) const {
        auto& localData = TLocalTensorSearchData::GetRef();
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

    void TDerivativeSetter::DoMap(
        NPar::IUserContext* /*ctx*/,
        int /*hostId*/,
        TInput* /*unused*/,
        TOutput* /*unused*/
    ) const {
        auto& localData = TLocalTensorSearchData::GetRef();
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

    void TBucketMultiUpdater::DoMap(
        NPar::IUserContext* /*ctx*/,
        int /*hostId*/,
        TInput* /*unused*/,
        TOutput* sums
    ) const {
        auto& localData = TLocalTensorSearchData::GetRef();
        const int approxDimension = localData.Progress->ApproxDimension;
        Y_ASSERT(approxDimension > 1);
        const auto error = BuildError(localData.Params, /*custom objective*/Nothing());
        const auto estimationMethod = localData.Params.ObliviousTreeOptions->LeavesEstimationMethod;

        for (auto& bucket : localData.MultiBuckets) {
            bucket.SetZeroDers();
        }
        if (estimationMethod == ELeavesEstimation::Newton) {
            UpdateBucketsMulti(
                AddSampleToBucketNewtonMulti,
                localData.Indices,
                localData.Progress->AveragingFold.LearnTarget,
                localData.Progress->AveragingFold.GetLearnWeights(),
                localData.Progress->AveragingFold.BodyTailArr[0].Approx,
                localData.ApproxDeltas,
                *error,
                localData.Progress->AveragingFold.BodyTailArr[0].BodyFinish,
                localData.GradientIteration,
                &localData.MultiBuckets);
        } else {
            Y_ASSERT(estimationMethod == ELeavesEstimation::Gradient);
            UpdateBucketsMulti(
                AddSampleToBucketGradientMulti,
                localData.Indices,
                localData.Progress->AveragingFold.LearnTarget,
                localData.Progress->AveragingFold.GetLearnWeights(),
                localData.Progress->AveragingFold.BodyTailArr[0].Approx,
                localData.ApproxDeltas,
                *error,
                localData.Progress->AveragingFold.BodyTailArr[0].BodyFinish,
                localData.GradientIteration,
                &localData.MultiBuckets);
        }
        sums->Data = std::make_pair(localData.MultiBuckets, TUnusedInitializedParam());
    }

    void TDeltaMultiUpdater::DoMap(
        NPar::IUserContext* /*ctx*/,
        int /*hostId*/,
        TInput* leafValues,
        TOutput* /*unused*/
    ) const {
        auto& localData = TLocalTensorSearchData::GetRef();
        UpdateApproxDeltasMulti(
            localData.StoreExpApprox,
            localData.Indices,
            localData.Progress->AveragingFold.BodyTailArr[0].BodyFinish,
            leafValues,
            &localData.ApproxDeltas);
        ++localData.GradientIteration; // gradient iteration completed
    }

    void TErrorCalcer::DoMap(
        NPar::IUserContext* ctx,
        int hostId,
        TInput* /*unused*/,
        TOutput* additiveStats
    ) const {
        const auto& localData = TLocalTensorSearchData::GetRef();
        const auto errors = CreateMetrics(
            localData.Params.MetricOptions,
            /*evalMetricDescriptor*/Nothing(),
            localData.Progress->ApproxDimension);
        const auto skipMetricOnTrain = GetSkipMetricOnTrain(errors);
        NPar::TCtxPtr<TTrainData> trainData(ctx, SHARED_ID_TRAIN_DATA, hostId);
        for (int errorIdx = 0; errorIdx < errors.ysize(); ++errorIdx) {
            if (!skipMetricOnTrain[errorIdx] && errors[errorIdx]->IsAdditiveMetric()) {
                const TString metricDescription = errors[errorIdx]->GetDescription();
                (*additiveStats)[metricDescription] = EvalErrors(
                    localData.Progress->AvrgApprox,
                    *GetTrainData(trainData)->TargetData->GetTargetForLoss(),
                    GetWeights(*GetTrainData(trainData)->TargetData),
                    GetTrainData(trainData)->TargetData->GetGroupInfo().GetOrElse(TConstArrayRef<TQueryInfo>()),
                    *errors[errorIdx],
                    &NPar::LocalExecutor());
            }
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
        const size_t leafCount = localData.Buckets.size();
        *leafWeights = SumLeafWeights(
            leafCount,
            localData.Indices,
            localData.Progress->AveragingFold.GetLearnPermutationArray(),
            GetWeights(*GetTrainData(trainData)->TargetData));
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
REGISTER_SAVELOAD_NM_CLASS(0xd66d50f, NCatboostDistributed, TBucketSimpleUpdater);
REGISTER_SAVELOAD_NM_CLASS(0xd66d4af, NCatboostDistributed, TDerivativeSetter);
REGISTER_SAVELOAD_NM_CLASS(0xd66d4b2, NCatboostDistributed, TDeltaMultiUpdater);
REGISTER_SAVELOAD_NM_CLASS(0xd66d4c1, NCatboostDistributed, TBucketMultiUpdater);

REGISTER_SAVELOAD_NM_CLASS(0xd66d4d1, NCatboostDistributed, TPairwiseScoreCalcer);
REGISTER_SAVELOAD_NM_CLASS(0xd66d4d2, NCatboostDistributed, TRemotePairwiseBinCalcer);
REGISTER_SAVELOAD_NM_CLASS(0xd66d4d3, NCatboostDistributed, TRemotePairwiseScoreCalcer);
REGISTER_SAVELOAD_NM_CLASS(0xd66d4d4, NCatboostDistributed, TErrorCalcer);

REGISTER_SAVELOAD_NM_CLASS(0xd66d4d6, NCatboostDistributed, TApproxReconstructor);
REGISTER_SAVELOAD_NM_CLASS(0xd66d4e0, NCatboostDistributed, TLeafWeightsGetter);

REGISTER_SAVELOAD_NM_CLASS(0xd66d4e1, NCatboostDistributed, TDatasetLoader);
