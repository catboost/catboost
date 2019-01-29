#include "mappers.h"

#include <catboost/libs/algo/approx_calcer.h>
#include <catboost/libs/algo/approx_calcer_multi.h>
#include <catboost/libs/algo/error_functions.h>
#include <catboost/libs/algo/score_calcer.h>
#include <catboost/libs/algo/learn_context.h>
#include <catboost/libs/algo/online_ctr.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/query_info_helper.h>
#include <catboost/libs/helpers/vector_helpers.h>

#include <utility>


namespace NCatboostDistributed {

    void TPlainFoldBuilder::DoMap(
        NPar::IUserContext* ctx,
        int hostId,
        TInput* /*unused*/,
        TOutput* /*unused*/
    ) const {
        NPar::TCtxPtr<TTrainData> trainData(ctx, SHARED_ID_TRAIN_DATA, hostId);
        auto& localData = TLocalTensorSearchData::GetRef();
        localData.Rand = new TRestorableFastRng64(trainData->RandomSeed + hostId);

        NJson::TJsonValue jsonParams;
        const bool jsonParamsOK = ReadJsonTree(trainData->StringParams, &jsonParams);
        Y_ASSERT(jsonParamsOK);
        localData.Params.Load(jsonParams);
        localData.StoreExpApprox = IsStoreExpApprox(
            localData.Params.LossFunctionDescription->GetLossFunction());

        localData.Progress.ApproxDimension = trainData->ApproxDimension;
        localData.Progress.AveragingFold = TFold::BuildPlainFold(
            *trainData->TrainData,
            trainData->TargetClassifiers,
            /*shuffle*/false,
            trainData->TrainData->GetObjectCount(),
            trainData->ApproxDimension,
            localData.StoreExpApprox,
            UsesPairsForCalculation(localData.Params.LossFunctionDescription->GetLossFunction()),
            *localData.Rand,
            &NPar::LocalExecutor());
        Y_ASSERT(localData.Progress.AveragingFold.BodyTailArr.ysize() == 1);

        auto baseline = GetBaseline(trainData->TrainData->TargetData);
        if (!baseline.empty()) {
            AssignRank2<float>(baseline, &localData.Progress.AvrgApprox);
        } else {
            localData.Progress.AvrgApprox.resize(
                trainData->ApproxDimension,
                TVector<double>(trainData->TrainData->GetObjectCount()));
        }

        localData.UseTreeLevelCaching = NeedToUseTreeLevelCaching(
            localData.Params,
            /*maxBodyTailCount=*/1,
            localData.Progress.AveragingFold.GetApproxDimension());

        const bool isPairwiseScoring = IsPairwiseScoring(
            localData.Params.LossFunctionDescription->GetLossFunction());
        const int defaultCalcStatsObjBlockSize =
            static_cast<int>(localData.Params.ObliviousTreeOptions->DevScoreCalcObjBlockSize);
        auto& plainFold = localData.Progress.AveragingFold;
        localData.SampledDocs.Create(
            { plainFold },
            isPairwiseScoring,
            defaultCalcStatsObjBlockSize,
            GetBernoulliSampleRate(localData.Params.ObliviousTreeOptions->BootstrapConfig));
        if (localData.UseTreeLevelCaching) {
            localData.SmallestSplitSideDocs.Create(
                { plainFold },
                isPairwiseScoring,
                defaultCalcStatsObjBlockSize);
            localData.PrevTreeLevelStats.Create(
                { plainFold },
                CountNonCtrBuckets(
                    *(trainData->TrainData->ObjectsData->GetQuantizedFeaturesInfo()),
                    localData.Params.CatFeatureParams->OneHotMaxSize.Get()),
                localData.Params.ObliviousTreeOptions->MaxDepth);
        }
        localData.Indices.yresize(plainFold.GetLearnSampleCount());
        localData.AllDocCount = trainData->AllDocCount;
        localData.SumAllWeights = trainData->SumAllWeights;
    }

    void TApproxReconstructor::DoMap(
        NPar::IUserContext* ctx,
        int hostId,
        TInput* valuedForest,
        TOutput* /*unused*/
    ) const {
        NPar::TCtxPtr<TTrainData> trainData(ctx, SHARED_ID_TRAIN_DATA, hostId);
        Y_ASSERT(!trainData->TrainData->MetaInfo.FeaturesLayout->GetCatFeatureCount());

        auto& localData = TLocalTensorSearchData::GetRef();
        Y_ASSERT(IsPlainMode(localData.Params.BoostingOptions->BoostingType));

        const auto& forest = valuedForest->Data.first;
        const auto& leafValues = valuedForest->Data.second;
        Y_ASSERT(forest.size() == leafValues.size());

        auto baseline = GetBaseline(trainData->TrainData->TargetData);
        if (!baseline.empty()) {
            AssignRank2<float>(baseline, &localData.Progress.AvrgApprox);
        }

        const ui32 learnSampleCount = trainData->TrainData->GetObjectCount();
        const bool storeExpApprox = IsStoreExpApprox(
            localData.Params.LossFunctionDescription->GetLossFunction());
        const auto& avrgFold = localData.Progress.AveragingFold;
        for (size_t treeIdx : xrange(forest.size())) {
            const auto leafIndices = BuildIndices(
                avrgFold,
                forest[treeIdx],
                trainData->TrainData, /*testData*/
                { },
                &NPar::LocalExecutor());
            UpdateAvrgApprox(
                storeExpApprox,
                learnSampleCount,
                leafIndices,
                leafValues[treeIdx], /*testData*/
                { },
                &localData.Progress,
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
            &localData.Progress.AveragingFold,
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
            *trainData->TrainData->ObjectsData,
            localData.Progress.AveragingFold.GetAllCtrs(),
            localData.SampledDocs,
            localData.SmallestSplitSideDocs,
            /*initialFold*/nullptr,
            /*pairs*/{},
            localData.Params,
            candidate.SplitCandidate,
            localData.Depth,
            localData.UseTreeLevelCaching,
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
            *trainData->TrainData->ObjectsData,
            localData.Progress.AveragingFold.GetAllCtrs(),
            localData.SampledDocs,
            localData.SmallestSplitSideDocs,
            /*initialFold*/nullptr,
            pairs,
            localData.Params,
            candidate.SplitCandidate,
            localData.Depth,
            localData.UseTreeLevelCaching,
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
        const auto pairs = UnpackPairsFromQueries(localData.Progress.AveragingFold.LearnQueriesInfo);
        auto calcPairwiseStats = [&](const TCandidateInfo& candidate, TPairwiseStats* pairwiseStats) {
            CalcPairwiseStats(trainData, pairs, candidate, pairwiseStats);
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
        const auto pairs = UnpackPairsFromQueries(localData.Progress.AveragingFold.LearnQueriesInfo);
        auto calcPairwiseStats = [&](const TCandidateInfo& candidate, TPairwiseStats* pairwiseStats) {
            CalcPairwiseStats(trainData, pairs, candidate, pairwiseStats);
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
                    ESplitType::FloatFeature,
                    localData.Params.ObliviousTreeOptions->L2Reg,
                    localData.Params.ObliviousTreeOptions->PairwiseNonDiagReg,
                    &scoreBins);
                *candidateScores = GetScores(scoreBins);
            };
        MapVector(getScores, *bucketStats, scores);
    }

    // subcandidates -> TStats4D
    void TRemoteBinCalcer::DoMap(
        NPar::IUserContext* ctx,
        int hostId,
        TInput* candidate,
        TOutput* bucketStats
    ) const {
        NPar::TCtxPtr<TTrainData> trainData(ctx, SHARED_ID_TRAIN_DATA, hostId);
        auto calcStats3D = [&](const TCandidateInfo& candidate, TStats3D* stats3D) {
            CalcStats3D(trainData, candidate, stats3D);
        };
        MapVector(calcStats3D, candidate->Candidates, bucketStats);
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
                        ESplitType::FloatFeature,
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
        TInput* bestSplitCandidate,
        TOutput* /*unused*/
    ) const {
        const TSplit bestSplit(
            bestSplitCandidate->Data.SplitCandidate,
            bestSplitCandidate->Data.BestBinBorderId);
        Y_ASSERT(bestSplit.Type != ESplitType::OnlineCtr);
        auto& localData = TLocalTensorSearchData::GetRef();
        NPar::TCtxPtr<TTrainData> trainData(ctx, SHARED_ID_TRAIN_DATA, hostId);
        SetPermutedIndices(
            bestSplit,
            *trainData->TrainData->ObjectsData,
            localData.Depth + 1,
            localData.Progress.AveragingFold,
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
        const int approxDimension = localData.Progress.ApproxDimension;
        Y_ASSERT(approxDimension == 1);
        const auto error = BuildError(localData.Params, /*custom objective*/Nothing());
        const auto estimationMethod = localData.Params.ObliviousTreeOptions->LeavesEstimationMethod;
        const int scratchSize =
            error->GetErrorType() == EErrorType::PerObjectError ?
                APPROX_BLOCK_SIZE * CB_THREAD_LIMIT :
                // plain boosting ==> not approx on full history
                localData.Progress.AveragingFold.BodyTailArr[0].BodyFinish;
        TVector<TDers> weightedDers;
        weightedDers.yresize(scratchSize);

        for (auto& bucket : localData.Buckets) {
            bucket.SetZeroDers();
        }
        CalcLeafDersSimple(
            localData.Indices,
            localData.Progress.AveragingFold,
            localData.Progress.AveragingFold.BodyTailArr[0],
            localData.Progress.AveragingFold.BodyTailArr[0].Approx[0],
            localData.ApproxDeltas[0],
            *error,
            localData.Progress.AveragingFold.BodyTailArr[0].BodyFinish,
            localData.Progress.AveragingFold.BodyTailArr[0].BodyQueryFinish,
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

    void TCalcApproxStarter::DoMap(
        NPar::IUserContext* ctx,
        int hostId,
        TInput* splitTree,
        TOutput* /*unused*/
    ) const {
        auto& localData = TLocalTensorSearchData::GetRef();
        NPar::TCtxPtr<TTrainData> trainData(ctx, SHARED_ID_TRAIN_DATA, hostId);
        localData.Indices = BuildIndices(
            localData.Progress.AveragingFold,
            splitTree->Data,
            trainData->TrainData,
            /*testDataPtrs*/{ },
            &NPar::LocalExecutor());
        const int approxDimension = localData.Progress.ApproxDimension;
        if (localData.ApproxDeltas.empty()) {
            localData.ApproxDeltas.resize(approxDimension); // 1D or nD
            for (auto& dimensionDelta : localData.ApproxDeltas) {
                dimensionDelta.yresize(localData.Progress.AveragingFold.BodyTailArr[0].TailFinish);
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
            TSumMulti(approxDimension, trainData->HessianType));
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
            localData.Progress.AveragingFold.BodyTailArr[0].TailFinish,
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
                &localData.Progress.AveragingFold);
        } else {
            UpdateBodyTailApprox</*StoreExpApprox*/false>(
                { localData.ApproxDeltas },
                localData.Params.BoostingOptions->LearningRate,
                &NPar::LocalExecutor(),
                &localData.Progress.AveragingFold);
        }
        TConstArrayRef<ui32> learnPermutationRef(localData.Progress.AveragingFold.GetLearnPermutationArray());
        TConstArrayRef<TIndexType> indicesRef(localData.Indices);
        const auto updateAvrgApprox =
            [=](TConstArrayRef<double> delta, TArrayRef<double> approx, size_t idx) {
                approx[learnPermutationRef[idx]] += delta[indicesRef[idx]];
            };
        UpdateApprox(
            updateAvrgApprox,
            *averageLeafValues,
            &localData.Progress.AvrgApprox,
            &NPar::LocalExecutor());
    }

    void TDerivativeSetter::DoMap(
        NPar::IUserContext* /*ctx*/,
        int /*hostId*/,
        TInput* /*unused*/,
        TOutput* /*unused*/
    ) const {
        auto& localData = TLocalTensorSearchData::GetRef();
        Y_ASSERT(localData.Progress.AveragingFold.BodyTailArr.ysize() == 1);
        const auto error = BuildError(localData.Params, /*custom objective*/Nothing());
        CalcWeightedDerivatives(
            *error,
            /*bodyTailIdx*/0,
            localData.Params,
            localData.Rand->GenRand(),
            &localData.Progress.AveragingFold,
            &NPar::LocalExecutor());
    }

    void TBucketMultiUpdater::DoMap(
        NPar::IUserContext* /*ctx*/,
        int /*hostId*/,
        TInput* /*unused*/,
        TOutput* sums
    ) const {
        auto& localData = TLocalTensorSearchData::GetRef();
        const int approxDimension = localData.Progress.ApproxDimension;
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
                localData.Progress.AveragingFold.LearnTarget,
                localData.Progress.AveragingFold.GetLearnWeights(),
                localData.Progress.AveragingFold.BodyTailArr[0].Approx,
                localData.ApproxDeltas,
                *error,
                localData.Progress.AveragingFold.BodyTailArr[0].BodyFinish,
                localData.GradientIteration,
                &localData.MultiBuckets);
        } else {
            Y_ASSERT(estimationMethod == ELeavesEstimation::Gradient);
            UpdateBucketsMulti(
                AddSampleToBucketGradientMulti,
                localData.Indices,
                localData.Progress.AveragingFold.LearnTarget,
                localData.Progress.AveragingFold.GetLearnWeights(),
                localData.Progress.AveragingFold.BodyTailArr[0].Approx,
                localData.ApproxDeltas,
                *error,
                localData.Progress.AveragingFold.BodyTailArr[0].BodyFinish,
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
            localData.Progress.AveragingFold.BodyTailArr[0].BodyFinish,
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
            localData.Params.LossFunctionDescription,
            localData.Params.MetricOptions,
            /*evalMetricDescriptor*/Nothing(),
            localData.Progress.ApproxDimension);
        const auto skipMetricOnTrain = GetSkipMetricOnTrain(errors);
        NPar::TCtxPtr<TTrainData> trainData(ctx, SHARED_ID_TRAIN_DATA, hostId);
        for (int errorIdx = 0; errorIdx < errors.ysize(); ++errorIdx) {
            if (!skipMetricOnTrain[errorIdx] && errors[errorIdx]->IsAdditiveMetric()) {
                const TString metricDescription = errors[errorIdx]->GetDescription();
                (*additiveStats)[metricDescription] = EvalErrors(
                    localData.Progress.AvrgApprox,
                    GetTarget(trainData->TrainData->TargetData),
                    GetWeights(trainData->TrainData->TargetData),
                    GetGroupInfo(trainData->TrainData->TargetData),
                    errors[errorIdx],
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
            localData.Progress.AveragingFold.GetLearnPermutationArray(),
            GetWeights(trainData->TrainData->TargetData));
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
