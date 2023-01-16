#include "learn_context.h"

#include "apply.h"
#include "approx_dimension.h"
#include "approx_updater_helpers.h"
#include "calc_score_cache.h"

#include "helpers.h"
#include "online_ctr.h"

#include <catboost/libs/helpers/checksum.h>
#include <catboost/libs/helpers/parallel_tasks.h>
#include <catboost/libs/helpers/progress_helper.h>
#include <catboost/libs/helpers/vector_helpers.h>
#include <catboost/libs/model/model.h>
#include <catboost/private/libs/algo_helpers/error_functions.h>
#include <catboost/private/libs/distributed/master.h>
#include <catboost/private/libs/index_range/index_range.h>
#include <catboost/private/libs/options/defaults_helper.h>

#include <library/cpp/digest/crc32c/crc32c.h>
#include <library/cpp/digest/md5/md5.h>
#include <library/cpp/threading/local_executor/local_executor.h>

#include <util/digest/multi.h>
#include <util/generic/algorithm.h>
#include <util/generic/cast.h>
#include <util/generic/guid.h>
#include <util/generic/xrange.h>
#include <util/folder/path.h>
#include <util/stream/file.h>
#include <util/system/fs.h>


using namespace NCB;


static bool IsPermutationNeeded(bool hasTime, bool hasCtrs, bool isOrderedBoosting, bool isAveragingFold) {
    if (hasTime) {
        return false;
    }
    if (hasCtrs) {
        return true;
    }
    return isOrderedBoosting && !isAveragingFold;
}

static int CountLearningFolds(int permutationCount, bool isPermutationNeededForLearning) {
    return isPermutationNeededForLearning ? Max<ui32>(1, permutationCount - 1) : 1;
}


TFoldsCreationParams::TFoldsCreationParams(
    const NCatboostOptions::TCatBoostOptions& params,
    const TQuantizedObjectsDataProvider& learnObjectsData,
    const TMaybe<TVector<double>>& startingApprox,
    bool isForWorkerLocalData)
    : IsOrderedBoosting(!IsPlainMode(params.BoostingOptions->BoostingType))
    , LearningFoldCount(0) // properly inited below
    , FoldPermutationBlockSize(0) // properly inited below
    , StoreExpApproxes(IsStoreExpApprox(params.LossFunctionDescription->GetLossFunction()))
    , HasPairwiseWeights(UsesPairsForCalculation(params.LossFunctionDescription->GetLossFunction()))
    , FoldLenMultiplier(params.BoostingOptions->FoldLenMultiplier)
    , IsAverageFoldPermuted(false) // properly inited below
    , StartingApprox(startingApprox)
    , LossFunction(params.LossFunctionDescription->LossFunction.Get())
{
    const bool hasTime = params.DataProcessingOptions->HasTimeFlag
        || (learnObjectsData.GetOrder() == EObjectsOrder::Ordered);

    const bool hasCtrs =
        learnObjectsData.GetQuantizedFeaturesInfo()->CalcMaxCategoricalFeaturesUniqueValuesCountOnLearn()
            > params.CatFeatureParams->OneHotMaxSize.Get();

    const bool isLearnFoldPermuted = IsPermutationNeeded(
        hasTime,
        hasCtrs,
        IsOrderedBoosting,
        /*isAveragingFold*/ false
    );
    LearningFoldCount = isForWorkerLocalData ?
        0 :
        CountLearningFolds(params.BoostingOptions->PermutationCount, isLearnFoldPermuted);

    //Todo(noxoomo): check and init
    const auto& boostingOptions = params.BoostingOptions.Get();
    const ui32 learnSampleCount = learnObjectsData.GetObjectCount();

    FoldPermutationBlockSize = boostingOptions.PermutationBlockSize;
    if (FoldPermutationBlockSize == FoldPermutationBlockSizeNotSet) {
        FoldPermutationBlockSize = DefaultFoldPermutationBlockSize(learnSampleCount);
    }
    if (!isLearnFoldPermuted) {
        FoldPermutationBlockSize = learnSampleCount;
    }

    IsAverageFoldPermuted = !isForWorkerLocalData && IsPermutationNeeded(
        hasTime,
        hasCtrs,
        IsOrderedBoosting,
        /*isAveragingFold*/ true
    );
}


ui32 TFoldsCreationParams::CalcCheckSum(
    const NCB::TObjectsGrouping& objectsGrouping,
    NPar::ILocalExecutor* localExecutor) const {

    ui32 checkSum = MultiHash(
        IsOrderedBoosting,
        LearningFoldCount,
        FoldPermutationBlockSize,
        StoreExpApproxes,
        HasPairwiseWeights,
        IsAverageFoldPermuted
    );

    if (IsOrderedBoosting) {
        checkSum = MultiHash(checkSum, FoldLenMultiplier);
        if (!objectsGrouping.IsTrivial()) {
            const auto groups = objectsGrouping.GetNonTrivialGroups();

            constexpr int GROUP_BLOCK_SIZE = 20000;

            TSimpleIndexRangesGenerator<int> indexRangeGenerator(
                TIndexRange<int>(SafeIntegerCast<int>(groups.size())),
                GROUP_BLOCK_SIZE
            );

            TVector<ui32> groupsBlockCheckSums(SafeIntegerCast<size_t>(indexRangeGenerator.RangesCount()), 0);

            localExecutor->ExecRangeWithThrow(
                [&, groups] (int blockIdx) {
                    ui32 blockCheckSum = 0;
                    for (auto i : indexRangeGenerator.GetRange(blockIdx).Iter()) {
                        blockCheckSum = MultiHash(blockCheckSum, groups[i].Begin, groups[i].End);
                    }
                    groupsBlockCheckSums[blockIdx] = blockCheckSum;
                },
                0,
                indexRangeGenerator.RangesCount(),
                NPar::TLocalExecutor::WAIT_COMPLETE
            );

            for (auto groupsBlockCheckSum : groupsBlockCheckSums) {
                checkSum = MultiHash(checkSum, groupsBlockCheckSum);
            }
        }
    }
    return checkSum;
}

static inline ui32 UpdateCheckSumImpl(ui32 init, const TNonSymmetricTreeStepNode& node) {
    return UpdateCheckSum(
        init,
        node.LeftSubtreeDiff,
        node.RightSubtreeDiff
    );
}


static inline ui32 UpdateCheckSumImpl(ui32 init, const TCatFeature& catFeature) {
    return UpdateCheckSum(
        init,
        catFeature.UsedInModel(),
        catFeature.Position.Index,
        catFeature.Position.FlatIndex
    );
}

static inline ui32 UpdateCheckSumImpl(ui32 init, const TFloatFeature& floatFeature) {
    return UpdateCheckSum(
        init,
        floatFeature.HasNans,
        floatFeature.Position.Index,
        floatFeature.Position.FlatIndex,
        floatFeature.Borders
    );
}

static inline ui32 UpdateCheckSumImpl(ui32 init, const TOneHotFeature& oneHotFeature) {
    return UpdateCheckSum(
        init,
        oneHotFeature.CatFeatureIndex,
        oneHotFeature.Values
    );
}

static inline ui32 UpdateCheckSumImpl(ui32 init, const TModelCtr& ctr) {
    return UpdateCheckSum(
        init,
        ctr.GetHash()
    );
}

static inline ui32 UpdateCheckSumImpl(ui32 init, const TCtrFeature& ctrFeature) {
    return UpdateCheckSum(
        init,
        ctrFeature.Ctr,
        ctrFeature.Borders
    );
}


static ui32 CalcCoreModelCheckSum(const TFullModel& model) {
    const auto& trees = *model.ModelTrees;

    return UpdateCheckSum(
        ui32(0),
        trees.GetDimensionsCount(),
        trees.GetModelTreeData()->GetTreeSplits(),
        trees.GetModelTreeData()->GetTreeSizes(),
        trees.GetModelTreeData()->GetTreeStartOffsets(),
        trees.GetModelTreeData()->GetNonSymmetricStepNodes(),
        trees.GetModelTreeData()->GetNonSymmetricNodeIdToLeafId(),
        trees.GetModelTreeData()->GetLeafValues(),
        trees.GetCatFeatures(),
        trees.GetFloatFeatures(),
        trees.GetOneHotFeatures(),
        trees.GetCtrFeatures()
    );
}


TLearnContext::TLearnContext(
    const NCatboostOptions::TCatBoostOptions& params,
    const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
    const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
    const NCatboostOptions::TOutputFilesOptions& outputOptions,
    const TTrainingDataProviders& data,
    TMaybe<TPrecomputedOnlineCtrData> precomputedSingleOnlineCtrDataForSingleFold,
    const TLabelConverter& labelConverter,
    const TMaybe<TVector<double>>& startingApprox,
    TMaybe<const TRestorableFastRng64*> initRand,
    TMaybe<TFullModel*> initModel,
    THolder<TLearnProgress> initLearnProgress, // will be modified if not non-nullptr
    NCB::TDataProviders initModelApplyCompatiblePools,
    NPar::ILocalExecutor* localExecutor,
    const TString& fileNamesPrefix)

    : TCommonContext(
        params,
        objectiveDescriptor,
        evalMetricDescriptor,
        data.Learn->ObjectsData->GetFeaturesLayout(),
        localExecutor)
    , OutputOptions(outputOptions)
    , Files(outputOptions, fileNamesPrefix)
    , Profile((int)Params.BoostingOptions->IterationCount)
    , UseTreeLevelCachingFlag(false)
    , HasWeights(data.Learn->MetaInfo.HasWeights) {

    ETaskType taskType = Params.GetTaskType();
    CB_ENSURE(taskType == ETaskType::CPU, "Error: expect learn on CPU task type, got " << taskType);

    THPTimer calcHashTimer;
    ui32 featuresCheckSum = data.CalcFeaturesCheckSum(localExecutor);
    CATBOOST_DEBUG_LOG << "Features checksum calculation time: " << calcHashTimer.Passed() << Endl;

    ui32 approxDimension = GetApproxDimension(Params, labelConverter, data.Learn->TargetData->GetTargetDimension());
    if (initLearnProgress) {
        CB_ENSURE(
            approxDimension == SafeIntegerCast<ui32>(initLearnProgress->ApproxDimension),
            "Attempt to continue learning with a different approx dimension"
        );
        CB_ENSURE(
            labelConverter == initLearnProgress->LabelConverter,
            "Attempt to continue learning with different class labels"
        );
    }

    const TFoldsCreationParams foldsCreationParams(
        params,
        *(data.Learn->ObjectsData),
        startingApprox,
        /*isForWorkerLocalData*/ false
    );
    const ui32 foldCreationParamsCheckSum = foldsCreationParams.CalcCheckSum(
        *data.Learn->ObjectsGrouping,
        localExecutor
    );

    auto lossFunction = Params.LossFunctionDescription->GetLossFunction();

    UpdateCtrsTargetBordersOption(lossFunction, approxDimension, &Params.CatFeatureParams.Get());

    CtrsHelper.InitCtrHelper(
        Params.CatFeatureParams,
        *Layout,
        data.Learn->TargetData->GetTarget(),
        lossFunction,
        ObjectiveDescriptor,
        Params.DataProcessingOptions->AllowConstLabel
    );

    // TODO(akhropov): implement effective RecalcApprox for shrinked models instead of completely new context
    if (initLearnProgress &&
        (initLearnProgress->LearnAndTestQuantizedFeaturesCheckSum == featuresCheckSum) &&
        (initLearnProgress->FoldCreationParamsCheckSum == foldCreationParamsCheckSum) &&
        initLearnProgress->IsFoldsAndApproxDataValid)
    {
        CATBOOST_DEBUG_LOG << "Continue with init LearnProgress\n";

        LearnProgress = std::move(initLearnProgress);
    } else {
        const bool isTrainingContinuation = initModel || initLearnProgress;
        const bool datasetsCanContainBaseline = !isTrainingContinuation ||
            /* it should be legal to pass the same initial datasets even if they contains baseline data
             * to the training continuation.
             * In this case baseline data will be ignored.
            */
            (initLearnProgress &&
             (initLearnProgress->LearnAndTestQuantizedFeaturesCheckSum == featuresCheckSum));

        if (initLearnProgress) {
            CB_ENSURE_INTERNAL(
                initModel,
                "Initial learn progress was specified but parameters and/or learning datasets have been"
                "  changed so initial model is required as well to continue learning"
            );

            // destroy old LearnProgress to save resources
            initLearnProgress.Destroy();
        }

        CATBOOST_DEBUG_LOG << "Create new LearnProgress\n";

        LearnProgress = MakeHolder<TLearnProgress>(
            /*isForWorkerLocalData*/ false,
            Params.SystemOptions->IsSingleHost(),
            data,
            SafeIntegerCast<int>(approxDimension),
            labelConverter,
            Params.RandomSeed,
            initRand,
            foldsCreationParams,
            datasetsCanContainBaseline,
            CtrsHelper.GetTargetClassifiers(),
            featuresCheckSum,
            foldCreationParamsCheckSum,
            /*estimatedFeaturesQuantizationOptions*/
                params.DataProcessingOptions->FloatFeaturesBinarization.Get(),
            std::move(precomputedSingleOnlineCtrDataForSingleFold),
            params.ObliviousTreeOptions.Get(),
            initModel,
            initModelApplyCompatiblePools,
            LocalExecutor
        );
    }
    LearnProgress->SerializedTrainParams = ToString(Params);
    LearnProgress->EnableSaveLoadApprox = Params.SystemOptions->IsSingleHost();

    const ui32 maxBodyTailCount = Max(1, GetMaxBodyTailCount(LearnProgress->Folds));
    UseTreeLevelCachingFlag = NeedToUseTreeLevelCaching(Params, maxBodyTailCount, LearnProgress->ApproxDimension);
}


void TLearnContext::SaveProgress(std::function<void(IOutputStream*)> onSaveSnapshot) {
    if (!OutputOptions.SaveSnapshot()) {
        return;
    }
    const auto snapshotBackup = Files.SnapshotFile + ".bak";
    TProgressHelper(ToString(ETaskType::CPU)).Write(
        snapshotBackup,
        [&](IOutputStream* out) {
            onSaveSnapshot(out);
            ::SaveMany(out, *LearnProgress, Profile.DumpProfileInfo());
        }
    );
    TFsPath(snapshotBackup).ForceRenameTo(Files.SnapshotFile);
}

bool TLearnContext::TryLoadProgress(std::function<bool(IInputStream*)> onLoadSnapshot) {
    if (!OutputOptions.SaveSnapshot() || !NFs::Exists(Files.SnapshotFile)) {
        return false;
    }
    try {
        TProgressHelper(ToString(ETaskType::CPU)).CheckedLoad(
            Files.SnapshotFile,
            [&](TIFStream* in) {
                if (!onLoadSnapshot(in)) {
                    return;
                }
                // use progress copy to avoid partial deserialization of corrupted progress file
                THolder<TLearnProgress> learnProgressRestored = MakeHolder<TLearnProgress>(*LearnProgress);
                TProfileInfoData ProfileRestored;

                // fail here does nothing with real LearnProgress
                ::LoadMany(in, *learnProgressRestored, ProfileRestored);

                const bool paramsCompatible = NCatboostOptions::IsParamsCompatible(
                    learnProgressRestored->SerializedTrainParams,
                    LearnProgress->SerializedTrainParams);
                CATBOOST_DEBUG_LOG
                    << LabeledOutput(learnProgressRestored->SerializedTrainParams) << ' '
                    << LabeledOutput(LearnProgress->SerializedTrainParams) << Endl;
                CB_ENSURE(paramsCompatible, "Current training params differ from the params saved in snapshot");

                const bool poolCompatible
                    = (learnProgressRestored->LearnAndTestQuantizedFeaturesCheckSum
                       == LearnProgress->LearnAndTestQuantizedFeaturesCheckSum);
                CB_ENSURE(
                    poolCompatible,
                    "Current learn and test datasets differ from the datasets used for snapshot "
                    << LabeledOutput(learnProgressRestored->LearnAndTestQuantizedFeaturesCheckSum) << ' '
                    << LabeledOutput(LearnProgress->LearnAndTestQuantizedFeaturesCheckSum)
                );

                LearnProgress = std::move(learnProgressRestored);
                Profile.InitProfileInfo(std::move(ProfileRestored));
                LearnProgress->SerializedTrainParams = ToString(Params); // substitute real
                CATBOOST_INFO_LOG << "Loaded progress file containing " << LearnProgress->TreeStruct.size()
                    << " trees" << Endl;
            }
        );
        return true;
    } catch(const TCatBoostException& e) {
        ythrow TCatBoostException() << "Can't load progress from snapshot file: " << Files.SnapshotFile
            << " : " << e.what();
    } catch (...) {
        CATBOOST_WARNING_LOG << "Can't load progress from snapshot file: " << Files.SnapshotFile
            << " exception: " << CurrentExceptionMessage() << Endl;
        return false;
    }
}

TLearnProgress::TLearnProgress() : Rand(0) {
}

TLearnProgress::TLearnProgress(
    bool isForWorkerLocalData,
    bool isSingleHost,
    const TTrainingDataProviders& data,
    int approxDimension,
    const TLabelConverter& labelConverter,
    ui64 randomSeed,
    TMaybe<const TRestorableFastRng64*> initRand,
    const TFoldsCreationParams& foldsCreationParams,
    bool datasetsCanContainBaseline,
    const TVector<TTargetClassifier>& targetClassifiers,
    ui32 featuresCheckSum,
    ui32 foldCreationParamsCheckSum,
    const NCatboostOptions::TBinarizationOptions& estimatedFeaturesQuantizationOptions,
    TMaybe<TPrecomputedOnlineCtrData> precomputedSingleOnlineCtrDataForSingleFold,
    const NCatboostOptions::TObliviousTreeLearnerOptions& trainOptions,
    TMaybe<TFullModel*> initModel,
    NCB::TDataProviders initModelApplyCompatiblePools,
    NPar::ILocalExecutor* localExecutor)
    : StartingApprox(foldsCreationParams.StartingApprox)
    , IsAveragingFoldPermuted(foldsCreationParams.IsAverageFoldPermuted)
    , FoldCreationParamsCheckSum(foldCreationParamsCheckSum)
    , CatFeatures(CreateCatFeatures(*data.Learn->ObjectsData->GetFeaturesLayout()))
    , FloatFeatures(
        CreateFloatFeatures(
            *data.Learn->ObjectsData->GetFeaturesLayout(),
            *data.Learn->ObjectsData->GetQuantizedFeaturesInfo()
        )
      )
    , TextFeatures(
          // use OriginalFeaturesLayout because ObjectsData contains tokenized features
          CreateTextFeatures(*data.Learn->OriginalFeaturesLayout)
      )
    , EmbeddingFeatures(CreateEmbeddingFeatures(*data.Learn->ObjectsData->GetFeaturesLayout()))
    , ApproxDimension(approxDimension)
    , LearnAndTestQuantizedFeaturesCheckSum(featuresCheckSum)
    , Rand(randomSeed) {

    LabelConverter = labelConverter;

    if (initRand) {
        Rand.Advance((**initRand).GetCallCount());
    }

    const ui32 learnSampleCount = data.Learn->GetObjectCount();

    CB_ENSURE_INTERNAL(
        !isForWorkerLocalData || (foldsCreationParams.LearningFoldCount == 0),
        "foldsCreationParams.LearningFoldCount != 0 for worker local data"
    );

    TQuantizedFeaturesInfoPtr onlineEstimatedQuantizedFeaturesInfo;

    TIntrusivePtr<TPrecomputedOnlineCtr> precomputedSingleOnlineCtrs;
    if (precomputedSingleOnlineCtrDataForSingleFold) {
        precomputedSingleOnlineCtrs = MakeIntrusive<TPrecomputedOnlineCtr>();
        precomputedSingleOnlineCtrs->Data = *precomputedSingleOnlineCtrDataForSingleFold;
    }

    InitApproxes(learnSampleCount, StartingApprox, ApproxDimension, false, &AvrgApprox);

    if (learnSampleCount) {

        Folds.reserve(foldsCreationParams.LearningFoldCount);

        if (foldsCreationParams.IsOrderedBoosting) {
            for (int foldIdx = 0; foldIdx < foldsCreationParams.LearningFoldCount; ++foldIdx) {
                Folds.emplace_back(
                    TFold::BuildDynamicFold(
                        data,
                        targetClassifiers,
                        foldIdx != 0,
                        foldsCreationParams.FoldPermutationBlockSize,
                        ApproxDimension,
                        foldsCreationParams.FoldLenMultiplier,
                        foldsCreationParams.StoreExpApproxes,
                        foldsCreationParams.HasPairwiseWeights,
                        StartingApprox,
                        estimatedFeaturesQuantizationOptions,
                        onlineEstimatedQuantizedFeaturesInfo,
                        &Rand,
                        localExecutor
                    )
                );
                if (foldIdx == 0) {
                    onlineEstimatedQuantizedFeaturesInfo
                        = Folds.back().GetOnlineEstimatedFeatures().GetQuantizedFeaturesInfo();
                } else {
                    Folds.back().GetOnlineEstimatedFeatures().Test
                        = Folds[0].GetOnlineEstimatedFeatures().Test;
                }
            }
        } else {
            for (int foldIdx = 0; foldIdx < foldsCreationParams.LearningFoldCount; ++foldIdx) {
                Folds.emplace_back(
                    TFold::BuildPlainFold(
                        data,
                        targetClassifiers,
                        foldIdx != 0,
                        isSingleHost ? foldsCreationParams.FoldPermutationBlockSize : learnSampleCount,
                        ApproxDimension,
                        foldsCreationParams.StoreExpApproxes,
                        foldsCreationParams.HasPairwiseWeights,
                        StartingApprox,
                        estimatedFeaturesQuantizationOptions,
                        onlineEstimatedQuantizedFeaturesInfo,
                        (!isSingleHost || (foldIdx == 0)) ? precomputedSingleOnlineCtrs : nullptr,
                        &Rand,
                        localExecutor
                    )
                );
                if (foldIdx == 0) {
                    onlineEstimatedQuantizedFeaturesInfo
                        = Folds.back().GetOnlineEstimatedFeatures().GetQuantizedFeaturesInfo();
                } else {
                    Folds.back().GetOnlineEstimatedFeatures().Test
                        = Folds[0].GetOnlineEstimatedFeatures().Test;
                }
            }
        }

        TMaybeData<TConstArrayRef<TConstArrayRef<float>>> learnBaseline = data.Learn->TargetData->GetBaseline();
        if (learnBaseline) {
            CB_ENSURE(
                datasetsCanContainBaseline,
                "Specifying baseline for training continuation is not supported"
            );
            if (!initModel) {
                AssignRank2<float>(*learnBaseline, &AvrgApprox);
            }
        }

        const auto externalFeaturesCount = data.Learn->ObjectsData->GetFeaturesLayout()->GetExternalFeatureCount();
        const auto objectsCount = data.Learn->ObjectsData->GetObjectCount();
        UsedFeatures.resize(externalFeaturesCount, false);
        // for symmetric tree features usage is equal for all objects, so we don't need to store it for each object individually
        if (trainOptions.GrowPolicy.Get() != EGrowPolicy::SymmetricTree) {
            const auto& featurePenaltiesOptions = trainOptions.FeaturePenalties.Get();
            for (const auto[featureIdx, penalty] : featurePenaltiesOptions.PerObjectFeaturePenalty.Get()) {
                UsedFeaturesPerObject[featureIdx].resize(objectsCount, false);
            }
        }
    }

    AveragingFold = TFold::BuildPlainFold(
        data,
        targetClassifiers,
        foldsCreationParams.IsAverageFoldPermuted,
        /*permuteBlockSize=*/ isSingleHost ? foldsCreationParams.FoldPermutationBlockSize : learnSampleCount,
        ApproxDimension,
        foldsCreationParams.StoreExpApproxes,
        foldsCreationParams.HasPairwiseWeights,
        StartingApprox,
        estimatedFeaturesQuantizationOptions,
        onlineEstimatedQuantizedFeaturesInfo,
        precomputedSingleOnlineCtrs,
        &Rand,
        localExecutor
    );
    if (Folds.size() > 0) {
        AveragingFold.GetOnlineEstimatedFeatures().Test = Folds[0].GetOnlineEstimatedFeatures().Test;
    }

    ResizeRank2(data.Test.size(), ApproxDimension, TestApprox);
    for (size_t testIdx = 0; testIdx < data.Test.size(); ++testIdx) {
        const auto* testData = data.Test[testIdx].Get();
        if (testData == nullptr || testData->GetObjectCount() == 0) {
            continue;
        }
        TMaybeData<TConstArrayRef<TConstArrayRef<float>>> testBaseline = testData->TargetData->GetBaseline();
        if (!testBaseline) {
            for (auto approxDim : xrange(TestApprox[testIdx].size())) {
                TestApprox[testIdx][approxDim].resize(
                    testData->GetObjectCount(),
                    StartingApprox ? (*StartingApprox)[approxDim] : 0
                );
            }
        } else {
            CB_ENSURE(
                datasetsCanContainBaseline,
                "Specifying baseline for training continuation is not supported"
            );
            if (!initModel) {
                AssignRank2<float>(*testBaseline, &TestApprox[testIdx]);
            }
        }
    }

    if (initModel) {
        SetSeparateInitModel(
            **initModel,
            initModelApplyCompatiblePools,
            foldsCreationParams.IsOrderedBoosting,
            foldsCreationParams.StoreExpApproxes,
            localExecutor
        );
    }

    EstimatedFeaturesContext.FeatureEstimators = data.FeatureEstimators;
    EstimatedFeaturesContext.OfflineEstimatedFeaturesLayout
        = data.EstimatedObjectsData.QuantizedEstimatedFeaturesInfo.Layout;
    EstimatedFeaturesContext.OnlineEstimatedFeaturesLayout
        = AveragingFold.GetOnlineEstimatedFeaturesInfo().Layout; // must be equal for all folds
}


void TLearnProgress::SetSeparateInitModel(
    const TFullModel& initModel,
    const TDataProviders& initModelApplyCompatiblePools,
    bool isOrderedBoosting,
    bool storeExpApproxes,
    NPar::ILocalExecutor* localExecutor) {

    CATBOOST_DEBUG_LOG << "TLearnProgress::SetSeparateInitModel\n";

    SeparateInitModelTreesSize = SafeIntegerCast<ui32>(initModel.GetTreeCount());
    SeparateInitModelCheckSum = CalcCoreModelCheckSum(initModel);

    // Calc approxes

    auto calcApproxFunction = [&] (const TDataProvider& data) -> TVector<TVector<double>> {
        return ApplyModelMulti(
            initModel,
            *data.ObjectsData,
            EPredictionType::RawFormulaVal,
            0,
            SafeIntegerCast<int>(initModel.GetTreeCount()),
            localExecutor,
            data.RawTargetData.GetBaseline()
        );
    };

    TVector<std::function<void()>> tasks;

    tasks.push_back(
        [&] () {
            const ui32 learnObjectCount = initModelApplyCompatiblePools.Learn->GetObjectCount();
            if (!learnObjectCount) {
                return;
            }

            AvrgApprox = calcApproxFunction(*initModelApplyCompatiblePools.Learn);

            TVector<TConstArrayRef<double>> approxRef(AvrgApprox.begin(), AvrgApprox.end());

            TVector<std::function<void()>> tasks;

            auto setFoldApproxes = [&] (bool isPlainFold, TFold* fold) {
                for (auto& bodyTail : fold->BodyTailArr) {
                    InitApproxFromBaseline(
                        isPlainFold ? learnObjectCount : SafeIntegerCast<ui32>(bodyTail.TailFinish),
                        TConstArrayRef<TConstArrayRef<double>>(approxRef),
                        fold->GetLearnPermutationArray(),
                        storeExpApproxes,
                        &bodyTail.Approx);
                }
            };

            for (auto foldIdx : xrange(Folds.size())) {
                tasks.push_back(
                    [&, foldIdx] () {
                        setFoldApproxes(!isOrderedBoosting, &Folds[foldIdx]);
                    }
                );
            }
            tasks.push_back(
                [&] () {
                    setFoldApproxes(/*isPlainFold*/ true, &AveragingFold);
                }
            );

            ExecuteTasksInParallel(&tasks, localExecutor);
        }
    );

    for (auto testIdx : xrange(initModelApplyCompatiblePools.Test.size())) {
        tasks.push_back(
            [&, testIdx] () {
                TestApprox[testIdx] = calcApproxFunction(
                    *initModelApplyCompatiblePools.Test[testIdx]);
            }
        );
    }

    ExecuteTasksInParallel(&tasks, localExecutor);
}

void TLearnProgress::PrepareForContinuation() {
    InitTreesSize = SafeIntegerCast<ui32>(TreeStruct.size());
    MetricsAndTimeHistory = TMetricsAndTimeLeftHistory();
}

void TLearnProgress::Save(IOutputStream* s) const {
    CB_ENSURE_INTERNAL(IsFoldsAndApproxDataValid, "Attempt to save TLearnProgress data in inconsistent state");

    ::Save(s, SerializedTrainParams);
    ::Save(s, EnableSaveLoadApprox);
    if (EnableSaveLoadApprox) {
        ui64 foldCount = Folds.size();
        ::Save(s, foldCount);
        for (ui64 i = 0; i < foldCount; ++i) {
            Folds[i].SaveApproxes(s);
        }
        AveragingFold.SaveApproxes(s);
        ::Save(s, AvrgApprox);
    }
    ::SaveMany(
        s,
        TestApprox,
        BestTestApprox,
        CatFeatures,
        FloatFeatures,
        ApproxDimension,
        TreeStruct,
        TreeStats,
        LeafValues,
        ModelShrinkHistory,
        InitTreesSize,
        MetricsAndTimeHistory,
        UsedCtrSplits,
        LearnAndTestQuantizedFeaturesCheckSum,
        SeparateInitModelTreesSize,
        SeparateInitModelCheckSum,
        Rand,
        StartingApprox,
        UsedFeatures,
        UsedFeaturesPerObject
    );
}

void TLearnProgress::Load(IInputStream* s) {
    ::Load(s, SerializedTrainParams);
    ::Load(s, EnableSaveLoadApprox);
    if (EnableSaveLoadApprox) {
        ui64 foldCount;
        ::Load(s, foldCount);
        Folds.resize(foldCount);
        for (ui64 i = 0; i < foldCount; ++i) {
            Folds[i].LoadApproxes(s);
        }
        AveragingFold.LoadApproxes(s);
        ::Load(s, AvrgApprox);
    }
    ::LoadMany(
        s,
        TestApprox,
        BestTestApprox,
        CatFeatures,
        FloatFeatures,
        ApproxDimension,
        TreeStruct,
        TreeStats,
        LeafValues,
        ModelShrinkHistory,
        InitTreesSize,
        MetricsAndTimeHistory,
        UsedCtrSplits,
        LearnAndTestQuantizedFeaturesCheckSum,
        SeparateInitModelTreesSize,
        SeparateInitModelCheckSum,
        Rand,
        StartingApprox,
        UsedFeatures,
        UsedFeaturesPerObject
    );
}

ui32 TLearnProgress::GetCurrentTrainingIterationCount() const {
    return SafeIntegerCast<ui32>(TreeStruct.size()) - InitTreesSize;
}

ui32 TLearnProgress::GetCompleteModelTreesSize() const {
    return SeparateInitModelTreesSize + SafeIntegerCast<ui32>(TreeStruct.size());
}

ui32 TLearnProgress::GetInitModelTreesSize() const {
    return SeparateInitModelTreesSize + InitTreesSize;
}

TQuantizedEstimatedFeaturesInfo TLearnProgress::GetOnlineEstimatedFeaturesInfo() const {
    return AveragingFold.GetOnlineEstimatedFeaturesInfo();
}


bool TLearnContext::UseTreeLevelCaching() const {
    return UseTreeLevelCachingFlag;
}

bool TLearnContext::GetHasWeights() const {
    return HasWeights;
}

bool NeedToUseTreeLevelCaching(
    const NCatboostOptions::TCatBoostOptions& params,
    ui32 maxBodyTailCount,
    ui32 approxDimension) {

    const ui32 maxLeafCount = 1 << params.ObliviousTreeOptions->MaxDepth;
    // TODO(nikitxskv): Pairwise scoring doesn't use statistics from previous tree level. Need to fix it.
    return (
        IsSamplingPerTree(params.ObliviousTreeOptions) &&
        !IsPairwiseScoring(params.LossFunctionDescription->GetLossFunction()) &&
        maxLeafCount * approxDimension * maxBodyTailCount < 64 * 1 * 10);
}

bool UseAveragingFoldAsFoldZero(const TLearnContext& ctx) {
    const auto lossFunction = ctx.Params.LossFunctionDescription->GetLossFunction();
    const bool usePairs = UsesPairsForCalculation(lossFunction);
    const bool isPlainBoosting = ctx.Params.BoostingOptions->BoostingType == EBoostingType::Plain;
    return isPlainBoosting && !ctx.LearnProgress->IsAveragingFoldPermuted && !usePairs;
}
