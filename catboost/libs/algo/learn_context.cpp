#include "learn_context.h"

#include "apply.h"
#include "approx_dimension.h"
#include "approx_updater_helpers.h"
#include "calc_score_cache.h"
#include "error_functions.h"
#include "helpers.h"
#include "online_ctr.h"

#include <catboost/libs/distributed/master.h>
#include <catboost/libs/helpers/checksum.h>
#include <catboost/libs/helpers/parallel_tasks.h>
#include <catboost/libs/helpers/progress_helper.h>
#include <catboost/libs/helpers/vector_helpers.h>
#include <catboost/libs/index_range/index_range.h>
#include <catboost/libs/model/model.h>
#include <catboost/libs/options/defaults_helper.h>

#include <library/digest/crc32c/crc32c.h>
#include <library/digest/md5/md5.h>
#include <library/threading/local_executor/local_executor.h>

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
    bool isForWorkerLocalData)
    : IsOrderedBoosting(!IsPlainMode(params.BoostingOptions->BoostingType))
    , LearningFoldCount(0) // properly inited below
    , FoldPermutationBlockSize(0) // properly inited below
    , StoreExpApproxes(IsStoreExpApprox(params.LossFunctionDescription->GetLossFunction()))
    , HasPairwiseWeights(UsesPairsForCalculation(params.LossFunctionDescription->GetLossFunction()))
    , FoldLenMultiplier(params.BoostingOptions->FoldLenMultiplier)
    , IsAverageFoldPermuted(false) // properly inited below
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
    NPar::TLocalExecutor* localExecutor) const {

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
        catFeature.UsedInModel,
        catFeature.FeatureIndex,
        catFeature.FlatFeatureIndex
    );
}

static inline ui32 UpdateCheckSumImpl(ui32 init, const TFloatFeature& floatFeature) {
    return UpdateCheckSum(
        init,
        floatFeature.HasNans,
        floatFeature.FeatureIndex,
        floatFeature.FlatFeatureIndex,
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
    const auto& trees = model.ObliviousTrees;

    return UpdateCheckSum(
        ui32(0),
        trees.ApproxDimension,
        trees.TreeSplits,
        trees.TreeSizes,
        trees.TreeStartOffsets,
        trees.NonSymmetricStepNodes,
        trees.NonSymmetricNodeIdToLeafId,
        trees.LeafValues,
        trees.CatFeatures,
        trees.FloatFeatures,
        trees.OneHotFeatures,
        trees.CtrFeatures
    );
}


TLearnContext::TLearnContext(
    const NCatboostOptions::TCatBoostOptions& params,
    const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
    const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
    const NCatboostOptions::TOutputFilesOptions& outputOptions,
    const TTrainingForCPUDataProviders& data,
    const TLabelConverter& labelConverter,
    TMaybe<const TRestorableFastRng64*> initRand,
    TMaybe<TFullModel*> initModel,
    THolder<TLearnProgress> initLearnProgress, // will be modified if not non-nullptr
    NCB::TDataProviders initModelApplyCompatiblePools,
    NPar::TLocalExecutor* localExecutor,
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
    , LearnAndTestDataPackingAreCompatible(false)
    , UseTreeLevelCachingFlag(false) {

    ETaskType taskType = Params.GetTaskType();
    CB_ENSURE(taskType == ETaskType::CPU, "Error: expect learn on CPU task type, got " << taskType);

    THPTimer calcHashTimer;
    ui32 featuresCheckSum = data.CalcFeaturesCheckSum(localExecutor);
    CATBOOST_DEBUG_LOG << "Features checksum calculation time: " << calcHashTimer.Passed() << Endl;

    ui32 approxDimension = GetApproxDimension(Params, labelConverter);
    if (initLearnProgress) {
        CB_ENSURE(
            approxDimension == SafeIntegerCast<ui32>(initLearnProgress->ApproxDimension),
            "Attempt to continue learning with a different approx dimension"
        );
        if (approxDimension > 1) {
            CB_ENSURE(
                labelConverter == initLearnProgress->LabelConverter,
                "Attempt to continue learning with different class labels"
            );
        }
    }

    const TFoldsCreationParams foldsCreationParams(
        params,
        *(data.Learn->ObjectsData),
        /*isForWorkerLocalData*/ false);
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
            initModel,
            initModelApplyCompatiblePools,
            LocalExecutor
        );
    }
    LearnProgress->SerializedTrainParams = ToString(Params);
    LearnProgress->EnableSaveLoadApprox = Params.SystemOptions->IsSingleHost();

    LearnAndTestDataPackingAreCompatible = true;
    for (const auto& testData : data.Test) {
        if (!testData->ObjectsData->IsPackingCompatibleWith(*data.Learn->ObjectsData)) {
            LearnAndTestDataPackingAreCompatible = false;
            break;
        }
    }

    const ui32 maxBodyTailCount = Max(1, GetMaxBodyTailCount(LearnProgress->Folds));
    UseTreeLevelCachingFlag = NeedToUseTreeLevelCaching(Params, maxBodyTailCount, LearnProgress->ApproxDimension);
}


TLearnContext::~TLearnContext() {
    if (Params.SystemOptions->IsMaster()) {
        FinalizeMaster(this);
    }
}

void TLearnContext::OutputMeta() {
    auto losses = CreateMetrics(
        Params.MetricOptions,
        EvalMetricDescriptor,
        LearnProgress->ApproxDimension
    );

    CreateMetaFile(Files, OutputOptions, GetConstPointers(losses), Params.BoostingOptions->IterationCount);
}

void TLearnContext::SaveProgress() {
    if (!OutputOptions.SaveSnapshot()) {
        return;
    }
    TProgressHelper(ToString(ETaskType::CPU)).Write(
        Files.SnapshotFile,
        [&](IOutputStream* out) {
            ::SaveMany(out, *LearnProgress, Profile.DumpProfileInfo());
        }
    );
}

bool TLearnContext::TryLoadProgress() {
    if (!OutputOptions.SaveSnapshot() || !NFs::Exists(Files.SnapshotFile)) {
        return false;
    }
    try {
        TProgressHelper(ToString(ETaskType::CPU)).CheckedLoad(
            Files.SnapshotFile,
            [&](TIFStream* in) {
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
                    "Current learn and test datasets differ from the datasets used for snapshot"
                    LabeledOutput(
                        learnProgressRestored->LearnAndTestQuantizedFeaturesCheckSum,
                        LearnProgress->LearnAndTestQuantizedFeaturesCheckSum
                    )
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


TLearnProgress::TLearnProgress(
    bool isForWorkerLocalData,
    bool isSingleHost,
    const TTrainingForCPUDataProviders& data,
    int approxDimension,
    const TLabelConverter& labelConverter,
    ui64 randomSeed,
    TMaybe<const TRestorableFastRng64*> initRand,
    const TFoldsCreationParams& foldsCreationParams,
    bool datasetsCanContainBaseline,
    const TVector<TTargetClassifier>& targetClassifiers,
    ui32 featuresCheckSum,
    ui32 foldCreationParamsCheckSum,
    TMaybe<TFullModel*> initModel,
    NCB::TDataProviders initModelApplyCompatiblePools,
    NPar::TLocalExecutor* localExecutor)
    : FoldCreationParamsCheckSum(foldCreationParamsCheckSum)
    , CatFeatures(CreateCatFeatures(*data.Learn->ObjectsData->GetQuantizedFeaturesInfo()))
    , FloatFeatures(CreateFloatFeatures(*data.Learn->ObjectsData->GetQuantizedFeaturesInfo()))
    , ApproxDimension(approxDimension)
    , LearnAndTestQuantizedFeaturesCheckSum(featuresCheckSum)
    , Rand(randomSeed) {

    if (ApproxDimension > 1) {
        LabelConverter = labelConverter;
    }

    if (initRand) {
        Rand.Advance((**initRand).GetCallCount());
    }

    const ui32 learnSampleCount = data.Learn->GetObjectCount();

    CB_ENSURE_INTERNAL(
        !isForWorkerLocalData || (foldsCreationParams.LearningFoldCount == 0),
        "foldsCreationParams.LearningFoldCount != 0 for worker local data"
    );

    Folds.reserve(foldsCreationParams.LearningFoldCount);

    if (foldsCreationParams.IsOrderedBoosting) {
        for (int foldIdx = 0; foldIdx < foldsCreationParams.LearningFoldCount; ++foldIdx) {
            Folds.emplace_back(
                TFold::BuildDynamicFold(
                    *data.Learn,
                    targetClassifiers,
                    foldIdx != 0,
                    foldsCreationParams.FoldPermutationBlockSize,
                    ApproxDimension,
                    foldsCreationParams.FoldLenMultiplier,
                    foldsCreationParams.StoreExpApproxes,
                    foldsCreationParams.HasPairwiseWeights,
                    &Rand,
                    localExecutor
                )
            );
        }
    } else {
        for (int foldIdx = 0; foldIdx < foldsCreationParams.LearningFoldCount; ++foldIdx) {
            Folds.emplace_back(
                TFold::BuildPlainFold(
                    *data.Learn,
                    targetClassifiers,
                    foldIdx != 0,
                    isSingleHost ? foldsCreationParams.FoldPermutationBlockSize : learnSampleCount,
                    ApproxDimension,
                    foldsCreationParams.StoreExpApproxes,
                    foldsCreationParams.HasPairwiseWeights,
                    &Rand,
                    localExecutor
                )
            );
        }
    }

    AveragingFold = TFold::BuildPlainFold(
        *data.Learn,
        targetClassifiers,
        foldsCreationParams.IsAverageFoldPermuted,
        /*permuteBlockSize=*/ isSingleHost ? foldsCreationParams.FoldPermutationBlockSize : learnSampleCount,
        ApproxDimension,
        foldsCreationParams.StoreExpApproxes,
        foldsCreationParams.HasPairwiseWeights,
        &Rand,
        localExecutor
    );

    AvrgApprox.resize(ApproxDimension, TVector<double>(learnSampleCount));
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
    ResizeRank2(data.Test.size(), ApproxDimension, TestApprox);
    for (size_t testIdx = 0; testIdx < data.Test.size(); ++testIdx) {
        const auto* testData = data.Test[testIdx].Get();
        if (testData == nullptr || testData->GetObjectCount() == 0) {
            continue;
        }
        TMaybeData<TConstArrayRef<TConstArrayRef<float>>> testBaseline = testData->TargetData->GetBaseline();
        if (!testBaseline) {
            for (auto& approxDim : TestApprox[testIdx]) {
                approxDim.resize(testData->GetObjectCount());
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
}


void TLearnProgress::SetSeparateInitModel(
    const TFullModel& initModel,
    const TDataProviders& initModelApplyCompatiblePools,
    bool isOrderedBoosting,
    bool storeExpApproxes,
    NPar::TLocalExecutor* localExecutor) {

    CATBOOST_DEBUG_LOG << "TLearnProgress::SetSeparateInitModel\n";

    SeparateInitModelTreesSize = SafeIntegerCast<ui32>(initModel.GetTreeCount());
    SeparateInitModelCheckSum = CalcCoreModelCheckSum(initModel);

    // Calc approxes

    auto calcApproxFunction = [&] (const TObjectsDataProvider& objectsData) -> TVector<TVector<double>> {
        // needed for ApplyModelMulti
        TIntrusiveConstPtr<TObjectsDataProvider> objectsDataWithConsecutiveFeaturesData;
        TMaybe<TVector<ui32>> srcPermutation;
        if (const auto* rawObjectsData = dynamic_cast<const TRawObjectsDataProvider*>(&objectsData)) {
            objectsDataWithConsecutiveFeaturesData
                = rawObjectsData->GetWithPermutedConsecutiveArrayFeaturesData(localExecutor, &srcPermutation);
        } else if (const auto* quantizedObjectsData
                       = dynamic_cast<const TQuantizedObjectsDataProvider*>(&objectsData))
        {
            objectsDataWithConsecutiveFeaturesData
                = quantizedObjectsData->GetWithPermutedConsecutiveArrayFeaturesData(
                    localExecutor,
                    &srcPermutation
                );
        } else {
            CB_ENSURE_INTERNAL(false, "Unknown ObjectsDataProvider type");
        }

        TVector<TVector<double>> approx = ApplyModelMulti(
            initModel,
            *objectsDataWithConsecutiveFeaturesData,
            EPredictionType::RawFormulaVal,
            0,
            SafeIntegerCast<int>(initModel.GetTreeCount()),
            localExecutor
        );

        if (srcPermutation) {
            CATBOOST_DEBUG_LOG << "srcPermutation present\n";

            TConstArrayRef<ui32> srcPermutationArray = *srcPermutation;
            const int objectCount = SafeIntegerCast<int>(approx.at(0).size());

            TVector<TVector<double>> resultApprox(approx.size());

            localExecutor->ExecRangeWithThrow(
                [&] (int approxDimension) {
                    resultApprox[approxDimension].yresize(objectCount);
                    TConstArrayRef<double> srcArray = approx[approxDimension];
                    TArrayRef<double> resultArray = resultApprox[approxDimension];
                    NPar::ParallelFor(
                        *localExecutor,
                        0,
                        objectCount,
                        [resultArray, srcPermutationArray, srcArray] (int i) {
                            resultArray[srcPermutationArray[i]] = srcArray[i];
                        }
                    );
                },
                0,
                SafeIntegerCast<int>(approx.size()),
                NPar::TLocalExecutor::WAIT_COMPLETE
            );
            return resultApprox;
        } else {
            return approx;
        }
    };

    TVector<std::function<void()>> tasks;

    tasks.push_back(
        [&] () {
            const ui32 learnObjectCount = initModelApplyCompatiblePools.Learn->GetObjectCount();

            AvrgApprox = calcApproxFunction(*initModelApplyCompatiblePools.Learn->ObjectsData);

            TVector<TConstArrayRef<double>> approxRef(AvrgApprox.begin(), AvrgApprox.end());

            TVector<std::function<void()>> tasks;

            auto setFoldApproxes = [&] (bool isPlainFold, TFold* fold) {
                for (auto& bodyTail : fold->BodyTailArr) {
                    InitApproxFromBaseline(
                        isPlainFold ? ui32(0) : SafeIntegerCast<ui32>(bodyTail.BodyFinish),
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
                    *initModelApplyCompatiblePools.Test[testIdx]->ObjectsData);
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
        ::SaveMany(s, AvrgApprox);
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
        InitTreesSize,
        MetricsAndTimeHistory,
        UsedCtrSplits,
        LearnAndTestQuantizedFeaturesCheckSum,
        SeparateInitModelTreesSize,
        SeparateInitModelCheckSum,
        Rand
    );
}

void TLearnProgress::Load(IInputStream* s) {
    ::Load(s, SerializedTrainParams);
    bool enableSaveLoadApprox;
    ::Load(s, enableSaveLoadApprox);
    CB_ENSURE(enableSaveLoadApprox == EnableSaveLoadApprox, "Cannot load progress from file");
    if (EnableSaveLoadApprox) {
        ui64 foldCount;
        ::Load(s, foldCount);
        CB_ENSURE(foldCount == Folds.size(), "Cannot load progress from file");
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
        InitTreesSize,
        MetricsAndTimeHistory,
        UsedCtrSplits,
        LearnAndTestQuantizedFeaturesCheckSum,
        SeparateInitModelTreesSize,
        SeparateInitModelCheckSum,
        Rand
    );
}

ui32 TLearnProgress::GetCurrentTrainingIterationCount() const {
    return SafeIntegerCast<ui32>(TreeStruct.size()) - InitTreesSize;
}

ui32 TLearnProgress::GetCompleteModelTreesSize() const {
    return SeparateInitModelTreesSize + SafeIntegerCast<ui32>(TreeStruct.size());
}


bool TLearnContext::UseTreeLevelCaching() const {
    return UseTreeLevelCachingFlag;
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
