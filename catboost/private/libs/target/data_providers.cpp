#include "data_providers.h"

#include "binarize_target.h"
#include "calc_class_weights.h"
#include "target_converter.h"

#include <catboost/libs/helpers/connected_components.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/logging/logging.h>
#include <catboost/libs/metrics/metric.h>
#include <catboost/private/libs/options/enum_helpers.h>
#include <catboost/private/libs/options/json_helper.h>
#include <catboost/private/libs/options/class_label_options.h>
#include <catboost/private/libs/options/data_processing_options.h>
#include <catboost/private/libs/pairs/util.h>

#include <util/generic/algorithm.h>
#include <util/generic/cast.h>
#include <util/generic/mapfindptr.h>
#include <util/generic/maybe.h>
#include <util/generic/xrange.h>
#include <util/generic/ymath.h>
#include <util/stream/labeled.h>


namespace NCB {

    inline bool IsSafeTarget(float value) {
        return Abs(value) < 1e6f;
    }


    // can be empty if target data is unavailable
    static TVector<TSharedVector<float>> ConvertTarget(
        TMaybeData<TConstArrayRef<TRawTarget>> maybeRawTarget,
        ERawTargetType targetType,
        bool isRealTarget,
        bool isClass,
        bool isMultiClass,
        TMaybe<float> targetBorder,
        bool classCountUnknown,
        const TVector<NJson::TJsonValue> inputClassLabels,
        TVector<NJson::TJsonValue>* outputClassLabels,
        NPar::TLocalExecutor* localExecutor,
        ui32* classCount)
    {
        if (!maybeRawTarget) {
            return {};
        }

        auto rawTarget = *maybeRawTarget;

        THolder<ITargetConverter> targetConverter = MakeTargetConverter(
            isRealTarget,
            isClass,
            isMultiClass,
            targetBorder,
            classCountUnknown ? Nothing() : TMaybe<ui32>(*classCount),
            inputClassLabels);

        const auto targetDim = rawTarget.size();
        TVector<TSharedVector<float>> trainingTarget(targetDim);
        for (auto targetIdx : xrange(targetDim)) {
            trainingTarget[targetIdx] = MakeAtomicShared<TVector<float>>(TVector<float>());
        }

        // TODO(akhropov): Process multidimensional target columns in parallel when possible
        for (auto targetIdx : xrange(targetDim)) {
            *trainingTarget[targetIdx] = targetConverter->Process(targetType, rawTarget[targetIdx], localExecutor);
        }

        if (isMultiClass && classCountUnknown) {
            *classCount = targetConverter->GetClassCount();
        }
        TMaybe<TVector<NJson::TJsonValue>> maybeOutputClassLabels = targetConverter->GetClassLabels();
        if (maybeOutputClassLabels) {
            *outputClassLabels = std::move(*maybeOutputClassLabels);
        }

        return trainingTarget;
    }

    static void CheckPreprocessedTarget(
        TConstArrayRef<TConstArrayRef<float>> convertedTarget,
        TStringBuf datasetName,
        bool isNonEmptyAndNonConst,
        bool allowConstLabel,
        bool needCheckTarget,
        TConstArrayRef<NCatboostOptions::TLossDescription> metricDescriptions)
    {
        if (convertedTarget.empty() || convertedTarget[0].empty()) {
            return;
        }

        if (needCheckTarget) {
            for (auto targetIdx : xrange(convertedTarget.size())) {
                for (auto objectIdx : xrange(convertedTarget[0].size())) {
                    const float value = convertedTarget[targetIdx][objectIdx];
                    if (!IsSafeTarget(value)) {
                        CATBOOST_WARNING_LOG
                            << "Got unsafe target "
                            << LabeledOutput(value)
                            << " at object #" << objectIdx << " of dataset " << datasetName << Endl;
                        break;
                    }
                }
            }
        }

        for (const auto& metricDescription : metricDescriptions) {
            for (const auto& target : convertedTarget) {
                ::CheckPreprocessedTarget(
                    target,
                    metricDescription,
                    isNonEmptyAndNonConst,
                    allowConstLabel
                );
            }
        }
    }

    TSharedWeights<float> MakeClassificationWeights(
        const TWeights<float>& rawWeights,
        const TWeights<float>& rawGroupWeights,
        ui32 classCount,
        bool isForGpu,

        // [objectIdx], should contain integer values, can be empty
        TMaybe<TConstArrayRef<float>> targetClasses,
        TConstArrayRef<float> classWeights, // [classIdx], empty if not specified
        NPar::TLocalExecutor* localExecutor)
    {
        CheckDataSize(classWeights.size(), (size_t)classCount, "class weights size", true, "class count");
        Y_VERIFY(!targetClasses || ((size_t)rawWeights.GetSize() == targetClasses->size()));

        if (classWeights.empty() && rawGroupWeights.IsTrivial()) {
            if (isForGpu && rawWeights.IsTrivial()) {
                // TODO(akhropov): make GPU also support trivial TWeights

                return MakeIntrusive<TWeights<float>>(TVector<float>(rawWeights.GetSize(), 1.0f));
            }
            return MakeIntrusive<TWeights<float>>(rawWeights);
        }

        CB_ENSURE(targetClasses, "Class weights have been specified but target class data is unavailable");

        TVector<float> classAdjustedWeights;
        classAdjustedWeights.yresize(rawWeights.GetSize());

        TConstArrayRef<float> targetClassesArray = *targetClasses; // optimization

        localExecutor->ExecRangeBlockedWithThrow(
            [&] (int i) {
                // TODO(annaveronika): check class weight not negative.
                CB_ENSURE(
                    (size_t)targetClassesArray[i] < classWeights.size(),
                    "class " + ToString((size_t)targetClassesArray[i]) + " is missing in class weights"
                );
                classAdjustedWeights[i]
                    = rawWeights[i]*rawGroupWeights[i]*classWeights[(size_t)targetClassesArray[i]];
            },
            0,
            (int)rawWeights.GetSize(),
            0,
            NPar::TLocalExecutor::WAIT_COMPLETE
        );

        return MakeIntrusive<TWeights<float>>(std::move(classAdjustedWeights));
    }

    TSharedWeights<float> MakeWeights(
        const TWeights<float>& rawWeights,
        const TWeights<float>& rawGroupWeights,
        bool isForGpu,
        NPar::TLocalExecutor* localExecutor
    ) {
        if (!isForGpu) {
            // TODO(akhropov): make GPU also support trivial TWeights

            if (rawGroupWeights.IsTrivial()) {
                return MakeIntrusive<TWeights<float>>(rawWeights);
            }
            if (rawWeights.IsTrivial()) {
                return MakeIntrusive<TWeights<float>>(rawGroupWeights);
            }
        }

        TVector<float> groupAdjustedWeights;
        groupAdjustedWeights.yresize(rawWeights.GetSize());

        localExecutor->ExecRangeBlockedWithThrow(
            [&] (int i) {
                groupAdjustedWeights[i] = rawWeights[i]*rawGroupWeights[i];
            },
            0,
            (int)rawWeights.GetSize(),
            0,
            NPar::TLocalExecutor::WAIT_COMPLETE
        );

        return MakeIntrusive<TWeights<float>>(std::move(groupAdjustedWeights));
    }



    TVector<TSharedVector<float>> MakeBaselines(TMaybeData<TBaselineArrayRef> baselines, ui32 classCount) {
        if (!baselines) {
            return {};
        }

        CheckDataSize(baselines->size(), (size_t)classCount, "baseline count", false, "classes count");

        TVector<TSharedVector<float>> result;
        for (auto baselineArray : *baselines) {
            result.push_back(MakeAtomicShared<TVector<float>>(baselineArray.begin(), baselineArray.end()));
        }
        return result;
    }

    TVector<TSharedVector<float>> MakeOneBaseline(TMaybeData<TBaselineArrayRef> baselines) {
        if (!baselines) {
            return {};
        }

        CheckDataSize(baselines->size(), (size_t)1, "baselines", false, "expected dimension");

        return TVector<TSharedVector<float>>(
            1,
            MakeAtomicShared<TVector<float>>((*baselines)[0].begin(), (*baselines)[0].end())
        );
    }

    TVector<TPair> GeneratePairs(
        const TObjectsGrouping& objectsGrouping,
        TConstArrayRef<float> targetData,
        int maxPairsCount,
        TRestorableFastRng64* rand)
    {
        CB_ENSURE(
            targetData,
            "Pool labels are not provided. Cannot generate pairs."
        );

        auto minMaxTarget = MinMaxElement(targetData.begin(), targetData.end());
        CB_ENSURE(
            *minMaxTarget.first != *minMaxTarget.second,
            "Target data is constant. Cannot generate pairs."
        );

        TVector<TPair> result;

        GeneratePairLogitPairs(
            objectsGrouping,
            targetData,
            maxPairsCount,
            rand,
            &result);

        return result;
    }


    TSharedVector<TQueryInfo> MakeGroupInfos(
        const TObjectsGrouping& objectsGrouping,
        TMaybeData<TConstArrayRef<TSubgroupId>> subgroupIds,
        const TWeights<float>& groupWeights,
        TConstArrayRef<TPair> pairs) // can be empty
    {
        CB_ENSURE(!objectsGrouping.IsTrivial(), "Groupwise loss/metrics require nontrivial groups");

        auto groupsBounds = objectsGrouping.GetNonTrivialGroups();

        TVector<TQueryInfo> result(groupsBounds.begin(), groupsBounds.end());

        TVector<ui32> objectToGroupIdxMap; // [objectIdx]->groupIdx  initialized only for pairs
        if (!pairs.empty()) {
            objectToGroupIdxMap.yresize(objectsGrouping.GetObjectCount());
        }

        for (auto groupIdx : xrange((ui32)groupsBounds.size())) {
            auto& group = result[groupIdx];

            if (group.GetSize()) {
                group.Weight = groupWeights[group.Begin];
                if (subgroupIds) {
                    group.SubgroupId.yresize(group.GetSize());
                    const auto subgroupIdsData = *subgroupIds;
                    for (auto objectInGroupIdx : xrange(group.GetSize())) {
                        group.SubgroupId[objectInGroupIdx] = subgroupIdsData[group.Begin + objectInGroupIdx];
                    }
                }
                if (!pairs.empty()) {
                    group.Competitors.resize(group.GetSize());
                    for (auto objectIdx : group.Iter()) {
                        objectToGroupIdxMap[objectIdx] = groupIdx;
                    }
                }
            }
        }

        if (!pairs.empty()) {
            for (const auto& pair : pairs) {
                ui32 groupIdx = objectToGroupIdxMap[pair.WinnerId];
                /* it has been already checked on RawTargetData creation that WinnerId and LoserId
                  belong to the same group, so don't recheck it in non-debug here
                */
                Y_ASSERT(objectToGroupIdxMap[pair.LoserId] == groupIdx);

                auto& group = result[groupIdx];
                group.Competitors[pair.WinnerId - group.Begin].emplace_back(
                    pair.LoserId - group.Begin,
                    pair.Weight);
            }
        }

        return MakeAtomicShared<TVector<TQueryInfo>>(std::move(result));
    }


    TTargetCreationOptions MakeTargetCreationOptions(
        const TRawTargetDataProvider& rawData,
        TConstArrayRef<NCatboostOptions::TLossDescription> metricDescriptions,
        TMaybe<ui32> knownModelApproxDimension,
        const TInputClassificationInfo& inputClassificationInfo
    ) {
        CB_ENSURE(!metricDescriptions.empty(), "No metrics specified");

        auto isAnyOfMetrics = [&](bool predicate(ELossFunction)) {
            return AnyOf(
                metricDescriptions,
                [&](const NCatboostOptions::TLossDescription& metricDescription) -> bool {
                    return predicate(metricDescription.LossFunction.Get());
                }
            );
        };

        bool hasClassificationOnlyMetrics = isAnyOfMetrics(IsClassificationOnlyMetric);
        bool hasBinClassOnlyMetrics = isAnyOfMetrics(IsBinaryClassOnlyMetric);
        bool hasMultiClassOnlyMetrics = isAnyOfMetrics(IsMultiClassOnlyMetric);
        bool hasMultiRegressionMetrics = isAnyOfMetrics(IsMultiRegressionMetric);
        bool hasGroupwiseMetrics = isAnyOfMetrics(IsGroupwiseMetric);
        bool hasUserDefinedMetrics = isAnyOfMetrics(IsUserDefined);

        if (!rawData.GetWeights().IsTrivial() && isAnyOfMetrics(UsesPairsForCalculation)) {
            CATBOOST_WARNING_LOG << "Pairwise losses don't support object weights." << '\n';
        }

        CB_ENSURE(
            !(hasBinClassOnlyMetrics && hasMultiClassOnlyMetrics),
            "Both binary classification-only and multiclassification-only loss function or metrics"
            " specified"
        );

        TMaybe<ui32> knownClassCount = inputClassificationInfo.KnownClassCount;
        bool classTargetData = (
            hasClassificationOnlyMetrics ||
            knownClassCount ||
            (inputClassificationInfo.ClassWeights.size() > 0) ||
            (inputClassificationInfo.ClassLabels.size() > 0) ||
            inputClassificationInfo.TargetBorder
        );

        bool multiClassTargetData = false;

        if (knownModelApproxDimension) {
            if (*knownModelApproxDimension == 1) {
                CB_ENSURE(
                    !hasMultiClassOnlyMetrics,
                    "Multiclassification-only metrics specified for a single-dimensional model"
                );
                multiClassTargetData = false;
            } else {
                for (const auto& metricDescription : metricDescriptions) {
                    auto metricLossFunction = metricDescription.GetLossFunction();
                    CB_ENSURE(
                        IsMultiClassCompatibleMetric(metricLossFunction) || IsMultiRegressionMetric(metricLossFunction),
                        "Non-Multiclassification and Non-Multiregression compatible metric (" << metricLossFunction
                        << ") specified for a multidimensional model"
                    );
                }
                multiClassTargetData = !hasMultiRegressionMetrics;
                if (multiClassTargetData && !knownClassCount) {
                    // because there might be missing classes in train
                    knownClassCount = *knownModelApproxDimension;
                }
            }
        } else if (hasMultiClassOnlyMetrics ||
            (knownClassCount && *knownClassCount > 2) ||
            (inputClassificationInfo.ClassWeights.size() > 2) ||
            (inputClassificationInfo.ClassLabels.size() > 2))
        {
            multiClassTargetData = true;
        }

        TMaybe<ui32> maxPairsCount;
        for (const auto& metricDescription : metricDescriptions) {
            if (IsPairwiseMetric(metricDescription.GetLossFunction())) {
                ui32 localMaxPairsCount = (ui32)NCatboostOptions::GetMaxPairCount(metricDescription);
                if (maxPairsCount) {
                    CB_ENSURE(
                        localMaxPairsCount == *maxPairsCount,
                        "Cannot generate pairs consistently - different metrics have different number"
                        "of max pairs"
                    );
                } else {
                    maxPairsCount = localMaxPairsCount;
                }
            }
        }

        ui32 classCount = (ui32)GetClassesCount(knownClassCount.GetOrElse(0), inputClassificationInfo.ClassLabels);
        TTargetCreationOptions options = {
            /*IsClass*/ classTargetData,
            /*IsMultiClass*/ multiClassTargetData,
            /*CreateBinClassTarget*/ (
                hasBinClassOnlyMetrics
                || (!hasMultiRegressionMetrics && !hasMultiClassOnlyMetrics && !multiClassTargetData && classCount == 2)
            ),
            /*CreateMultiClassTarget*/ (
                hasMultiClassOnlyMetrics
                || (!hasMultiRegressionMetrics && !hasBinClassOnlyMetrics && (multiClassTargetData || classCount > 2))
            ),
            /*CreateGroups*/ (
                hasGroupwiseMetrics
                || (!rawData.GetObjectsGrouping()->IsTrivial() && hasUserDefinedMetrics)
            ),
            /*CreatePairs*/ isAnyOfMetrics(IsPairwiseMetric),
            /*MaxPairsCount*/ maxPairsCount
        };
        return options;
    }


    void CheckTargetConsistency(
        TTargetDataProviderPtr targetDataProvider,
        TConstArrayRef<NCatboostOptions::TLossDescription> metricDescriptions,
        const NCatboostOptions::TLossDescription* mainLossFunction, // can be nullptr
        bool needTargetDataForCtrs,
        bool metricsThatRequireTargetCanBeSkipped,
        TStringBuf datasetName,
        bool isNonEmptyAndNonConst,
        bool allowConstLabel)
    {
        auto isAnyOfMetrics = [&](bool predicate(ELossFunction)) {
            return AnyOf(
                metricDescriptions,
                [&](const NCatboostOptions::TLossDescription& metricDescription) -> bool {
                    return predicate(metricDescription.LossFunction.Get());
                }
            );
        };

        auto target = targetDataProvider->GetTarget();
        bool hasUserDefinedMetrics = isAnyOfMetrics(IsUserDefined);
        bool hasGroupNonPairwiseDataMetrics = isAnyOfMetrics(
            [](ELossFunction lossFunction) {
                return IsGroupwiseMetric(lossFunction) && !IsPairwiseMetric(lossFunction);
            }
        );

        if (needTargetDataForCtrs) {
            CB_ENSURE(target, "CTR features require Target data");
        }

        if (isAnyOfMetrics(IsRegressionMetric)) {
            CB_ENSURE(
                metricsThatRequireTargetCanBeSkipped || target,
                "Regression loss/metrics require target data"
            );
        }

        if ((mainLossFunction && IsUserDefined(mainLossFunction->GetLossFunction())) ||
            hasUserDefinedMetrics)
        {
            CB_ENSURE(target, "User defined objective/metrics require Target data");
        }

        if (hasGroupNonPairwiseDataMetrics) {
            CB_ENSURE(
                metricsThatRequireTargetCanBeSkipped || target,
                (hasGroupNonPairwiseDataMetrics ? "Groupwise" : "User defined")
                << " loss/metrics require target data"
            );
        }

        const bool needCheckTarget = !mainLossFunction || !IsRegressionObjective(mainLossFunction->GetLossFunction());
        // TODO(akhropov): Will be split by target type. MLTOOLS-2337.
        if (target) {
            CheckPreprocessedTarget(
                *target,
                datasetName,
                isNonEmptyAndNonConst,
                allowConstLabel,
                needCheckTarget,
                metricDescriptions
            );
        }
    }


    TTargetDataProviderPtr CreateTargetDataProvider(
        const TRawTargetDataProvider& rawData,
        TMaybeData<TConstArrayRef<TSubgroupId>> subgroupIds,
        bool isForGpu,
        const NCatboostOptions::TLossDescription* mainLossFunction, // can be nullptr
        bool metricsThatRequireTargetCanBeSkipped,
        TMaybe<ui32> knownModelApproxDimension,
        const TTargetCreationOptions& targetCreationOptions,
        const TInputClassificationInfo& inputClassificationInfo,
        TOutputClassificationInfo* outputClassificationInfo,
        TRestorableFastRng64* rand, // for possible pairs generation
        NPar::TLocalExecutor* localExecutor,
        TOutputPairsInfo* outputPairsInfo) {
        
        if (mainLossFunction) {
            CB_ENSURE(
                IsMultiRegressionObjective(mainLossFunction->GetLossFunction()) || rawData.GetTargetDimension() <= 1,
                "Currently only multi-regression objectives work with multidimensional target"
            );
        }
        
        TMaybe<ui32> knownClassCount = inputClassificationInfo.KnownClassCount;

        bool isRealTarget = !inputClassificationInfo.TargetBorder;
        if (targetCreationOptions.IsClass) {
            isRealTarget
                = mainLossFunction
                    && (mainLossFunction->GetLossFunction() == ELossFunction::CrossEntropy);

            if (!isRealTarget && !knownClassCount && knownModelApproxDimension > 1) {
                knownClassCount = knownModelApproxDimension;
            }
        }

        ui32 classCount = knownClassCount.GetOrElse(0);

        auto maybeConvertedTarget = ConvertTarget(
            rawData.GetTarget(),
            rawData.GetTargetType(),
            isRealTarget,
            targetCreationOptions.IsClass,
            targetCreationOptions.IsMultiClass,
            inputClassificationInfo.TargetBorder,
            !knownClassCount,
            inputClassificationInfo.ClassLabels,
            &outputClassificationInfo->ClassLabels,
            localExecutor,
            &classCount
        );

        // TODO(akhropov): make GPU also support absence of Target data
        if (isForGpu && maybeConvertedTarget.empty()) {
            maybeConvertedTarget = TVector<TSharedVector<float>>{MakeAtomicShared<TVector<float>>(TVector<float>(rawData.GetObjectCount(), 0.0f))};
        }

        classCount = (ui32)GetClassesCount((int)classCount, outputClassificationInfo->ClassLabels);
        bool createClassTarget = targetCreationOptions.CreateBinClassTarget || targetCreationOptions.CreateMultiClassTarget;
        TProcessedTargetData processedTargetData;

        /*
         * TODO(akhropov)
         * until target data provider selection supported by metrics use common data for all target
         *  data providers. MLTOOLS-2337.
         */

        // Targets
        if (targetCreationOptions.CreateBinClassTarget) {
            CB_ENSURE(
                metricsThatRequireTargetCanBeSkipped || rawData.GetTarget(),
                "Binary classification loss/metrics require label data"
            );

            if (!(*outputClassificationInfo->LabelConverter)->IsInitialized()) {
                (*outputClassificationInfo->LabelConverter)->InitializeBinClass();
            }

            if (!maybeConvertedTarget.empty()) {
                processedTargetData.TargetsClassCount.emplace("", 2);
            }
        }

        if (targetCreationOptions.CreateMultiClassTarget) {
            CB_ENSURE(
                metricsThatRequireTargetCanBeSkipped || rawData.GetTarget(),
                "Multi classification loss/metrics require label data"
            );
            CB_ENSURE(
                classCount >= 2,
                "Multiclass metric/loss specified but target data does not have more than one different labels"
            );

            if (!(*outputClassificationInfo->LabelConverter)->IsInitialized()) {
                (*outputClassificationInfo->LabelConverter)->InitializeMultiClass(
                    *maybeConvertedTarget[0],
                    classCount
                );
            }
            PrepareTargetCompressed(**outputClassificationInfo->LabelConverter, &*maybeConvertedTarget[0]);

            if (!maybeConvertedTarget.empty()) {
                processedTargetData.TargetsClassCount.emplace("", (**outputClassificationInfo->LabelConverter).GetApproxDimension());
            }
        }

        if (!maybeConvertedTarget.empty()) {
            processedTargetData.Targets.emplace("", maybeConvertedTarget);
        }

        // Weights
        {
            if (createClassTarget && (!inputClassificationInfo.ClassWeights.empty() ||
                inputClassificationInfo.AutoClassWeightsType != EAutoClassWeightsType::None))
            {
                auto targetClasses = !maybeConvertedTarget.empty()
                    ? TMaybe<TConstArrayRef<float>>(*maybeConvertedTarget[0])
                    : Nothing();

                TConstArrayRef<float> classWeights = targetClasses
                    ? inputClassificationInfo.ClassWeights
                    : TConstArrayRef<float>();

                TVector<float> autoClassWeights;
                if (targetClasses && classWeights.empty() &&
                    inputClassificationInfo.AutoClassWeightsType != EAutoClassWeightsType::None)
                {
                    classWeights = autoClassWeights = CalculateClassWeights(
                        *targetClasses,
                        rawData.GetWeights(),
                        classCount,
                        inputClassificationInfo.AutoClassWeightsType,
                        localExecutor);

                    if (outputClassificationInfo->ClassWeights) {
                        outputClassificationInfo->ClassWeights = autoClassWeights;
                    }
                }

                processedTargetData.Weights.emplace(
                    "",
                    MakeClassificationWeights(
                        rawData.GetWeights(),
                        rawData.GetGroupWeights(),
                        targetCreationOptions.CreateMultiClassTarget ? classCount : ui32(2),
                        isForGpu,
                        targetClasses,
                        classWeights,
                        localExecutor
                    )
                );
            } else {
                processedTargetData.Weights.emplace(
                    "",
                    MakeWeights(
                        rawData.GetWeights(),
                        rawData.GetGroupWeights(),
                        isForGpu,
                        localExecutor
                    )
                );
            }
        }


        // Baseline
        {
            auto maybeBaseline = rawData.GetBaseline();
            if (maybeBaseline) {
                if (targetCreationOptions.CreateMultiClassTarget) {
                    processedTargetData.Baselines.emplace(
                        "",
                        MakeBaselines(rawData.GetBaseline(), (ui32)classCount));
                } else {
                    processedTargetData.Baselines.emplace("", MakeOneBaseline(maybeBaseline));
                }
            }
        }

        outputPairsInfo->HasPairs = false;

        // GroupInfos
        // TODO(akhropov): always create group data if it is available (for compatibility)
        if (targetCreationOptions.CreateGroups ||
            (!rawData.GetObjectsGrouping()->IsTrivial() && !maybeConvertedTarget.empty()))
        {
            TConstArrayRef<TPair> pairs = rawData.GetPairs();
            TVector<TPair> generatedPairs;

            if (pairs.empty() && targetCreationOptions.CreatePairs) {
                CB_ENSURE(rawData.GetTarget(), "Pool labels are not provided. Cannot generate pairs.");

                generatedPairs = GeneratePairs(
                    *rawData.GetObjectsGrouping(),
                    *maybeConvertedTarget[0],
                    *targetCreationOptions.MaxPairsCount,
                    rand);

                pairs = generatedPairs;
            }

            if (!pairs.empty()) {
                outputPairsInfo->HasPairs = true;
            }

            if (rawData.GetObjectsGrouping()->IsTrivial() && outputPairsInfo->HasPairs) {
                ui32 docCount = rawData.GetObjectCount();
                TVector<ui32> fakeGroupsBounds;
                ConstructConnectedComponents(docCount, pairs, &fakeGroupsBounds, &outputPairsInfo->PermutationForGrouping, &outputPairsInfo->PairsInPermutedDataset);
                TVector<TGroupBounds> groups;
                groups.reserve(fakeGroupsBounds.size());
                ui32 leftBound = 0;
                for (const auto& rightBound : fakeGroupsBounds) {
                    groups.emplace_back(leftBound, rightBound);
                    leftBound = rightBound;
                }
                outputPairsInfo->FakeObjectsGrouping = TObjectsGrouping(std::move(groups));
            }

            if (!(rawData.GetObjectsGrouping()->IsTrivial() && outputPairsInfo->HasPairs)) {
                processedTargetData.GroupInfos.emplace(
                    "",
                    MakeGroupInfos(
                        *rawData.GetObjectsGrouping(),
                        subgroupIds,
                        rawData.GetGroupWeights(),
                        pairs
                    )
                );
            }
        }

        return MakeIntrusive<TTargetDataProvider>(
            rawData.GetObjectsGrouping(),
            std::move(processedTargetData)
        );
    }


    void InitClassesParams(
        const NJson::TJsonValue& param,
        TVector<float>* classWeights,
        TVector<NJson::TJsonValue>* classLabels,
        TMaybe<ui32>* classCount,
        EAutoClassWeightsType* autoClassWeights
    ) {
        if (param.Has("class_weights")) {
            classWeights->clear();
            for (const auto& token : param["class_weights"].GetArraySafe()) {
                classWeights->push_back((float)token.GetDoubleSafe());
            }
        }
        if (param.Has("class_names")) { // "class_names" used for compatibility
            const NJson::TJsonValue::TArray& classLabelsJsonArray = param["class_names"].GetArraySafe();
            classLabels->assign(classLabelsJsonArray.begin(), classLabelsJsonArray.end());
        }
        if (param.Has("classes_count")) {
            const ui32 classesCountValue = SafeIntegerCast<ui32>(param["classes_count"].GetUIntegerSafe());
            if (classesCountValue) { // compatibility with existing saved data
                classCount->ConstructInPlace(classesCountValue);
            }
        }
        if (param.Has("auto_class_weights")) {
            *autoClassWeights = FromString<EAutoClassWeightsType>(param["auto_class_weights"].GetStringSafe());
        }
    }


    static void ApplyGrouping(
        const TOutputPairsInfo& outputPairsInfo,
        ui64 cpuRamLimit,
        TProcessedDataProvider* processedDataProvider,
        NPar::TLocalExecutor* localExecutor
    ) {
        *processedDataProvider = *processedDataProvider->GetSubset(
            TObjectsGroupingSubset(
                processedDataProvider->TargetData->GetObjectsGrouping(),
                TArraySubsetIndexing<ui32>(TIndexedSubset<ui32>(outputPairsInfo.PermutationForGrouping)),
                EObjectsOrder::Undefined
            ),
            cpuRamLimit,
            localExecutor
        );
        processedDataProvider->TargetData->UpdateGroupInfos(
            MakeGroupInfos(
                outputPairsInfo.FakeObjectsGrouping,
                Nothing(),
                TWeights(outputPairsInfo.PermutationForGrouping.size()),
                TConstArrayRef<TPair>(outputPairsInfo.PairsInPermutedDataset)
            )
        );
    }


    TProcessedDataProvider CreateModelCompatibleProcessedDataProvider(
        const TDataProvider& srcData,
        TConstArrayRef<NCatboostOptions::TLossDescription> metricDescriptions,
        const TFullModel& model,
        ui64 cpuRamLimit,
        TRestorableFastRng64* rand, // for possible pairs generation
        NPar::TLocalExecutor* localExecutor) {

        TVector<NCatboostOptions::TLossDescription> updatedMetricsDescriptions(
            metricDescriptions.begin(),
            metricDescriptions.end());

        TVector<float> classWeights;
        TVector<NJson::TJsonValue> classLabels;
        TMaybe<ui32> classCount = Nothing();
        EAutoClassWeightsType autoClassWeights = EAutoClassWeightsType::None;

        TMaybe<NCatboostOptions::TLossDescription> modelLossDescription;
        if (const auto* modelInfoParams = MapFindPtr(model.ModelInfo, "params")) {
            NJson::TJsonValue paramsJson = ReadTJsonValue(*modelInfoParams);

            if (paramsJson.Has("data_processing_options")) {
                InitClassesParams(
                    paramsJson["data_processing_options"],
                    &classWeights,
                    &classLabels,
                    &classCount,
                    &autoClassWeights);
            }

            if (paramsJson.Has("loss_function")) {
                modelLossDescription.ConstructInPlace();
                modelLossDescription->Load(paramsJson["loss_function"]);

                if (!classCount && IsBinaryClassOnlyMetric(modelLossDescription->LossFunction)) {
                    CB_ENSURE_INTERNAL(
                        model.GetDimensionsCount() == 1,
                        "model trained with binary classification function has ApproxDimension="
                        << model.GetDimensionsCount()
                    );
                }

                if (updatedMetricsDescriptions.empty()) {
                    updatedMetricsDescriptions.push_back(*modelLossDescription);
                }
            }
        }

        TLabelConverter labelConverter;

        // "class_params" is new, more generic option, used for binclass as well
        for (const auto& paramName : {"class_params", "multiclass_params"}) {
            if (model.ModelInfo.contains(paramName)) {
                const auto& classParamsJsonAsString = model.ModelInfo.at(paramName);
                labelConverter.Initialize(model.GetDimensionsCount() > 1, classParamsJsonAsString);
                TClassLabelOptions classOptions;
                classOptions.Load(ReadTJsonValue(classParamsJsonAsString));
                if (classOptions.ClassLabels.IsSet()) {
                    classLabels = classOptions.ClassLabels;
                }
                if (labelConverter.IsMultiClass()) {
                    classCount.ConstructInPlace(GetClassesCount(classOptions.ClassesCount.Get(), classLabels));
                }
                break;
            }
        }

        if (!labelConverter.IsInitialized()) {
            const bool isClassOnlyMetrics = AnyOf(
                updatedMetricsDescriptions,
                [](const NCatboostOptions::TLossDescription& lossDescription) {
                    return IsClassificationOnlyMetric(lossDescription.GetLossFunction());
                }
            );
            if (isClassOnlyMetrics) {
                if (model.GetDimensionsCount() > 1) {
                    labelConverter.InitializeMultiClass(SafeIntegerCast<int>(model.GetDimensionsCount()));
                    classCount.ConstructInPlace(model.GetDimensionsCount());
                } else {
                    labelConverter.InitializeBinClass();
                }
            }
        }

        TMaybe<float> targetBorder;
        if (const auto* modelInfoParams = MapFindPtr(model.ModelInfo, "params")) {
            NJson::TJsonValue paramsJson = ReadTJsonValue(*modelInfoParams);

            const auto& dataProcessingOptions = paramsJson["data_processing_options"];
            const bool haveTargetBorder = dataProcessingOptions.Has("target_border")
                && !dataProcessingOptions["target_border"].IsNull();
            if (haveTargetBorder) {
                targetBorder = dataProcessingOptions["target_border"].GetDouble();
            }
        }

        const bool shouldBinarizeLabel = AnyOf(
            updatedMetricsDescriptions,
            [](const NCatboostOptions::TLossDescription& lossDescription) {
                return ShouldBinarizeLabel(lossDescription.GetLossFunction());
            }
        );
        if (shouldBinarizeLabel && !targetBorder && classLabels.empty() && !classCount) {
            targetBorder = GetDefaultTargetBorder();
            CATBOOST_WARNING_LOG << "Cannot restore border parameter, falling to default border = " << *targetBorder << Endl;
        }

        TProcessedDataProvider result;
        result.MetaInfo = srcData.MetaInfo;
        result.ObjectsGrouping = srcData.ObjectsGrouping;
        result.ObjectsData = srcData.ObjectsData;
        TOutputPairsInfo outputPairsInfo;
        TInputClassificationInfo inputClassificationInfo{
            classCount,
            classWeights,
            autoClassWeights,
            classLabels,
            targetBorder,
        };
        TOutputClassificationInfo outputClassificationInfo {
            classLabels,
            &labelConverter,
            Nothing(),
            Nothing()
        };
        auto targetCreationOptions = MakeTargetCreationOptions(
            srcData.RawTargetData,
            updatedMetricsDescriptions,
            model.GetDimensionsCount(),
            inputClassificationInfo
        );
        result.TargetData = CreateTargetDataProvider(
            srcData.RawTargetData,
            srcData.ObjectsData->GetSubgroupIds(),
            /*isForGpu*/ false,
            modelLossDescription.Get(),
            /*metricsThatRequireTargetCanBeSkipped*/ false,
            (ui32)model.GetDimensionsCount(),
            targetCreationOptions,
            inputClassificationInfo,
            &outputClassificationInfo,
            rand,
            localExecutor,
            &outputPairsInfo
        );
        CheckTargetConsistency(
            result.TargetData,
            updatedMetricsDescriptions,
            modelLossDescription.Get(),
            /*needTargetDataForCtrs*/ false,
            /*metricsThatRequireTargetCanBeSkipped*/ false,
            /*datasetName*/ TStringBuf(),
            /*isNonEmptyAndNonConst*/false,
            /*allowConstLabel*/ true
        );

        result.MetaInfo.HasPairs = outputPairsInfo.HasPairs;
        classLabels = outputClassificationInfo.ClassLabels;

        if (outputPairsInfo.HasFakeGroupIds()) {
            ApplyGrouping(outputPairsInfo, cpuRamLimit, &result, localExecutor);
        }

        return result;
    }

    TProcessedDataProvider CreateClassificationCompatibleDataProvider(
        const TDataProvider& srcData,
        const TFullModel& model,
        ui64 cpuRamLimit,
        TRestorableFastRng64* rand,
        NPar::TLocalExecutor* localExecutor
    ) {
        const TString ParamsJsonKey = "params";
        const TString DataProcessingOptionsJsonKey = "data_processing_options";
        const TString TargetBorderJsonKey = "target_border";
        const TString LossJsonKey = "loss_function";

        NCatboostOptions::TLossDescription lossDescription;
        if (model.ModelInfo.contains(ParamsJsonKey)) {
            const auto& params = ReadTJsonValue(model.ModelInfo.at(ParamsJsonKey));
            if (params.Has(LossJsonKey)) {
                lossDescription.Load(params[LossJsonKey]);
            }
        }

        if (!lossDescription.LossFunction.IsSet()) {
            auto loss = model.GetDimensionsCount() == 1 ? ELossFunction::Logloss : ELossFunction::MultiClass;
            lossDescription.LossFunction.Set(loss);
        }

        CB_ENSURE(IsClassificationObjective(lossDescription.GetLossFunction()),
                  "Attempt to create classification data provider for non classification model");

        TMaybe<float> targetBorder;
        TVector<float> classWeights;
        TVector<NJson::TJsonValue> classLabels;
        TMaybe<ui32> classCount = Nothing();
        EAutoClassWeightsType autoClassWeights;

        if (const auto* modelInfoParams = MapFindPtr(model.ModelInfo, ParamsJsonKey)) {
            NJson::TJsonValue paramsJson = ReadTJsonValue(*modelInfoParams);
            if (paramsJson.Has(DataProcessingOptionsJsonKey)) {
                InitClassesParams(
                    paramsJson[DataProcessingOptionsJsonKey],
                    &classWeights,
                    &classLabels,
                    &classCount,
                    &autoClassWeights);

                if (classLabels.empty() && !classCount) {
                    if (paramsJson[DataProcessingOptionsJsonKey].Has(TargetBorderJsonKey)) {
                        targetBorder = paramsJson[DataProcessingOptionsJsonKey][TargetBorderJsonKey].GetDouble();
                    }
                }
            }
        }

        if (ShouldBinarizeLabel(lossDescription.GetLossFunction()) &&
            classLabels.empty() &&
            !classCount &&
            !targetBorder)
        {
            targetBorder = GetDefaultTargetBorder();
            CATBOOST_WARNING_LOG << "Cannot restore border parameter, falling to default border = " << *targetBorder << Endl;
        }

        TLabelConverter labelConverter;

        // "class_params" is new, more generic option, used for binclass as well
        for (const auto& paramName : {"class_params", "multiclass_params"}) {
            if (model.ModelInfo.contains(paramName)) {
                const auto& classParamsJsonAsString = model.ModelInfo.at(paramName);
                labelConverter.Initialize(model.GetDimensionsCount() > 1, classParamsJsonAsString);
                TClassLabelOptions classOptions;
                classOptions.Load(ReadTJsonValue(classParamsJsonAsString));
                if (classOptions.ClassLabels.IsSet()) {
                    classLabels = classOptions.ClassLabels;
                }
                if (labelConverter.IsMultiClass()) {
                    classCount.ConstructInPlace(GetClassesCount(classOptions.ClassesCount.Get(), classLabels));
                }
                break;
            }
        }

        if (!labelConverter.IsInitialized()) {
            if (model.GetDimensionsCount() > 1) {
                labelConverter.InitializeMultiClass(model.GetDimensionsCount());
                classCount.ConstructInPlace(model.GetDimensionsCount());
            } else {
                labelConverter.InitializeBinClass();
            }
        }

        TVector<NCatboostOptions::TLossDescription> metricsDescriptions = {lossDescription};

        TProcessedDataProvider result;
        result.MetaInfo = srcData.MetaInfo;
        result.ObjectsGrouping = srcData.ObjectsGrouping;
        result.ObjectsData = srcData.ObjectsData;
        TOutputPairsInfo outputPairsInfo;
        TInputClassificationInfo inputClassificationInfo{
            classCount,
            classWeights,
            autoClassWeights,
            classLabels,
            targetBorder
        };
        TOutputClassificationInfo outputClassificationInfo {
            classLabels,
            &labelConverter,
            Nothing(),
            Nothing()
        };
        auto targetCreationOptions = MakeTargetCreationOptions(
            srcData.RawTargetData,
            metricsDescriptions,
            model.GetDimensionsCount(),
            inputClassificationInfo
        );
        result.TargetData = CreateTargetDataProvider(
            srcData.RawTargetData,
            srcData.ObjectsData->GetSubgroupIds(),
            /*isForGpu*/ false,
            &lossDescription,
            /*metricsThatRequireTargetCanBeSkipped*/ false,
            (ui32)model.GetDimensionsCount(),
            targetCreationOptions,
            inputClassificationInfo,
            &outputClassificationInfo,
            rand,
            localExecutor,
            &outputPairsInfo
        );
        CheckTargetConsistency(
            result.TargetData,
            metricsDescriptions,
            &lossDescription,
            /*needTargetDataForCtrs*/ false,
            /*metricsThatRequireTargetCanBeSkipped*/ false,
            /*datasetName*/ TStringBuf(),
            /*isNonEmptyAndNonConst*/false,
            /*allowConstLabel*/ true
        );

        result.MetaInfo.HasPairs = outputPairsInfo.HasPairs;
        classLabels = outputClassificationInfo.ClassLabels;

        if (outputPairsInfo.HasFakeGroupIds()) {
            ApplyGrouping(outputPairsInfo, cpuRamLimit, &result, localExecutor);
        }

        return result;
    }
}
