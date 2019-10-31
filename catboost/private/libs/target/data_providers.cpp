#include "data_providers.h"

#include "binarize_target.h"
#include "target_converter.h"

#include <catboost/libs/helpers/connected_components.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/logging/logging.h>
#include <catboost/libs/metrics/metric.h>
#include <catboost/private/libs/options/enum_helpers.h>
#include <catboost/private/libs/options/json_helper.h>
#include <catboost/private/libs/options/multiclass_label_options.h>
#include <catboost/private/libs/options/data_processing_options.h>
#include <catboost/private/libs/pairs/util.h>

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
        TMaybeData<TConstArrayRef<TVector<TString>>> maybeRawTarget,
        bool isClass,
        bool isMultiClass,
        bool classCountUnknown,
        const TVector<TString> inputClassNames,
        TVector<TString>* outputClassNames,
        NPar::TLocalExecutor* localExecutor,
        ui32* classCount)
    {
        if (!maybeRawTarget) {
            return {};
        }

        auto rawTarget = *maybeRawTarget;

        auto targetConverter = MakeTargetConverter(
            isClass,
            isMultiClass,
            classCountUnknown,
            inputClassNames,
            outputClassNames);

        const auto targetDim = rawTarget.size();
        TVector<TSharedVector<float>> trainingTarget(targetDim);
        for (auto targetIdx : xrange(targetDim)) {
            trainingTarget[targetIdx] = MakeAtomicShared<TVector<float>>(TVector<float>());
        }

        switch (targetConverter.GetTargetPolicy()) {
            case EConvertTargetPolicy::MakeClassNames: {
                for (auto targetIdx : xrange(targetDim)) {
                    *trainingTarget[targetIdx] = targetConverter.PostprocessLabels(rawTarget[targetIdx]);
                }
                if (isClass && outputClassNames) {
                    targetConverter.SetOutputClassNames();
                }
                break;
            }
            case EConvertTargetPolicy::CastFloat:
                if (isClass) {
                    // can't use parallel processing because of UniqueLabels update
                    for (auto targetIdx : xrange(targetDim)) {
                        trainingTarget[targetIdx]->reserve(rawTarget[targetIdx].size());
                        for (const auto& rawTargetElement : rawTarget[targetIdx]) {
                            trainingTarget[targetIdx]->push_back(targetConverter.ConvertLabel(rawTargetElement));
                        }
                    }
                    break;
                }
            case EConvertTargetPolicy::UseClassNames: {
                for (auto targetIdx : xrange(targetDim)) {
                    trainingTarget[targetIdx]->yresize(rawTarget[targetIdx].size());
                    localExecutor->ExecRangeBlockedWithThrow(
                        [&] (int idx) { (*trainingTarget[targetIdx])[idx] = targetConverter.ConvertLabel(rawTarget[targetIdx][idx]); },
                        0,
                        SafeIntegerCast<int>(rawTarget[targetIdx].size()),
                        0,
                        NPar::TLocalExecutor::WAIT_COMPLETE
                    );
                }
                break;
            }
            default: {
                CB_ENSURE(
                    false,
                    "Unsupported convert target policy "
                    << ToString<EConvertTargetPolicy>(targetConverter.GetTargetPolicy())
                );
            }
        }
        if (isMultiClass && classCountUnknown) {
            *classCount = targetConverter.GetClassCount();
        }

        return trainingTarget;
    }

    static void CheckPreprocessedTarget(
        const TVector<TSharedVector<float>>& convertedTarget,
        TStringBuf datasetName,
        bool isNonEmptyAndNonConst,
        bool allowConstLabel,
        bool needCheckTarget,
        TConstArrayRef<NCatboostOptions::TLossDescription> metricDescriptions)
    {
        if (convertedTarget.empty() || convertedTarget[0]->empty()) {
            return;
        }

        if (needCheckTarget) {
            for (auto targetIdx : xrange(convertedTarget.size())) {
                for (auto objectIdx : xrange(convertedTarget[0]->size())) {
                    const float value = (*convertedTarget[targetIdx])[objectIdx];
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
                    *target,
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
        TConstArrayRef<NCatboostOptions::TLossDescription> metricDescriptions,
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

        CB_ENSURE_INTERNAL(maxPairsCount, "GeneratePairs was called but no loss/metric needs it");

        TVector<TPair> result;

        GeneratePairLogitPairs(
            objectsGrouping,
            targetData,
            (ui32)*maxPairsCount,
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

        TVector<TQueryInfo> result;
        result.reserve(groupsBounds.size());

        TVector<ui32> objectToGroupIdxMap; // [objectIdx]->groupIdx  initialized only for pairs
        if (!pairs.empty()) {
            objectToGroupIdxMap.yresize(objectsGrouping.GetObjectCount());
        }

        for (auto groupIdx : xrange((ui32)groupsBounds.size())) {
            result.emplace_back(groupsBounds[groupIdx].Begin, groupsBounds[groupIdx].End);
            auto& group = result.back();
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


    TTargetDataProviderPtr CreateTargetDataProvider(
        const TRawTargetDataProvider& rawData,
        TMaybeData<TConstArrayRef<TSubgroupId>> subgroupIds,
        bool isForGpu,
        bool isNonEmptyAndNonConst,
        TStringBuf datasetName,
        TConstArrayRef<NCatboostOptions::TLossDescription> metricDescriptions,
        TMaybe<NCatboostOptions::TLossDescription*> mainLossFunction,
        bool allowConstLabel,
        bool metricsThatRequireTargetCanBeSkipped,
        bool needTargetDataForCtrs,
        TMaybe<ui32> knownModelApproxDimension,
        const TInputClassificationInfo& inputClassificationInfo,
        TOutputClassificationInfo* outputClassificationInfo,
        TRestorableFastRng64* rand, // for possible pairs generation
        NPar::TLocalExecutor* localExecutor,
        TOutputPairsInfo* outputPairsInfo) {

        if (isNonEmptyAndNonConst) {
            CB_ENSURE(rawData.GetObjectCount() > 0, "Train dataset is empty");
        }
        CB_ENSURE(!metricDescriptions.empty(), "No metrics specified");

        auto isAnyOfMetrics = [&](bool predicate(ELossFunction)) {
            return AnyOf(
                metricDescriptions,
                [&](const NCatboostOptions::TLossDescription& metricDescription) -> bool {
                    return predicate(metricDescription.LossFunction.Get());
                }
            );
        };

        CB_ENSURE_INTERNAL(
            !mainLossFunction ||
            ((*mainLossFunction)->GetLossFunction() == ELossFunction::PythonUserDefinedPerObject) ||
            FindPtr(metricDescriptions, **mainLossFunction),
            "mainLossFunction is not in metricDescriptions"
        );

        bool hasClassificationOnlyMetrics = isAnyOfMetrics(IsClassificationOnlyMetric);
        bool hasBinClassOnlyMetrics = isAnyOfMetrics(IsBinaryClassOnlyMetric);
        bool hasMultiClassOnlyMetrics = isAnyOfMetrics(IsMultiClassOnlyMetric);
        bool hasGroupwiseMetrics = isAnyOfMetrics(IsGroupwiseMetric);
        bool hasUserDefinedMetrics = isAnyOfMetrics(IsUserDefined);

        CB_ENSURE(
            !(hasBinClassOnlyMetrics && hasMultiClassOnlyMetrics),
            "Both binary classification -only and multiclassification -only loss function or metrics"
            " specified"
        );

        bool multiClassTargetData = false;
        ui32 knownClassCount = inputClassificationInfo.KnownClassCount ? *inputClassificationInfo.KnownClassCount : 0;

        if (knownModelApproxDimension) {
            if (*knownModelApproxDimension == 1) {
                CB_ENSURE(
                    !hasMultiClassOnlyMetrics,
                    "Multiclassification-only metrics specified for a single-dimensional model"
                );
            } else {
                for (const auto& metricDescription : metricDescriptions) {
                    auto metricLossFunction = metricDescription.GetLossFunction();
                    CB_ENSURE(
                        IsMultiClassCompatibleMetric(metricLossFunction),
                        "Non-Multiclassification compatible metric (" << metricLossFunction
                        << ") specified for a multidimensional model"
                    );
                }
                if (!knownClassCount) { // because there might be missing classes in train
                    knownClassCount = *knownModelApproxDimension;
                }
                multiClassTargetData = true;
            }
        } else {
            multiClassTargetData = (
                hasMultiClassOnlyMetrics || (knownClassCount > 2) ||
                (inputClassificationInfo.ClassWeights.size() > 2) ||
                (inputClassificationInfo.ClassNames.size() > 2)
            );
        }

        ui32 classCountInData = 0;

        auto maybeConvertedTarget = ConvertTarget(
            rawData.GetMultiTarget(),
            hasClassificationOnlyMetrics || multiClassTargetData,
            multiClassTargetData,
            knownClassCount == 0,
            inputClassificationInfo.ClassNames,
            &outputClassificationInfo->ClassNames,
            localExecutor,
            &classCountInData
        );

        if (needTargetDataForCtrs) {
            CB_ENSURE(!maybeConvertedTarget.empty(), "CTR features require Target data");
        }
        if ((mainLossFunction && IsUserDefined((*mainLossFunction)->GetLossFunction())) ||
            hasUserDefinedMetrics)
        {
            CB_ENSURE(!maybeConvertedTarget.empty(), "User defined objective/metrics require Target data");
        }

        // TODO(akhropov): make GPU also support absence of Target data
        if (isForGpu && maybeConvertedTarget.empty()) {
            maybeConvertedTarget = TVector<TSharedVector<float>>{MakeAtomicShared<TVector<float>>(TVector<float>(rawData.GetObjectCount(), 0.0f))};
        }


        if (knownClassCount == 0) {
            knownClassCount = classCountInData;
        }
        ui32 classCount = (ui32)GetClassesCount((int)knownClassCount, outputClassificationInfo->ClassNames);


        bool createBinClassTarget = false;
        bool createMultiClassTarget = false;

        if (hasBinClassOnlyMetrics) {
            createBinClassTarget = true;
        } else if (hasMultiClassOnlyMetrics) {
            createMultiClassTarget = true;
        } else {
            if (multiClassTargetData || (classCount > 2)) {
                createMultiClassTarget = true;
            } else if (classCount == 2) {
                createBinClassTarget = true;
            }
        }
        bool createClassTarget = createBinClassTarget || createMultiClassTarget;

        TProcessedTargetData processedTargetData;

        /*
         * TODO(akhropov)
         * until target data provider selection supported by metrics use common data for all target
         *  data providers. MLTOOLS-2337.
         */

        // Targets
        if (createBinClassTarget) {
            CB_ENSURE(
                metricsThatRequireTargetCanBeSkipped || rawData.GetTarget(),
                "Binary classification loss/metrics require label data"
            );
            CB_ENSURE(
                (classCount == 0) || (classCount == 2),
                "Binary classification loss/metric specified but target data is neither positive class"
                " probability nor 2-class labels"
            );

            if (
                rawData.GetTarget() &&
                mainLossFunction &&
                ShouldBinarizeLabel((*mainLossFunction)->GetLossFunction())
            ) {
                float realTargetBorder;
                if (inputClassificationInfo.TargetBorder) {
                    realTargetBorder = *inputClassificationInfo.TargetBorder;
                } else {
                    THashSet<float> uniqueValues;
                    for (auto target : *maybeConvertedTarget[0]) {
                        if (uniqueValues.size() < 3) {
                            uniqueValues.insert(target);
                        } else {
                            break;
                        }
                    }
                    CB_ENSURE_INTERNAL(!uniqueValues.empty(), "Target vector is empty, nothing to binarize.");
                    CB_ENSURE(uniqueValues.size() != 1, "Data has constant target, so it's impossible to binarize it.");
                    CB_ENSURE(uniqueValues.size() == 2, "You should specify target border parameter for target binarization.");
                    const auto firstValue = *uniqueValues.begin();
                    const auto secondValue = uniqueValues.size() == 1 ? firstValue : *(++uniqueValues.begin());
                    realTargetBorder = (firstValue + secondValue) / 2;
                    outputClassificationInfo->TargetBorder.ConstructInPlace(realTargetBorder);
                    CATBOOST_DEBUG_LOG << "Target border set to " << realTargetBorder << Endl;
                }
                PrepareTargetBinary(*maybeConvertedTarget[0], realTargetBorder, &*maybeConvertedTarget[0]);
            }

            if (!maybeConvertedTarget.empty()) {
                processedTargetData.TargetsClassCount.emplace("", 2);
            }
        }

        if (createMultiClassTarget) {
            CB_ENSURE(
                metricsThatRequireTargetCanBeSkipped || rawData.GetTarget(),
                "Multi classification loss/metrics require label data"
            );
            CB_ENSURE(
                classCount >= 2,
                "Multiclass metric/loss specified but target data does not have more than one different labels"
            );

            if (!(*outputClassificationInfo->LabelConverter)->IsInitialized()) {
                (*outputClassificationInfo->LabelConverter)->Initialize(*maybeConvertedTarget[0], classCount);
            }
            PrepareTargetCompressed(**outputClassificationInfo->LabelConverter, &*maybeConvertedTarget[0]);

            if (!maybeConvertedTarget.empty()) {
                processedTargetData.TargetsClassCount.emplace("", (**outputClassificationInfo->LabelConverter).GetApproxDimension());
            }
        }

        if (isAnyOfMetrics(IsRegressionMetric)) {
            CB_ENSURE(
                metricsThatRequireTargetCanBeSkipped || rawData.GetTarget(),
                "Regression loss/metrics require target data"
            );
        }

        if (!maybeConvertedTarget.empty()) {
            processedTargetData.Targets.emplace("", maybeConvertedTarget);
        }

        // Weights
        {
            if (!rawData.GetWeights().IsTrivial() && isAnyOfMetrics(UsesPairsForCalculation)) {
                CATBOOST_WARNING_LOG << "Pairwise losses don't support object weights." << '\n';
            }

            if (createClassTarget && !inputClassificationInfo.ClassWeights.empty()) {
                processedTargetData.Weights.emplace(
                    "",
                    MakeClassificationWeights(
                        rawData.GetWeights(),
                        rawData.GetGroupWeights(),
                        createMultiClassTarget ? classCount : ui32(2),
                        isForGpu,
                        !maybeConvertedTarget.empty() ? TMaybe<TConstArrayRef<float>>(*maybeConvertedTarget[0]) : Nothing(),
                        rawData.GetTarget() ? inputClassificationInfo.ClassWeights : TConstArrayRef<float>(),
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
                if (createMultiClassTarget) {
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
        if (hasGroupwiseMetrics ||
            (!rawData.GetObjectsGrouping()->IsTrivial() && (!maybeConvertedTarget.empty() || hasUserDefinedMetrics)))
        {
            bool hasGroupNonPairwiseDataMetrics = isAnyOfMetrics(
                [](ELossFunction lossFunction) {
                    return IsGroupwiseMetric(lossFunction) && !IsPairwiseMetric(lossFunction);
                });

            if (hasGroupNonPairwiseDataMetrics ||
                (!rawData.GetObjectsGrouping()->IsTrivial() && (!maybeConvertedTarget.empty() || hasUserDefinedMetrics)))
            {
                CB_ENSURE(
                    metricsThatRequireTargetCanBeSkipped || !maybeConvertedTarget.empty(),
                    (hasGroupNonPairwiseDataMetrics ? "Groupwise" : "User defined")
                    << " loss/metrics require target data"
                );
            }

            TConstArrayRef<TPair> pairs = rawData.GetPairs();

            TVector<TPair> generatedPairs;

            if (pairs.empty() && isAnyOfMetrics(IsPairwiseMetric)) {
                CB_ENSURE(rawData.GetTarget(), "Pool labels are not provided. Cannot generate pairs.");

                generatedPairs = GeneratePairs(
                    *rawData.GetObjectsGrouping(),
                    *maybeConvertedTarget[0],
                    metricDescriptions,
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


        const bool needCheckTarget = !mainLossFunction || !IsRegressionObjective((*mainLossFunction)->GetLossFunction());
        // TODO(akhropov): Will be split by target type. MLTOOLS-2337.
        CheckPreprocessedTarget(
            maybeConvertedTarget,
            datasetName,
            isNonEmptyAndNonConst,
            allowConstLabel,
            needCheckTarget,
            metricDescriptions);

        return MakeIntrusive<TTargetDataProvider>(
            rawData.GetObjectsGrouping(),
            std::move(processedTargetData)
        );
    }


    void InitClassesParams(
        const NJson::TJsonValue& param,
        TVector<float>* classWeights,
        TVector<TString>* classNames,
        TMaybe<ui32>* classCount
    ) {
        if (param.Has("class_weights")) {
            classWeights->clear();
            for (const auto& token : param["class_weights"].GetArraySafe()) {
                classWeights->push_back((float)token.GetDoubleSafe());
            }
        }
        if (param.Has("class_names")) {
            classNames->clear();
            for (const auto& token : param["class_names"].GetArraySafe()) {
                classNames->push_back(token.GetStringSafe());
            }
        }
        if (param.Has("classes_count")) {
            classCount->ConstructInPlace(SafeIntegerCast<ui32>(param["classes_count"].GetUIntegerSafe()));
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
        TVector<TString> classNames;
        TMaybe<ui32> classCount = Nothing();

        if (const auto* modelInfoParams = MapFindPtr(model.ModelInfo, "params")) {
            NJson::TJsonValue paramsJson = ReadTJsonValue(*modelInfoParams);

            if (paramsJson.Has("data_processing_options")) {
                InitClassesParams(
                    paramsJson["data_processing_options"],
                    &classWeights,
                    &classNames,
                    &classCount);
            }

            if (paramsJson.Has("loss_function")) {
                updatedMetricsDescriptions.resize(1);
                NCatboostOptions::TLossDescription modelLossDescription;
                modelLossDescription.Load(paramsJson["loss_function"]);

                if (!classCount && IsBinaryClassOnlyMetric(modelLossDescription.LossFunction)) {
                    CB_ENSURE_INTERNAL(
                        model.GetDimensionsCount() == 1,
                        "model trained with binary classification function has ApproxDimension="
                        << model.GetDimensionsCount()
                    );
                }

                if (updatedMetricsDescriptions.empty()) {
                    updatedMetricsDescriptions.push_back(std::move(modelLossDescription));
                }
            }
        }

        TLabelConverter labelConverter;
        if (model.GetDimensionsCount() > 1) {  // is multiclass?
            if (model.ModelInfo.contains("multiclass_params")) {
                const auto& multiclassParamsJsonAsString = model.ModelInfo.at("multiclass_params");
                labelConverter.Initialize(multiclassParamsJsonAsString);
                TMulticlassLabelOptions multiclassOptions;
                multiclassOptions.Load(ReadTJsonValue(multiclassParamsJsonAsString));
                if (multiclassOptions.ClassNames.IsSet()) {
                    classNames = multiclassOptions.ClassNames;
                }
                classCount.ConstructInPlace(GetClassesCount(multiclassOptions.ClassesCount.Get(), classNames));
            } else {
                labelConverter.Initialize(model.GetDimensionsCount());
                classCount.ConstructInPlace(model.GetDimensionsCount());
            }
        }

        TProcessedDataProvider result;
        result.MetaInfo = srcData.MetaInfo;
        result.ObjectsGrouping = srcData.ObjectsGrouping;
        result.ObjectsData = srcData.ObjectsData;
        TOutputPairsInfo outputPairsInfo;
        TInputClassificationInfo inputClassificationInfo{
            classCount,
            classWeights,
            classNames,
            Nothing()
        };
        TOutputClassificationInfo outputClassificationInfo {
            classNames,
            &labelConverter,
            Nothing()
        };
        result.TargetData = CreateTargetDataProvider(
            srcData.RawTargetData,
            srcData.ObjectsData->GetSubgroupIds(),
            /*isForGpu*/ false,
            /*isLearn*/ false,
            /*datasetName*/ TStringBuf(),
            updatedMetricsDescriptions,
            /*mainLossFunction*/ Nothing(),
            /*allowConstLabel*/ true,
            /*metricsThatRequireTargetCanBeSkipped*/ false,
            /*needTargetDataForCtrs*/ false,
            (ui32)model.GetDimensionsCount(),
            inputClassificationInfo,
            &outputClassificationInfo,
            rand,
            localExecutor,
            &outputPairsInfo
        );
        result.MetaInfo.HasPairs = outputPairsInfo.HasPairs;
        classNames = outputClassificationInfo.ClassNames;

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
        const TString MultiClassParamsJsonKey = "multiclass_params";
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
        TVector<TString> classNames;
        TMaybe<ui32> classCount = Nothing();

        if (const auto* modelInfoParams = MapFindPtr(model.ModelInfo, ParamsJsonKey)) {
            NJson::TJsonValue paramsJson = ReadTJsonValue(*modelInfoParams);
            if (paramsJson.Has(DataProcessingOptionsJsonKey)) {
                InitClassesParams(
                    paramsJson[DataProcessingOptionsJsonKey],
                    &classWeights,
                    &classNames,
                    &classCount);

                if (paramsJson[DataProcessingOptionsJsonKey].Has(TargetBorderJsonKey)) {
                    targetBorder = paramsJson[DataProcessingOptionsJsonKey][TargetBorderJsonKey].GetDouble();
                }
            }
        }

        if (ShouldBinarizeLabel(lossDescription.GetLossFunction()) && !targetBorder) {
            targetBorder = GetDefaultTargetBorder();
            CATBOOST_WARNING_LOG << "Cannot restore border parameter, falling to default border = " << *targetBorder << Endl;
        }

        TLabelConverter labelConverter;
        if (model.GetDimensionsCount() > 1) {
            if (model.ModelInfo.contains(MultiClassParamsJsonKey)) {
                const auto& multiclassParamsJsonAsString = model.ModelInfo.at(MultiClassParamsJsonKey);
                labelConverter.Initialize(multiclassParamsJsonAsString);
                TMulticlassLabelOptions multiclassOptions;
                multiclassOptions.Load(ReadTJsonValue(multiclassParamsJsonAsString));
                if (multiclassOptions.ClassNames.IsSet()) {
                    classNames = multiclassOptions.ClassNames;
                }
                classCount.ConstructInPlace(GetClassesCount(multiclassOptions.ClassesCount.Get(), classNames));
            } else {
                labelConverter.Initialize(model.GetDimensionsCount());
                classCount.ConstructInPlace(model.GetDimensionsCount());
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
            classNames,
            targetBorder
        };
        TOutputClassificationInfo outputClassificationInfo {
            classNames,
            &labelConverter,
            Nothing()
        };
        result.TargetData = CreateTargetDataProvider(
            srcData.RawTargetData,
            srcData.ObjectsData->GetSubgroupIds(),
            /*isForGpu*/ false,
            /*isLearn*/ false,
            /*datasetName*/ TStringBuf(),
            metricsDescriptions,
            &lossDescription,
            /*allowConstLabel*/ true,
            /*metricsThatRequireTargetCanBeSkipped*/ false,
            /*needTargetDataForCtrs*/ false,
            (ui32)model.GetDimensionsCount(),
            inputClassificationInfo,
            &outputClassificationInfo,
            rand,
            localExecutor,
            &outputPairsInfo
        );
        result.MetaInfo.HasPairs = outputPairsInfo.HasPairs;
        classNames = outputClassificationInfo.ClassNames;

        if (outputPairsInfo.HasFakeGroupIds()) {
            ApplyGrouping(outputPairsInfo, cpuRamLimit, &result, localExecutor);
        }

        return result;
    }
}
