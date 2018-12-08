#include "data_providers.h"

#include "binarize_target.h"
#include "target_converter.h"

#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/logging/logging.h>
#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/options/enum_helpers.h>
#include <catboost/libs/options/json_helper.h>
#include <catboost/libs/options/multiclass_label_options.h>
#include <catboost/libs/pairs/util.h>

#include <util/generic/cast.h>
#include <util/generic/xrange.h>
#include <util/generic/ymath.h>
#include <util/stream/labeled.h>


namespace NCB {

    inline bool IsSafeTarget(float value) {
        return Abs(value) < 1e6f;
    }


    // can return TAtomicSharedPtr(nullptr) is not target data is available
    static TAtomicSharedPtr<TVector<float>> ConvertTarget(
        TMaybeData<TConstArrayRef<TString>> maybeRawTarget,
        bool isClass,
        bool isMultiClass,
        bool classCountUnknown,
        TVector<TString>* classNames,
        NPar::TLocalExecutor* localExecutor,
        ui32* classCount)
    {
        if (!maybeRawTarget) {
            return TAtomicSharedPtr<TVector<float>>();
        }

        auto rawTarget = *maybeRawTarget;

        auto targetConverter = MakeTargetConverter(
            isClass,
            isMultiClass,
            classCountUnknown,
            classNames);

        TVector<float> trainingTarget;

        switch (targetConverter.GetTargetPolicy()) {
            case EConvertTargetPolicy::MakeClassNames: {
                trainingTarget = targetConverter.PostprocessLabels(rawTarget);
                if (isClass && classNames) {
                    targetConverter.SetOutputClassNames();
                }
                break;
            }
            case EConvertTargetPolicy::CastFloat:
                if (isClass) {
                    // can't use parallel processing because of UniqueLabels update
                    trainingTarget.reserve(rawTarget.size());
                    for (const auto& rawTargetElement : rawTarget) {
                        trainingTarget.push_back(targetConverter.ConvertLabel(rawTargetElement));
                    }
                    break;
                }
            case EConvertTargetPolicy::UseClassNames: {
                trainingTarget.yresize(rawTarget.size());
                localExecutor->ExecRangeWithThrow(
                    [&] (int idx) { trainingTarget[idx] = targetConverter.ConvertLabel(rawTarget[idx]); },
                    0,
                    SafeIntegerCast<int>(rawTarget.size()),
                    NPar::TLocalExecutor::WAIT_COMPLETE
                );
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
        if (isClass && classCountUnknown) {
            *classCount = targetConverter.GetClassCount();
        }

        return MakeAtomicShared<TVector<float>>(std::move(trainingTarget));
    }


    static void CheckPreprocessedTarget(
        const TSharedVector<float>& maybeConvertedTarget,
        TStringBuf datasetName,
        bool isLearnData,
        bool allowConstLabel,
        TConstArrayRef<NCatboostOptions::TLossDescription> metricDescriptions)
    {
        if (!maybeConvertedTarget) {
            return;
        }

        TConstArrayRef<float> convertedTarget = *maybeConvertedTarget;
        if (convertedTarget.empty()) {
            return;
        }

        for (auto objectIdx : xrange(convertedTarget.size())) {
            const float value = convertedTarget[objectIdx];
            if (!IsSafeTarget(value)) {
                CATBOOST_WARNING_LOG
                    << "Got unsafe target "
                    << LabeledOutput(value)
                    << " at object #" << objectIdx << " of dataset " << datasetName << Endl;
                break;
            }
        }

        for (const auto& metricDescription : metricDescriptions) {
            ::CheckPreprocessedTarget(
                convertedTarget,
                metricDescription.GetLossFunction(),
                isLearnData,
                allowConstLabel);
        }
    }

    TSharedWeights<float> MakeClassificationWeights(
        const TWeights<float>& rawWeights,
        const TWeights<float>& rawGroupWeights,
        ui32 classCount,
        bool isForGpu,
        TConstArrayRef<float> targetClasses, // [objectIdx], should contain integer values
        TConstArrayRef<float> classWeights, // [classIdx], empty if not specified
        NPar::TLocalExecutor* localExecutor)
    {
        CheckDataSize(classWeights.size(), (size_t)classCount, "class weights size", true, "class count");
        Y_VERIFY(rawWeights.GetSize() == targetClasses.Size());

        if (classWeights.empty() && rawGroupWeights.IsTrivial()) {
            if (isForGpu && rawWeights.IsTrivial()) {
                // TODO(akhropov): make GPU also support trivial TWeights

                return MakeIntrusive<TWeights<float>>(TVector<float>(rawWeights.GetSize(), 1.0f));
            }
            return MakeIntrusive<TWeights<float>>(rawWeights);
        }

        TVector<float> classAdjustedWeights;
        classAdjustedWeights.yresize(rawWeights.GetSize());

        localExecutor->ExecRangeWithThrow(
            [&] (int i) {
                // TODO(annaveronika): check class weight not negative.
                CB_ENSURE(
                    (size_t)targetClasses[i] < classWeights.size(),
                    "class " + ToString((size_t)targetClasses[i]) + " is missing in class weights"
                );
                classAdjustedWeights[i]
                    = rawWeights[i]*rawGroupWeights[i]*classWeights[(size_t)targetClasses[i]];
            },
            0,
            (int)rawWeights.GetSize(),
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

        localExecutor->ExecRangeWithThrow(
            [&] (int i) {
                groupAdjustedWeights[i] = rawWeights[i]*rawGroupWeights[i];
            },
            0,
            (int)rawWeights.GetSize(),
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

    TSharedVector<float> MakeOneBaseline(TMaybeData<TBaselineArrayRef> baselines) {
        if (!baselines) {
            return nullptr;
        }

        CheckDataSize(baselines->size(), (size_t)1, "baselines", false, "expected dimension");

        return MakeAtomicShared<TVector<float>>((*baselines)[0].begin(), (*baselines)[0].end());
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

        TMaybe<ui32> maxPairsCount;

        for (const auto& metricDescription : metricDescriptions) {
            if (IsPairLogit(metricDescription.GetLossFunction())) {
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


    TTargetDataProviders CreateTargetDataProviders(
        const TRawTargetDataProvider& rawData,
        TMaybeData<TConstArrayRef<TSubgroupId>> subgroupIds,
        bool isForGpu,
        bool isLearnData,
        TStringBuf datasetName,
        TConstArrayRef<NCatboostOptions::TLossDescription> metricDescriptions,
        TMaybe<NCatboostOptions::TLossDescription*> mainLossFunction,
        bool allowConstLabel,
        ui32 knownClassCount,
        TConstArrayRef<float> classWeights, // [classIdx], empty if not specified
        TVector<TString>* classNames,
        TMaybe<TLabelConverter*> labelConverter, // needed only for multiclass
        TRestorableFastRng64* rand, // for possible pairs generation
        NPar::TLocalExecutor* localExecutor) {

        if (isLearnData) {
            CB_ENSURE(rawData.GetObjectCount() > 0, "Train dataset is empty");
        }

        auto isAnyOfMetrics = [&](auto&& predicate) {
            return AnyOf(
                metricDescriptions,
                [&](const NCatboostOptions::TLossDescription& metricDescription) -> bool {
                    return predicate(metricDescription.LossFunction.Get());
                }
            );
        };

        CB_ENSURE_INTERNAL(
            !mainLossFunction || FindPtr(metricDescriptions, **mainLossFunction),
            "mainLossFunction is not in metricDescriptions"
        );

        bool hasClassificationMetrics = isAnyOfMetrics(IsClassificationMetric);
        bool hasMultiClassMetrics = isAnyOfMetrics(IsMultiClassMetric);
        ui32 classCountInData = 0;

        auto maybeConvertedTarget = ConvertTarget(
            rawData.GetTarget(),
            hasClassificationMetrics,
            hasMultiClassMetrics,
            knownClassCount == 0,
            classNames,
            localExecutor,
            &classCountInData);

        if (knownClassCount == 0) {
            knownClassCount = classCountInData;
        }
        ui32 classCount = (ui32)GetClassesCount((int)knownClassCount, *classNames);

        // some metrics are both binclass and multiclass (e.g. HingeLoss)

        bool hasBinClassOnlyMetrics = isAnyOfMetrics(
            [](ELossFunction lossFunction) {
                return IsBinaryClassMetric(lossFunction) && !IsMultiClassMetric(lossFunction);
            });

        bool hasMultiClassOnlyMetrics = isAnyOfMetrics(
            [](ELossFunction lossFunction) {
                return IsMultiClassMetric(lossFunction) && !IsBinaryClassMetric(lossFunction);
            });

        CB_ENSURE(
            !(hasBinClassOnlyMetrics && hasMultiClassOnlyMetrics),
            "Both binary classification -only and multiclassification -only loss function or metrics"
            " specified"
        );

        bool createBinClassTarget = false;
        bool createMultiClassTarget = false;

        if (hasBinClassOnlyMetrics) {
            createBinClassTarget = true;
        } else if (hasMultiClassOnlyMetrics) {
            createMultiClassTarget = true;
        } else {
            if (hasClassificationMetrics) {
                if (classCount > 2) {
                    createMultiClassTarget = true;
                } else {
                    createBinClassTarget = true;
                }
            }
        }


        /*
         * TODO(akhropov)
         * until target data provider selection supported by metrics use common data for all target
         *  data providers. MLTOOLS-2337.
         */
        TSharedWeights<float> adjustedWeights;
        TSharedVector<float> oneBaseline;
        TSharedVector<TQueryInfo> groupInfos;

        TTargetDataProviders result;

        if (createBinClassTarget) { // binclass
            CB_ENSURE(maybeConvertedTarget, "Binary classification loss/metrics require label data");
            CB_ENSURE(
                classCount == 2,
                "Binary classification loss/metric specified but target data is not 2-class labels"
            );

            adjustedWeights = MakeClassificationWeights(
                rawData.GetWeights(),
                rawData.GetGroupWeights(),
                (ui32)2,
                isForGpu,
                *maybeConvertedTarget,
                classWeights,
                localExecutor);

            oneBaseline = MakeOneBaseline(rawData.GetBaseline());

            if (mainLossFunction && ((*mainLossFunction)->GetLossFunction() == ELossFunction::Logloss)) {
                PrepareTargetBinary(
                    *maybeConvertedTarget,
                    NCatboostOptions::GetLogLossBorder(**mainLossFunction),
                    &*maybeConvertedTarget);

                TTargetDataSpecification specification(ETargetType::BinClass, "Binary");
                result.emplace(
                    specification,
                    MakeIntrusive<TBinClassTarget>(
                        specification.Description,
                        rawData.GetObjectsGrouping(),
                        maybeConvertedTarget,
                        adjustedWeights,
                        oneBaseline));
            }

            TTargetDataSpecification specification(ETargetType::BinClass);
            result.emplace(
                specification,
                MakeIntrusive<TBinClassTarget>(
                    specification.Description,
                    rawData.GetObjectsGrouping(),
                    maybeConvertedTarget,
                    adjustedWeights,
                    oneBaseline));
        }

        if (createMultiClassTarget) {
            CB_ENSURE(maybeConvertedTarget, "Multi classification loss/metrics require label data");
            CB_ENSURE(
                classCount >= 2,
                "Multiclass metric/loss specified but target data does not have more than one different labels"
            );

            adjustedWeights = MakeClassificationWeights(
                rawData.GetWeights(),
                rawData.GetGroupWeights(),
                (ui32)classCount,
                isForGpu,
                *maybeConvertedTarget,
                classWeights,
                localExecutor);

            if (!(*labelConverter)->IsInitialized()) {
                (*labelConverter)->Initialize(*maybeConvertedTarget, classCount);
            }
            PrepareTargetCompressed(**labelConverter, &*maybeConvertedTarget);

            TTargetDataSpecification specification(ETargetType::MultiClass);
            result.emplace(
                specification,
                MakeIntrusive<TMultiClassTarget>(
                    specification.Description,
                    rawData.GetObjectsGrouping(),
                    (ui32)(**labelConverter).GetApproxDimension(),
                    maybeConvertedTarget,
                    adjustedWeights,
                    MakeBaselines(rawData.GetBaseline(), (ui32)classCount)
                )
            );
        }

        if (!adjustedWeights) {
            adjustedWeights = MakeWeights(
                rawData.GetWeights(),
                rawData.GetGroupWeights(),
                isForGpu,
                localExecutor
            );
        }

        if (isAnyOfMetrics(IsRegressionMetric)) {
            CB_ENSURE(maybeConvertedTarget, "Regression loss/metrics require target data");

            if (!oneBaseline) {
                oneBaseline = MakeOneBaseline(rawData.GetBaseline());
            }

            TTargetDataSpecification specification(ETargetType::Regression);
            result.emplace(
                specification,
                MakeIntrusive<TRegressionTarget>(
                    specification.Description,
                    rawData.GetObjectsGrouping(),
                    maybeConvertedTarget,
                    adjustedWeights,
                    oneBaseline));
        }

        auto makeGroupInfos = [&]() {
            if (!groupInfos) {
                TConstArrayRef<TPair> pairs = rawData.GetPairs();

                TVector<TPair> generatedPairs;

                if (pairs.empty() && isAnyOfMetrics(IsPairwiseMetric)) {
                    CB_ENSURE(maybeConvertedTarget, "Pool labels are not provided. Cannot generate pairs.");

                    generatedPairs = GeneratePairs(
                        *rawData.GetObjectsGrouping(),
                        *maybeConvertedTarget,
                        metricDescriptions,
                        rand);

                    pairs = generatedPairs;
                }

                groupInfos = MakeGroupInfos(
                    *rawData.GetObjectsGrouping(),
                    subgroupIds,
                    rawData.GetGroupWeights(),
                    pairs);
            }
        };


        if (!rawData.GetWeights().IsTrivial() && isAnyOfMetrics(UsesPairsForCalculation)) {
            CATBOOST_WARNING_LOG << "Pairwise losses don't support object weights.";
        }


        bool hasGroupNonPairwiseDataMetrics = isAnyOfMetrics(
            [](ELossFunction lossFunction) {
                return IsGroupwiseMetric(lossFunction) && !IsPairwiseMetric(lossFunction);
            });

        // TODO(akhropov): always create group data if it is available (for compatibility)
        if (hasGroupNonPairwiseDataMetrics ||
            (!rawData.GetObjectsGrouping()->IsTrivial() && maybeConvertedTarget))
        {
            CB_ENSURE(maybeConvertedTarget, "Groupwise loss/metrics require target data");

            if (!oneBaseline) {
                oneBaseline = MakeOneBaseline(rawData.GetBaseline());
            }

            makeGroupInfos();

            TTargetDataSpecification specification(ETargetType::GroupwiseRanking);
            result.emplace(
                specification,
                MakeIntrusive<TGroupwiseRankingTarget>(
                    specification.Description,
                    rawData.GetObjectsGrouping(),
                    maybeConvertedTarget,
                    adjustedWeights,
                    oneBaseline,
                    groupInfos));
        }

        // TODO(akhropov): always create group data if it is available (for compatibility)
        if (isAnyOfMetrics(IsPairwiseMetric) ||
            (!rawData.GetObjectsGrouping()->IsTrivial() && !groupInfos))
        {
            if (!oneBaseline) {
                oneBaseline = MakeOneBaseline(rawData.GetBaseline());
            }

            makeGroupInfos();

            TTargetDataSpecification specification(ETargetType::GroupPairwiseRanking);
            result.emplace(
                specification,
                MakeIntrusive<TGroupPairwiseRankingTarget>(
                    specification.Description,
                    rawData.GetObjectsGrouping(),
                    oneBaseline,
                    groupInfos));
        }

        // TODO(akhropov): Will be split by target type. MLTOOLS-2337.
        CheckPreprocessedTarget(
            maybeConvertedTarget,
            datasetName,
            isLearnData,
            allowConstLabel,
            metricDescriptions);

        return result;
    }


    void InitClassesParams(
        const NJson::TJsonValue& param,
        TVector<float>* classWeights,
        TVector<TString>* classNames,
        ui32* classCount
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
            *classCount = SafeIntegerCast<ui32>(param["classes_count"].GetUIntegerSafe());
        }
    }


    TProcessedDataProvider CreateModelCompatibleProcessedDataProvider(
        const TDataProvider& srcData,
        const TFullModel& model,
        TRestorableFastRng64* rand, // for possible pairs generation
        NPar::TLocalExecutor* localExecutor) {

        const TString& modelInfoParams = model.ModelInfo.at("params");
        NJson::TJsonValue paramsJson = ReadTJsonValue(modelInfoParams);

        CB_ENSURE(paramsJson.Has("loss_function"), "No loss_function in model metadata params");

        TVector<NCatboostOptions::TLossDescription> metrics(1);
        metrics[0].Load(paramsJson["loss_function"]);

        TVector<float> classWeights;
        TVector<TString> classNames;
        ui32 classCount = 0;

        if (paramsJson.Has("data_processing_options")) {
            InitClassesParams(paramsJson["data_processing_options"], &classWeights, &classNames, &classCount);
        }

        TLabelConverter labelConverter;
        if (model.ObliviousTrees.ApproxDimension > 1) {  // is multiclass?
            if (model.ModelInfo.contains("multiclass_params")) {
                const auto& multiclassParamsJsonAsString = model.ModelInfo.at("multiclass_params");
                labelConverter.Initialize(multiclassParamsJsonAsString);
                TMulticlassLabelOptions multiclassOptions;
                multiclassOptions.Load(ReadTJsonValue(multiclassParamsJsonAsString));
                if (multiclassOptions.ClassNames.IsSet()) {
                    classNames = multiclassOptions.ClassNames;
                }
                if (multiclassOptions.ClassesCount.IsSet()) {
                    classCount = multiclassOptions.ClassesCount;
                }
            } else {
                labelConverter.Initialize(model.ObliviousTrees.ApproxDimension);
                classCount = model.ObliviousTrees.ApproxDimension;
            }
        }

        TProcessedDataProvider result;
        result.MetaInfo = srcData.MetaInfo;
        result.ObjectsGrouping = srcData.ObjectsGrouping;
        result.ObjectsData = srcData.ObjectsData;
        result.TargetData = CreateTargetDataProviders(
            srcData.RawTargetData,
            srcData.ObjectsData->GetSubgroupIds(),
            /*isForGpu*/ false,
            /*isLearn*/ false,
            /*datasetName*/ TStringBuf(),
            metrics,
            /*mainLossFunction*/ Nothing(),
            /*allowConstLabel*/ true,
            classCount,
            classWeights,
            &classNames,
            &labelConverter,
            rand,
            localExecutor
        );

        // in case pairs were generated
        if (result.TargetData.contains(TTargetDataSpecification(ETargetType::GroupPairwiseRanking))) {
            result.MetaInfo.HasPairs = true;
        }

        return result;
    }

}
