#pragma once

#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/data/pairs.h>
#include <catboost/libs/data/util.h>
#include <catboost/private/libs/data_types/groupid.h>
#include <catboost/libs/helpers/restorable_rng.h>
#include <catboost/private/libs/labels/label_converter.h>
#include <catboost/libs/model/model.h>
#include <catboost/private/libs/options/loss_description.h>

#include <library/cpp/json/json_value.h>
#include <library/cpp/threading/local_executor/local_executor.h>

#include <util/generic/fwd.h>


namespace NCB {
    struct TTargetCreationOptions {
        bool IsClass;
        bool IsMultiClass;
        bool IsMultiLabel;
        bool CreateBinClassTarget;
        bool CreateMultiClassTarget;
        bool CreateMultiLabelTarget;
        bool CreateGroups;
        bool CreatePairs;
        bool SkipMinMaxPairsCheck;
        TMaybe<ui32> MaxPairsCount;
        bool AllowConstLabel;
    };

    struct TInputClassificationInfo {
        TMaybe<ui32> KnownClassCount;
        TConstArrayRef<float> ClassWeights; // [classIdx], empty if not specified
        EAutoClassWeightsType AutoClassWeightsType;
        TVector<NJson::TJsonValue> ClassLabels; // can be Integers, Floats or Strings
        TMaybe<float> TargetBorder;
    };

    struct TOutputClassificationInfo {
        TVector<NJson::TJsonValue> ClassLabels; // can be Integers, Floats or Strings
        TMaybe<TLabelConverter*> LabelConverter; // needed only for multiclass
        TMaybe<float> TargetBorder; // TODO(isaf27): delete it from output parameters
        TMaybe<TVector<float>> ClassWeights;
    };

    struct TOutputPairsInfo {
        bool HasPairs;
        TObjectsGrouping FakeObjectsGrouping;
        TVector<ui32> PermutationForGrouping;
        TVector<TPair> PairsInPermutedDataset;

        bool HasFakeGroupIds() const {
            return !PermutationForGrouping.empty();
        }
    };

    TTargetCreationOptions MakeTargetCreationOptions(
        bool dataHasWeights,
        ui32 dataTargetDimension,
        bool dataHasGroups,
        TConstArrayRef<NCatboostOptions::TLossDescription> metricDescriptions,
        TMaybe<ui32> knownModelApproxDimension,
        bool knownIsClassification,
        const TInputClassificationInfo& inputClassificationInfo,
        bool allowConstLabel,
        bool hasGraph,
        bool skipMinMaxPairsCheck=false);

    TTargetCreationOptions MakeTargetCreationOptions(
        const TRawTargetDataProvider &rawData,
        TConstArrayRef<NCatboostOptions::TLossDescription> metricDescriptions,
        TMaybe<ui32> knownModelApproxDimension,
        const TInputClassificationInfo& inputClassificationInfo,
        bool allowConstLabel,
        bool hasGraph,
        bool skipMinMaxPairsCheck=false);

    void CheckTargetConsistency(
        TTargetDataProviderPtr targetDataProvider,
        TConstArrayRef<NCatboostOptions::TLossDescription> metricDescriptions,
        const NCatboostOptions::TLossDescription* mainLossFunction, // can be nullptr
        bool needTargetDataForCtrs,
        bool metricsThatRequireTargetCanBeSkipped,
        TStringBuf datasetName,
        bool isNonEmptyAndNonConst,
        bool allowConstLabel);

    TSharedWeights<float> MakeWeights(
        const TWeights<float>& rawWeights,
        const TWeights<float>& rawGroupWeights,
        bool isForGpu,
        NPar::ILocalExecutor* localExecutor);

    TTargetDataProviderPtr CreateTargetDataProvider(
        const TRawTargetDataProvider& rawData,
        TMaybeData<TConstArrayRef<TSubgroupId>> subgroupIds,
        bool isForGpu,

        /* used to select whether to convert target to binary or not
         * pass nullptr if target providers are created not for training
         * TODO(akhropov): will be removed with proper multi-target support. MLTOOLS-2337.
         */
        const NCatboostOptions::TLossDescription* mainLossFuncion,
        bool metricsThatRequireTargetCanBeSkipped,
        TMaybe<ui32> knownModelApproxDimension,
        const TTargetCreationOptions& targetCreationOptions,
        const TInputClassificationInfo& inputClassificationInfo,
        TOutputClassificationInfo* outputClassificationInfo,
        TRestorableFastRng64* rand, // for possible pairs generation
        NPar::ILocalExecutor* localExecutor,
        TOutputPairsInfo* outputPairsInfo);


    TProcessedDataProvider CreateModelCompatibleProcessedDataProvider(
        const TDataProvider& srcData,

        // can be empty, then try to get the metric from loss_function parameter of the model
        TConstArrayRef<NCatboostOptions::TLossDescription> metricDescriptions,
        const TFullModel& model,
        ui64 cpuRamLimit,
        TRestorableFastRng64* rand, // for possible pairs generation
        NPar::ILocalExecutor* localExecutor,
        bool metricsThatRequireTargetCanBeSkipped=false,
        bool skipMinMaxPairsCheck=false,
        bool skipTargetConsistencyCheck=false);

    TProcessedDataProvider CreateClassificationCompatibleDataProvider(
        const TDataProvider& srcData,
        const TFullModel& model,
        ui64 cpuRamLimit,
        TRestorableFastRng64* rand, // for possible pairs generation
        NPar::ILocalExecutor* localExecutor);


    TSharedVector<TQueryInfo> MakeGroupInfos(
        const TObjectsGrouping& objectsGrouping,
        TMaybeData<TConstArrayRef<TSubgroupId>> subgroupIds,
        const TWeights<float>& groupWeights,
        TMaybe<TRawPairsDataRef> pairs);

    void UpdateTargetProcessingParams(
        const TInputClassificationInfo& inputClassificationInfo,
        const TTargetCreationOptions& targetCreationOptions,
        TMaybe<ui32> knownApproxDimension,
        const NCatboostOptions::TLossDescription* mainLossFunction, // can be nullptr
        bool* isRealTarget,
        TMaybe<ui32>* knownClassCount,
        TInputClassificationInfo* updatedInputClassificationInfo);

    TVector<TSharedVector<float>> ConvertTarget(
        TMaybeData<TConstArrayRef<TRawTarget>> maybeRawTarget,
        ERawTargetType targetType,
        bool isRealTarget,
        bool isClass,
        bool isMultiClass,
        bool isMultiLabel,
        TMaybe<float> targetBorder,
        bool classCountUnknown,
        const TVector<NJson::TJsonValue> inputClassLabels,
        bool allowConstLabel,
        TVector<NJson::TJsonValue>* outputClassLabels,
        NPar::ILocalExecutor* localExecutor,
        ui32* classCount);
}
