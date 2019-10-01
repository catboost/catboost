#pragma once

#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/data/util.h>
#include <catboost/libs/data_types/groupid.h>
#include <catboost/libs/helpers/restorable_rng.h>
#include <catboost/libs/labels/label_converter.h>
#include <catboost/libs/model/model.h>
#include <catboost/libs/options/loss_description.h>

#include <library/threading/local_executor/local_executor.h>

#include <util/generic/fwd.h>


namespace NCB {

    struct TInputClassificationInfo {
        TMaybe<ui32> KnownClassCount;
        TConstArrayRef<float> ClassWeights; // [classIdx], empty if not specified
        TVector<TString> ClassNames;
        TMaybe<float> TargetBorder;
    };

    struct TOutputClassificationInfo {
        TVector<TString> ClassNames;
        TMaybe<TLabelConverter*> LabelConverter; // needed only for multiclass
        TMaybe<float> TargetBorder; // TODO(isaf27): delete it from output parameters
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

    TTargetDataProviderPtr CreateTargetDataProvider(
        const TRawTargetDataProvider& rawData,
        TMaybeData<TConstArrayRef<TSubgroupId>> subgroupIds,
        bool isForGpu,
        bool isNonEmptyAndNonConst,
        TStringBuf datasetName,
        TConstArrayRef<NCatboostOptions::TLossDescription> metricDescriptions, // must be non-empty

        /* used to select whether to convert target to binary or not
         * pass nothing if target providers are created not for training
         * TODO(akhropov): will be removed with proper multi-target support. MLTOOLS-2337.
         */
        TMaybe<NCatboostOptions::TLossDescription*> mainLossFuncion,
        bool allowConstLabel,
        bool metricsThatRequireTargetCanBeSkipped,
        bool needTargetDataForCtrs,
        TMaybe<ui32> knownModelApproxDimension,
        const TInputClassificationInfo& inputClassificationInfo,
        TOutputClassificationInfo* outputClassificationInfo,
        TRestorableFastRng64* rand, // for possible pairs generation
        NPar::TLocalExecutor* localExecutor,
        TOutputPairsInfo* outputPairsInfo);


    TProcessedDataProvider CreateModelCompatibleProcessedDataProvider(
        const TDataProvider& srcData,

        // can be empty, then try to get the metric from loss_function parameter of the model
        TConstArrayRef<NCatboostOptions::TLossDescription> metricDescriptions,
        const TFullModel& model,
        ui64 cpuRamLimit,
        TRestorableFastRng64* rand, // for possible pairs generation
        NPar::TLocalExecutor* localExecutor);

    TProcessedDataProvider CreateClassificationCompatibleDataProvider(
        const TDataProvider& srcData,
        const TFullModel& model,
        ui64 cpuRamLimit,
        TRestorableFastRng64* rand, // for possible pairs generation
        NPar::TLocalExecutor* localExecutor);


    TSharedVector<TQueryInfo> MakeGroupInfos(
        const TObjectsGrouping& objectsGrouping,
        TMaybeData<TConstArrayRef<TSubgroupId>> subgroupIds,
        const TWeights<float>& groupWeights,
        TConstArrayRef<TPair> pairs);

}
