#pragma once

#include <catboost/libs/data_new/data_provider.h>
#include <catboost/libs/data_new/util.h>
#include <catboost/libs/data_types/groupid.h>
#include <catboost/libs/helpers/restorable_rng.h>
#include <catboost/libs/labels/label_converter.h>
#include <catboost/libs/model/model.h>
#include <catboost/libs/options/loss_description.h>

#include <library/threading/local_executor/local_executor.h>

#include <util/generic/fwd.h>


namespace NCB {

    TTargetDataProviderPtr CreateTargetDataProvider(
        const TRawTargetDataProvider& rawData,
        TMaybeData<TConstArrayRef<TSubgroupId>> subgroupIds,
        bool isForGpu,
        bool isLearnData,
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
        ui32 knownClassCount, // == 0 if unknown
        TConstArrayRef<float> classWeights, // [classIdx], empty if not specified
        TVector<TString>* classNames, // inout parameter
        TMaybe<TLabelConverter*> labelConverter, // needed only for multiclass
        TRestorableFastRng64* rand, // for possible pairs generation
        NPar::TLocalExecutor* localExecutor,
        bool* hasPairs);


    TProcessedDataProvider CreateModelCompatibleProcessedDataProvider(
        const TDataProvider& srcData,

        // can be empty, then try to get the metric from loss_function parameter of the model
        TConstArrayRef<NCatboostOptions::TLossDescription> metricDescriptions,
        const TFullModel& model,
        TRestorableFastRng64* rand, // for possible pairs generation
        NPar::TLocalExecutor* localExecutor);

}
