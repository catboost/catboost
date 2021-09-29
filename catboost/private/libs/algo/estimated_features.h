#pragma once

#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/data/feature_estimators.h>
#include <catboost/libs/data/quantized_features_info.h>

#include <util/generic/array_ref.h>
#include <util/generic/fwd.h>
#include <util/generic/maybe.h>
#include <util/generic/vector.h>
#include <util/system/types.h>


namespace NPar {
    class ILocalExecutor;
}

namespace NCatboostOptions {
    struct TBinarizationOptions;
}

struct TRestorableFastRng64;


namespace NCB {

    TEstimatedForCPUObjectsDataProviders CreateEstimatedFeaturesData(
        const NCatboostOptions::TBinarizationOptions& quantizationOptions,
        ui32 maxSubsetSizeForBuildBordersAlgorithms,
        NCB::TQuantizedFeaturesInfoPtr quantizedFeaturesInfo, // can be nullptr
        const TTrainingDataProviders& trainingDataProviders,
        TFeatureEstimatorsPtr featureEstimators,

        // calculate online features if defined, offline - otherwise
        TMaybe<TConstArrayRef<ui32>> learnPermutation,
        NPar::ILocalExecutor* localExecutor,
        TRestorableFastRng64* rand
    );

    struct TCombinedEstimatedFeaturesContext {
        TFeatureEstimatorsPtr FeatureEstimators;
        TVector<TEstimatedFeatureId> OfflineEstimatedFeaturesLayout;
        TVector<TEstimatedFeatureId> OnlineEstimatedFeaturesLayout;
    };
}
