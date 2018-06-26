#pragma once

#include "binarizations_manager.h"
#include "data_utils.h"

#include <catboost/cuda/utils/compression_helpers.h>
#include <catboost/libs/helpers/exception.h>
#include <util/stream/file.h>
#include <util/system/spinlock.h>
#include <util/system/sem.h>
#include <util/random/shuffle.h>

namespace NCatboostCuda {
    class TCatFeaturesPerfectHashHelper {
    public:
        explicit TCatFeaturesPerfectHashHelper(TBinarizedFeaturesManager& featuresManager)
            : FeaturesManager(featuresManager)
        {
        }

        ui32 GetUniqueValues(ui32 dataProviderId) const {
            const ui32 featureId = FeaturesManager.GetFeatureManagerIdForCatFeature(dataProviderId);
            return FeaturesManager.CatFeaturesPerfectHash.GetUniqueValues(featureId);
        }

        TVector<ui32> UpdatePerfectHashAndBinarize(ui32 dataProviderId,
                                                   const float* hashesFloat,
                                                   ui32 hashesSize);

    private:
        TBinarizedFeaturesManager& FeaturesManager;
        TAdaptiveLock UpdateLock;
    };



}
