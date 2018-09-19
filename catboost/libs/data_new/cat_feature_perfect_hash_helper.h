#pragma once

#include "quantizations_manager.h"

#include <catboost/libs/helpers/array_subset.h>

#include <util/generic/maybe.h>
#include <util/generic/ptr.h>
#include <util/generic/vector.h>
#include <util/system/spinlock.h>
#include <util/system/types.h>


namespace NCB {

    class TCatFeaturesPerfectHashHelper {
    public:
        explicit TCatFeaturesPerfectHashHelper(TIntrusivePtr<TQuantizedFeaturesManager> featuresManager)
            : FeaturesManager(std::move(featuresManager))
        {
        }

        ui32 GetUniqueValues(ui32 dataProviderId) const {
            const ui32 featureId = FeaturesManager->GetFeatureManagerIdForCatFeature(dataProviderId);
            return FeaturesManager->CatFeaturesPerfectHash.GetUniqueValues(featureId);
        }

        void UpdatePerfectHashAndMaybeQuantize(
            ui32 dataProviderId,
            TMaybeOwningArraySubset<ui32, ui32> hashedCatArraySubset,
            TMaybe<TVector<ui32>*> dstBins
        );

    private:
        TIntrusivePtr<TQuantizedFeaturesManager> FeaturesManager;
        TAdaptiveLock UpdateLock;
    };

}
