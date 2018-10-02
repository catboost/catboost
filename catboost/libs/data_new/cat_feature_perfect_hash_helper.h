#pragma once

#include "quantized_features_info.h"

#include <catboost/libs/helpers/array_subset.h>

#include <util/generic/maybe.h>
#include <util/generic/ptr.h>
#include <util/generic/vector.h>
#include <util/system/spinlock.h>
#include <util/system/types.h>


namespace NCB {

    class TCatFeaturesPerfectHashHelper {
    public:
        explicit TCatFeaturesPerfectHashHelper(TQuantizedFeaturesInfoPtr quantizedFeaturesInfo)
            : QuantizedFeaturesInfo(std::move(quantizedFeaturesInfo))
        {
        }

        ui32 GetUniqueValues(const TCatFeatureIdx catFeatureIdx) const {
            return QuantizedFeaturesInfo->CatFeaturesPerfectHash.GetUniqueValues(catFeatureIdx);
        }

        void UpdatePerfectHashAndMaybeQuantize(
            const TCatFeatureIdx catFeatureIdx,
            TMaybeOwningArraySubset<ui32, ui32> hashedCatArraySubset,
            TMaybe<TVector<ui32>*> dstBins
        );

    private:
        TQuantizedFeaturesInfoPtr QuantizedFeaturesInfo;
        TAdaptiveLock UpdateLock;
    };

}
