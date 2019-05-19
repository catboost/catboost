#pragma once

#include "quantized_features_info.h"

#include <catboost/libs/helpers/array_subset.h>

#include <util/generic/array_ref.h>
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

        TCatFeatureUniqueValuesCounts GetUniqueValuesCounts(const TCatFeatureIdx catFeatureIdx) const {
            return QuantizedFeaturesInfo->CatFeaturesPerfectHash.GetUniqueValuesCounts(catFeatureIdx);
        }

        // thread-safe w.r.t. QuantizedFeaturesInfo
        void UpdatePerfectHashAndMaybeQuantize(
            const TCatFeatureIdx catFeatureIdx,
            TMaybeOwningConstArraySubset<ui32, ui32> hashedCatArraySubset,
            bool mapMostFrequentValueTo0,
            TMaybe<TArrayRef<ui32>*> dstBins
        );

    private:
        TQuantizedFeaturesInfoPtr QuantizedFeaturesInfo;
    };

}
