#pragma once

#include "quantized_features_info.h"

#include <catboost/libs/helpers/polymorphic_type_containers.h>

#include <library/cpp/grid_creator/binarization.h>

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
            const ITypedArraySubset<ui32>& hashedCatArraySubset,
            bool mapMostFrequentValueTo0,
            TMaybe<TDefaultValue<ui32>> hashedCatDefaultValue,
            TMaybe<float> quantizedDefaultBinFraction,
            TMaybe<TArrayRef<ui32>*> dstBins
        );

    private:
        TQuantizedFeaturesInfoPtr QuantizedFeaturesInfo;
    };

}
