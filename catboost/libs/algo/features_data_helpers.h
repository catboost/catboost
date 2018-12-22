#pragma once

#include <catboost/libs/data_new/objects.h>
#include <catboost/libs/helpers/exception.h>


namespace NCB {

    inline const float* GetRawFeatureDataBeginPtr(
        const TRawObjectsDataProvider& rawObjectsData,
        const TFeaturesLayout& featuresLayout,
        ui32 consecutiveSubsetBegin,
        ui32 flatFeatureIdx) {

        const ui32 internalFeatureIdx = featuresLayout.GetInternalFeatureIdx(flatFeatureIdx);
        if (featuresLayout.GetExternalFeatureType(flatFeatureIdx) == EFeatureType::Float) {
            return (*(*(**rawObjectsData.GetFloatFeature(internalFeatureIdx)).GetArrayData().GetSrc()
                    )).data() + consecutiveSubsetBegin;
        } else {
            return reinterpret_cast<const float*>((*(*(**rawObjectsData.GetCatFeature(internalFeatureIdx))
                    .GetArrayData().GetSrc())).data()) + consecutiveSubsetBegin;
        }
    }

    inline ui32 GetConsecutiveSubsetBegin(const TRawObjectsDataProvider& rawObjectsData) {
        const auto maybeConsecutiveSubsetBegin =
            rawObjectsData.GetFeaturesArraySubsetIndexing().GetConsecutiveSubsetBegin();
        CB_ENSURE_INTERNAL(
            maybeConsecutiveSubsetBegin,
            "Only consecutive feature data is supported for apply"
        );
        return *maybeConsecutiveSubsetBegin;
    }

}
