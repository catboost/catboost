#include "quantized_features_info.h"

#include "feature_index.h"

#include <util/generic/xrange.h>
#include <util/stream/output.h>


namespace NCB {

    bool TQuantizedFeaturesInfo::operator==(const TQuantizedFeaturesInfo& rhs) const {
        return (*FeaturesLayout == *rhs.FeaturesLayout) &&
            (FloatFeaturesBinarization == rhs.FloatFeaturesBinarization) && (Borders == rhs.Borders) &&
            (NanModes == rhs.NanModes) && (CatFeaturesPerfectHash == rhs.CatFeaturesPerfectHash);
    }

    ENanMode TQuantizedFeaturesInfo::ComputeNanMode(const TFloatValuesHolder& feature) const {
        if (FloatFeaturesBinarization.NanMode == ENanMode::Forbidden) {
            return ENanMode::Forbidden;
        }
        TConstMaybeOwningArraySubset<float, ui32> arrayData = feature.GetArrayData();

        bool hasNans = arrayData.Find([] (size_t /*idx*/, float value) { return IsNan(value); });
        if (hasNans) {
            return FloatFeaturesBinarization.NanMode;
        }
        return ENanMode::Forbidden;
    }

    void TQuantizedFeaturesInfo::SetOrCheckNanMode(const TFloatValuesHolder& feature, ENanMode nanMode)  {
        const auto floatFeatureIdx = GetPerTypeFeatureIdx<EFeatureType::Float>(feature);
        if (!NanModes.has(*floatFeatureIdx)) {
            NanModes[*floatFeatureIdx] = nanMode;
        } else {
            CB_ENSURE(NanModes.at(*floatFeatureIdx) == nanMode, "NaN mode should be consistent " << nanMode);
        }
    }

    ENanMode TQuantizedFeaturesInfo::GetOrComputeNanMode(const TFloatValuesHolder& feature)  {
        const auto floatFeatureIdx = GetPerTypeFeatureIdx<EFeatureType::Float>(feature);
        if (!NanModes.has(*floatFeatureIdx)) {
            NanModes[*floatFeatureIdx] = ComputeNanMode(feature);
        }
        return NanModes.at(*floatFeatureIdx);
    }

    ENanMode TQuantizedFeaturesInfo::GetNanMode(const TFloatFeatureIdx floatFeatureIdx) const  {
        CheckCorrectPerTypeFeatureIdx(floatFeatureIdx);
        ENanMode nanMode = ENanMode::Forbidden;
        if (NanModes.has(*floatFeatureIdx)) {
            nanMode = NanModes.at(*floatFeatureIdx);
        }
        return nanMode;
    }

    const TVector<float>& TQuantizedFeaturesInfo::GetBorders(const TFloatValuesHolder& feature) const {
        const auto floatFeatureIdx = GetPerTypeFeatureIdx<EFeatureType::Float>(feature);
        return Borders.at(*floatFeatureIdx);
    }

}
