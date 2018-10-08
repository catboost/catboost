#include "quantized_features_info.h"

#include "feature_index.h"

#include <catboost/libs/helpers/dbg_output.h>
#include <catboost/libs/helpers/vector_helpers.h>

#include <library/dbg_output/dump.h>

#include <util/generic/mapfindptr.h>
#include <util/generic/xrange.h>
#include <util/stream/output.h>


namespace NCB {
    static bool ApproximatelyEqualBorders(
        const TMap<ui32, TVector<float>>& lhs,
        const TMap<ui32, TVector<float>>& rhs
    ) {
        constexpr auto EPS = 1.e-6f;

        if (lhs.size() != rhs.size()) {
            return false;
        }

        for (const auto& featureIdxAndValue : lhs) {
            auto* rhsValue = MapFindPtr(rhs, featureIdxAndValue.first);
            if (!rhsValue) {
                return false;
            }
            if (!ApproximatelyEqual<float>(featureIdxAndValue.second, *rhsValue, EPS)) {
                return false;
            }
        }
        for (const auto& featureIdxAndValue : rhs) {
            if (!lhs.has(featureIdxAndValue.first)) {
                return false;
            }
        }
        return true;
    }


    bool TQuantizedFeaturesInfo::operator==(const TQuantizedFeaturesInfo& rhs) const {
        return (*FeaturesLayout == *rhs.FeaturesLayout) &&
            (FloatFeaturesBinarization == rhs.FloatFeaturesBinarization) &&
            ApproximatelyEqualBorders(Borders, rhs.Borders) && (NanModes == rhs.NanModes) &&
            (CatFeaturesPerfectHash == rhs.CatFeaturesPerfectHash);
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

}
