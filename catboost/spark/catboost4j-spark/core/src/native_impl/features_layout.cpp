#include "features_layout.h"

#include <util/generic/cast.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>

using namespace NCB;


TFeatureMetaInfo MakeFeatureMetaInfo(
    EFeatureType type,
    const TString& name,
    bool isSparse,
    bool isIgnored,
    bool isAvailable
) throw (yexception) {
    return TFeatureMetaInfo(type, name, isSparse, isIgnored, isAvailable);
}

TFeaturesLayout MakeFeaturesLayout(TVector<TFeatureMetaInfo>* data) throw (yexception) {
    return TFeaturesLayout(data);
}

TFeaturesLayout MakeFeaturesLayout(
    const int featureCount,
    const TVector<TString>& featureNames,
    const TVector<i32>& ignoredFeatures
) throw (yexception) {
    TFeaturesLayout result(SafeIntegerCast<ui32>(featureCount), /*catFeatureIndices*/ {}, featureNames);

    for (auto i : ignoredFeatures) {
        result.IgnoreExternalFeature(SafeIntegerCast<ui32>(i));
    }

    return result;
}

TVector<i32> GetAvailableFloatFeatures(const TFeaturesLayout& featuresLayout) throw (yexception) {
    TVector<i32> result;
    result.reserve(featuresLayout.GetFloatFeatureCount());
    featuresLayout.IterateOverAvailableFeatures<EFeatureType::Float>(
        [&] (TFloatFeatureIdx idx) {
            result.push_back(SafeIntegerCast<i32>(*idx));
        }
    );

    return result;
}
