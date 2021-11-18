#include "features_layout.h"

#include <util/generic/string.h>


using namespace NCB;


TFeatureMetaInfo MakeFeatureMetaInfo(
    EFeatureType type,
    const TString& name,
    bool isSparse,
    bool isIgnored,
    bool isAvailable
) {
    return TFeatureMetaInfo(type, name, isSparse, isIgnored, isAvailable);
}

TFeaturesLayout MakeFeaturesLayout(TVector<TFeatureMetaInfo>* data) {
    return TFeaturesLayout(data);
}

TFeaturesLayout MakeFeaturesLayout(
    const int featureCount,
    const TVector<TString>& featureNames,
    const TVector<i32>& ignoredFeatures
) {
    TFeaturesLayout result(SafeIntegerCast<ui32>(featureCount), /*catFeatureIndices*/ {}, featureNames);

    for (auto i : ignoredFeatures) {
        result.IgnoreExternalFeature(SafeIntegerCast<ui32>(i));
    }

    return result;
}

NCB::TFeaturesLayoutPtr CloneWithSelectedFeatures(
    const NCB::TFeaturesLayout& featuresLayout,
    TConstArrayRef<i32> selectedFeatures
) {
    TConstArrayRef<TFeatureMetaInfo> srcFeaturesMetaInfo = featuresLayout.GetExternalFeaturesMetaInfo();
    TVector<TFeatureMetaInfo> dstFeaturesMetaInfo;
    dstFeaturesMetaInfo.reserve(srcFeaturesMetaInfo.size());
    for (const auto& srcFeatureMetaInfo : srcFeaturesMetaInfo) {
        dstFeaturesMetaInfo.push_back(srcFeatureMetaInfo);
        dstFeaturesMetaInfo.back().IsIgnored = true;
        dstFeaturesMetaInfo.back().IsAvailable = false;
    }
    for (auto i : selectedFeatures) {
        dstFeaturesMetaInfo[i].IsIgnored = false;
        dstFeaturesMetaInfo[i].IsAvailable = true;
    }
    return MakeIntrusive<TFeaturesLayout>(&dstFeaturesMetaInfo);
}
