#include "features.h"

#include "flatbuffers_serializer_helper.h"

TVector<TFloatFeature> CreateFloatFeatures(
    size_t allFeaturesCount,
    const THashSet<int>& catFeatures,
    const TVector<TString>& featureIds
) {
    TVector<TFloatFeature> floatFeatures(allFeaturesCount - catFeatures.size());
    int floatFeatureIdx = 0;
    for (size_t flatFeatureIdx = 0; flatFeatureIdx < allFeaturesCount; ++flatFeatureIdx) {
        if (catFeatures.has(flatFeatureIdx)) {
            continue;
        }
        auto& floatFeature = floatFeatures[floatFeatureIdx];
        floatFeature.FeatureIndex = floatFeatureIdx;
        floatFeature.FlatFeatureIndex = flatFeatureIdx;
        if (flatFeatureIdx < featureIds.size()) {
            floatFeature.FeatureId = featureIds[flatFeatureIdx];
        }
        ++floatFeatureIdx;
    }
    return floatFeatures;
}

flatbuffers::Offset<NCatBoostFbs::TCtrFeature> TCtrFeature::FBSerialize(TModelPartsCachingSerializer& serializer) const {
    return NCatBoostFbs::CreateTCtrFeatureDirect(
        serializer.FlatbufBuilder,
        serializer.GetOffset(Ctr),
        &Borders
    );
}
