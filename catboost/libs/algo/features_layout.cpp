#include "features_layout.h"

#include <catboost/libs/helpers/exception.h>
#include <util/generic/algorithm.h>

TFeaturesLayout::TFeaturesLayout(const int featureCount, std::vector<int> catFeatureIndices, const TVector<TString>& featureId) {
    Sort(catFeatureIndices.begin(), catFeatureIndices.end());
    if (!catFeatureIndices.empty()) {
        CB_ENSURE(catFeatureIndices.back() < featureCount, "Invalid cat feature index " << catFeatureIndices.back());
        CB_ENSURE(catFeatureIndices[0] >= 0, "Cat feature indices should be >= 0");
    }

    FeatureType.resize(featureCount, EFeatureType::Float);
    InternalFeatureIdx.resize(featureCount);

    int catFeatureCount = catFeatureIndices.size();
    CatFeatureExternalId.resize(catFeatureCount);
    FloatFeatureExternalId.resize(featureCount - catFeatureCount);

    for (size_t i = 0; i < catFeatureIndices.size(); ++i) {
        int externalIdx = catFeatureIndices[i];
        Y_VERIFY(externalIdx < featureCount, "Cat feature indices must be less than feature count");
        FeatureType[externalIdx] = EFeatureType::Categorical;
        InternalFeatureIdx[externalIdx] = i;
        CatFeatureExternalId[i] = externalIdx;
    }

    int seenFloatFeatures = 0;
    for (int externalIdx = 0; externalIdx < featureCount; ++externalIdx) {
        if (FeatureType[externalIdx] == EFeatureType::Float) {
            FloatFeatureExternalId[seenFloatFeatures] = externalIdx;
            InternalFeatureIdx[externalIdx] = seenFloatFeatures;
            seenFloatFeatures++;
        }
    }
    FeatureId = featureId;
}
