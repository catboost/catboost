#include "features_layout.h"

#include <catboost/libs/helpers/exception.h>
#include <util/generic/algorithm.h>



TFeaturesLayout::TFeaturesLayout(const int featureCount, std::vector<int> catFeatureIndices, const TVector<TString>& featureId)
    : ExternalIdxToFeatureId(featureId)
{
    InitIndices(featureCount, std::move(catFeatureIndices));
}

TFeaturesLayout::TFeaturesLayout(const TVector<TFloatFeature>& floatFeatures, const TVector<TCatFeature>& catFeatures) {
    for (const TFloatFeature& floatFeature : floatFeatures) {
        CB_ENSURE(floatFeature.FlatFeatureIndex != -1, "floatFeature.FlatFeatureIndex == -1");
        if ((size_t)floatFeature.FlatFeatureIndex >= ExternalIdxToFeatureId.size()) {
            ExternalIdxToFeatureId.resize(floatFeature.FlatFeatureIndex + 1);
        }
        ExternalIdxToFeatureId[floatFeature.FlatFeatureIndex] = floatFeature.FeatureId;
    }

    TVector<int> catFeatureIndices;
    for (const TCatFeature& catFeature : catFeatures) {
        CB_ENSURE(catFeature.FlatFeatureIndex != -1, "catFeature.FlatFeatureIndex == -1");
        if ((size_t)catFeature.FlatFeatureIndex >= ExternalIdxToFeatureId.size()) {
            ExternalIdxToFeatureId.resize(catFeature.FlatFeatureIndex + 1);
        }
        ExternalIdxToFeatureId[catFeature.FlatFeatureIndex] = catFeature.FeatureId;
        catFeatureIndices.push_back(catFeature.FlatFeatureIndex);
    }

    InitIndices(ExternalIdxToFeatureId.size(), std::move(catFeatureIndices));
}


void TFeaturesLayout::InitIndices(const int featureCount, std::vector<int> catFeatureIndices) {
    Sort(catFeatureIndices.begin(), catFeatureIndices.end());
    if (!catFeatureIndices.empty()) {
        CB_ENSURE(catFeatureIndices.back() < featureCount, "Invalid cat feature index " << catFeatureIndices.back());
        CB_ENSURE(catFeatureIndices[0] >= 0, "Cat feature indices should be >= 0");
    }

    ExternalIdxToFeatureType.resize(featureCount, EFeatureType::Float);
    FeatureExternalIdxToInternalIdx.resize(featureCount);

    int catFeatureCount = catFeatureIndices.size();
    CatFeatureInternalIdxToExternalIdx.resize(catFeatureCount);
    FloatFeatureInternalIdxToExternalIdx.resize(featureCount - catFeatureCount);

    for (size_t i = 0; i < catFeatureIndices.size(); ++i) {
        int externalIdx = catFeatureIndices[i];
        Y_VERIFY(externalIdx < featureCount, "Cat feature indices must be less than feature count");
        ExternalIdxToFeatureType[externalIdx] = EFeatureType::Categorical;
        FeatureExternalIdxToInternalIdx[externalIdx] = i;
        CatFeatureInternalIdxToExternalIdx[i] = externalIdx;
    }

    int seenFloatFeatures = 0;
    for (int externalIdx = 0; externalIdx < featureCount; ++externalIdx) {
        if (ExternalIdxToFeatureType[externalIdx] == EFeatureType::Float) {
            FloatFeatureInternalIdxToExternalIdx[seenFloatFeatures] = externalIdx;
            FeatureExternalIdxToInternalIdx[externalIdx] = seenFloatFeatures;
            seenFloatFeatures++;
        }
    }
}
