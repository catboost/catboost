#include "features_layout.h"

#include "util.h"

#include <catboost/libs/helpers/exception.h>
#include <util/generic/algorithm.h>
#include <util/generic/xrange.h>

using namespace NCB;


TFeaturesLayout::TFeaturesLayout(const ui32 featureCount, TVector<ui32> catFeatureIndices, const TVector<TString>& featureId)
{
    CheckDataSize(featureId.size(), (size_t)featureCount, "feature Ids", true, "feature count");

    ExternalIdxToMetaInfo.reserve((size_t)featureCount);
    for (auto externalFeatureIdx : xrange(featureCount)) {
        // cat feature will be set later
        ExternalIdxToMetaInfo.emplace_back(
            EFeatureType::Float,
            !featureId.empty() ? featureId[externalFeatureIdx] : ""
        );
    }
    for (auto catFeatureExternalIdx : catFeatureIndices) {
        CB_ENSURE(
            (catFeatureExternalIdx >= 0) && (catFeatureExternalIdx < featureCount),
            "Cat feature index (" << catFeatureExternalIdx << ") is out of valid range [0,"
            << featureCount << ')'
        );
        ExternalIdxToMetaInfo[catFeatureExternalIdx].Type = EFeatureType::Categorical;
    }

    InitIndices();
}

TFeaturesLayout::TFeaturesLayout(const TVector<TFloatFeature>& floatFeatures, const TVector<TCatFeature>& catFeatures) {
    TFeatureMetaInfo defaultIgnoredMetaInfo(EFeatureType::Float, TString(), true);

    for (const TFloatFeature& floatFeature : floatFeatures) {
        CB_ENSURE(floatFeature.FlatFeatureIndex >= 0, "floatFeature.FlatFeatureIndex is negative");
        if ((size_t)floatFeature.FlatFeatureIndex >= ExternalIdxToMetaInfo.size()) {
            CB_ENSURE(
                (size_t)floatFeature.FlatFeatureIndex < (size_t)Max<ui32>(),
                "floatFeature.FlatFeatureIndex is greater than maximum allowed index: " << (Max<ui32>() - 1)
            );
            ExternalIdxToMetaInfo.resize(floatFeature.FlatFeatureIndex + 1, defaultIgnoredMetaInfo);
        }
        ExternalIdxToMetaInfo[floatFeature.FlatFeatureIndex] =
            TFeatureMetaInfo(EFeatureType::Float, floatFeature.FeatureId);
    }

    for (const TCatFeature& catFeature : catFeatures) {
        CB_ENSURE(catFeature.FlatFeatureIndex >= 0, "catFeature.FlatFeatureIndex is negative");
        if ((size_t)catFeature.FlatFeatureIndex >= ExternalIdxToMetaInfo.size()) {
            CB_ENSURE(
                (size_t)catFeature.FlatFeatureIndex < (size_t)Max<ui32>(),
                "catFeature.FlatFeatureIndex is greater than maximum allowed index: " << (Max<ui32>() - 1)
            );
            ExternalIdxToMetaInfo.resize(catFeature.FlatFeatureIndex + 1, defaultIgnoredMetaInfo);
        }
        ExternalIdxToMetaInfo[catFeature.FlatFeatureIndex] =
            TFeatureMetaInfo(EFeatureType::Categorical, catFeature.FeatureId);
    }

    InitIndices();
}


TVector<TString> TFeaturesLayout::GetExternalFeatureIds() const {
    TVector<TString> result;
    result.reserve(ExternalIdxToMetaInfo.size());
    for (const auto& metaInfo : ExternalIdxToMetaInfo) {
        result.push_back(metaInfo.Name);
    }
    return result;
}

void TFeaturesLayout::InitIndices() {
    for (auto externalFeatureIdx : xrange(ExternalIdxToMetaInfo.size())) {
        if (ExternalIdxToMetaInfo[externalFeatureIdx].Type == EFeatureType::Float) {
            FeatureExternalIdxToInternalIdx.push_back((ui32)FloatFeatureInternalIdxToExternalIdx.size());
            FloatFeatureInternalIdxToExternalIdx.push_back(externalFeatureIdx);
        } else {
            FeatureExternalIdxToInternalIdx.push_back((ui32)CatFeatureInternalIdxToExternalIdx.size());
            CatFeatureInternalIdxToExternalIdx.push_back(externalFeatureIdx);
        }
    }
}
