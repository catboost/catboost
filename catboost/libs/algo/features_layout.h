#pragma once


#include <catboost/libs/options/enums.h>
#include <util/generic/vector.h>
#include <util/generic/string.h>
#include <util/system/yassert.h>

class TFeaturesLayout {
public:
    TFeaturesLayout(const int featureCount, std::vector<int> catFeatureIndices, const TVector<TString>& featureId);

    TString GetExternalFeatureDescription(int internalFeatureIdx, EFeatureType type) const {
        if (FeatureId.empty()) {
            return TString();
        } else {
            int featureIdx = GetFeature(internalFeatureIdx, type);
            return FeatureId[featureIdx];
        }
    }
    int GetFeature(int internalFeatureIdx, EFeatureType type) const {
        if (type == EFeatureType::Float) {
            return FloatFeatureExternalId[internalFeatureIdx];
        } else {
            return CatFeatureExternalId[internalFeatureIdx];
        }
    }
    int GetInternalFeatureIdx(int feature) const {
        Y_ASSERT(IsCorrectFeatureIdx(feature));
        return InternalFeatureIdx[feature];
    }
    EFeatureType GetFeatureType(int feature) const {
        Y_ASSERT(IsCorrectFeatureIdx(feature));
        return FeatureType[feature];
    }
    bool IsCorrectFeatureIdx(int feature) const {
        return feature >= 0 && feature < FeatureType.ysize();
    }
    int GetCatFeatureCount() const {
        return CatFeatureExternalId.ysize();
    }

private:
    TVector<EFeatureType> FeatureType;
    TVector<int> InternalFeatureIdx;
    TVector<int> CatFeatureExternalId;
    TVector<int> FloatFeatureExternalId;
    TVector<TString> FeatureId;
};
