#pragma once

#include <catboost/libs/options/enums.h>
#include <catboost/libs/model/features.h>

#include <util/generic/vector.h>
#include <util/generic/string.h>
#include <util/system/yassert.h>


namespace NCB {

    class TFeaturesLayout {
    public:
        // needed because of default init in Cython
        TFeaturesLayout() = default;

        TFeaturesLayout(const int featureCount, TVector<int> catFeatureIndices, const TVector<TString>& featureId);
        TFeaturesLayout(const TVector<TFloatFeature>& floatFeatures, const TVector<TCatFeature>& catFeatures);


        TString GetExternalFeatureDescription(int internalFeatureIdx, EFeatureType type) const {
            return ExternalIdxToFeatureId[GetExternalFeatureIdx(internalFeatureIdx, type)];
        }
        const TVector<TString>& GetExternalFeatureIds() const {
            return ExternalIdxToFeatureId;
        }
        int GetExternalFeatureIdx(int internalFeatureIdx, EFeatureType type) const {
            if (type == EFeatureType::Float) {
                return FloatFeatureInternalIdxToExternalIdx[internalFeatureIdx];
            } else {
                return CatFeatureInternalIdxToExternalIdx[internalFeatureIdx];
            }
        }
        int GetInternalFeatureIdx(int externalFeatureIdx) const {
            Y_ASSERT(IsCorrectExternalFeatureIdx(externalFeatureIdx));
            return FeatureExternalIdxToInternalIdx[externalFeatureIdx];
        }
        EFeatureType GetExternalFeatureType(int externalFeatureIdx) const {
            Y_ASSERT(IsCorrectExternalFeatureIdx(externalFeatureIdx));
            return ExternalIdxToFeatureType[externalFeatureIdx];
        }
        bool IsCorrectExternalFeatureIdx(int externalFeatureIdx) const {
            return externalFeatureIdx >= 0 && externalFeatureIdx < ExternalIdxToFeatureType.ysize();
        }
        int GetCatFeatureCount() const {
            return CatFeatureInternalIdxToExternalIdx.ysize();
        }
        int GetExternalFeatureCount() const {
            return ExternalIdxToFeatureType.ysize();
        }

    private:
        void InitIndices(const int featureCount, TVector<int> catFeatureIndices);

    private:
        TVector<EFeatureType> ExternalIdxToFeatureType;
        TVector<int> FeatureExternalIdxToInternalIdx;
        TVector<int> CatFeatureInternalIdxToExternalIdx;
        TVector<int> FloatFeatureInternalIdxToExternalIdx;
        TVector<TString> ExternalIdxToFeatureId;
    };

}

