#pragma once

#include <catboost/libs/options/enums.h>
#include <catboost/libs/model/features.h>

#include <util/generic/array_ref.h>
#include <util/generic/vector.h>
#include <util/generic/string.h>
#include <util/system/yassert.h>


namespace NCB {

    struct TFeatureMetaInfo {
        EFeatureType Type;
        TString Name;
        bool IsIgnored;

        /* some datasets can contain only part of all features present in the whole dataset
         * (e.g. workers in distributed processing)
         * ignored features are always unavailable
         */
        bool IsAvailable;

    public:
        TFeatureMetaInfo(
            EFeatureType type,
            const TString& name,
            bool isIgnored = false,
            bool isAvailable = true // isIgnored = true overrides this parameter
        )
            : Type(type)
            , Name(name)
            , IsIgnored(isIgnored)
            , IsAvailable(!isIgnored && isAvailable)
        {}
    };


    class TFeaturesLayout {
    public:
        // needed because of default init in Cython
        TFeaturesLayout() = default;

        TFeaturesLayout(const int featureCount, TVector<int> catFeatureIndices, const TVector<TString>& featureId);
        TFeaturesLayout(const TVector<TFloatFeature>& floatFeatures, const TVector<TCatFeature>& catFeatures);


        const TFeatureMetaInfo& GetInternalFeatureMetaInfo(
            int internalFeatureIdx,
            EFeatureType type
        ) const {
            return ExternalIdxToMetaInfo[GetExternalFeatureIdx(internalFeatureIdx, type)];
        }

        // prefer this method to GetExternalFeatureIds
        TConstArrayRef<TFeatureMetaInfo> GetExternalFeaturesMetaInfo() const {
            return ExternalIdxToMetaInfo;
        }

        TString GetExternalFeatureDescription(int internalFeatureIdx, EFeatureType type) const {
            return ExternalIdxToMetaInfo[GetExternalFeatureIdx(internalFeatureIdx, type)].Name;
        }
        TVector<TString> GetExternalFeatureIds() const;

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
            return ExternalIdxToMetaInfo[externalFeatureIdx].Type;
        }
        bool IsCorrectExternalFeatureIdx(int externalFeatureIdx) const {
            return externalFeatureIdx >= 0 && externalFeatureIdx < ExternalIdxToMetaInfo.ysize();
        }
        int GetFloatFeatureCount() const {
            return FloatFeatureInternalIdxToExternalIdx.ysize();
        }
        int GetCatFeatureCount() const {
            return CatFeatureInternalIdxToExternalIdx.ysize();
        }
        int GetExternalFeatureCount() const {
            return ExternalIdxToMetaInfo.ysize();
        }

        int GetFeatureCount(EFeatureType type) const {
            if (type == EFeatureType::Float) {
                return GetFloatFeatureCount();
            } else {
                return GetCatFeatureCount();
            }
        }

        void IgnoreExternalFeature(int externalFeatureIdx) {
            auto& metaInfo = ExternalIdxToMetaInfo[externalFeatureIdx];
            metaInfo.IsIgnored = true;
            metaInfo.IsAvailable = false;
        }

    private:
        void InitIndices();

    private:
        TVector<TFeatureMetaInfo> ExternalIdxToMetaInfo;
        TVector<int> FeatureExternalIdxToInternalIdx;
        TVector<int> CatFeatureInternalIdxToExternalIdx;
        TVector<int> FloatFeatureInternalIdxToExternalIdx;
    };

}

