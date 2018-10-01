#pragma once

#include <catboost/libs/options/enums.h>
#include <catboost/libs/model/features.h>

#include <util/generic/array_ref.h>
#include <util/generic/ptr.h>
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

        bool operator==(const TFeatureMetaInfo& rhs) const;
    };


    class TFeaturesLayout final : public TAtomicRefCount<TFeaturesLayout> {
    public:
        // needed because of default init in Cython
        TFeaturesLayout() = default;

        TFeaturesLayout(const ui32 featureCount, TVector<ui32> catFeatureIndices, const TVector<TString>& featureId);
        TFeaturesLayout(const TVector<TFloatFeature>& floatFeatures, const TVector<TCatFeature>& catFeatures);

        bool operator==(const TFeaturesLayout& rhs) const;

        const TFeatureMetaInfo& GetInternalFeatureMetaInfo(
            ui32 internalFeatureIdx,
            EFeatureType type
        ) const {
            return ExternalIdxToMetaInfo[GetExternalFeatureIdx(internalFeatureIdx, type)];
        }

        // prefer this method to GetExternalFeatureIds
        TConstArrayRef<TFeatureMetaInfo> GetExternalFeaturesMetaInfo() const {
            return ExternalIdxToMetaInfo;
        }

        TString GetExternalFeatureDescription(ui32 internalFeatureIdx, EFeatureType type) const {
            return ExternalIdxToMetaInfo[GetExternalFeatureIdx(internalFeatureIdx, type)].Name;
        }
        TVector<TString> GetExternalFeatureIds() const;

        ui32 GetExternalFeatureIdx(ui32 internalFeatureIdx, EFeatureType type) const {
            if (type == EFeatureType::Float) {
                return FloatFeatureInternalIdxToExternalIdx[internalFeatureIdx];
            } else {
                return CatFeatureInternalIdxToExternalIdx[internalFeatureIdx];
            }
        }
        ui32 GetInternalFeatureIdx(ui32 externalFeatureIdx) const {
            Y_ASSERT(IsCorrectExternalFeatureIdx(externalFeatureIdx));
            return FeatureExternalIdxToInternalIdx[externalFeatureIdx];
        }
        EFeatureType GetExternalFeatureType(ui32 externalFeatureIdx) const {
            Y_ASSERT(IsCorrectExternalFeatureIdx(externalFeatureIdx));
            return ExternalIdxToMetaInfo[externalFeatureIdx].Type;
        }
        bool IsCorrectExternalFeatureIdx(ui32 externalFeatureIdx) const {
            return (size_t)externalFeatureIdx < ExternalIdxToMetaInfo.size();
        }

        bool IsCorrectInternalFeatureIdx(ui32 internalFeatureIdx, EFeatureType type) const {
            if (type == EFeatureType::Float) {
                return (size_t)internalFeatureIdx < FloatFeatureInternalIdxToExternalIdx.size();
            } else {
                return (size_t)internalFeatureIdx < CatFeatureInternalIdxToExternalIdx.size();
            }
        }
        bool IsCorrectExternalFeatureIdxAndType(ui32 externalFeatureIdx, EFeatureType type) const {
            if ((size_t)externalFeatureIdx >= ExternalIdxToMetaInfo.size()) {
                return false;
            }
            return ExternalIdxToMetaInfo[externalFeatureIdx].Type == type;
        }

        ui32 GetFloatFeatureCount() const {
            // cast is safe because of size invariant established in constructors
            return (ui32)FloatFeatureInternalIdxToExternalIdx.size();
        }
        ui32 GetCatFeatureCount() const {
            // cast is safe because of size invariant established in constructors
            return (ui32)CatFeatureInternalIdxToExternalIdx.size();
        }
        ui32 GetExternalFeatureCount() const {
            // cast is safe because of size invariant established in constructors
            return (ui32)ExternalIdxToMetaInfo.size();
        }

        ui32 GetFeatureCount(EFeatureType type) const {
            if (type == EFeatureType::Float) {
                return GetFloatFeatureCount();
            } else {
                return GetCatFeatureCount();
            }
        }

        void IgnoreExternalFeature(ui32 externalFeatureIdx) {
            auto& metaInfo = ExternalIdxToMetaInfo[externalFeatureIdx];
            metaInfo.IsIgnored = true;
            metaInfo.IsAvailable = false;
        }

    private:
        void InitIndices();

    private:
        TVector<TFeatureMetaInfo> ExternalIdxToMetaInfo;
        TVector<ui32> FeatureExternalIdxToInternalIdx;
        TVector<ui32> CatFeatureInternalIdxToExternalIdx;
        TVector<ui32> FloatFeatureInternalIdxToExternalIdx;
    };

    using TFeaturesLayoutPtr = TIntrusivePtr<TFeaturesLayout>;
}

