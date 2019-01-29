#pragma once

#include "feature_index.h"

#include <catboost/libs/model/features.h>
#include <catboost/libs/options/enums.h>

#include <library/binsaver/bin_saver.h>
#include <library/dbg_output/dump.h>

#include <util/generic/array_ref.h>
#include <util/generic/ptr.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/generic/xrange.h>
#include <util/generic/ymath.h>
#include <util/system/yassert.h>

namespace NCB {
    struct TPoolQuantizationSchema;
}

namespace NCB {

    struct TFeatureMetaInfo {
        EFeatureType Type;
        TString Name;
        bool IsIgnored = false;

        /* some datasets can contain only part of all features present in the whole dataset
         * (e.g. workers in distributed processing)
         * ignored features are always unavailable
         */
        bool IsAvailable = true;

    public:
        // needed for BinSaver
        TFeatureMetaInfo() = default;

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

        SAVELOAD(Type, Name, IsIgnored, IsAvailable);
    };

}

template <>
struct TDumper<NCB::TFeatureMetaInfo> {
    template <class S>
    static inline void Dump(S& s, const NCB::TFeatureMetaInfo& featureMetaInfo) {
        s << "Type=" << featureMetaInfo.Type << "\tName=" << featureMetaInfo.Name
          << "\tIsIgnored=" << featureMetaInfo.IsIgnored << "\tIsAvailable=" << featureMetaInfo.IsAvailable;
    }
};



namespace NCB {

    class TQuantizedFeatureIndexRemapper {
    public:
        TQuantizedFeatureIndexRemapper() = default;
        explicit TQuantizedFeatureIndexRemapper(ui32 indexOffset, ui32 binCount);

        SAVELOAD(IndexOffset_, BinCount_)

        ui32 GetIdxOffset() const {
            return IndexOffset_;
        }

        ui32 GetBinsPerArtificialFeature() const {
            // Leave one extra bin for "missing" artificial feature
            return 255;
        }

        ui32 GetRemappedIdx(ui32 binIdx) const {
            return IndexOffset_ + binIdx / GetBinsPerArtificialFeature();
        }

        ui32 GetArtificialFeatureCount() const {
            if (BinCount_) {
                return CeilDiv<ui32>(BinCount_, GetBinsPerArtificialFeature()) - 1;
            }

            return 0;
        }

        bool HasArtificialFeatures() const {
            return GetArtificialFeatureCount() > 0;
        }

    private:
        ui32 IndexOffset_ = 0;
        ui32 BinCount_ = 0;
    };

    class TFeaturesLayout final : public TAtomicRefCount<TFeaturesLayout> {
    public:
        // needed because of default init in Cython and because of BinSaver
        TFeaturesLayout() = default;
        explicit TFeaturesLayout(const ui32 featureCount);
        TFeaturesLayout(
            const ui32 featureCount,
            const TVector<ui32>& catFeatureIndices,
            const TVector<TString>& featureId,
            const NCB::TPoolQuantizationSchema* quantizationSchema);
        TFeaturesLayout(
            const TVector<TFloatFeature>& floatFeatures,
            const TVector<TCatFeature>& catFeatures);

        bool operator==(const TFeaturesLayout& rhs) const;

        SAVELOAD(
            ExternalIdxToMetaInfo,
            FeatureExternalIdxToInternalIdx,
            CatFeatureInternalIdxToExternalIdx,
            FloatFeatureInternalIdxToExternalIdx)

        const TFeatureMetaInfo& GetInternalFeatureMetaInfo(
            ui32 internalFeatureIdx,
            EFeatureType type) const;

        // prefer this method to GetExternalFeatureIds
        TConstArrayRef<TFeatureMetaInfo> GetExternalFeaturesMetaInfo() const;

        TString GetExternalFeatureDescription(ui32 internalFeatureIdx, EFeatureType type) const;

        TVector<TString> GetExternalFeatureIds() const;

        // needed for python-package
        void SetExternalFeatureIds(TConstArrayRef<TString> featureIds);

        ui32 GetExternalFeatureIdx(ui32 internalFeatureIdx, EFeatureType type) const;

        ui32 GetInternalFeatureIdx(ui32 externalFeatureIdx) const;

        template <EFeatureType FeatureType>
        TFeatureIdx<FeatureType> GetInternalFeatureIdx(ui32 externalFeatureIdx) const {
            Y_ASSERT(IsCorrectExternalFeatureIdxAndType(externalFeatureIdx, FeatureType));
            return TFeatureIdx<FeatureType>(FeatureExternalIdxToInternalIdx[externalFeatureIdx]);
        }

        EFeatureType GetExternalFeatureType(ui32 externalFeatureIdx) const;

        bool IsCorrectExternalFeatureIdx(ui32 externalFeatureIdx) const;

        bool IsCorrectInternalFeatureIdx(ui32 internalFeatureIdx, EFeatureType type) const;

        bool IsCorrectExternalFeatureIdxAndType(ui32 externalFeatureIdx, EFeatureType type) const;

        ui32 GetFloatFeatureCount() const;

        ui32 GetCatFeatureCount() const;

        ui32 GetExternalFeatureCount() const;

        ui32 GetFeatureCount(EFeatureType type) const;

        void IgnoreExternalFeature(ui32 externalFeatureIdx);

        // indices in list can be outside of range of features in layout - such features are ignored
        void IgnoreExternalFeatures(TConstArrayRef<ui32> ignoredFeatures);

        // Function must get one param -  TFeatureIdx<FeatureType>
        template <EFeatureType FeatureType, class Function>
        void IterateOverAvailableFeatures(Function&& f) const {
            const ui32 perTypeFeatureCount = GetFeatureCount(FeatureType);

            for (auto perTypeFeatureIdx : xrange(perTypeFeatureCount)) {
                if (GetInternalFeatureMetaInfo(perTypeFeatureIdx, FeatureType).IsAvailable) {
                    f(TFeatureIdx<FeatureType>(perTypeFeatureIdx));
                }
            }
        }

        TConstArrayRef<ui32> GetCatFeatureInternalIdxToExternalIdx() const;

        bool HasAvailableAndNotIgnoredFeatures() const;

        TQuantizedFeatureIndexRemapper GetQuantizedFeatureIndexRemapper(
            ui32 externalFeatureIdx) const;

    private:
        TVector<TQuantizedFeatureIndexRemapper> ExternalIdxRemappers;
        TVector<TFeatureMetaInfo> ExternalIdxToMetaInfo;
        TVector<ui32> FeatureExternalIdxToInternalIdx;
        TVector<ui32> CatFeatureInternalIdxToExternalIdx;
        TVector<ui32> FloatFeatureInternalIdxToExternalIdx;
    };

    using TFeaturesLayoutPtr = TIntrusivePtr<TFeaturesLayout>;

    void CheckCompatibleForApply(
        const TFeaturesLayout& learnFeaturesLayout,
        const TFeaturesLayout& applyFeaturesLayout,
        const TString& applyDataName
    );
}


template <>
struct TDumper<NCB::TFeaturesLayout> {
    template <class S>
    static inline void Dump(S& s, const NCB::TFeaturesLayout& featuresLayout) {
        auto externalFeaturesMetaInfo = featuresLayout.GetExternalFeaturesMetaInfo();
        for (auto externalFeatureIdx : xrange(externalFeaturesMetaInfo.size())) {
            s << "externalFeatureIdx=" << externalFeatureIdx
              << "\tinternalFeatureIdx=" << featuresLayout.GetInternalFeatureIdx(externalFeatureIdx)
              << "\tMetaInfo={" << DbgDump(externalFeaturesMetaInfo[externalFeatureIdx]) << "}\n";
        }
    }
};


