#pragma once

#include "feature_index.h"

#include <catboost/libs/column_description/column.h>
#include <catboost/libs/column_description/feature_tag.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/model/features.h>
#include <catboost/private/libs/options/enums.h>

#include <library/cpp/binsaver/bin_saver.h>
#include <library/cpp/dbg_output/dump.h>
#include <library/cpp/json/json_value.h>

#include <util/generic/array_ref.h>
#include <util/generic/ptr.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/generic/xrange.h>
#include <util/generic/ymath.h>
#include <util/system/compiler.h>
#include <util/system/yassert.h>

namespace NCB {

    struct TFeatureMetaInfo {
        EFeatureType Type;
        TString Name;

        /* Means that the distribution of values of this feature contains a large
         * probability for a single value (often called default value).
         * Note that IsSparse == true does not mean that data for this feature will always be stored
         * as sparse. For some applications dense storage might be necessary/more effective.
         */
        bool IsSparse = false;

        bool IsIgnored = false;

        /* some datasets can contain only part of all features present in the whole dataset
         * (e.g. workers in distributed processing)
         * ignored features are always unavailable
         */
        bool IsAvailable = true;

        /* feature is aggregated from neighboring vertices in the graph
         * see paper Learning on Graphs with Tabular Features
         */
        bool IsAggregated = false;

    public:
        // needed for BinSaver
        TFeatureMetaInfo() = default;

        TFeatureMetaInfo(
            EFeatureType type,
            const TString& name,
            bool isSparse = false,
            bool isIgnored = false,
            bool isAvailable = true, // isIgnored = true overrides this parameter
            bool isAggregated = false // feature is aggregated using graph
        )
            : Type(type)
            , Name(name)
            , IsSparse(isSparse)
            , IsIgnored(isIgnored)
            , IsAvailable(!isIgnored && isAvailable)
            , IsAggregated(!isIgnored && isAggregated)
        {}

        bool EqualTo(const TFeatureMetaInfo& rhs, bool ignoreSparsity = false) const;

        bool operator==(const TFeatureMetaInfo& rhs) const {
            return EqualTo(rhs);
        }

        SAVELOAD(Type, Name, IsSparse, IsIgnored, IsAvailable, IsAggregated);

        operator NJson::TJsonValue() const;
    };

}

template <>
struct TDumper<NCB::TFeatureMetaInfo> {
    template <class S>
    static inline void Dump(S& s, const NCB::TFeatureMetaInfo& featureMetaInfo) {
        s << "Type=" << featureMetaInfo.Type << "\tName=" << featureMetaInfo.Name
          << "\tIsSparse=" << featureMetaInfo.IsSparse
          << "\tIsIgnored=" << featureMetaInfo.IsIgnored
          << "\tIsAvailable=" << featureMetaInfo.IsAvailable
          << "\tIsAggregated=" << featureMetaInfo.IsAggregated;
    }
};



namespace NCB {
    class TFeaturesLayout;
    using TFeaturesLayoutPtr = TIntrusivePtr<TFeaturesLayout>;

    class TFeaturesLayout final : public TAtomicRefCount<TFeaturesLayout> {
    public:
        // needed because of default init in Cython and because of BinSaver
        TFeaturesLayout() = default;
        explicit TFeaturesLayout(const ui32 featureCount);
        TFeaturesLayout( // TODO(d-kruchinin) Temporary fix while DataProvider interface is not private
            const ui32 featureCount,
            const TVector<ui32>& catFeatureIndices,
            const TVector<TString>& featureId)
            : TFeaturesLayout(featureCount, catFeatureIndices, {}, {}, featureId, /*graph*/ false) {}
        TFeaturesLayout(
            const ui32 featureCount,
            const TVector<ui32>& catFeatureIndices,
            const TVector<ui32>& textFeatureIndices,
            const TVector<ui32>& embeddingFeatureIndices,
            const TVector<TString>& featureId,
            bool hasGraph = false,
            const THashMap<TString, TTagDescription>& featureTags = {},
            bool allFeaturesAreSparse = false);
        TFeaturesLayout(
            const TVector<TFloatFeature>& floatFeatures,
            const TVector<TCatFeature>& catFeatures);
        TFeaturesLayout(
            TConstArrayRef<TFloatFeature> floatFeatures,
            TConstArrayRef<TCatFeature> catFeatures,
            TConstArrayRef<TTextFeature> textFeatures,
            TConstArrayRef<TEmbeddingFeature> embeddingFeatures);

        // data is moved into - poor substitute to && because SWIG does not support it
        TFeaturesLayout(TVector<TFeatureMetaInfo>* data);

        // needed for SWIG wrapper deserialization
        // data is moved into - poor substitute to && because SWIG does not support it
        void Init(TVector<TFeatureMetaInfo>* data);

        // create from columns info
        static TFeaturesLayoutPtr CreateFeaturesLayout(
            TConstArrayRef<TColumn> columns,
            TMaybe<const TVector<TString>*> featureNames = Nothing(),
            TMaybe<const THashMap<TString, TTagDescription>*> featureTags = Nothing(),
            bool hasGraph = false);

        bool EqualTo(const TFeaturesLayout& rhs, bool ignoreSparsity = false) const;

        bool operator==(const TFeaturesLayout& rhs) const {
            return EqualTo(rhs);
        }

        SAVELOAD(
            ExternalIdxToMetaInfo,
            FeatureExternalIdxToInternalIdx,
            CatFeatureInternalIdxToExternalIdx,
            FloatFeatureInternalIdxToExternalIdx,
            TextFeatureInternalIdxToExternalIdx,
            EmbeddingFeatureInternalIdxToExternalIdx,
            TagToExternalIndices,
            HasGraph);

        operator NJson::TJsonValue() const;

        const TFeatureMetaInfo& GetInternalFeatureMetaInfo(
            ui32 internalFeatureIdx,
            EFeatureType type) const;

        const TFeatureMetaInfo& GetExternalFeatureMetaInfo(ui32 externalFeatureIdx) const;

        // prefer this method to GetExternalFeatureIds
        TConstArrayRef<TFeatureMetaInfo> GetExternalFeaturesMetaInfo() const noexcept;

        TString GetExternalFeatureDescription(ui32 internalFeatureIdx, EFeatureType type) const;

        TVector<TString> GetExternalFeatureIds() const;

        // needed for python-package
        void SetExternalFeatureIds(TConstArrayRef<TString> featureIds);

        ui32 GetExternalFeatureIdx(ui32 internalFeatureIdx, EFeatureType type) const {
            switch (type) {
                case EFeatureType::Float:
                    return FloatFeatureInternalIdxToExternalIdx[internalFeatureIdx];
                case EFeatureType::Categorical:
                    return CatFeatureInternalIdxToExternalIdx[internalFeatureIdx];
                case EFeatureType::Text:
                    return TextFeatureInternalIdxToExternalIdx[internalFeatureIdx];
                case EFeatureType::Embedding:
                    return EmbeddingFeatureInternalIdxToExternalIdx[internalFeatureIdx];
            }
            Y_UNREACHABLE();
        }

        ui32 GetInternalFeatureIdx(ui32 externalFeatureIdx) const;

        template <EFeatureType FeatureType>
        TFeatureIdx<FeatureType> GetInternalFeatureIdx(ui32 externalFeatureIdx) const {
            Y_ASSERT(IsCorrectExternalFeatureIdxAndType(externalFeatureIdx, FeatureType));
            return TFeatureIdx<FeatureType>(FeatureExternalIdxToInternalIdx[externalFeatureIdx]);
        }

        /* when externalFeatureIdx can be outside known range
         * return index as if feature of this type as added at the end
         * WARNING: this is only correct if there is single expanding feature type!
         */
        template <EFeatureType FeatureType>
        TFeatureIdx<FeatureType> GetExpandingInternalFeatureIdx(ui32 externalFeatureIdx) const {
            if (externalFeatureIdx >= FeatureExternalIdxToInternalIdx.size()) {
                ui32 otherTypesSize = 0;
                if constexpr (FeatureType == EFeatureType::Float) {
                    otherTypesSize = ExternalIdxToMetaInfo.size() - FloatFeatureInternalIdxToExternalIdx.size();
                } else if constexpr (FeatureType == EFeatureType::Categorical) {
                    otherTypesSize = ExternalIdxToMetaInfo.size() - CatFeatureInternalIdxToExternalIdx.size();
                } else if constexpr (FeatureType == EFeatureType::Text) {
                    otherTypesSize = ExternalIdxToMetaInfo.size() - TextFeatureInternalIdxToExternalIdx.size();
                } else {
                    otherTypesSize
                        = ExternalIdxToMetaInfo.size() - EmbeddingFeatureInternalIdxToExternalIdx.size();
                }
                return TFeatureIdx<FeatureType>(externalFeatureIdx - otherTypesSize);
            } else {
                return TFeatureIdx<FeatureType>(FeatureExternalIdxToInternalIdx[externalFeatureIdx]);
            }
        }

        EFeatureType GetExternalFeatureType(ui32 externalFeatureIdx) const;

        bool IsCorrectExternalFeatureIdx(ui32 externalFeatureIdx) const noexcept;

        bool IsCorrectInternalFeatureIdx(ui32 internalFeatureIdx, EFeatureType type) const noexcept;

        bool IsCorrectExternalFeatureIdxAndType(ui32 externalFeatureIdx, EFeatureType type) const noexcept;

        ui32 GetFloatFeatureCount() const noexcept;

        ui32 GetCatFeatureCount() const noexcept;

        ui32 GetTextFeatureCount() const noexcept;

        ui32 GetEmbeddingFeatureCount() const noexcept;

        ui32 GetExternalFeatureCount() const noexcept;

        ui32 GetFloatAggregatedFeatureCount() const noexcept;

        ui32 GetAggregatedFeatureCount(EFeatureType type) const noexcept;

        bool HasGraphForAggregatedFeatures() const noexcept {
            return HasGraph;
        }

        ui32 GetFeatureCount(EFeatureType type) const noexcept;

        bool HasSparseFeatures(bool checkOnlyAvailable = true) const noexcept;

        void IgnoreExternalFeature(ui32 externalFeatureIdx) noexcept;

        // indices in list can be outside of range of features in layout - such features are ignored
        void IgnoreExternalFeatures(TConstArrayRef<ui32> ignoredFeatures) noexcept;

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

        TConstArrayRef<ui32> GetFloatFeatureInternalIdxToExternalIdx() const noexcept;

        TConstArrayRef<ui32> GetCatFeatureInternalIdxToExternalIdx() const noexcept;

        TConstArrayRef<ui32> GetTextFeatureInternalIdxToExternalIdx() const noexcept;

        TConstArrayRef<ui32> GetEmbeddingFeatureInternalIdxToExternalIdx() const noexcept;

        const THashMap<TString, TVector<ui32>>& GetTagToExternalIndices() const noexcept;

        bool HasAvailableAndNotIgnoredFeatures() const noexcept;

        void AddFeature(TFeatureMetaInfo&& featureMetaInfo);

    private:
        TVector<TFeatureMetaInfo> ExternalIdxToMetaInfo;
        TVector<ui32> FeatureExternalIdxToInternalIdx;
        TVector<ui32> FloatFeatureInternalIdxToExternalIdx;
        TVector<ui32> CatFeatureInternalIdxToExternalIdx;
        TVector<ui32> TextFeatureInternalIdxToExternalIdx;
        TVector<ui32> EmbeddingFeatureInternalIdxToExternalIdx;
        THashMap<TString, TVector<ui32>> TagToExternalIndices;
        bool HasGraph = false;

        template <class TFeatureElement>
        inline void UpdateFeaturesMetaInfo(
            TConstArrayRef<TFeatureElement> features,
            EFeatureType featureType)
        {
            const TFeatureMetaInfo defaultIgnoredMetaInfo(
                EFeatureType::Float,
                /*name*/ TString(),
                /*isSparse*/ false,
                /*isIgnored*/ true
            );
            const ui32 internalOrExternalIndexPlaceholder = Max<ui32>();
            TVector<ui32>& featureInternalIdxToExternalIdx = [&]()->TVector<ui32>& {
                switch (featureType) {
                    case EFeatureType::Float:
                        return FloatFeatureInternalIdxToExternalIdx;
                    case EFeatureType::Categorical:
                        return CatFeatureInternalIdxToExternalIdx;
                    case EFeatureType::Text:
                        return TextFeatureInternalIdxToExternalIdx;
                    case EFeatureType::Embedding:
                        return EmbeddingFeatureInternalIdxToExternalIdx;
                    default:
                        CB_ENSURE(false, "Unsupported feature type " << featureType << " for layout");
                }
            }();
            for (const auto& feature : features) {
                CB_ENSURE(feature.Position.FlatIndex >= 0, "feature.Position.FlatIndex is negative");
                CB_ENSURE(feature.Position.Index >= 0, "feature.Position.Index is negative");
                if ((size_t)feature.Position.FlatIndex >= ExternalIdxToMetaInfo.size()) {
                    CB_ENSURE(
                        (size_t)feature.Position.FlatIndex < (size_t)Max<ui32>(),
                        "feature.Position.FlatIndex is greater than maximum allowed index: "
                        << (Max<ui32>() - 1)
                    );
                    ExternalIdxToMetaInfo.resize(feature.Position.FlatIndex + 1, defaultIgnoredMetaInfo);
                    FeatureExternalIdxToInternalIdx.resize(
                        feature.Position.FlatIndex + 1,
                        internalOrExternalIndexPlaceholder
                    );
                }
                ExternalIdxToMetaInfo[feature.Position.FlatIndex] =
                    TFeatureMetaInfo(featureType, feature.FeatureId);
                FeatureExternalIdxToInternalIdx[feature.Position.FlatIndex] = feature.Position.Index;
                if ((size_t)feature.Position.Index >= featureInternalIdxToExternalIdx.size()) {
                    featureInternalIdxToExternalIdx.resize(
                        (size_t)feature.Position.Index + 1,
                        internalOrExternalIndexPlaceholder
                    );
                }
                featureInternalIdxToExternalIdx[feature.Position.Index] = feature.Position.FlatIndex;
            }
        }
    };

    void CheckCompatibleForApply(
        const TFeaturesLayout& learnFeaturesLayout,
        const TFeaturesLayout& applyFeaturesLayout,
        const TString& applyDataName
    );

    void CheckCompatibleForQuantize(
        const TFeaturesLayout& dataFeaturesLayout,
        const TFeaturesLayout& quantizedFeaturesLayout,
        const TString& dataName
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
