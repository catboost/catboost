#pragma once

#include "columns.h"
#include "cat_feature_perfect_hash.h"
#include "feature_index.h"
#include "features_layout.h"

#include <catboost/libs/helpers/dbg_output.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/options/binarization_options.h>
#include <catboost/libs/options/enums.h>
#include <catboost/libs/quantization/utils.h>

#include <library/dbg_output/dump.h>

#include <util/generic/guid.h>
#include <util/generic/map.h>
#include <util/generic/ptr.h>
#include <util/generic/vector.h>
#include <util/generic/xrange.h>
#include <util/string/builder.h>
#include <util/system/types.h>


namespace NCB {
    //stores expression for quantized features calculations and mapping from this expression to unique ids
    //WARNING: not thread-safe

    // TODO(akhropov): try to replace TMap with THashMap - MLTOOLS-2278.
    class TQuantizedFeaturesInfo : public TThrRefBase {
    public:
        TQuantizedFeaturesInfo(TFeaturesLayoutPtr featuresLayout,
                               const NCatboostOptions::TBinarizationOptions floatFeaturesBinarization)
            : FeaturesLayout(featuresLayout)
            , FloatFeaturesBinarization(floatFeaturesBinarization)
            , CatFeaturesPerfectHash(featuresLayout->GetCatFeatureCount(),
                                     TStringBuilder() << "cat_feature_index." << CreateGuidAsString() << ".tmp")
        {
        }

        bool operator==(const TQuantizedFeaturesInfo& rhs) const;

        const TFeaturesLayout& GetFeaturesLayout() const {
            return *FeaturesLayout;
        }

        template <EFeatureType FeatureType>
        TFeatureIdx<FeatureType> GetPerTypeFeatureIdx(const IFeatureValuesHolder& feature) const {
            CB_ENSURE_INTERNAL(
                feature.GetFeatureType() == FeatureType,
                "feature #" << feature.GetId() << " has feature type "
                << feature.GetFeatureType() << " but GetPerTypeFeatureIdx was called with FeatureType "
                << FeatureType
            );
            CheckCorrectFeature(feature);
            return TFeatureIdx<FeatureType>(FeaturesLayout->GetInternalFeatureIdx(feature.GetId()));
        }

        template <class TBuilder>
        const TVector<float>& GetOrCreateBorders(const TFloatValuesHolder& feature,
                                                 TBuilder&& builder) {
            const auto floatFeatureIdx = GetPerTypeFeatureIdx<EFeatureType::Float>(feature);
            if (!Borders.has(*floatFeatureIdx)) {
                Borders[*floatFeatureIdx] = builder(GetFloatFeatureBinarization());
            }
            return Borders[*floatFeatureIdx];
        }

        bool HasBorders(const TFloatValuesHolder& feature) const {
            return Borders.has(*GetPerTypeFeatureIdx<EFeatureType::Float>(feature));
        }

        void SetOrCheckNanMode(const TFloatValuesHolder& feature,
                               ENanMode nanMode);

        void SetNanMode(const TFloatFeatureIdx floatFeatureIdx, ENanMode nanMode) {
            CheckCorrectPerTypeFeatureIdx(floatFeatureIdx);
            NanModes[*floatFeatureIdx] = nanMode;
        }

        ENanMode GetOrComputeNanMode(const TFloatValuesHolder& feature);

        ENanMode GetNanMode(const TFloatFeatureIdx floatFeatureIdx) const;

        ENanMode GetNanMode(const TFloatValuesHolder& feature);

        const TVector<float>& GetBorders(const TFloatValuesHolder& feature) const;

        bool HasBorders(const TFloatFeatureIdx floatFeatureIdx) const {
            CheckCorrectPerTypeFeatureIdx(floatFeatureIdx);
            return Borders.has(*floatFeatureIdx);
        }

        void SetBorders(const TFloatFeatureIdx floatFeatureIdx,
                        TVector<float>&& borders) {
            CheckCorrectPerTypeFeatureIdx(floatFeatureIdx);
            Borders[*floatFeatureIdx] = std::move(borders);
        }

        void SetBorders(const TFloatValuesHolder& feature,
                        TVector<float>&& borders) {
            Borders[*GetPerTypeFeatureIdx<EFeatureType::Float>(feature)] = std::move(borders);
        }

        const TVector<float>& GetBorders(const TFloatFeatureIdx floatFeatureIdx) const {
            CheckCorrectPerTypeFeatureIdx(floatFeatureIdx);
            return Borders.at(*floatFeatureIdx);
        }

        const NCatboostOptions::TBinarizationOptions& GetFloatFeatureBinarization() const {
            return FloatFeaturesBinarization;
        }

        ui32 GetBinCount(const TFloatFeatureIdx floatFeatureIdx) const {
            return NCB::GetBinCount(GetBorders(floatFeatureIdx), GetNanMode(floatFeatureIdx));
        }


        ui32 GetUniqueValues(const TCatFeatureIdx catFeatureIdx) const {
            CheckCorrectPerTypeFeatureIdx(catFeatureIdx);
            return CatFeaturesPerfectHash.GetUniqueValues(catFeatureIdx);
        }

        const TMap<int, ui32>& GetCategoricalFeaturesPerfectHash(const TCatFeatureIdx catFeatureIdx) const {
            CheckCorrectPerTypeFeatureIdx(catFeatureIdx);
            return CatFeaturesPerfectHash.GetFeaturePerfectHash(catFeatureIdx);
        };

        void UnloadCatFeaturePerfectHashFromRam() const {
            CatFeaturesPerfectHash.FreeRam();
        }

    private:
        template <EFeatureType FeatureType>
        void CheckCorrectPerTypeFeatureIdx(TFeatureIdx<FeatureType> perTypeFeatureIdx) const {
            CB_ENSURE_INTERNAL(
                FeaturesLayout->IsCorrectInternalFeatureIdx(*perTypeFeatureIdx, FeatureType),
                perTypeFeatureIdx << " is not present in featuresLayout"
            );
        }

        void CheckCorrectFeature(const IFeatureValuesHolder& feature) const {
            CB_ENSURE_INTERNAL(
                IsConsistentWithLayout(feature, *FeaturesLayout),
                "feature #" << feature.GetId() << " is not consistent with featuresLayout"
            );
        }

        friend class TCatFeaturesPerfectHashHelper;

        inline ENanMode ComputeNanMode(const TFloatValuesHolder& feature) const;

    private:
        TFeaturesLayoutPtr FeaturesLayout;

        // it's common for all float features
        NCatboostOptions::TBinarizationOptions FloatFeaturesBinarization;

        TMap<ui32, TVector<float>> Borders; // [floatFeatureIdx]
        TMap<ui32, ENanMode> NanModes; // [floatFeatureIdx]

        TCatFeaturesPerfectHash CatFeaturesPerfectHash;
    };

    using TQuantizedFeaturesInfoPtr = TIntrusivePtr<TQuantizedFeaturesInfo>;
}


template <>
struct TDumper<NCB::TQuantizedFeaturesInfo> {
    template <class S>
    static inline void Dump(S& s, const NCB::TQuantizedFeaturesInfo& quantizedFeaturesInfo) {
        const auto& featuresLayout = quantizedFeaturesInfo.GetFeaturesLayout();

        s << "FeaturesLayout:\n" << DbgDump(featuresLayout);

        const auto& floatFeaturesBinarization = quantizedFeaturesInfo.GetFloatFeatureBinarization();
        s << "\nFloatFeaturesBinarization: {BorderSelectionType="
            << floatFeaturesBinarization.BorderSelectionType
            << ", BorderCount=" << floatFeaturesBinarization.BorderCount
            << ", NanMode=" << floatFeaturesBinarization.NanMode << "}\n";

        for (auto i : xrange(featuresLayout.GetFloatFeatureCount())) {
            auto floatFeatureIdx = NCB::TFloatFeatureIdx(i);

            if (!featuresLayout.GetInternalFeatureMetaInfo(
                *floatFeatureIdx,
                EFeatureType::Float).IsAvailable)
            {
                continue;
            }

            s << "floatFeatureIdx=" << *floatFeatureIdx << "\tBorders=";
            if (quantizedFeaturesInfo.HasBorders(floatFeatureIdx)) {
                s << '{' << NCB::DbgDumpWithIndices<float>(quantizedFeaturesInfo.GetBorders(floatFeatureIdx))
                  << '}';
            } else {
                s << '-';
            }
            s << "\tnanMode=" << quantizedFeaturesInfo.GetNanMode(floatFeatureIdx) << Endl;
        }

        for (auto i : xrange(featuresLayout.GetCatFeatureCount())) {
            auto catFeatureIdx = NCB::TCatFeatureIdx(i);

            if (!featuresLayout.GetInternalFeatureMetaInfo(
                *catFeatureIdx,
                EFeatureType::Categorical).IsAvailable)
            {
                continue;
            }

            s << "catFeatureIdx=" << *catFeatureIdx << "\tPerfectHash="
              << DbgDump(quantizedFeaturesInfo.GetCategoricalFeaturesPerfectHash(catFeatureIdx))
              << "\nuniqueValues=" << quantizedFeaturesInfo.GetUniqueValues(catFeatureIdx) << Endl;
        }
    }
};
