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

#include <library/binsaver/bin_saver.h>

#include <library/dbg_output/dump.h>
#include <library/threading/local_executor/local_executor.h>

#include <util/generic/guid.h>
#include <util/generic/map.h>
#include <util/generic/ptr.h>
#include <util/generic/vector.h>
#include <util/generic/xrange.h>
#include <util/string/builder.h>
#include <util/system/rwlock.h>
#include <util/system/types.h>


namespace NCB {
    // [catFeatureIdx][perfectHashIdx] -> hashedCatValue
    using TPerfectHashedToHashedCatValuesMap = TVector<TVector<ui32>>;


    //stores expression for quantized features calculations and mapping from this expression to unique ids
    //WARNING: not thread-safe, use RWMutex from GetRWMutex for mutable shared access

    // TODO(akhropov): try to replace TMap with THashMap - MLTOOLS-2278.
    class TQuantizedFeaturesInfo : public TThrRefBase {
    public:

        // featuresLayout copy is needed because some features might become ignored during quantization
        TQuantizedFeaturesInfo(
            const TFeaturesLayout& featuresLayout,
            TConstArrayRef<ui32> ignoredFeatures,
            NCatboostOptions::TBinarizationOptions floatFeaturesBinarization,
            bool floatFeaturesAllowNansInTestOnly = true,
            bool allowWriteFiles = true);

        bool operator==(const TQuantizedFeaturesInfo& rhs) const;

        TRWMutex& GetRWMutex() {
            return RWMutex;
        }


        const TFeaturesLayoutPtr GetFeaturesLayout() const {
            return FeaturesLayout;
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


        bool HasNanMode(const TFloatFeatureIdx floatFeatureIdx) const {
            CheckCorrectPerTypeFeatureIdx(floatFeatureIdx);
            return NanModes.contains(*floatFeatureIdx);
        }

        void SetNanMode(const TFloatFeatureIdx floatFeatureIdx, ENanMode nanMode) {
            CheckCorrectPerTypeFeatureIdx(floatFeatureIdx);
            NanModes[*floatFeatureIdx] = nanMode;
        }

        ENanMode GetOrComputeNanMode(const TFloatValuesHolder& feature);

        ENanMode GetNanMode(const TFloatFeatureIdx floatFeatureIdx) const;


        bool HasBorders(const TFloatFeatureIdx floatFeatureIdx) const {
            CheckCorrectPerTypeFeatureIdx(floatFeatureIdx);
            return Borders.contains(*floatFeatureIdx);
        }

        void SetBorders(const TFloatFeatureIdx floatFeatureIdx,
                        TVector<float>&& borders) {
            CheckCorrectPerTypeFeatureIdx(floatFeatureIdx);
            Borders[*floatFeatureIdx] = std::move(borders);
        }

        const TVector<float>& GetBorders(const TFloatFeatureIdx floatFeatureIdx) const {
            CheckCorrectPerTypeFeatureIdx(floatFeatureIdx);
            return Borders.at(*floatFeatureIdx);
        }

        const NCatboostOptions::TBinarizationOptions& GetFloatFeatureBinarization() const {
            return FloatFeaturesBinarization;
        }

        bool GetFloatFeaturesAllowNansInTestOnly() const {
            return FloatFeaturesAllowNansInTestOnly;
        }

        ui32 GetBinCount(const TFloatFeatureIdx floatFeatureIdx) const {
            CheckCorrectPerTypeFeatureIdx(floatFeatureIdx);
            return ui32(Borders.at(*floatFeatureIdx).size() + 1);
        }


        TCatFeatureUniqueValuesCounts GetUniqueValuesCounts(const TCatFeatureIdx catFeatureIdx) const {
            CheckCorrectPerTypeFeatureIdx(catFeatureIdx);
            return CatFeaturesPerfectHash.GetUniqueValuesCounts(catFeatureIdx);
        }

        ui32 CalcMaxCategoricalFeaturesUniqueValuesCountOnLearn() const;

        const TMap<ui32, ui32>& GetCategoricalFeaturesPerfectHash(const TCatFeatureIdx catFeatureIdx) const {
            CheckCorrectPerTypeFeatureIdx(catFeatureIdx);
            return CatFeaturesPerfectHash.GetFeaturePerfectHash(catFeatureIdx);
        };

        void UpdateCategoricalFeaturesPerfectHash(const TCatFeatureIdx catFeatureIdx,
                                                  TMap<ui32, ui32>&& perfectHash) {
            CheckCorrectPerTypeFeatureIdx(catFeatureIdx);
            CatFeaturesPerfectHash.UpdateFeaturePerfectHash(catFeatureIdx, std::move(perfectHash));
        };

        void SetAllowWriteFiles(bool allowWriteFiles) {
            CatFeaturesPerfectHash.SetAllowWriteFiles(allowWriteFiles);
        }

        void LoadCatFeaturePerfectHashToRam() const {
            CatFeaturesPerfectHash.Load();
        }

        void UnloadCatFeaturePerfectHashFromRamIfPossible() const {
            CatFeaturesPerfectHash.FreeRamIfPossible();
        }

        TPerfectHashedToHashedCatValuesMap CalcPerfectHashedToHashedCatValuesMap(
            NPar::TLocalExecutor* localExecutor
        ) const;

        ui32 CalcCheckSum() const;

    private:
        void LoadNonSharedPart(IBinSaver* binSaver);
        void SaveNonSharedPart(IBinSaver* binSaver) const;

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
        friend class TObjectsSerialization;

        inline ENanMode ComputeNanMode(const TFloatValuesHolder& feature) const;

    private:
        // use for shared mutable access
        TRWMutex RWMutex;

        TFeaturesLayoutPtr FeaturesLayout;

        // it's common for all float features
        NCatboostOptions::TBinarizationOptions FloatFeaturesBinarization;

        bool FloatFeaturesAllowNansInTestOnly;

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
        const auto& featuresLayout = *quantizedFeaturesInfo.GetFeaturesLayout();

        s << "FeaturesLayout:\n" << DbgDump(featuresLayout);

        const auto& floatFeaturesBinarization = quantizedFeaturesInfo.GetFloatFeatureBinarization();
        s << "\nFloatFeaturesBinarization: {BorderSelectionType="
            << floatFeaturesBinarization.BorderSelectionType
            << ", BorderCount=" << floatFeaturesBinarization.BorderCount
            << ", NanMode=" << floatFeaturesBinarization.NanMode << "}\n";

        s << "FloatFeaturesAllowNansInTestOnly="
          << quantizedFeaturesInfo.GetFloatFeaturesAllowNansInTestOnly() << Endl;

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
              << DbgDump(quantizedFeaturesInfo.GetCategoricalFeaturesPerfectHash(catFeatureIdx));

            auto uniqueValuesCounts = quantizedFeaturesInfo.GetUniqueValuesCounts(catFeatureIdx);
            s << "\nuniqueValuesCounts={OnAll=" << uniqueValuesCounts.OnAll
              << ", OnLearnOnly=" << uniqueValuesCounts.OnLearnOnly << "}\n";
        }
    }
};
