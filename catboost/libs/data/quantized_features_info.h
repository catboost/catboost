#pragma once

#include "columns.h"
#include "cat_feature_perfect_hash.h"
#include "feature_index.h"
#include "features_layout.h"

#include <catboost/libs/helpers/dbg_output.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/private/libs/options/binarization_options.h>
#include <catboost/private/libs/options/enums.h>
#include <catboost/private/libs/options/text_processing_options.h>
#include <catboost/private/libs/options/runtime_text_options.h>
#include <catboost/private/libs/text_processing/text_digitizers.h>
#include <catboost/private/libs/quantization/utils.h>
#include <catboost/private/libs/quantization_schema/schema.h>

#include <catboost/private/libs/options/runtime_embedding_options.h>

#include <library/cpp/binsaver/bin_saver.h>
#include <library/cpp/grid_creator/binarization.h>
#include <library/cpp/dbg_output/dump.h>
#include <library/cpp/threading/local_executor/local_executor.h>

#include <util/generic/fwd.h>
#include <util/generic/guid.h>
#include <util/generic/map.h>
#include <util/generic/ptr.h>
#include <util/generic/vector.h>
#include <util/generic/set.h>
#include <util/generic/xrange.h>
#include <util/string/builder.h>
#include <util/system/rwlock.h>
#include <util/system/types.h>


namespace NCB {
    // [catFeatureIdx][perfectHashIdx] -> hashedCatValue
    using TPerfectHashedToHashedCatValuesMap = TVector<TVector<ui32>>;

    // because grid_creator library should not depend on binSaver
    struct TQuantizationWithSerialization : public NSplitSelection::TQuantization {
    public:
        TQuantizationWithSerialization() = default;

        explicit TQuantizationWithSerialization(NSplitSelection::TQuantization&& quantization)
            : NSplitSelection::TQuantization(std::move(quantization))
        {}

        SAVELOAD(Borders, DefaultQuantizedBin);
    };


    //stores expression for quantized features calculations and mapping from this expression to unique ids
    //WARNING: not thread-safe, use RWMutex from GetRWMutex for mutable shared access

    // TODO(akhropov): try to replace TMap with THashMap - MLTOOLS-2278.
    class TQuantizedFeaturesInfo : public TThrRefBase {
    public:
        // for BinSaver
        TQuantizedFeaturesInfo() = default;

        // featuresLayout copy is needed because some features might become ignored during quantization
        TQuantizedFeaturesInfo(
            const TFeaturesLayout& featuresLayout,
            TConstArrayRef<ui32> ignoredFeatures,
            NCatboostOptions::TBinarizationOptions commonFloatFeaturesBinarization,
            TMap<ui32, NCatboostOptions::TBinarizationOptions> perFloatFeatureQuantization=TMap<ui32, NCatboostOptions::TBinarizationOptions>(),
            bool floatFeaturesAllowNansInTestOnly = true);

        TQuantizedFeaturesInfo(
            const TFeaturesLayout& featuresLayout,
            TConstArrayRef<ui32> ignoredFeatures,
            NCatboostOptions::TBinarizationOptions commonFloatFeaturesBinarization,
            TMap<ui32, NCatboostOptions::TBinarizationOptions> perFloatFeatureQuantization,
            const NCatboostOptions::TTextProcessingOptions& textFeaturesProcessing,
            const NCatboostOptions::TEmbeddingProcessingOptions& embeddingFeatureProcessing,
            bool floatFeaturesAllowNansInTestOnly = true);

        /* for Java deserialization
         *  ignored features are already set in featuresLayout
         */
        void Init(TFeaturesLayout* featuresLayout); // featuresLayout is moved into

        bool EqualTo(const TQuantizedFeaturesInfo& rhs, bool ignoreSparsity = false) const;

        bool operator==(const TQuantizedFeaturesInfo& rhs) const {
            return EqualTo(rhs);
        }

        // for Spark
        bool EqualWithoutOptionsTo(const TQuantizedFeaturesInfo& rhs, bool ignoreSparsity = false) const;

        int operator&(IBinSaver& binSaver);

        // const because can be used with TReadGuard without changing the object
        TRWMutex& GetRWMutex() const {
            return RWMutex;
        }

        /* Note that availability and ignored status of features represents their state with respect
         *  to quantization.
         * When working with datasets use GetFeaturesLayout from their classes because additional
         *  features might be set as ignored or unavailable there.
         */
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

        bool HasQuantization(const TFloatFeatureIdx floatFeatureIdx) const {
            CheckCorrectPerTypeFeatureIdx(floatFeatureIdx);
            return Quantization.contains(*floatFeatureIdx);
        }

        void SetQuantization(const TFloatFeatureIdx floatFeatureIdx,
                             NSplitSelection::TQuantization&& quantization) {
            CheckCorrectPerTypeFeatureIdx(floatFeatureIdx);
            Quantization[*floatFeatureIdx] = TQuantizationWithSerialization(std::move(quantization));
        }

        const NSplitSelection::TQuantization& GetQuantization(const TFloatFeatureIdx floatFeatureIdx) const {
            CheckCorrectPerTypeFeatureIdx(floatFeatureIdx);
            return Quantization.at(*floatFeatureIdx);
        }

        bool HasBorders(const TFloatFeatureIdx floatFeatureIdx) const {
            return HasQuantization(floatFeatureIdx);
        }

        void SetBorders(const TFloatFeatureIdx floatFeatureIdx,
                        TVector<float>&& borders) {
            SetQuantization(floatFeatureIdx, NSplitSelection::TQuantization(std::move(borders)));
        }

        const TVector<float>& GetBorders(const TFloatFeatureIdx floatFeatureIdx) const {
            CheckCorrectPerTypeFeatureIdx(floatFeatureIdx);
            return Quantization.at(*floatFeatureIdx).Borders;
        }

        const NCatboostOptions::TBinarizationOptions& GetFloatFeatureBinarization(ui32 featureIndex) const {
            if (auto optsPtr = PerFloatFeatureQuantization.FindPtr(featureIndex)) {
                return *optsPtr;
            }
            return CommonFloatFeaturesBinarization;
        }

        bool GetFloatFeaturesAllowNansInTestOnly() const {
            return FloatFeaturesAllowNansInTestOnly;
        }

        ui32 GetBinCount(const TFloatFeatureIdx floatFeatureIdx) const {
            return ui32(GetBorders(floatFeatureIdx).size() + 1);
        }


        TCatFeatureUniqueValuesCounts GetUniqueValuesCounts(const TCatFeatureIdx catFeatureIdx) const {
            CheckCorrectPerTypeFeatureIdx(catFeatureIdx);
            return CatFeaturesPerfectHash.GetUniqueValuesCounts(catFeatureIdx);
        }

        ui32 CalcMaxCategoricalFeaturesUniqueValuesCountOnLearn() const;

        const TCatFeaturePerfectHash& GetCategoricalFeaturesPerfectHash(
            const TCatFeatureIdx catFeatureIdx
        ) const {
            CheckCorrectPerTypeFeatureIdx(catFeatureIdx);
            return CatFeaturesPerfectHash.GetFeaturePerfectHash(catFeatureIdx);
        };

        void UpdateCategoricalFeaturesPerfectHash(const TCatFeatureIdx catFeatureIdx,
                                                  TCatFeaturePerfectHash&& perfectHash) {
            CheckCorrectPerTypeFeatureIdx(catFeatureIdx);
            CatFeaturesPerfectHash.UpdateFeaturePerfectHash(catFeatureIdx, std::move(perfectHash));
        };

        void LoadCatFeaturePerfectHashToRam() const {
            CatFeaturesPerfectHash.Load();
        }

        void UnloadCatFeaturePerfectHashFromRam(const TString& tmpDir) const {
            CatFeaturesPerfectHash.FreeRam(tmpDir);
        }

        TPerfectHashedToHashedCatValuesMap CalcPerfectHashedToHashedCatValuesMap(
            NPar::ILocalExecutor* localExecutor
        ) const;

        ui32 CalcCheckSum() const;

        TTextDigitizers* GetTextDigitizersMutable() {
            return &TextDigitizers;
        }

        const TTextDigitizers& GetTextDigitizers() const {
            return TextDigitizers;
        }

        const NCatboostOptions::TRuntimeTextOptions& GetTextProcessingOptions() const {
            return RuntimeTextProcessingOptions;
        }

        const NCatboostOptions::TRuntimeEmbeddingOptions& GetEmbeddingProcessingOptions() const {
            return EmbeddingEstimatorsOptions;
        }

        ui32 GetTokenizedFeatureCount() const {
            return RuntimeTextProcessingOptions.TokenizedFeatureCount();
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
                FeaturesLayout->IsCorrectExternalFeatureIdxAndType(
                    feature.GetId(), feature.GetFeatureType()
                ),
                "feature #" << feature.GetId() << " is not consistent with featuresLayout"
            );
        }

        friend class TCatFeaturesPerfectHashHelper;

        inline ENanMode ComputeNanMode(const TFloatValuesHolder& feature) const;

    private:
        // use for shared mutable access
        mutable TRWMutex RWMutex;

        TFeaturesLayoutPtr FeaturesLayout;

        NCatboostOptions::TBinarizationOptions CommonFloatFeaturesBinarization;
        TMap<ui32, NCatboostOptions::TBinarizationOptions> PerFloatFeatureQuantization;

        bool FloatFeaturesAllowNansInTestOnly = false;

        TMap<ui32, TQuantizationWithSerialization> Quantization; // [floatFeatureIdx]
        TMap<ui32, ENanMode> NanModes; // [floatFeatureIdx]

        TCatFeaturesPerfectHash CatFeaturesPerfectHash;

        NCatboostOptions::TRuntimeTextOptions RuntimeTextProcessingOptions;
        TTextDigitizers TextDigitizers;

        NCatboostOptions::TRuntimeEmbeddingOptions EmbeddingEstimatorsOptions;
    };

    using TQuantizedFeaturesInfoPtr = TIntrusivePtr<TQuantizedFeaturesInfo>;


    // compatibility, probably better switch to TQuantizedFeaturesInfo everywhere
    TPoolQuantizationSchema GetPoolQuantizationSchema(
        const TQuantizedFeaturesInfo& quantizedFeaturesInfo,
        const TVector<NJson::TJsonValue>& classNames // can be empty
    );
}


template <>
struct TDumper<NSplitSelection::TQuantization> {
    template <class S>
    static inline void Dump(S& s, const NSplitSelection::TQuantization& quantization) {
        s << "Borders=" << NCB::DbgDumpWithIndices<float>(quantization.Borders, true)
          << ",DefaultQuantizedBin=";
        if (quantization.DefaultQuantizedBin) {
            s << "{Idx=" << quantization.DefaultQuantizedBin->Idx << ",Fraction="
              << quantization.DefaultQuantizedBin->Fraction << '}';
        } else {
            s << '-';
        }
        s << Endl;
    }
};


template <>
struct TDumper<NCB::TQuantizedFeaturesInfo> {
    template <class S>
    static inline void Dump(S& s, const NCB::TQuantizedFeaturesInfo& quantizedFeaturesInfo) {
        const auto& featuresLayout = *quantizedFeaturesInfo.GetFeaturesLayout();

        s << "FeaturesLayout:\n" << DbgDump(featuresLayout);

        const auto& floatFeaturesBinarization = quantizedFeaturesInfo.GetFloatFeatureBinarization(Max<ui32>());
        s << "\nCommonFloatFeaturesBinarization: {BorderSelectionType="
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

            s << "floatFeatureIdx=" << *floatFeatureIdx << "\tQuantization=";
            if (quantizedFeaturesInfo.HasQuantization(floatFeatureIdx)) {
                s << DbgDump(quantizedFeaturesInfo.GetQuantization(floatFeatureIdx));
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
