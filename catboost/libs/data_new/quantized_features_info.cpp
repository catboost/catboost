#include "quantized_features_info.h"

#include "feature_index.h"

#include <catboost/libs/helpers/checksum.h>
#include <catboost/libs/helpers/dbg_output.h>
#include <catboost/libs/helpers/serialization.h>
#include <catboost/libs/helpers/vector_helpers.h>

#include <library/dbg_output/dump.h>

#include <util/generic/cast.h>
#include <util/generic/mapfindptr.h>
#include <util/generic/xrange.h>
#include <util/stream/output.h>


inline int operator&(NCatboostOptions::TBinarizationOptions& binarizationOptions, IBinSaver& binSaver) {
    EBorderSelectionType borderSelectionType;
    ui32 borderCount;
    ENanMode nanMode;
    if (!binSaver.IsReading()) {
        borderSelectionType = binarizationOptions.BorderSelectionType;
        borderCount = binarizationOptions.BorderCount;
        nanMode = binarizationOptions.NanMode;
    }
    binSaver.AddMulti(borderSelectionType, borderCount, nanMode);
    if (binSaver.IsReading()) {
        binarizationOptions.BorderSelectionType = borderSelectionType;
        binarizationOptions.BorderCount = borderCount;
        binarizationOptions.NanMode = nanMode;
    }
    return 0;
}


namespace NCB {
    static bool ApproximatelyEqualBorders(
        const TMap<ui32, TVector<float>>& lhs,
        const TMap<ui32, TVector<float>>& rhs
    ) {
        constexpr auto EPS = 1.e-6f;

        if (lhs.size() != rhs.size()) {
            return false;
        }

        for (const auto& featureIdxAndValue : lhs) {
            auto* rhsValue = MapFindPtr(rhs, featureIdxAndValue.first);
            if (!rhsValue) {
                return false;
            }
            if (!ApproximatelyEqual<float>(featureIdxAndValue.second, *rhsValue, EPS)) {
                return false;
            }
        }
        for (const auto& featureIdxAndValue : rhs) {
            if (!lhs.contains(featureIdxAndValue.first)) {
                return false;
            }
        }
        return true;
    }

    TQuantizedFeaturesInfo::TQuantizedFeaturesInfo(
        const TFeaturesLayout& featuresLayout,
        TConstArrayRef<ui32> ignoredFeatures,
        NCatboostOptions::TBinarizationOptions floatFeaturesBinarization,
        bool floatFeaturesAllowNansInTestOnly,
        bool allowWriteFiles)
        : FeaturesLayout(MakeIntrusive<TFeaturesLayout>(featuresLayout))
        , FloatFeaturesBinarization(std::move(floatFeaturesBinarization))
        , FloatFeaturesAllowNansInTestOnly(floatFeaturesAllowNansInTestOnly)
        , CatFeaturesPerfectHash(
            featuresLayout.GetCatFeatureCount(),
            TString::Join("cat_feature_index.", CreateGuidAsString(), ".tmp"),
            allowWriteFiles)
    {
        FeaturesLayout->IgnoreExternalFeatures(ignoredFeatures);
    }


    bool TQuantizedFeaturesInfo::operator==(const TQuantizedFeaturesInfo& rhs) const {
        return (*FeaturesLayout == *rhs.FeaturesLayout) &&
            (FloatFeaturesBinarization == rhs.FloatFeaturesBinarization) &&
            ApproximatelyEqualBorders(Borders, rhs.Borders) && (NanModes == rhs.NanModes) &&
            (CatFeaturesPerfectHash == rhs.CatFeaturesPerfectHash);
    }

    ENanMode TQuantizedFeaturesInfo::ComputeNanMode(const TFloatValuesHolder& feature) const {
        if (FloatFeaturesBinarization.NanMode == ENanMode::Forbidden) {
            return ENanMode::Forbidden;
        }
        TMaybeOwningConstArraySubset<float, ui32> arrayData = feature.GetArrayData();

        bool hasNans = arrayData.Find([] (size_t /*idx*/, float value) { return IsNan(value); });
        if (hasNans) {
            return FloatFeaturesBinarization.NanMode;
        }
        return ENanMode::Forbidden;
    }

    ENanMode TQuantizedFeaturesInfo::GetOrComputeNanMode(const TFloatValuesHolder& feature)  {
        const auto floatFeatureIdx = GetPerTypeFeatureIdx<EFeatureType::Float>(feature);
        if (!NanModes.contains(*floatFeatureIdx)) {
            NanModes[*floatFeatureIdx] = ComputeNanMode(feature);
        }
        return NanModes.at(*floatFeatureIdx);
    }

    ENanMode TQuantizedFeaturesInfo::GetNanMode(const TFloatFeatureIdx floatFeatureIdx) const  {
        CheckCorrectPerTypeFeatureIdx(floatFeatureIdx);
        ENanMode nanMode = ENanMode::Forbidden;
        if (NanModes.contains(*floatFeatureIdx)) {
            nanMode = NanModes.at(*floatFeatureIdx);
        }
        return nanMode;
    }

    ui32 TQuantizedFeaturesInfo::CalcMaxCategoricalFeaturesUniqueValuesCountOnLearn() const {
        ui32 maxCategoricalFeaturesUniqueValuesCountOnLearn = 0;
        FeaturesLayout->IterateOverAvailableFeatures<EFeatureType::Categorical>(
            [&] (TCatFeatureIdx catFeatureIdx) {
                auto learnOnlyCount = CatFeaturesPerfectHash.GetUniqueValuesCounts(catFeatureIdx).OnLearnOnly;
                if (learnOnlyCount > maxCategoricalFeaturesUniqueValuesCountOnLearn) {
                    maxCategoricalFeaturesUniqueValuesCountOnLearn = learnOnlyCount;
                }
            }
        );
        return maxCategoricalFeaturesUniqueValuesCountOnLearn;
    }

    TPerfectHashedToHashedCatValuesMap TQuantizedFeaturesInfo::CalcPerfectHashedToHashedCatValuesMap(
        NPar::TLocalExecutor* localExecutor
    ) const {
        // load once and then work with all features in parallel
        LoadCatFeaturePerfectHashToRam();

        const auto& featuresLayout = *GetFeaturesLayout();
        TPerfectHashedToHashedCatValuesMap result(featuresLayout.GetCatFeatureCount());

        localExecutor->ExecRangeWithThrow(
            [&] (int catFeatureIdx) {
                if (!featuresLayout.GetInternalFeatureMetaInfo(
                        (ui32)catFeatureIdx,
                        EFeatureType::Categorical
                    ).IsAvailable)
                {
                    return;
                }

                const auto& catFeaturePerfectHash = GetCategoricalFeaturesPerfectHash(
                    TCatFeatureIdx((ui32)catFeatureIdx)
                );
                auto& perFeatureResult = result[catFeatureIdx];
                perFeatureResult.yresize(catFeaturePerfectHash.size());
                for (const auto [hashedCatValue, perfectHash] : catFeaturePerfectHash) {
                    perFeatureResult[perfectHash] = hashedCatValue;
                }
            },
            0,
            SafeIntegerCast<int>(featuresLayout.GetCatFeatureCount()),
            NPar::TLocalExecutor::WAIT_COMPLETE
        );

        UnloadCatFeaturePerfectHashFromRamIfPossible();

        return result;
    }

    ui32 TQuantizedFeaturesInfo::CalcCheckSum() const {
        ui32 checkSum = 0;
        checkSum = UpdateCheckSum(checkSum, FloatFeaturesBinarization.BorderSelectionType.Get());
        checkSum = UpdateCheckSum(checkSum, FloatFeaturesBinarization.BorderCount.Get());
        checkSum = UpdateCheckSum(checkSum, FloatFeaturesBinarization.NanMode.Get());
        checkSum = UpdateCheckSum(checkSum, FloatFeaturesAllowNansInTestOnly);
        checkSum = UpdateCheckSum(checkSum, Borders);
        checkSum = UpdateCheckSum(checkSum, NanModes);
        return checkSum ^ CatFeaturesPerfectHash.CalcCheckSum();
    }

    void TQuantizedFeaturesInfo::LoadNonSharedPart(IBinSaver* binSaver) {
        LoadMulti(
            binSaver,
            &FloatFeaturesBinarization,
            &FloatFeaturesAllowNansInTestOnly,
            &Borders,
            &NanModes,
            &CatFeaturesPerfectHash
        );
    }

    void TQuantizedFeaturesInfo::SaveNonSharedPart(IBinSaver* binSaver) const {
        SaveMulti(
            binSaver,
            FloatFeaturesBinarization,
            FloatFeaturesAllowNansInTestOnly,
            Borders,
            NanModes,
            CatFeaturesPerfectHash
        );
    }

}
