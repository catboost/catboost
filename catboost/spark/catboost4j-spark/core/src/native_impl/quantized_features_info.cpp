#include "quantized_features_info.h"

#include <catboost/libs/cat_feature/cat_feature.h>

#include <library/cpp/dbg_output/dump.h>

#include <util/generic/cast.h>
#include <util/generic/ptr.h>
#include <util/generic/xrange.h>
#include <util/stream/file.h>
#include <util/system/rwlock.h>
#include <util/system/yassert.h>


using namespace NCB;


TQuantizedFeaturesInfoPtr MakeQuantizedFeaturesInfo(
    const TFeaturesLayout& featuresLayout
) {
    return MakeIntrusive<TQuantizedFeaturesInfo>(
        featuresLayout,
        /*ignoredFeatures*/ TConstArrayRef<ui32>(),
        NCatboostOptions::TBinarizationOptions()
    );
}

TQuantizedFeaturesInfoPtr MakeEstimatedQuantizedFeaturesInfo(i32 featureCount) {
    /* In fact they are 1/256, 2/256 ... 255/256 but they are not really used now so they are left
     * constant for simplicity
     */
    static const TVector<float> STANDARD_BORDERS(255, 1.0f);

    TFeaturesLayout estimatedFeaturesLayout(featureCount);
    auto estimatedQuantizedFeaturesInfo = MakeIntrusive<TQuantizedFeaturesInfo>();
    estimatedQuantizedFeaturesInfo->Init(&estimatedFeaturesLayout);
    for (auto featureIdx : xrange(SafeIntegerCast<ui32>(featureCount))) {
        estimatedQuantizedFeaturesInfo->SetBorders(
            TFloatFeatureIdx(featureIdx),
            TVector<float>(STANDARD_BORDERS)
        );
    }
    return estimatedQuantizedFeaturesInfo;
}

void UpdateCatFeaturesInfo(
    TConstArrayRef<i32> catFeaturesUniqValueCounts,
    bool isInitialization,
    NCB::TQuantizedFeaturesInfo* quantizedFeaturesInfo
) {
    TVector<ui32> integerValueHashes; // hashes for "0", "1" ... etc.

    auto& featuresLayout = *(quantizedFeaturesInfo->GetFeaturesLayout());

    featuresLayout.IterateOverAvailableFeatures<EFeatureType::Categorical>(
        [&] (TCatFeatureIdx catFeatureIdx) {
            auto flatFeatureIdx = featuresLayout.GetExternalFeatureIdx(
                *catFeatureIdx,
                EFeatureType::Categorical
            );

            i32 uniqValuesCount = catFeaturesUniqValueCounts[flatFeatureIdx];
            Y_ASSERT(uniqValuesCount > 0);
            if ((size_t)uniqValuesCount >= integerValueHashes.size()) {
                for (auto i : xrange((ui32)integerValueHashes.size(), (ui32)uniqValuesCount)) {
                    integerValueHashes.push_back(CalcCatFeatureHash(ToString(i)));
                }
            }

            TCatFeaturePerfectHash catFeaturePerfectHash;
            for (auto i : xrange((ui32)uniqValuesCount)) {
                catFeaturePerfectHash.Map.emplace(integerValueHashes[i], TValueWithCount{i, 1});
            }

            {
                TWriteGuard guard(quantizedFeaturesInfo->GetRWMutex());

                if (isInitialization && (uniqValuesCount == 1)) {
                    // it is safe - we've already got to this element in iteration
                    featuresLayout.IgnoreExternalFeature(flatFeatureIdx);
                } else {
                    quantizedFeaturesInfo->UpdateCategoricalFeaturesPerfectHash(
                        catFeatureIdx,
                        std::move(catFeaturePerfectHash)
                    );
                }
            }
        }
    );
}

i32 CalcMaxCategoricalFeaturesUniqueValuesCountOnLearn(
    const TQuantizedFeaturesInfo& quantizedFeaturesInfo
) {
    return SafeIntegerCast<i32>(quantizedFeaturesInfo.CalcMaxCategoricalFeaturesUniqueValuesCountOnLearn());
}

TVector<i32> GetCategoricalFeaturesUniqueValuesCounts(
    const NCB::TQuantizedFeaturesInfo& quantizedFeaturesInfo
) {
    const auto& featuresLayout = *(quantizedFeaturesInfo.GetFeaturesLayout());

    TVector<i32> catFeaturesUniqueValuesCounts(featuresLayout.GetExternalFeatureCount(), 0);

    featuresLayout.IterateOverAvailableFeatures<EFeatureType::Categorical>(
        [&] (TCatFeatureIdx catFeatureIdx) {
            auto flatFeatureIdx = featuresLayout.GetExternalFeatureIdx(
                *catFeatureIdx,
                EFeatureType::Categorical
            );
            catFeaturesUniqueValuesCounts[flatFeatureIdx]
                = quantizedFeaturesInfo.GetUniqueValuesCounts(catFeatureIdx).OnAll;
        }
    );

    return catFeaturesUniqueValuesCounts;
}

void DbgDump(const NCB::TQuantizedFeaturesInfo& quantizedFeaturesInfo, const TString& fileName) {
    TFileOutput out(fileName);
    out << DbgDump(quantizedFeaturesInfo);
}


