#include <catboost/libs/data/external_columns.h>

#include <catboost/libs/data/cat_feature_perfect_hash_helper.h>

#include <catboost/libs/cat_feature/cat_feature.h>
#include <catboost/libs/gpu_config/interface/get_gpu_device_count.h>
#include <catboost/private/libs/quantization/utils.h>

#include <library/cpp/testing/unittest/registar.h>


using namespace NCB;


Y_UNIT_TEST_SUITE(ExternalColumns) {
    Y_UNIT_TEST(TestExternalFloatValuesHolder) {

        TVector<float> v = {10.0f, 11.1f, 12.2f, 13.3f, 14.4f, 15.5f, 16.6f, 17.7f, 18.8f, 19.9f};

        NCB::TArraySubsetIndexing<ui32> vSubsetIndexing( NCB::TFullSubset<ui32>{(ui32)v.size()} );

        const NCatboostOptions::TBinarizationOptions binarizationOptions(
            EBorderSelectionType::GreedyLogSum,
            /*discretization*/3
        );

        TFeaturesLayout featuresLayout(ui32(1), {}, {}, {}, {});
        auto quantizedFeaturesInfo = MakeIntrusive<TQuantizedFeaturesInfo>(
            featuresLayout,
            TConstArrayRef<ui32>(),
            binarizationOptions,
            TMap<ui32, NCatboostOptions::TBinarizationOptions>()
        );

        const ui32 featureId = 0;

        {
            TFloatArrayValuesHolder floatValuesHolder(
                featureId,
                TMaybeOwningConstArrayHolder<float>::CreateNonOwning(v),
                &vSubsetIndexing
            );

            quantizedFeaturesInfo->GetOrComputeNanMode(floatValuesHolder);
        }

        auto borders = BuildBorders(v, featureId /*seed */, binarizationOptions);

        quantizedFeaturesInfo->SetBorders(TFloatFeatureIdx(featureId), std::move(borders));

        TExternalFloatValuesHolder externalFloatValuesHolder(
            featureId,
            MakeIntrusive<TTypeCastArraySubset<float, float>>(
                TMaybeOwningConstArrayHolder<float>::CreateOwning(std::move(v)),
                &vSubsetIndexing
            ),
            quantizedFeaturesInfo
        );

        NPar::TLocalExecutor localExecutor;
        localExecutor.RunAdditionalThreads(2);

        TVector<ui8> quantizedValues = externalFloatValuesHolder.ExtractValues<ui8>(
            &localExecutor
        );

        TVector<ui8> expectedQuantizedValues = {0, 0, 1, 1, 1, 2, 2, 3, 3, 3};

        UNIT_ASSERT_EQUAL(quantizedValues, expectedQuantizedValues);
    }

    Y_UNIT_TEST(TestExternalCatValuesHolder) {
        TVector<TString> srcCatValues = {"Austria", "Germany", "UK", "USA", "Germany", "UK", "Russia"};

        TVector<ui32> hashedCatValues;

        for (const auto& srcCatValue : srcCatValues) {
            hashedCatValues.push_back( (ui32)CalcCatFeatureHash(srcCatValue) );
        }

        auto hashedArrayNonOwningHolder = TMaybeOwningConstArrayHolder<ui32>::CreateNonOwning(hashedCatValues);

        NCB::TArraySubsetIndexing<ui32> vSubsetIndexing( NCB::TFullSubset<ui32>{(ui32)hashedCatValues.size()} );

        TTypeCastArraySubset<ui32, ui32> arraySubset(hashedArrayNonOwningHolder, &vSubsetIndexing);

        const NCatboostOptions::TBinarizationOptions binarizationOptions;

        TFeaturesLayout featuresLayout(ui32(1), TVector<ui32>{0}, {}, {}, {});

        for (auto mapMostFrequentValueTo0 : {false, true}) {

            auto quantizedFeaturesInfo = MakeIntrusive<TQuantizedFeaturesInfo>(
                featuresLayout,
                TConstArrayRef<ui32>(),
                binarizationOptions
            );

            const ui32 featureId = 0;

            TCatFeaturesPerfectHashHelper catFeaturesPerfectHashHelper(quantizedFeaturesInfo);

            catFeaturesPerfectHashHelper.UpdatePerfectHashAndMaybeQuantize(
                TCatFeatureIdx(featureId),
                arraySubset,
                mapMostFrequentValueTo0,
                /*hashedCatDefaultValue*/ Nothing(),
                /*quantizedDefaultBinFraction*/ Nothing(),
                /*dstBins*/ Nothing()
            );

            TExternalCatValuesHolder externalCatValuesHolder(
                featureId,
                MakeIntrusive<TTypeCastArraySubset<ui32, ui32>>(
                    TMaybeOwningConstArrayHolder<ui32>::CreateOwning(TVector<ui32>(hashedCatValues)),
                    &vSubsetIndexing
                ),
                quantizedFeaturesInfo
            );

            NPar::TLocalExecutor localExecutor;
            localExecutor.RunAdditionalThreads(2);

            TVector<ui32> bins = externalCatValuesHolder.ExtractValues<ui32>(&localExecutor);

            auto expectedBins = mapMostFrequentValueTo0 ?
                TVector<ui32>{1, 0, 2, 3, 0, 2, 4} :
                TVector<ui32>{0, 1, 2, 3, 1, 2, 4};

            UNIT_ASSERT_EQUAL(bins, expectedBins);
        }
    }
}
