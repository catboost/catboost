#include <catboost/libs/data_new/external_columns.h>

#include <catboost/libs/data_new/cat_feature_perfect_hash_helper.h>

#include <catboost/libs/cat_feature/cat_feature.h>
#include <catboost/libs/gpu_config/interface/get_gpu_device_count.h>
#include <catboost/libs/quantization/utils.h>

#include <library/unittest/registar.h>


using namespace NCB;


Y_UNIT_TEST_SUITE(ExternalColumns) {
    TVector<ETaskType> GetTaskTypes() {
        TVector<ETaskType> result = {ETaskType::CPU};
        if (GetGpuDeviceCount() > 0) {
            result.push_back(ETaskType::GPU);
        }
        return result;
    }

    Y_UNIT_TEST(TestExternalFloatValuesHolder) {
        for (ETaskType taskType : GetTaskTypes()) {

            TVector<float> v = {10.0f, 11.1f, 12.2f, 13.3f, 14.4f, 15.5f, 16.6f, 17.7f, 18.8f, 19.9f};

            NCB::TArraySubsetIndexing<ui32> vSubsetIndexing( NCB::TFullSubset<ui32>{(ui32)v.size()} );

            const NCatboostOptions::TCatFeatureParams catFeatureOptions(taskType);
            const NCatboostOptions::TBinarizationOptions binarizationOptions(
                EBorderSelectionType::GreedyLogSum,
                /*discretization*/3
            );

            auto featuresManager = MakeIntrusive<TQuantizedFeaturesManager>(
                catFeatureOptions,
                binarizationOptions
            );

            const ui32 featureId = 0;
            featuresManager->RegisterDataProviderFloatFeature(featureId);

            {
                TFloatValuesHolder floatValuesHolder(
                    featureId,
                    TMaybeOwningArrayHolder<float>::CreateNonOwning(v),
                    &vSubsetIndexing
                );

                featuresManager->GetOrComputeNanMode(floatValuesHolder);
            }

            auto borders = BuildBorders(v, featureId /*seed */, binarizationOptions);

            featuresManager->SetFloatFeatureBordersForDataProviderId(featureId, std::move(borders));

            TExternalFloatValuesHolder externalFloatValuesHolder(
                featureId,
                TMaybeOwningArrayHolder<float>::CreateOwning(std::move(v)),
                &vSubsetIndexing,
                featuresManager
            );

            NPar::TLocalExecutor localExecutor;
            localExecutor.RunAdditionalThreads(2);

            TMaybeOwningArrayHolder<ui8> quantizedValues = externalFloatValuesHolder.ExtractValues(
                &localExecutor
            );

            TVector<ui8> expectedQuantizedValues = {0, 0, 1, 1, 1, 2, 2, 3, 3, 3};

            UNIT_ASSERT_EQUAL(*quantizedValues, TArrayRef<ui8>(expectedQuantizedValues));
        }
    }

    Y_UNIT_TEST(TestExternalCatValuesHolder) {
        for (ETaskType taskType : GetTaskTypes()) {
            TVector<TString> srcCatValues = {"Austria", "Germany", "UK", "USA", "Germany", "UK", "Russia"};

            TVector<ui32> hashedCatValues;

            for (const auto& srcCatValue : srcCatValues) {
                hashedCatValues.push_back( (ui32)CalcCatFeatureHash(srcCatValue) );
            }

            auto hashedArrayNonOwningHolder = TMaybeOwningArrayHolder<ui32>::CreateNonOwning(hashedCatValues);

            NCB::TArraySubsetIndexing<ui32> vSubsetIndexing( NCB::TFullSubset<ui32>{(ui32)hashedCatValues.size()} );

            TMaybeOwningArraySubset<ui32, ui32> arraySubset(
                &hashedArrayNonOwningHolder,
                &vSubsetIndexing
            );

            const NCatboostOptions::TCatFeatureParams catFeatureOptions(taskType);
            const NCatboostOptions::TBinarizationOptions binarizationOptions;

            auto featuresManager = MakeIntrusive<TQuantizedFeaturesManager>(
                catFeatureOptions,
                binarizationOptions
            );

            const ui32 featureId = 0;
            featuresManager->RegisterDataProviderCatFeature(featureId);

            TCatFeaturesPerfectHashHelper catFeaturesPerfectHashHelper(featuresManager);

            catFeaturesPerfectHashHelper.UpdatePerfectHashAndMaybeQuantize(featureId, arraySubset, Nothing());

            TExternalCatValuesHolder externalCatValuesHolder(
                featureId,
                TMaybeOwningArrayHolder<ui32>::CreateOwning(std::move(hashedCatValues)),
                &vSubsetIndexing,
                featuresManager
            );

            NPar::TLocalExecutor localExecutor;
            localExecutor.RunAdditionalThreads(2);

            TMaybeOwningArrayHolder<ui32> bins = externalCatValuesHolder.ExtractValues(&localExecutor);

            TVector<ui32> expectedBins = {0, 1, 2, 3, 1, 2, 4};

            UNIT_ASSERT_EQUAL(*bins, TConstArrayRef<ui32>(expectedBins));
        }
    }
}
