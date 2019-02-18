#include <catboost/libs/data_new/model_dataset_compatibility.h>

#include <catboost/libs/data_new/features_layout.h>
#include <catboost/libs/helpers/vector_helpers.h>

#include <library/unittest/registar.h>


using namespace NCB;


Y_UNIT_TEST_SUITE(GetFloatFeaturesBordersRemap) {
    Y_UNIT_TEST(Test) {
        bool hasNans = false;
        TFullModel model;
        model.ObliviousTrees.FloatFeatures = {
            TFloatFeature(hasNans, 0, 0, {1e-9, 1.f, 2.f}),
        };

        TFeaturesLayout featuresLayout(ui32(3), TVector<ui32>(), TVector<TString>(), nullptr);
        TQuantizedFeaturesInfo quantizedFeaturesInfo(featuresLayout, TConstArrayRef<ui32>(), NCatboostOptions::TBinarizationOptions());

        quantizedFeaturesInfo.SetBorders(TFloatFeatureIdx(0), {-0.1f, 0.f, 1.f, 1.5f, 2.f, 3.f});
        quantizedFeaturesInfo.SetBorders(TFloatFeatureIdx(1), {0.f});

        auto floatBinsRemap =GetFloatFeaturesBordersRemap(model, quantizedFeaturesInfo);

        UNIT_ASSERT(Equal<ui8>(floatBinsRemap[0], {0, 0, 1, 2, 2, 3, 3}));
    }

    Y_UNIT_TEST(InconsistentBorders) {
        bool hasNans = false;
        TFullModel model;

        TFeaturesLayout featuresLayout(ui32(3), TVector<ui32>(), TVector<TString>(), nullptr);
        TQuantizedFeaturesInfo quantizedFeaturesInfo(featuresLayout, TConstArrayRef<ui32>(), NCatboostOptions::TBinarizationOptions());

        quantizedFeaturesInfo.SetBorders(TFloatFeatureIdx(0), {0.f});

        model.ObliviousTrees.FloatFeatures = {
            TFloatFeature(hasNans, 0, 0, {1.f}),
        };
        UNIT_ASSERT_EXCEPTION(GetFloatFeaturesBordersRemap(model, quantizedFeaturesInfo), TCatBoostException);

        model.ObliviousTrees.FloatFeatures = {
            TFloatFeature(hasNans, 0, 0, {1.f}),
            TFloatFeature(hasNans, 1, 1, {0.f})
        };
        UNIT_ASSERT_EXCEPTION(GetFloatFeaturesBordersRemap(model, quantizedFeaturesInfo), TCatBoostException);

        quantizedFeaturesInfo.SetBorders(TFloatFeatureIdx(0), {0.f, 1.f});

        model.ObliviousTrees.FloatFeatures = {
            TFloatFeature(hasNans, 0, 0, {-1.f, 0.f, 1.f}),
        };
        UNIT_ASSERT_EXCEPTION(GetFloatFeaturesBordersRemap(model, quantizedFeaturesInfo), TCatBoostException);

        model.ObliviousTrees.FloatFeatures = {
            TFloatFeature(hasNans, 0, 0, {0.f, 1.f, 2.f}),
        };
        UNIT_ASSERT_EXCEPTION(GetFloatFeaturesBordersRemap(model, quantizedFeaturesInfo), TCatBoostException);

        model.ObliviousTrees.FloatFeatures = {
            TFloatFeature(hasNans, 0, 0, {0.f, 0.5f, 1.f}),
        };
        UNIT_ASSERT_EXCEPTION(GetFloatFeaturesBordersRemap(model, quantizedFeaturesInfo), TCatBoostException);
    }
}
