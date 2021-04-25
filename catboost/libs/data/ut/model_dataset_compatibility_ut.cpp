#include <catboost/libs/data/model_dataset_compatibility.h>

#include <catboost/libs/data/features_layout.h>
#include <catboost/libs/helpers/vector_helpers.h>

#include <library/cpp/testing/unittest/registar.h>


using namespace NCB;


Y_UNIT_TEST_SUITE(GetFloatFeaturesBordersRemap) {
    void Check(TVector<float> poolBorders, TVector<float> modelBorders, TVector<ui8> expectedMap) {
        bool hasNans = false;
        TFullModel model;
        model.ModelTrees.GetMutable()->SetFloatFeatures({ TFloatFeature(hasNans, 0, 0, modelBorders) });

        THashMap<ui32, ui32> columnIndexesReorderMap{{0, 0}};

        TFeaturesLayout featuresLayout(ui32(3), {}, {}, {}, {});
        TQuantizedFeaturesInfo quantizedFeaturesInfo(featuresLayout, TConstArrayRef<ui32>(), NCatboostOptions::TBinarizationOptions());

        quantizedFeaturesInfo.SetBorders(TFloatFeatureIdx(0), std::move(poolBorders));

        if (expectedMap.empty()) {

            // Cerr << "Expecting EXCEPTION for P=" << poolBorders << ", M=" << modelBorders << Endl;
            UNIT_ASSERT_EXCEPTION(
                GetFloatFeaturesBordersRemap(model, columnIndexesReorderMap, quantizedFeaturesInfo),
                TCatBoostException
            );

        } else {

           auto floatBinsRemap = GetFloatFeaturesBordersRemap(
               model,
               columnIndexesReorderMap,
               quantizedFeaturesInfo
           );
           UNIT_ASSERT_VALUES_EQUAL(floatBinsRemap[0], expectedMap);

        }
    }

    Y_UNIT_TEST(TestThreeBorders) {

        const float a = 0;
        const float b = 1;
        const float c = 2;
        Check({a, b, c}, {a}, {0, 1, 1, 1});
        Check({a, b, c}, {b}, {0, 0, 1, 1});
        Check({a, b, c}, {c}, {0, 0, 0, 1});
        Check({a, b, c}, {a, b}, {0, 1, 2, 2});
        Check({a, b, c}, {a, c}, {0, 1, 1, 2});
        Check({a, b, c}, {b, c}, {0, 0, 1, 2});

        const float d = -0.5;
        const float e = 0.5;
        const float f = 2.5;
        Check({a, b, c}, {d}, {});
        Check({a, b, c}, {e}, {});
        Check({a, b, c}, {f}, {});
        Check({a, b, c}, {d, a, b, c}, {});
        Check({a, b, c}, {a, e, b, c}, {});
        Check({a, b, c}, {a, b, c, f}, {});
    }

    Y_UNIT_TEST(Test) {
        bool hasNans = false;
        TFullModel model;
        model.ModelTrees.GetMutable()->SetFloatFeatures(
            {
                TFloatFeature(hasNans, 0, 0, {1e-9, 1.f, 2.f}),
            }
        );

        THashMap<ui32, ui32> columnIndexesReorderMap{{0, 0}};

        TFeaturesLayout featuresLayout(ui32(3), {}, {}, {}, {});
        TQuantizedFeaturesInfo quantizedFeaturesInfo(featuresLayout, TConstArrayRef<ui32>(), NCatboostOptions::TBinarizationOptions());

        quantizedFeaturesInfo.SetBorders(TFloatFeatureIdx(0), {-0.1f, 1e-9f, 1.f, 1.5f, 2.f, 3.f});
        quantizedFeaturesInfo.SetBorders(TFloatFeatureIdx(1), {0.f});

        auto floatBinsRemap = GetFloatFeaturesBordersRemap(model, columnIndexesReorderMap, quantizedFeaturesInfo);

        UNIT_ASSERT(Equal<ui8>(floatBinsRemap[0], {0, 0, 1, 2, 2, 3, 3}));
    }

    Y_UNIT_TEST(InconsistentBorders) {
        bool hasNans = false;
        TFullModel model;

        THashMap<ui32, ui32> columnIndexesReorderMap;

        TFeaturesLayout featuresLayout(ui32(3), {}, {}, {}, {});
        TQuantizedFeaturesInfo quantizedFeaturesInfo(featuresLayout, TConstArrayRef<ui32>(), NCatboostOptions::TBinarizationOptions());

        quantizedFeaturesInfo.SetBorders(TFloatFeatureIdx(0), {0.f});

        model.ModelTrees.GetMutable()->SetFloatFeatures(
            {
                TFloatFeature(hasNans, 0, 0, {1.f}),
            }
        );
        columnIndexesReorderMap = {{0, 0}};
        UNIT_ASSERT_EXCEPTION(GetFloatFeaturesBordersRemap(model, columnIndexesReorderMap, quantizedFeaturesInfo), TCatBoostException);

        model.ModelTrees.GetMutable()->SetFloatFeatures(
            {
                TFloatFeature(hasNans, 0, 0, {1.f}),
                TFloatFeature(hasNans, 1, 1, {0.f})
            }
        );
        columnIndexesReorderMap = {{0, 0}, {1, 1}};
        UNIT_ASSERT_EXCEPTION(GetFloatFeaturesBordersRemap(model, columnIndexesReorderMap, quantizedFeaturesInfo), TCatBoostException);

        quantizedFeaturesInfo.SetBorders(TFloatFeatureIdx(0), {0.f, 1.f});

        model.ModelTrees.GetMutable()->SetFloatFeatures(
            {
                TFloatFeature(hasNans, 0, 0, {-1.f, 0.f, 1.f}),
            }
        );
        columnIndexesReorderMap = {{0, 0}};
        UNIT_ASSERT_EXCEPTION(GetFloatFeaturesBordersRemap(model, columnIndexesReorderMap, quantizedFeaturesInfo), TCatBoostException);

        model.ModelTrees.GetMutable()->SetFloatFeatures(
            {
                TFloatFeature(hasNans, 0, 0, {0.f, 1.f, 2.f}),
            }
        );
        columnIndexesReorderMap = {{0, 0}};
        UNIT_ASSERT_EXCEPTION(GetFloatFeaturesBordersRemap(model, columnIndexesReorderMap, quantizedFeaturesInfo), TCatBoostException);

        model.ModelTrees.GetMutable()->SetFloatFeatures(
            {
                TFloatFeature(hasNans, 0, 0, {0.f, 0.5f, 1.f}),
            }
        );
        columnIndexesReorderMap = {{0, 0}};
        UNIT_ASSERT_EXCEPTION(GetFloatFeaturesBordersRemap(model, columnIndexesReorderMap, quantizedFeaturesInfo), TCatBoostException);

        quantizedFeaturesInfo.SetBorders(TFloatFeatureIdx(0), {0.f, 1.f, 3.f, 4.f});

        model.ModelTrees.GetMutable()->SetFloatFeatures(
            {
                TFloatFeature(hasNans, 0, 0, {0.f, 0.5f, 0.7f, 1.f}),
            }
        );
        columnIndexesReorderMap = {{0, 0}};
        UNIT_ASSERT_EXCEPTION(GetFloatFeaturesBordersRemap(model, columnIndexesReorderMap, quantizedFeaturesInfo), TCatBoostException);
    }

    Y_UNIT_TEST(Precision) {
        bool hasNans = false;
        TFullModel model;

        THashMap<ui32, ui32> columnIndexesReorderMap;

        TFeaturesLayout featuresLayout(ui32(3), {}, {}, {}, {});
        TQuantizedFeaturesInfo quantizedFeaturesInfo(featuresLayout, TConstArrayRef<ui32>(), NCatboostOptions::TBinarizationOptions());

        const float a = 0.0000006269f;
        const float a_plus_eps = std::nextafterf(a, 1.0);
        const float b = 0.000000746f;
        const float b_plus_eps = std::nextafterf(b, 1.0);

        quantizedFeaturesInfo.SetBorders(TFloatFeatureIdx(0), {a, a_plus_eps, b, b_plus_eps});

        model.ModelTrees.GetMutable()->SetFloatFeatures(
            {
                TFloatFeature(hasNans, 0, 0, {a}),
            }
        );
        columnIndexesReorderMap = {{0, 0}};
        auto floatBinsRemap = GetFloatFeaturesBordersRemap(model, columnIndexesReorderMap, quantizedFeaturesInfo);

        UNIT_ASSERT(Equal<ui8>(floatBinsRemap[0], {0, 1, 1, 1, 1}));

        model.ModelTrees.GetMutable()->SetFloatFeatures(
            {
                TFloatFeature(hasNans, 0, 0, {a_plus_eps}),
            }
        );
        columnIndexesReorderMap = {{0, 0}};
        floatBinsRemap = GetFloatFeaturesBordersRemap(model, columnIndexesReorderMap, quantizedFeaturesInfo);
        UNIT_ASSERT(Equal<ui8>(floatBinsRemap[0], {0, 0, 1, 1, 1}));

        model.ModelTrees.GetMutable()->SetFloatFeatures(
            {
                TFloatFeature(hasNans, 0, 0, {b_plus_eps}),
            }
        );
        columnIndexesReorderMap = {{0, 0}};
        floatBinsRemap = GetFloatFeaturesBordersRemap(model, columnIndexesReorderMap, quantizedFeaturesInfo);
        UNIT_ASSERT(Equal<ui8>(floatBinsRemap[0], {0, 0, 0, 0, 1}));
    }
}
