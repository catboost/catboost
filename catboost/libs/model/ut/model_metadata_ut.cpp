#include <library/cpp/testing/unittest/registar.h>

#include <catboost/libs/model/model.h>

Y_UNIT_TEST_SUITE(TModelTreesMetadata) {

    Y_UNIT_TEST(TestMetadataUpdate) {
        TFullModel model;
        TModelTrees* trees = model.ModelTrees.GetMutable();
        trees->SetFloatFeatures(
            {
                TFloatFeature {
                    false, 0, 0,
                    {1.f, 2.f}, // bin splits 0, 1
                    ""
                },
                TFloatFeature {
                    false, 1, 2,
                    {}, // ignored feature
                    ""
                },
                TFloatFeature {
                    false, 2, 4,
                    {0.5f}, // bin split 2
                    ""
                },
                TFloatFeature {
                    false, 3, 6,
                    {}, // ignored feature
                    ""
                }
            }
        );
        trees->SetCatFeatures(
            {
                TCatFeature {
                    false,
                    0, 1,
                    ""
                },
                TCatFeature {
                    true,
                    1, 3,
                    ""
                },
                TCatFeature {
                    true,
                    2, 5,
                    ""
                },
                TCatFeature {
                    false,
                    3, 7,
                    ""
                },
                TCatFeature {
                    true,
                    4, 8,
                    ""
                },
                TCatFeature {
                    false,
                    5, 9,
                    ""
                }
            }
        );
        model.UpdateDynamicData();
        model.UpdateDynamicData();// we run update metadata to detect non zeroing of some counters
        UNIT_ASSERT_EQUAL(model.GetMinimalSufficientFloatFeaturesVectorSize(), 3);
        UNIT_ASSERT_EQUAL(model.GetMinimalSufficientCatFeaturesVectorSize(), 5);
        UNIT_ASSERT_EQUAL(model.GetUsedFloatFeaturesCount(), 2);
        UNIT_ASSERT_EQUAL(model.GetUsedCatFeaturesCount(), 3);
        UNIT_ASSERT_EQUAL(model.GetNumFloatFeatures(), 4);
        UNIT_ASSERT_EQUAL(model.GetNumCatFeatures(), 6);
        trees->DropUnusedFeatures();
        UNIT_ASSERT_EQUAL(model.GetMinimalSufficientFloatFeaturesVectorSize(), 3);
        UNIT_ASSERT_EQUAL(model.GetMinimalSufficientCatFeaturesVectorSize(), 5);
        UNIT_ASSERT_EQUAL(model.GetUsedFloatFeaturesCount(), 2);
        UNIT_ASSERT_EQUAL(model.GetUsedCatFeaturesCount(), 3);
        UNIT_ASSERT_EQUAL(model.GetNumFloatFeatures(), model.GetMinimalSufficientFloatFeaturesVectorSize());
        UNIT_ASSERT_EQUAL(model.GetNumCatFeatures(), model.GetMinimalSufficientCatFeaturesVectorSize());
    }
}
