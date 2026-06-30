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
        trees->SetTextFeatures(
            {
                TTextFeature {
                    false,
                    0,
                    10,
                    ""
                },
                 TTextFeature {
                    true,
                    1,
                    12,
                    ""
                }
            }
        );
        trees->SetEmbeddingFeatures(
            {
                TEmbeddingFeature {
                    true,
                    0,
                    11,
                    "",
                    0
                },
                TEmbeddingFeature {
                    false,
                    1,
                    13,
                    "",
                    0
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

        auto floatIndices = GetModelFloatFeaturesIndices(model);
        TVector<size_t> expectedFloatIndices = {0, 2, 4, 6};
        UNIT_ASSERT_EQUAL(floatIndices.size(), expectedFloatIndices.size());
        for (size_t i = 0; i != floatIndices.size(); ++i) {
            UNIT_ASSERT_EQUAL(expectedFloatIndices[i], floatIndices[i]);
        }

        auto categoryIndices = GetModelCatFeaturesIndices(model);
        TVector<size_t> expectedCategoryIndices = {1, 3, 5, 7, 8, 9};
        UNIT_ASSERT_EQUAL(categoryIndices.size(), expectedCategoryIndices.size());
        for (size_t i = 0; i != categoryIndices.size(); ++i) {
            UNIT_ASSERT_EQUAL(expectedCategoryIndices[i], categoryIndices[i]);
        }

        auto textIndices = GetModelTextFeaturesIndices(model);
        TVector<size_t> expectedTextIndices = {10, 12};
        UNIT_ASSERT_EQUAL(categoryIndices.size(), expectedCategoryIndices.size());
        for (size_t i = 0; i != textIndices.size(); ++i) {
            UNIT_ASSERT_EQUAL(expectedTextIndices[i], textIndices[i]);
        }

        auto embeddingIndices = GetModelEmbeddingFeaturesIndices(model);
        TVector<size_t> expectedEmbeddingIndices = {11, 13};
        UNIT_ASSERT_EQUAL(embeddingIndices.size(), expectedEmbeddingIndices.size());
        for (size_t i = 0; i != embeddingIndices.size(); ++i) {
            UNIT_ASSERT_EQUAL(expectedEmbeddingIndices[i], embeddingIndices[i]);
        }

        trees->DropUnusedFeatures();
        UNIT_ASSERT_EQUAL(model.GetMinimalSufficientFloatFeaturesVectorSize(), 3);
        UNIT_ASSERT_EQUAL(model.GetMinimalSufficientCatFeaturesVectorSize(), 5);
        UNIT_ASSERT_EQUAL(model.GetUsedFloatFeaturesCount(), 2);
        UNIT_ASSERT_EQUAL(model.GetUsedCatFeaturesCount(), 3);
        UNIT_ASSERT_EQUAL(model.GetNumFloatFeatures(), model.GetMinimalSufficientFloatFeaturesVectorSize());
        UNIT_ASSERT_EQUAL(model.GetNumCatFeatures(), model.GetMinimalSufficientCatFeaturesVectorSize());

        floatIndices = GetModelFloatFeaturesIndices(model);
        expectedFloatIndices = {0, 4};
        UNIT_ASSERT_EQUAL(floatIndices.size(), expectedFloatIndices.size());
        for (size_t i = 0; i != floatIndices.size(); ++i) {
            UNIT_ASSERT_EQUAL(expectedFloatIndices[i], floatIndices[i]);
        }

        categoryIndices = GetModelCatFeaturesIndices(model);
        expectedCategoryIndices = {3, 5, 8};
        UNIT_ASSERT_EQUAL(categoryIndices.size(), expectedCategoryIndices.size());
        for (size_t i = 0; i != categoryIndices.size(); ++i) {
            UNIT_ASSERT_EQUAL(expectedCategoryIndices[i], categoryIndices[i]);
        }

        textIndices = GetModelTextFeaturesIndices(model);
        expectedTextIndices = {12};
        UNIT_ASSERT_EQUAL(categoryIndices.size(), expectedCategoryIndices.size());
        for (size_t i = 0; i != textIndices.size(); ++i) {
            UNIT_ASSERT_EQUAL(expectedTextIndices[i], textIndices[i]);
        }

        embeddingIndices = GetModelEmbeddingFeaturesIndices(model);
        expectedEmbeddingIndices = {11};
        UNIT_ASSERT_EQUAL(embeddingIndices.size(), expectedEmbeddingIndices.size());
        for (size_t i = 0; i != embeddingIndices.size(); ++i) {
            UNIT_ASSERT_EQUAL(expectedEmbeddingIndices[i], embeddingIndices[i]);
        }
    }
}
