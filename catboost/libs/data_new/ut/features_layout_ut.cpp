
#include <catboost/libs/data_new/features_layout.h>

#include <library/unittest/registar.h>


Y_UNIT_TEST_SUITE(TestFeaturesLayout) {
    Y_UNIT_TEST(Test) {
        {
            TVector<ui32> catFeatures = {1, 5, 9};
            ui32 featuresCount = 10;
            NCB::TFeaturesLayout layout(featuresCount, catFeatures, TVector<TString>());
            UNIT_ASSERT_EQUAL(layout.GetExternalFeatureType(0), EFeatureType::Float);
            UNIT_ASSERT_EQUAL(layout.GetExternalFeatureType(1), EFeatureType::Categorical);
            UNIT_ASSERT_EQUAL(layout.GetExternalFeatureType(3), EFeatureType::Float);
            UNIT_ASSERT_EQUAL(layout.GetExternalFeatureType(9), EFeatureType::Categorical);

            UNIT_ASSERT_EQUAL(layout.GetInternalFeatureIdx(9), 2);
            UNIT_ASSERT_EQUAL(layout.GetInternalFeatureIdx(2), 1);

            UNIT_ASSERT_EQUAL(layout.GetExternalFeatureIdx(2, EFeatureType::Categorical), 9);
            UNIT_ASSERT_EQUAL(layout.GetExternalFeatureIdx(1, EFeatureType::Float), 2);
        }

        {
            TVector<TFloatFeature> floatFeatures(4);

            floatFeatures[0].FeatureIndex = 0;
            floatFeatures[0].FlatFeatureIndex = 0;
            floatFeatures[0].FeatureId = "f0";

            floatFeatures[1].FeatureIndex = 1;
            floatFeatures[1].FlatFeatureIndex = 2;
            floatFeatures[1].FeatureId = "f1_name1";

            floatFeatures[2].FeatureIndex = 2;
            floatFeatures[2].FlatFeatureIndex = 3;
            floatFeatures[2].FeatureId = "f2";

            floatFeatures[3].FeatureIndex = 3;
            floatFeatures[3].FlatFeatureIndex = 5;
            floatFeatures[3].FeatureId = "f3";


            TVector<TCatFeature> catFeatures(3);

            catFeatures[0].FeatureIndex = 0;
            catFeatures[0].FlatFeatureIndex = 1;
            catFeatures[0].FeatureId = "c0_catname0";

            catFeatures[1].FeatureIndex = 1;
            catFeatures[1].FlatFeatureIndex = 4;
            catFeatures[1].FeatureId = "c1";

            catFeatures[2].FeatureIndex = 2;
            catFeatures[2].FlatFeatureIndex = 6;
            catFeatures[2].FeatureId = "c2";

            NCB::TFeaturesLayout layout(floatFeatures, catFeatures);

            UNIT_ASSERT_EQUAL(layout.GetExternalFeatureDescription(0, EFeatureType::Float), "f0");
            UNIT_ASSERT_EQUAL(layout.GetExternalFeatureDescription(1, EFeatureType::Float), "f1_name1");
            UNIT_ASSERT_EQUAL(layout.GetExternalFeatureDescription(2, EFeatureType::Float), "f2");
            UNIT_ASSERT_EQUAL(layout.GetExternalFeatureDescription(3, EFeatureType::Float), "f3");

            UNIT_ASSERT_EQUAL(layout.GetExternalFeatureDescription(0, EFeatureType::Categorical), "c0_catname0");
            UNIT_ASSERT_EQUAL(layout.GetExternalFeatureDescription(1, EFeatureType::Categorical), "c1");
            UNIT_ASSERT_EQUAL(layout.GetExternalFeatureDescription(2, EFeatureType::Categorical), "c2");

            {
                TVector<TString> featureIds{"f0", "c0_catname0", "f1_name1", "f2", "c1", "f3", "c2"};
                UNIT_ASSERT_EQUAL(layout.GetExternalFeatureIds(), featureIds);
            }

            UNIT_ASSERT_EQUAL(layout.GetExternalFeatureIdx(0, EFeatureType::Float), 0);
            UNIT_ASSERT_EQUAL(layout.GetExternalFeatureIdx(1, EFeatureType::Float), 2);
            UNIT_ASSERT_EQUAL(layout.GetExternalFeatureIdx(2, EFeatureType::Float), 3);
            UNIT_ASSERT_EQUAL(layout.GetExternalFeatureIdx(3, EFeatureType::Float), 5);

            UNIT_ASSERT_EQUAL(layout.GetExternalFeatureIdx(0, EFeatureType::Categorical), 1);
            UNIT_ASSERT_EQUAL(layout.GetExternalFeatureIdx(1, EFeatureType::Categorical), 4);
            UNIT_ASSERT_EQUAL(layout.GetExternalFeatureIdx(2, EFeatureType::Categorical), 6);

            UNIT_ASSERT_EQUAL(layout.GetInternalFeatureIdx(0), 0);
            UNIT_ASSERT_EQUAL(layout.GetInternalFeatureIdx(2), 1);
            UNIT_ASSERT_EQUAL(layout.GetInternalFeatureIdx(3), 2);
            UNIT_ASSERT_EQUAL(layout.GetInternalFeatureIdx(5), 3);

            UNIT_ASSERT_EQUAL(layout.GetInternalFeatureIdx(1), 0);
            UNIT_ASSERT_EQUAL(layout.GetInternalFeatureIdx(4), 1);
            UNIT_ASSERT_EQUAL(layout.GetInternalFeatureIdx(6), 2);

            UNIT_ASSERT_EQUAL(layout.GetExternalFeatureType(0), EFeatureType::Float);
            UNIT_ASSERT_EQUAL(layout.GetExternalFeatureType(2), EFeatureType::Float);
            UNIT_ASSERT_EQUAL(layout.GetExternalFeatureType(3), EFeatureType::Float);
            UNIT_ASSERT_EQUAL(layout.GetExternalFeatureType(5), EFeatureType::Float);

            UNIT_ASSERT_EQUAL(layout.GetExternalFeatureType(1), EFeatureType::Categorical);
            UNIT_ASSERT_EQUAL(layout.GetExternalFeatureType(4), EFeatureType::Categorical);
            UNIT_ASSERT_EQUAL(layout.GetExternalFeatureType(6), EFeatureType::Categorical);

            UNIT_ASSERT(layout.IsCorrectExternalFeatureIdx(0));
            UNIT_ASSERT(layout.IsCorrectExternalFeatureIdx(1));
            UNIT_ASSERT(layout.IsCorrectExternalFeatureIdx(5));
            UNIT_ASSERT(!layout.IsCorrectExternalFeatureIdx(10));
            UNIT_ASSERT(!layout.IsCorrectExternalFeatureIdx(-1));

            UNIT_ASSERT_EQUAL(layout.GetCatFeatureCount(), 3);

            UNIT_ASSERT_EQUAL(layout.GetExternalFeatureCount(), 7);
        }
    }
}
