#include <catboost/libs/train_lib/train_model.h>
#include <catboost/libs/algo/features_layout.h>
#include <catboost/libs/options/plain_options_helper.h>

#include <library/unittest/registar.h>
#include <library/json/json_reader.h>

#include <util/random/fast.h>
#include <util/generic/vector.h>

Y_UNIT_TEST_SUITE(TTrainTest) {
    Y_UNIT_TEST(TestRepeatableTrain) {
        const size_t TestDocCount = 1000;
        const size_t FactorCount = 10;

        TReallyFastRng32 rng(123);
        TPool pool;
        pool.Docs.Resize(TestDocCount, FactorCount, /*baseline dimension*/ 0, /*has queryId*/ false, /*has subgroupId*/ false);
        for (size_t i = 0; i < TestDocCount; ++i) {
            pool.Docs.Target[i] = rng.GenRandReal2();
            for (size_t j = 0; j < FactorCount; ++j) {
                pool.Docs.Factors[j][i] = rng.GenRandReal2();
            }
        }
        TPool poolCopy(pool);
        NJson::TJsonValue plainFitParams;
        plainFitParams.InsertValue("random_seed", 5);
        plainFitParams.InsertValue("iterations", 1);
        plainFitParams.InsertValue("train_dir", ".");
        NJson::TJsonValue metadata;
        metadata["a"] = "b";
        plainFitParams.InsertValue("metadata", metadata);
        std::vector<int> emptyCatFeatures;
        TEvalResult testApprox;
        TPool testPool;
        TFullModel model;
        TrainModel(plainFitParams, Nothing(), Nothing(), pool, false, testPool, "", &model, &testApprox);
        {
            TrainModel(plainFitParams, Nothing(), Nothing(), pool, false, testPool, "model_for_test.cbm", nullptr, &testApprox);
            TFullModel otherCallVariant = ReadModel("model_for_test.cbm");
            UNIT_ASSERT(model.ModelInfo.has("a"));
            UNIT_ASSERT_VALUES_EQUAL(model.ModelInfo["a"], "b");
            UNIT_ASSERT_EQUAL(model, otherCallVariant);
        }
        UNIT_ASSERT_EQUAL(pool.Docs.GetDocCount(), poolCopy.Docs.GetDocCount());
        UNIT_ASSERT_EQUAL(pool.Docs.GetEffectiveFactorCount(), poolCopy.Docs.GetEffectiveFactorCount());
        for (int j = 0; j < pool.Docs.GetEffectiveFactorCount(); ++j) {
            const auto& factors = pool.Docs.Factors[j];
            const auto& factorsCopy = poolCopy.Docs.Factors[j];
            for (size_t i = 0; i < pool.Docs.GetDocCount(); ++i) {
                UNIT_ASSERT_EQUAL(factors[i], factorsCopy[i]);
            }
        }
    }
    Y_UNIT_TEST(TestFeaturesLayout) {
        {
            std::vector<int> catFeatures = {1, 5, 9};
            int featuresCount = 10;
            TFeaturesLayout layout(featuresCount, catFeatures, TVector<TString>());
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

            TFeaturesLayout layout(floatFeatures, catFeatures);

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
