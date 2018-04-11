#include <catboost/libs/train_lib/train_model.h>
#include <catboost/libs/algo/features_layout.h>
#include <catboost/libs/options/plain_options_helper.h>

#include <library/unittest/registar.h>
#include <library/json/json_reader.h>

#include <util/random/fast.h>
#include <util/generic/vector.h>

SIMPLE_UNIT_TEST_SUITE(TTrainTest) {
    SIMPLE_UNIT_TEST(TestRepeatableTrain) {
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
        std::vector<int> emptyCatFeatures;

        TEvalResult testApprox;
        TPool testPool;
        TFullModel model;
        TrainModel(plainFitParams, Nothing(), Nothing(), pool, false, testPool, "", &model, &testApprox);
        {
            TrainModel(plainFitParams, Nothing(), Nothing(), pool, false, testPool, "model_for_test.cbm", nullptr, &testApprox);
            TFullModel otherCallVariant = ReadModel("model_for_test.cbm");
            UNIT_ASSERT_EQUAL(model, otherCallVariant);
        }
        UNIT_ASSERT_EQUAL(pool.Docs.GetDocCount(), poolCopy.Docs.GetDocCount());
        UNIT_ASSERT_EQUAL(pool.Docs.GetFactorsCount(), poolCopy.Docs.GetFactorsCount());
        for (int j = 0; j < pool.Docs.GetFactorsCount(); ++j) {
            const auto& factors = pool.Docs.Factors[j];
            const auto& factorsCopy = poolCopy.Docs.Factors[j];
            for (size_t i = 0; i < pool.Docs.GetDocCount(); ++i) {
                UNIT_ASSERT_EQUAL(factors[i], factorsCopy[i]);
            }
        }
    }
    SIMPLE_UNIT_TEST(TestFeaturesLayout) {
        std::vector<int> catFeatures = {1, 5, 9};
        int featuresCount = 10;
        TFeaturesLayout layout(featuresCount, catFeatures, TVector<TString>());
        UNIT_ASSERT_EQUAL(layout.GetFeatureType(0), EFeatureType::Float);
        UNIT_ASSERT_EQUAL(layout.GetFeatureType(1), EFeatureType::Categorical);
        UNIT_ASSERT_EQUAL(layout.GetFeatureType(3), EFeatureType::Float);
        UNIT_ASSERT_EQUAL(layout.GetFeatureType(9), EFeatureType::Categorical);

        UNIT_ASSERT_EQUAL(layout.GetInternalFeatureIdx(9), 2);
        UNIT_ASSERT_EQUAL(layout.GetInternalFeatureIdx(2), 1);

        UNIT_ASSERT_EQUAL(layout.GetFeature(2, EFeatureType::Categorical), 9);
        UNIT_ASSERT_EQUAL(layout.GetFeature(1, EFeatureType::Float), 2);
    }
}
