#include <catboost/libs/algo/train_model.h>
#include <catboost/libs/algo/features_layout.h>

#include <library/unittest/registar.h>
#include <library/json/json_reader.h>

#include <util/random/fast.h>
#include <util/generic/vector.h>

SIMPLE_UNIT_TEST_SUITE(TTrainTest) {
    SIMPLE_UNIT_TEST(TestRepeatableTrain) {
        const size_t TestDocCount = 1000;
        const size_t FactorCount = 10;

        TReallyFastRng32 rng(123);
        yvector<TDocInfo> documents;
        for (size_t i = 0; i < TestDocCount; ++i) {
            TDocInfo doc;
            doc.Target = (float)rng.GenRandReal2();
            doc.Factors.resize(FactorCount);
            for (size_t j = 0; j < FactorCount; ++j) {
                doc.Factors[j] = (float)rng.GenRandReal2();
            }
            documents.emplace_back(std::move(doc));
        }

        TPool pool;
        pool.Docs = documents;
        TPool poolCopy(pool);

        TPool testPool;

        NJson::TJsonValue fitParams;
        fitParams.InsertValue("random_seed", 5);
        fitParams.InsertValue("iterations", 0);

        std::vector<int> emptyCatFeatures;

        yvector<yvector<double>> testApprox;
        TFullModel model;
        TrainModel(fitParams, Nothing(), Nothing(), pool, testPool, "", &model, &testApprox);
        {
            TrainModel(fitParams, Nothing(), Nothing(), pool, testPool, "model_for_test.cbm", nullptr, &testApprox);
            TFullModel otherCallVariant = ReadModel("model_for_test.cbm");
            UNIT_ASSERT_EQUAL(model, otherCallVariant);
        }
        UNIT_ASSERT_EQUAL(pool.Docs.size(), poolCopy.Docs.size());
        for (size_t i = 0; i < pool.Docs.size(); ++i) {
            const auto& doc1 = pool.Docs[i];
            const auto& doc2 = poolCopy.Docs[i];
            UNIT_ASSERT_EQUAL(doc1.Target, doc2.Target);
            UNIT_ASSERT_EQUAL(doc1.Factors.size(), doc2.Factors.size());
            for (size_t j = 0; j < doc1.Factors.size(); ++j) {
                UNIT_ASSERT_EQUAL(doc1.Factors[j], doc2.Factors[j]);
            }
        }
    }
    SIMPLE_UNIT_TEST(TestFeaturesLayout) {
        std::vector<int> catFeatures = {1, 5, 9};
        int featuresCount = 10;
        TFeaturesLayout layout(featuresCount, catFeatures, yvector<TString>());
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
