#include <catboost/libs/train_lib/train_model.h>

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
        TEvalResult testApprox;
        TPool testPool;
        TFullModel model;
        TrainModel(
            plainFitParams,
            Nothing(),
            Nothing(),
            TClearablePoolPtrs(pool, {&testPool}),
            "",
            &model,
            {&testApprox}
        );
        {
            TrainModel(
                plainFitParams,
                Nothing(),
                Nothing(),
                TClearablePoolPtrs(pool, {&testPool}),
                "model_for_test.cbm",
                nullptr,
                {&testApprox}
            );
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
}
