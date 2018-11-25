#include <catboost/libs/model/model.h>
#include <catboost/libs/train_lib/train_model.h>

#include <library/unittest/registar.h>

#include <util/folder/tempdir.h>
#include <util/generic/array_ref.h>
#include <util/random/fast.h>

#include <limits>

template <typename Prng>
static void FillWithRandom(TArrayRef<TVector<float>> matrix, Prng& prng) {
    for (auto& row : matrix) {
        for (auto& cell : row) {
            cell = prng.GenRandReal1();
        }
    }
}

template <typename Prng>
static void FillWithRandom(TArrayRef<float> array, Prng& prng) {
    for (auto& v : array) {
        v = prng.GenRandReal1();
    }
}

Y_UNIT_TEST_SUITE(TrainModelTests) {
    Y_UNIT_TEST(TrainWithoutNansTestWithNans) {
        // Train doesn't have NaNs, so TrainModel implicitly forbids them (during quantization), but
        // test data has NaNs and we just allow that
        //
        // See MLTOOLS-1602 and MLTOOLS-2235 for details (though there aren't much details).
        //
        TTempDir trainDir;

        TPool learn;
        learn.Docs.Resize(/*doc count*/3, /*factors count*/ 3, /*baseline dimension*/ 0, /*has queryId*/ false, /*has subgroupId*/ false);
        learn.Docs.Factors[0] = {+0.5f, +1.5f, -2.5f};
        learn.Docs.Factors[1] = {+0.7f, +6.4f, +2.4f};
        learn.Docs.Factors[2] = {-2.0f, -1.0f, +6.0f};
        learn.Docs.Target = {1.0f, 0.0f, 0.2f};

        TPool test;
        test.Docs.Resize(/*doc count*/1, /*factors count*/ 3, /*baseline dimension*/ 0, /*has queryId*/ false, /*has subgroupId*/ false);
        test.Docs.Factors[0] = {std::numeric_limits<float>::quiet_NaN()};
        test.Docs.Factors[1] = {+1.5f};
        test.Docs.Factors[2] = {-2.5f};
        test.Docs.Target = {1.0f};

        TFullModel model;
        TEvalResult evalResult;
        NJson::TJsonValue params;
        params.InsertValue("iterations", 5);
        params.InsertValue("random_seed", 1);
        params.InsertValue("train_dir", trainDir.Name());

        const auto f = [&] {
            TrainModel(
                params,
                {},
                {},
                TClearablePoolPtrs(learn, {&test}),
                "",
                &model,
                {&evalResult}
            );
        };

        UNIT_ASSERT_NO_EXCEPTION(f());
    }

    Y_UNIT_TEST(TrainWithoutNansApplyWithNans) {
        // Train doesn't have NaNs, so TrainModel implicitly forbids them (during quantization), but
        // during model application we allow NaNs (because it's too expensive to check for their
        // presence).
        //
        // See MLTOOLS-1602 and MLTOOLS-2235 for details (though there aren't much details).
        //
        TTempDir trainDir;

        TPool learn;
        learn.Docs.Resize(/*doc count*/3, /*factors count*/ 3, /*baseline dimension*/ 0, /*has queryId*/ false, /*has subgroupId*/ false);
        learn.Docs.Factors[0] = {+0.5f, +1.5f, -2.5f};
        learn.Docs.Factors[1] = {+0.7f, +6.4f, +2.4f};
        learn.Docs.Factors[2] = {-2.0f, -1.0f, +6.0f};
        learn.Docs.Target = {1.0f, 0.0f, 0.2f};

        TFullModel model;
        TEvalResult evalResult;
        NJson::TJsonValue params;
        params.InsertValue("iterations", 5);
        params.InsertValue("random_seed", 1);
        params.InsertValue("train_dir", trainDir.Name());
        TrainModel(
            params,
            {},
            {},
            TClearablePoolPtrs(learn, {&learn}),
            "",
            &model,
            {&evalResult}
        );

        const float numeric[] = {std::numeric_limits<float>::quiet_NaN(), +1.5f, -2.5f};
        double predictions[1];
        const auto f = [&] { model.Calc(numeric, {}, predictions); };
        UNIT_ASSERT_NO_EXCEPTION(f());
    }

    Y_UNIT_TEST(TrainWithDifferentRandomStrength) {
        // In general models trained with different random strength (--random-strength) should be
        // different.
        //
        // issue: MLTOOLS-2464

        const ui64 seed = 20181029;
        const size_t objectCount = 100;
        const size_t numericFeatureCount = 2;
        const double randomStrength[2] = {2., 5000.};

        TFullModel models[2];
        for (size_t i = 0; i < 2; ++i) {
            TTempDir trainDir;

            TPool learn;
            learn.Docs.Resize(objectCount, numericFeatureCount, /*baseline dimension*/ 0, /*has queryId*/ false, /*has subgroupId*/ false);

            TFastRng<ui64> prng(seed);
            FillWithRandom(learn.Docs.Factors, prng);
            FillWithRandom(learn.Docs.Target, prng);

            TEvalResult evalResult;
            NJson::TJsonValue params;
            params.InsertValue("iterations", 20);
            params.InsertValue("random_seed", 1);
            params.InsertValue("train_dir", trainDir.Name());
            params.InsertValue("random_strength", randomStrength[i]);
            params.InsertValue("boosting_type", "Plain");
            TrainModel(
                params,
                {},
                {},
                TClearablePoolPtrs(learn, {&learn}),
                "",
                &models[i],
                {&evalResult}
            );
        }

        TVector<float> object(numericFeatureCount);
        {
            TFastRng<ui64> prng(seed);
            prng.Advance(objectCount * numericFeatureCount);
            FillWithRandom(object, prng);
        }

        double predictions[2][1];
        models[0].Calc(object, {}, predictions[0]);
        models[1].Calc(object, {}, predictions[1]);

        UNIT_ASSERT_VALUES_UNEQUAL(predictions[0][0], predictions[1][0]);
    }
}
