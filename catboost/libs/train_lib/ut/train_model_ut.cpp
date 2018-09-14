#include <catboost/libs/model/model.h>
#include <catboost/libs/train_lib/train_model.h>

#include <library/unittest/registar.h>

#include <util/folder/tempdir.h>

#include <limits>

Y_UNIT_TEST_SUITE(TrainModelTests) {
    Y_UNIT_TEST(TrainWithoutNansTestWithNans) {
        // Train doesn't have NaNs, so TrainModel implicitly forbids them (during quantization), and
        // test data have NaN feature, so the entire training process fails.
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
        test.Docs.Factors[0] = {std::numeric_limits<float>::quiet_NaN(), +1.5f, -2.5f};
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

        UNIT_ASSERT_EXCEPTION_CONTAINS(f(), TCatboostException, "There are NaNs in test dataset");
    }

    Y_UNIT_TEST(TrainWithoutNansApplyWithNans) {
        // Train doesn't have NaNs, so TrainModel implicitly forbids them (during quantization), but
        // during model application we allow NaNs (because it's too expensive to check for their
        // presence)
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
}
