#include <catboost/libs/model/ut/lib/model_test_helpers.h>

#include <catboost/private/libs/algo/apply.h>
#include <catboost/libs/train_lib/train_model.h>

#include <library/cpp/testing/unittest/registar.h>

using namespace std;
using namespace NCB;


Y_UNIT_TEST_SUITE(TShrinkModel) {
    Y_UNIT_TEST(TestTruncateModel) {
        NJson::TJsonValue params;
        params.InsertValue("learning_rate", 0.3);
        params.InsertValue("iterations", 7);
        TFullModel model, model2;
        TEvalResult evalResult;

        TDataProviderPtr pool = GetAdultPool();

        TrainModel(
            params,
            nullptr,
            Nothing(),
            Nothing(),
            Nothing(),
            TDataProviders{pool, {pool}},
            /*initModel*/ Nothing(),
            /*initLearnProgress*/ nullptr,
            "",
            &model,
            {&evalResult});
        params.InsertValue("iterations", 5);
        TrainModel(
            params,
            nullptr,
            Nothing(),
            Nothing(),
            Nothing(),
            TDataProviders{pool, {pool}},
            /*initModel*/ Nothing(),
            /*initLearnProgress*/ nullptr,
            "",
            &model2,
            {&evalResult});

        model.Truncate(0, 5);
        auto result = ApplyModelMulti(model, *pool)[0];
        auto result2 = ApplyModelMulti(model2, *pool)[0];
        UNIT_ASSERT_EQUAL(result.ysize(), result2.ysize());
        for (int idx = 0; idx < result.ysize(); ++idx) {
            UNIT_ASSERT_DOUBLES_EQUAL(result[idx], result2[idx], 1e-6);
        }
        model.Truncate(1, 3);
    }
}
