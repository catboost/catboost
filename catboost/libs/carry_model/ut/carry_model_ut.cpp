#include <catboost/libs/model/ut/lib/model_test_helpers.h>
#include <catboost/libs/carry_model/carry.h>

#include <library/cpp/testing/unittest/registar.h>


Y_UNIT_TEST_SUITE(TModelCarryTests) {
    Y_UNIT_TEST(SimpleModelCarry) {
        const auto model = SimpleFloatModel();
        const auto carry = CarryModelByFlatIndex(model, {1}, {{1}});

        TVector<TVector<float>> originPool = {
            {0, 1, 1},
            {1, 1, 0},
        };

        TVector<TVector<float>> carriedPool = {
            {0, 1},
            {1, 0},
        };

        TVector<double> originScores(2);
        TVector<double> carriedScores(2);

        model.CalcFlat(originPool, originScores);
        carry.CalcFlat(carriedPool, carriedScores);

        UNIT_ASSERT_EQUAL(originScores, carriedScores);
    }

    Y_UNIT_TEST(SimpleModelUplift) {
        const auto model = SimpleFloatModel(100);
        const auto uplift = UpliftModelByFlatIndex(model, {1}, {0}, {1});

        TVector<TVector<float>> originPool = {{0, 1, 1}, {1, 1, 0}};
        TVector<TVector<float>> carriedPool = {{0, 1}, {1, 0}};
        TVector<TVector<double>> originScores(2, TVector<double>(2));
        TVector<double> upliftScores(2);


        // calc scores in case feature #1 equal 0
        for (auto& row : originPool) { row[1] = 0; }
        model.CalcFlat(originPool, originScores[0]);

        // calc scores in case feature #1 equal 1
        for (auto& row : originPool) { row[1] = 1; }
        model.CalcFlat(originPool, originScores[1]);

        uplift.CalcFlat(carriedPool, upliftScores);
        UNIT_ASSERT_EQUAL(originScores[1][0] - originScores[0][0], upliftScores[0]);
        UNIT_ASSERT_EQUAL(originScores[1][1] - originScores[0][1], upliftScores[1]);
    }
}
