#include <library/cpp/testing/unittest/registar.h>
#include <catboost/private/libs/algo_helpers/pairwise_leaves_calculation.h>

static TArray2D<double> Convert(const TVector<TVector<double>>& matrix) {
    if (matrix.empty()) {
        return {};
    }
    TArray2D<double> result(matrix.ysize(), matrix[0].ysize());
    for (int i = 0; i < matrix.ysize(); ++i){
        for (int j = 0; j < matrix[0].ysize(); ++j){
            result[i][j] = matrix[i][j];
        }
    }
    return result;
}

Y_UNIT_TEST_SUITE(PairwiseLeafCalculationTest) {
    Y_UNIT_TEST(PairwiseLeafCalculationTestSmallMatrix) {
        const TArray2D<double> pairwiseWeightSums = Convert({{5.0, -5.0}, {-5.0, 5.0}});
        const TVector<double> derSums = {-2.0, 2.0};
        const float l2DiagReg = 0.3;
        const float pairwiseNonDiagReg = 0.1;

        const TVector<double> leafValues = CalculatePairwiseLeafValues(pairwiseWeightSums, derSums, l2DiagReg, pairwiseNonDiagReg);

        UNIT_ASSERT_DOUBLES_EQUAL(leafValues[0], -0.1869158874, 1e-6);
        UNIT_ASSERT_DOUBLES_EQUAL(leafValues[1], 0.1869158874, 1e-6);
    }

    Y_UNIT_TEST(PairwiseLeafCalculationTestLargeMatrix) {
        const TArray2D<double> pairwiseWeightSums = Convert({
            {2.0, -2.0, 0.0, 0.0},
            {-2.0, 3.0, -1.0, 0.0},
            {0.0, -1.0, 5.0, -4.0},
            {0.0, 0.0, -4.0, 4.0}
        });
        const TVector<double> derSums = {16.0, -32.0, 32.0, -16};
        const float l2DiagReg = 0.3;
        const float pairwiseNonDiagReg = 0.1;

        const TVector<double> leafValues = CalculatePairwiseLeafValues(pairwiseWeightSums, derSums, l2DiagReg, pairwiseNonDiagReg);

        UNIT_ASSERT_DOUBLES_EQUAL(leafValues[0], 0.7374471593, 1e-6);
        UNIT_ASSERT_DOUBLES_EQUAL(leafValues[1], -7.279036944, 1e-6);
        UNIT_ASSERT_DOUBLES_EQUAL(leafValues[2], 5.448432894, 1e-6);
        UNIT_ASSERT_DOUBLES_EQUAL(leafValues[3], 1.093156891, 1e-6);
    }
}
