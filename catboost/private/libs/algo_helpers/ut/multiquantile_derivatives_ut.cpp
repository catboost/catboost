#include <library/cpp/testing/unittest/registar.h>
#include <catboost/private/libs/algo_helpers/error_functions.h>

static void CalcDersAtTarget(const TVector<double>& approx, double target, TVector<double>* der, THessianInfo* der2) {
    const TVector<double> alpha = {0.25, 0.5, 0.75};
    const double delta = 0.001;

    TMultiQuantileError derCalcer(alpha, delta, /*isExpApprox*/false);

    const float weight = 1.0f;

    derCalcer.CalcDersMulti(approx, target, weight, der, der2);

    UNIT_ASSERT_DOUBLES_EQUAL(der2->Data[0], 0.0, 1e-6);
    UNIT_ASSERT_DOUBLES_EQUAL(der2->Data[1], 0.0, 1e-6);
    UNIT_ASSERT_DOUBLES_EQUAL(der2->Data[2], 0.0, 1e-6);
}

Y_UNIT_TEST_SUITE(MultiQuantileDerivativesTest) {
    Y_UNIT_TEST(MultiQuantileDerivativesTest) {
        const TVector<double> approx = {0, 1, 2};
        TVector<double> der(approx.size());
        THessianInfo der2(approx.size(), EHessianType::Diagonal);

        CalcDersAtTarget(approx, 0, &der, &der2);

        UNIT_ASSERT_DOUBLES_EQUAL(der[0], +0.00, 1e-6);
        UNIT_ASSERT_DOUBLES_EQUAL(der[1], -0.50, 1e-6);
        UNIT_ASSERT_DOUBLES_EQUAL(der[2], -0.25, 1e-6);

        CalcDersAtTarget(approx, 0.5, &der, &der2);

        UNIT_ASSERT_DOUBLES_EQUAL(der[0], +0.25, 1e-6);
        UNIT_ASSERT_DOUBLES_EQUAL(der[1], -0.50, 1e-6);
        UNIT_ASSERT_DOUBLES_EQUAL(der[2], -0.25, 1e-6);

        CalcDersAtTarget(approx, 1.5, &der, &der2);

        UNIT_ASSERT_DOUBLES_EQUAL(der[0], +0.25, 1e-6);
        UNIT_ASSERT_DOUBLES_EQUAL(der[1], +0.50, 1e-6);
        UNIT_ASSERT_DOUBLES_EQUAL(der[2], -0.25, 1e-6);
    }
}
