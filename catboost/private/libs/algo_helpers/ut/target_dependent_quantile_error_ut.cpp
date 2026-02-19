#include <library/cpp/testing/unittest/registar.h>
#include <catboost/private/libs/algo_helpers/error_functions.h>


Y_UNIT_TEST_SUITE(TTargetDependentQuantileErrorTest) {

    double q1 = 0.2;
    double q2 = 0.5;
    double q3 = 0.8;

    TVector<float> targets={1.,1.,3.,6.,7.,11.,12.};
    TVector<double> approxes={1.0001,3.,1.,7.,6.,12.,11.};
    TVector<double> ders1(7);
    TVector<double> ders2(7);
    TVector<double> ders3(7);

    TTargetDependentQuantileError error(TVector<double>{5., 10.}, TVector<double>{q1, q2, q3}, 1e-3, false);
    ui32 docCount = SafeIntegerCast<ui32>(targets.size());
    TVector<TDers> derivatives(docCount);

    //check that small values are forced to zero
    Y_UNIT_TEST(TargetDependentQuantileErrorDelta) {
        error.CalcDersRange(0,docCount,true,approxes.data(),nullptr,targets.data(),nullptr,derivatives.data());
        UNIT_ASSERT_DOUBLES_EQUAL(derivatives[0].Der1,0.,1e-6);
    }
    //check that we get the right derivative for each window, check right and left derivatives
    Y_UNIT_TEST(TargetDependentQuantileErrorQ1L) {
        UNIT_ASSERT_DOUBLES_EQUAL(derivatives[1].Der1,q1-1.,1e-6);
    }
    Y_UNIT_TEST(TargetDependentQuantileErrorQ1R) {
        UNIT_ASSERT_DOUBLES_EQUAL(derivatives[2].Der1,q1,1e-6);
    }
    Y_UNIT_TEST(TargetDependentQuantileErrorQ2R) {
        UNIT_ASSERT_DOUBLES_EQUAL(derivatives[3].Der1,q2-1.,1e-6);
    }
    Y_UNIT_TEST(TargetDependentQuantileErrorQ2L) {
        UNIT_ASSERT_DOUBLES_EQUAL(derivatives[4].Der1,q2,1e-6);
    }
    Y_UNIT_TEST(TargetDependentQuantileErrorQ3R) {
        UNIT_ASSERT_DOUBLES_EQUAL(derivatives[5].Der1,q3-1.,1e-6);
    }
    Y_UNIT_TEST(TargetDependentQuantileErrorQ3L) {
        UNIT_ASSERT_DOUBLES_EQUAL(derivatives[6].Der1,q3,1e-6);
    }

    //check that second and third derivative are zero
    Y_UNIT_TEST(TargetDependentQuantileErrorDer2) {
        UNIT_ASSERT_DOUBLES_EQUAL(derivatives[1].Der2,0,1e-6);
    }
    Y_UNIT_TEST(TargetDependentQuantileErrorDer3) {
        UNIT_ASSERT_DOUBLES_EQUAL(derivatives[1].Der3,0,1e-6);
    }

}
