#include <catboost/libs/metrics/dcg.h>
#include <catboost/libs/metrics/sample.h>

#include <library/unittest/registar.h>

#include <util/generic/array_ref.h>

Y_UNIT_TEST_SUITE(TMetricTest) {
Y_UNIT_TEST(DCGTest) {
    {
        TVector<double> approx{1.0, 0.0, 2.0};
        TVector<double> target{1.0, 0.0, 2.0};
        TVector<NMetrics::TSample> samples = NMetrics::TSample::FromVectors(target, approx);
        UNIT_ASSERT_EQUAL(CalcNDCG(samples), 1);
    }
    {
        TVector<double> approx{1.0, 1.0, 2.0};
        TVector<double> target{1.0, 0.0, 2.0};
        TVector<NMetrics::TSample> samples = NMetrics::TSample::FromVectors(target, approx);
        UNIT_ASSERT_DOUBLES_EQUAL(CalcNDCG(samples), 0.9751172084, 1e-5);
    }
}
}
