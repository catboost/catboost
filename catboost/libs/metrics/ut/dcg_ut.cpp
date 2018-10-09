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
        UNIT_ASSERT_EQUAL(CalcNdcg(samples, ENdcgMetricType::Base), 1);
        UNIT_ASSERT_EQUAL(CalcNdcg(samples, ENdcgMetricType::Exp), 1);
    }
    {
        TVector<double> approx{1.0, 1.0, 2.0};
        TVector<double> target{1.0, 0.0, 2.0};
        TVector<NMetrics::TSample> samples = NMetrics::TSample::FromVectors(target, approx);
        UNIT_ASSERT_DOUBLES_EQUAL(CalcNdcg(samples, ENdcgMetricType::Base), 0.9502344168, 1e-5);
        UNIT_ASSERT_DOUBLES_EQUAL(CalcNdcg(samples, ENdcgMetricType::Exp), 0.9639404333, 1e-5);
    }
}
}
