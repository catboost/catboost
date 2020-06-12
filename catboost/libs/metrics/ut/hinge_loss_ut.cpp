#include <library/cpp/testing/unittest/registar.h>

#include <catboost/libs/helpers/vector_helpers.h>
#include <catboost/libs/metrics/hinge_loss.h>
#include <catboost/libs/metrics/metric_holder.h>

#include <util/generic/array_ref.h>

//The benchmark value was calculated by sklearn.metrics.hinge_loss
Y_UNIT_TEST_SUITE(HingeLossMetricTest) {
Y_UNIT_TEST(HingeLossTest) {
    {
        TVector<TVector<double>> approx{{-2.19722458,  2.19722458,  1.38629436, -0.84729786}};
        TVector<float> target{0, 1, 1, 0};
        TVector<float> weight{1, 1, 1, 1};

        TMetricHolder score = ComputeHingeLossMetric(To2DConstArrayRef<double>(approx),
                                                     TConstArrayRef<float>(target.begin(), target.end()),
                                                     TConstArrayRef<float>(weight.begin(), weight.end()),
                                                     0, 4, 0.5);

        UNIT_ASSERT_DOUBLES_EQUAL(score.Stats[0] / score.Stats[1], 0.03817553500000001, 1e-6);
    }
    {
        TVector<TVector<double>> approx{{0, 100, 100, 0}};
        TVector<float> target{0, 1, 1, 0};
        TVector<float> weight{1, 1, 1, 1};

        TMetricHolder score = ComputeHingeLossMetric(To2DConstArrayRef<double>(approx),
                                                     {target.begin(), target.end()},
                                                     {weight.begin(), weight.end()},
                                                     0, 4, 0.5);

        UNIT_ASSERT_DOUBLES_EQUAL(score.Stats[0] / score.Stats[1], 0.5, 1e-6);
    }
    {
        TVector<TVector<double>> approx{{0.3, 0.9, 0.1, 0.2}, {0.5, 0.05, 0.8, 0.1}, {0.2, 0.05, 0.1, 0.7}};
        TVector<float> target{1, 0, 1, 2};
        TVector<float> weight{1, 1, 1, 1};

        TMetricHolder score = ComputeHingeLossMetric(To2DConstArrayRef<double>(approx),
                                                     {target.begin(), target.end()},
                                                     {weight.begin(), weight.end()},
                                                     0, 4, 0.5);

        UNIT_ASSERT_DOUBLES_EQUAL(score.Stats[0] / score.Stats[1], 1.75/4.0, 1e-6);
    }
}
}
