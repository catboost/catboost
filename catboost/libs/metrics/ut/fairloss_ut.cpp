#include <library/unittest/registar.h>
#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/metrics/metric_holder.h>

Y_UNIT_TEST_SUITE(FairLossMetricTest) {
        Y_UNIT_TEST(FairLossTest) {
            {
                TVector <TVector<double>> approx{{1.0, 2, 3}};
                TVector<float> target{1.0, 2, 3};
                TVector<float> weight{1, 1, 1};
                NPar::TLocalExecutor executor;

                TFairLossMetric metric;
                TMetricHolder score = metric.Eval(approx, target, weight, {}, 0, target.size(), executor);

                UNIT_ASSERT_DOUBLES_EQUAL(metric.GetFinalError(score), 0.0, 1e-5);
            }
            {
                TVector <TVector<double>> approx{{2.0, 3, 4}};
                TVector<float> target{1.0, 2, 3};
                TVector<float> weight{1, 1, 1};
                NPar::TLocalExecutor executor;

                TFairLossMetric metric;
                TMetricHolder score = metric.Eval(approx, target, weight, {}, 0, target.size(), executor);

                UNIT_ASSERT_DOUBLES_EQUAL(metric.GetFinalError(score), 0.3781395, 1e-5);
            }
            {
                TVector <TVector<double>> approx{{0.0, 0, 0}};
                TVector<float> target{1.0, 2, 3};
                TVector<float> weight{1, 1, 1};
                NPar::TLocalExecutor executor;

                TFairLossMetric metric;
                TMetricHolder score = metric.Eval(approx, target, weight, {}, 0, target.size(), executor);

                UNIT_ASSERT_DOUBLES_EQUAL(metric.GetFinalError(score), 1.3134626, 1e-5);
            }
            {
                TVector <TVector<double>> approx{{2.0, 3, 4}};
                TVector<float> target{1.0, 2, 3};
                TVector<float> weight{1, 1, 1};
                NPar::TLocalExecutor executor;

                TFairLossMetric metric(5);
                TMetricHolder score = metric.Eval(approx, target, weight, {}, 0, target.size(), executor);

                UNIT_ASSERT_DOUBLES_EQUAL(metric.GetFinalError(score), 0.441961, 1e-5);
            }
        }
}

