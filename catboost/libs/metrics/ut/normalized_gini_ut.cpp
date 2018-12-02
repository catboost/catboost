#include <library/unittest/registar.h>
#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/metrics/metric_holder.h>

//https://www.kaggle.com/c/ClaimPredictionChallenge/discussion/703, Jeff Moser comment with tests
Y_UNIT_TEST_SUITE(NormalizedGiniMetricTest) {
Y_UNIT_TEST(NormalizedGiniTest) {
    {
        TVector<TVector<double>> approx{{ 0.9, 0.3, 0.8, 0.75, 0.65, 0.6, 0.78, 0.7, 0.05, 0.4, 0.4, 0.05, 0.5, 0.1, 0.1 }};
        TVector<float> target{1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        TVector<float> weight{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
        NPar::TLocalExecutor executor;

        TNormalizedGINIMetric metric;
        TMetricHolder score = metric.Eval(approx, target, weight, {}, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric.GetFinalError(score), 0.629629, 1e-5);
    }
    {
        TVector<TVector<double>> approx{{ 0.9, 0.8, 0.75, 0.65, 0.6, 0.78, 0.7, 0.05, 0.4, 0.4, 0.05, 0.5, 0.1, 0.1, 0.3 }};
        TVector<float> target{1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
        TVector<float> weight{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
        NPar::TLocalExecutor executor;

        TNormalizedGINIMetric metric;
        TMetricHolder score = metric.Eval(approx, target, weight, {}, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric.GetFinalError(score), 0.629629, 1e-5);
    }
    {
        TVector<TVector<double>> approx{{ 0.9, 0.8, 0.75, 0.5}};
        TVector<float> target{1, 0, 1, 0};
        TVector<float> weight{1, 1, 1, 1};
        NPar::TLocalExecutor executor;

        TNormalizedGINIMetric metric;
        TMetricHolder score = metric.Eval(approx, target, weight, {}, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric.GetFinalError(score), 0.5, 1e-5);
    }
}
}