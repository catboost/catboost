#include <library/unittest/registar.h>
#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/metrics/metric_holder.h>

//The benchmark value was calculated by sklearn.metrics.median_absolute_error
Y_UNIT_TEST_SUITE(NormalizedGiniMetricTest) {
Y_UNIT_TEST(NormalizedGiniTest) {
    {
        TVector<TVector<double>> approx{{ 10.0, 20, 30 }};
        TVector<float> target{1.0, 2, 3};
        TVector<float> weight{1, 1, 1};
        NPar::TLocalExecutor executor;

        TNormalizedGINI metric;
        TMetricHolder score = metric.Eval(approx, target, weight, {}, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric.GetFinalError(score), 1.0, 1e-5);
    }
    {
        TVector<TVector<double>> approx{{ 30.0, 20, 10 }};
        TVector<float> target{1.0, 2, 3};
        TVector<float> weight{1, 1, 1};
        NPar::TLocalExecutor executor;

        TNormalizedGINI metric;
        TMetricHolder score = metric.Eval(approx, target, weight, {}, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric.GetFinalError(score), -1.0, 1e-5);
    }
//    {
//        TVector<TVector<double>> approx{{ 0.0, 0, 0 }};
//        TVector<float> target{1.0, 2, 3};
//        TVector<float> weight{1, 1, 1};
//        NPar::TLocalExecutor executor;
//
//        TNormalizedGINI metric;
//        TMetricHolder score = metric.Eval(approx, target, weight, {}, 0, target.size(), executor);
//
//        UNIT_ASSERT_DOUBLES_EQUAL(metric.GetFinalError(score), -1.0, 1e-5);
//    }
//    {
//        TVector<TVector<double>> approx{{ 0.0, 0, 0 }};
//        TVector<float> target{1.0, 2, 3};
//        TVector<float> weight{1, 1, 1};
//        NPar::TLocalExecutor executor;
//
//        TNormalizedGINI metric;
//        TMetricHolder score = metric.Eval(approx, target, weight, {}, 0, target.size(), executor);
//
//        UNIT_ASSERT_DOUBLES_EQUAL(metric.GetFinalError(score), 1.0, 1e-5);
//    }
//    {
//        TVector<TVector<double>> approx{{ 0.0, 0, 0, 0 }};
//        TVector<float> target{1.0, 2, 4, 3};
//        TVector<float> weight{1, 1, 1, 1};
//        NPar::TLocalExecutor executor;
//
//        TNormalizedGINI metric;
//        TMetricHolder score = metric.Eval(approx, target, weight, {}, 0, target.size(), executor);
//
//        UNIT_ASSERT_DOUBLES_EQUAL(metric.GetFinalError(score), -0.8, 1e-5);
//    }
//    {
//        TVector<TVector<double>> approx{{ 0.0, 0, 2, 1 }};
//        TVector<float> target{2.0, 1, 4, 3};
//        TVector<float> weight{1, 1, 1, 1};
//        NPar::TLocalExecutor executor;
//
//        TNormalizedGINI metric;
//        TMetricHolder score = metric.Eval(approx, target, weight, {}, 0, target.size(), executor);
//
//        UNIT_ASSERT_DOUBLES_EQUAL(metric.GetFinalError(score), 1, 1e-5);
//    }
//    {
//        TVector<TVector<double>> approx{{40.0, 40.0, 10.0, 5, 5}};
//        TVector<float> target{ 0.0, 20, 40, 0, 10};
//        TVector<float> weight{1, 1, 1, 1, 1};
//        NPar::TLocalExecutor executor;
//
//        TNormalizedGINI metric;
//        TMetricHolder score = metric.Eval(approx, target, weight, {}, 0, target.size(), executor);
//
//        UNIT_ASSERT_DOUBLES_EQUAL(metric.GetFinalError(score), 0, 1e-5);
//    }
//    {
//        TVector<TVector<double>> approx{{ 1000000.0, 40, 40, 5, 5 }};
//        TVector<float> target{ 40.0, 0, 20, 0, 10 };
//        TVector<float> weight{1, 1, 1, 1, 1};
//        NPar::TLocalExecutor executor;
//
//        TNormalizedGINI metric;
//        TMetricHolder score = metric.Eval(approx, target, weight, {}, 0, target.size(), executor);
//
//        UNIT_ASSERT_DOUBLES_EQUAL(metric.GetFinalError(score), 0.6, 1e-5);
//    }
//    {
//        TVector<TVector<double>> approx{ { 40.0, 20, 10, 0, 0 }};
//        TVector<float> target{ 40.0, 20, 10, 0, 0 };
//        TVector<float> weight{1, 1, 1, 1, 1};
//        NPar::TLocalExecutor executor;
//
//        TNormalizedGINI metric;
//        TMetricHolder score = metric.Eval(approx, target, weight, {}, 0, target.size(), executor);
//
//        UNIT_ASSERT_DOUBLES_EQUAL(metric.GetFinalError(score), 1, 1e-5);
//    }
//    {
//        TVector<TVector<double>> approx{{0, 0, 10, 20, 40.0}};
//        TVector<float> target{ 40.0, 20, 10, 0, 0 };
//        TVector<float> weight{1, 1, 1, 1, 1};
//        NPar::TLocalExecutor executor;
//
//        TNormalizedGINI metric;
//        TMetricHolder score = metric.Eval(approx, target, weight, {}, 0, target.size(), executor);
//
//        UNIT_ASSERT_DOUBLES_EQUAL(metric.GetFinalError(score), -1, 1e-5);
//    }
//    {
//        TVector<TVector<double>> approx{{0.86, 0.26, 0.52, 0.32}};
//        TVector<float> target{ 1.0, 1.0, 0.0, 1.0};
//        TVector<float> weight{1, 1, 1, 1};
//        NPar::TLocalExecutor executor;
//
//        TNormalizedGINI metric;
//        TMetricHolder score = metric.Eval(approx, target, weight, {}, 0, target.size(), executor);
//
//        UNIT_ASSERT_DOUBLES_EQUAL(metric.GetFinalError(score), -0.33333333, 1e-5);
//    }
}
}