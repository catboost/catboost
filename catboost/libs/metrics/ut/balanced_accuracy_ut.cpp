#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/metrics/metric_holder.h>

#include <library/unittest/registar.h>

// use balanced_accuracy_score from sklearn to compute benchmark value
Y_UNIT_TEST_SUITE(BalancedAccuracyMetricTest) {
Y_UNIT_TEST(BalancedAccuracyTest) {
    {
        TVector<TVector<double>> approx{{0, 1, 0, 0, 1, 0}};
        TVector<float> target{0, 1, 0, 0, 0, 1};
        TVector<float> weight{1, 1, 1, 1, 1, 1};
        TVector<TQueryInfo> q;

        auto metric = TBalancedAccuracyMetric::CreateBinClassMetric();
        TMetricHolder score = metric->EvalSingleThread(approx, target, weight, q, 0, target.size());

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0.625, 1e-3);
    }
    {
        TVector<TVector<double>> approx{{0, 0, 1}};
        TVector<float> target{0, 1, 1};
        TVector<float> weight{1, 1, 1};
        TVector<TQueryInfo> q;

        auto metric = TBalancedAccuracyMetric::CreateBinClassMetric();
        TMetricHolder score = metric->EvalSingleThread(approx, target, weight, q, 0, target.size());

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0.75, 1e-2);
    }
    {
        TVector<TVector<double>> approx{{1, 1, 1, 0}};
        TVector<float> target{1, 1, 1, 0};
        TVector<float> weight{1, 1, 1, 1};
        TVector<TQueryInfo> q;

        auto metric = TBalancedAccuracyMetric::CreateBinClassMetric();
        TMetricHolder score = metric->EvalSingleThread(approx, target, weight, q, 0, target.size());

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 1, 1e-1);
    }
    {
        TVector<TVector<double>> approx{{1, 1, 1, 1}};
        TVector<float> target{1, 1, 1, 1};
        TVector<float> weight{1, 1, 1, 1};
        TVector<TQueryInfo> q;

        auto metric = TBalancedAccuracyMetric::CreateBinClassMetric();
        TMetricHolder score = metric->EvalSingleThread(approx, target, weight, q, 1, target.size());

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 1, 1e-1);
    }
    {
        TVector<TVector<double>> approx{{0, 0, 0, 0}};
        TVector<float> target{0, 0, 0, 0};
        TVector<float> weight{1, 1, 1, 1};
        TVector<TQueryInfo> q;

        auto metric = TBalancedAccuracyMetric::CreateBinClassMetric();
        TMetricHolder score = metric->EvalSingleThread(approx, target, weight, q, 1, target.size());

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 1, 1e-1);
    }
}
}
