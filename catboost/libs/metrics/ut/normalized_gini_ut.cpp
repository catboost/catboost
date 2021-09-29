#include <library/cpp/testing/unittest/registar.h>
#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/metrics/metric_holder.h>

// test created based on sklearn.metrics.roc_auc_score and fact that
// Gini_normalized = 2*AUC - 1
Y_UNIT_TEST_SUITE(NormalizedGiniMetricTest) {
Y_UNIT_TEST(BinClassTest) {
    {
        TVector<TVector<double>> approx{{0, 2, 1}};
        TVector<float> target{0, 0, 1};
        TVector<float> weight;
        TVector<TQueryInfo> queries;

        NPar::TLocalExecutor executor;
        const auto metric = std::move(CreateSingleTargetMetric(ELossFunction::NormalizedGini, TLossParams(), 1)[0]);
        TMetricHolder score = metric->Eval(approx, target, weight, queries, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0.0, 1e-6);
    }
    {
        TVector<TVector<double>> approx{{8, 7, 6, 5, 4, 3, 2, 1}};
        TVector<float> target{0, 1, 1, 1, 0, 0, 0, 1};
        TVector<float> weight;
        TVector<TQueryInfo> queries;

        NPar::TLocalExecutor executor;
        const auto metric = std::move(CreateSingleTargetMetric(ELossFunction::NormalizedGini, TLossParams(), 1)[0]);
        TMetricHolder score = metric->Eval(approx, target, weight, queries, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0.125, 1e-6);
    }
    {
        TVector<TVector<double>> approx{{8, 7, 6, 5, 4, 3, 2, 1}};
        TVector<float> target{0, 1, 1, 1, 0, 0, 0, 1};
        TVector<float> weight{0.0, 1, 1, 1, 1, 1, 1, 0.0};
        TVector<TQueryInfo> queries;

        NPar::TLocalExecutor executor;
        const auto metric = std::move(CreateSingleTargetMetric(ELossFunction::NormalizedGini, TLossParams(), 1)[0]);
        metric->UseWeights = true;
        TMetricHolder score = metric->Eval(approx, target, weight, queries, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 1.0, 1e-6);
    }
    {
        TVector<TVector<double>> approx{{8, 7, 6, 5, 4, 3, 2, 1}};
        TVector<float> target{0, 1, 1, 1, 0, 0, 0, 1};
        TVector<float> weight{0.1, 1, 1, 1, 1, 1, 1, 0.1};
        TVector<TQueryInfo> queries;

        NPar::TLocalExecutor executor;
        const auto metric = std::move(CreateSingleTargetMetric(ELossFunction::NormalizedGini, TLossParams(), 1)[0]);
        metric->UseWeights = true;
        TMetricHolder score = metric->Eval(approx, target, weight, queries, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0.8730489073881373, 1e-6);
    }
    {
        TVector<TVector<double>> approx{{8, 7, 6, 5, 4, 3, 2, 1}};
        TVector<float> target{1, 1, 1, 1, 0, 0, 0, 0};
        TVector<float> weight;
        TVector<TQueryInfo> queries;

        NPar::TLocalExecutor executor;
        const auto metric = std::move(CreateSingleTargetMetric(ELossFunction::NormalizedGini, TLossParams(), 1)[0]);
        TMetricHolder score = metric->Eval(approx, target, weight, queries, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 1.0, 1e-6);
    }
}
Y_UNIT_TEST(MultiClassTest) {
    {
        TVector<TVector<double>> approx{{0, 1/3.0, 2/3.0}, {0, 2/3.0, 1/3.0}};
        TVector<float> target{0, 0, 1};
        TVector<float> weight;
        TVector<TQueryInfo> queries;

        NPar::TLocalExecutor executor;

        // Metric with target class = 1.
        const auto metric = std::move(CreateSingleTargetMetric(ELossFunction::NormalizedGini, TLossParams(), 2)[1]);
        TMetricHolder score = metric->Eval(approx, target, weight, queries, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0.0, 1e-6);
    }
    {
        TVector<TVector<double>> approx{{0, 1, 2, 3, 4, 5, 6, 7}, {8, 7, 6, 5, 4, 3, 2, 1}};
        TVector<float> target{0, 1, 1, 1, 0, 0, 0, 1};
        TVector<float> weight;
        TVector<TQueryInfo> queries;

        NPar::TLocalExecutor executor;

        // Metric with target class = 1.
        const auto metric = std::move(CreateSingleTargetMetric(ELossFunction::NormalizedGini, TLossParams(), 2)[1]);
        TMetricHolder score = metric->Eval(approx, target, weight, queries, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0.125, 1e-6);
    }
    {
        TVector<TVector<double>> approx{{0, 1, 2, 3, 4, 5, 6, 7}, {8, 7, 6, 5, 4, 3, 2, 1}};
        TVector<float> target{1, 1, 1, 1, 0, 0, 0, 0};
        TVector<float> weight;
        TVector<TQueryInfo> queries;

        NPar::TLocalExecutor executor;

        // Metric with target class = 1.
        const auto metric = std::move(CreateSingleTargetMetric(ELossFunction::NormalizedGini, TLossParams(), 2)[1]);
        TMetricHolder score = metric->Eval(approx, target, weight, queries, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 1.0, 1e-6);
    }
    {
        TVector<TVector<double>> approx{{0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5}, {8, 7, 6, 5, 4, 3, 2, 1}, {0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5}};
        TVector<float> target{0, 1, 1, 1, 0, 0, 0, 1};
        TVector<float> weight;
        TVector<TQueryInfo> queries;

        NPar::TLocalExecutor executor;

        // Metric with target class = 1.
        const auto metric = std::move(CreateSingleTargetMetric(ELossFunction::NormalizedGini, TLossParams(), 2)[1]);
        TMetricHolder score = metric->Eval(approx, target, weight, queries, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0.125, 1e-6);
    }
}
}
