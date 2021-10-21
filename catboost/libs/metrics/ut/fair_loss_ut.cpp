#include <library/cpp/testing/unittest/registar.h>
#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/metrics/metric_holder.h>

// tests created based on python implementation of loss
// since formulas in python are more readable and hence less likely to be incorrect
Y_UNIT_TEST_SUITE(FairLossMetricTest) {
Y_UNIT_TEST(FairLossTest) {
    {
        TVector<TVector<double>> approx{{0, 1, 2}};
        TVector<float> target{0, 1, 2};
        TVector<float> weight;
        TVector<TQueryInfo> queries;

        NPar::TLocalExecutor executor;
        const auto metric = std::move(CreateSingleTargetMetric(ELossFunction::FairLoss, TLossParams::FromVector({{"smoothness", "1.0"}}), /*approxDimension=*/1)[0]);
        TMetricHolder score = metric->Eval(approx, target, weight, queries, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0.0, 1e-6);
    }
    {
        TVector<TVector<double>> approx{{1, 2, 3}};
        TVector<float> target{3, 2, 1};
        TVector<float> weight{1, 8, 1};
        TVector<TQueryInfo> queries;

        NPar::TLocalExecutor executor;
        const auto metric = std::move(CreateSingleTargetMetric(ELossFunction::FairLoss, TLossParams::FromVector({{"smoothness", "1.0"}}), /*approxDimension=*/1)[0]);
        metric->UseWeights = true;
        TMetricHolder score = metric->Eval(approx, target, weight, queries, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0.18027754226637804, 1e-6);
    }
    {
        TVector<TVector<double>> approx{{1, 2, 3}};
        TVector<float> target{3, 2, 1};
        TVector<float> weight{1, 8, 1};
        TVector<TQueryInfo> queries;

        NPar::TLocalExecutor executor;
        const auto metric = std::move(CreateSingleTargetMetric(ELossFunction::FairLoss, TLossParams::FromVector({{"smoothness", "10.0"}}), /*approxDimension=*/1)[0]);
        metric->UseWeights = true;
        TMetricHolder score = metric->Eval(approx, target, weight, queries, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0.3535688641209084, 1e-6);
    }
}
}
