#include <library/cpp/testing/unittest/registar.h>
#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/metrics/metric_holder.h>

Y_UNIT_TEST_SUITE(HuberLossMetricTest) {
Y_UNIT_TEST(HuberLossTest) {
    {
        TVector<TVector<double>> approx{{0, 0, 0, 0}};
        TVector<float> target{0, 0, 0, 0};
        TVector<float> weight;

        NPar::TLocalExecutor executor;
        const auto metric = std::move(CreateSingleTargetMetric(ELossFunction::Huber, TLossParams::FromVector({{"delta", "1.0"}}), /*approxDimension=*/1)[0]);
        TMetricHolder score = metric->Eval(approx, target, weight, {}, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0, 1e-6);
    }
    {
        TVector<TVector<double>> approx{{0, 0, 0, 0}};
        TVector<float> target{1, 2, 3, 4};
        TVector<float> weight{0.26705f, 0.666578f, 0.6702279f, 0.3976618f};

        NPar::TLocalExecutor executor;
        const auto metric = std::move(CreateSingleTargetMetric(ELossFunction::Huber, TLossParams::FromVector({{"delta", "1.0"}}), /*approxDimension=*/1)[0]);
        metric->UseWeights = true;
        TMetricHolder score = metric->Eval(approx, target, weight, {}, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 2.0987963, 1e-6);
    }
}
}
