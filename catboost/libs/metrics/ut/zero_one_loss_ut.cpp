#include <library/cpp/testing/unittest/registar.h>
#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/metrics/metric_holder.h>

//The benchmark value was calculated by sklearn.metrics.zero_one_loss
Y_UNIT_TEST_SUITE(ZeroOneLossCachingMetricTest) {
Y_UNIT_TEST(ZeroOneLossTest) {
    {
        TVector<TVector<double>> approx{{0, 1, 1, 0}};
        TVector<float> target{0, 1, 1, 0};
        TVector<float> weight{1, 1, 1, 1};

        NPar::TLocalExecutor executor;
        const auto metric = std::move(CreateSingleTargetMetric(ELossFunction::ZeroOneLoss, TLossParams(), /*approxDimensions=*/1).front());
        TMetricHolder score = metric->Eval(approx, target, weight, {}, 0, 4, executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0, 1e-1);
    }
    {
        TVector<TVector<double>> approx{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
        TVector<float> target{0, 0, 2, 3};
        TVector<float> weight{1, 1, 1, 1};

        NPar::TLocalExecutor executor;
        const auto metric = std::move(CreateSingleTargetMetric(ELossFunction::ZeroOneLoss, TLossParams(), approx.size()).front());
        TMetricHolder score = metric->Eval(approx, target, weight, {}, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0.25, 1e-1);
    }
    {
        TVector<TVector<double>> approx{{1, 1, 1, 1}, {0, 0, 0, 0}};
        TVector<float> target{0, 0, 0, 0};
        TVector<float> weight{0.26705f, 0.666578f, 0.6702279f, 0.3976618f};

        NPar::TLocalExecutor executor;
        const auto metric = std::move(CreateSingleTargetMetric(ELossFunction::ZeroOneLoss, TLossParams(), approx.size()).front());
        TMetricHolder score = metric->Eval(approx, target, weight, {}, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0, 1e-1);
    }
    {
        TVector<TVector<double>> approx{{1, 0, 0, 1}};
        TVector<float> target{0, 1, 1, 0};
        TVector<float> weight{1, 1, 1, 1};

        NPar::TLocalExecutor executor;
        const auto metric = std::move(CreateSingleTargetMetric(ELossFunction::ZeroOneLoss, TLossParams(), approx.size()).front());
        TMetricHolder score = metric->Eval(approx, target, weight, {}, 0, 4, executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 1, 1e-1);
    }
    {
        TVector<TVector<double>> approx{{1, 1, 0, 0}};
        TVector<float> target{0, 1, 1, 0};
        TVector<float> weight{1, 1, 1, 1};

        NPar::TLocalExecutor executor;
        const auto metric = std::move(CreateSingleTargetMetric(ELossFunction::ZeroOneLoss, TLossParams(), approx.size()).front());
        TMetricHolder score = metric->Eval(approx, target, weight, {}, 0, 4, executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0.5, 1e-1);
    }
}
}
