#include <library/cpp/testing/unittest/registar.h>

#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/metrics/metric_holder.h>

#include <util/generic/array_ref.h>


//The benchmark value was calculated by sklearn.metrics.brier_score_loss
Y_UNIT_TEST_SUITE(BrierScoreMetricTest) {
Y_UNIT_TEST(BrierScoreTest) {
    {
        TVector<TVector<double>> approx{{-2.19722458,  2.19722458,  1.38629436, -0.84729786}};
        TVector<float> target{0, 1, 1, 0};
        TVector<float> weight{1, 1, 1, 1};

        NPar::TLocalExecutor executor;
        const auto metric = std::move(CreateSingleTargetMetric(ELossFunction::BrierScore, TLossParams(), /*approxDimension=*/1)[0]);
        TMetricHolder score = metric->Eval(approx, target, weight, {}, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0.03749999999999999, 1e-5);
    }
    {
        TVector<TVector<double>> approx{{100, 100, 100, 100}};
        TVector<float> target{1, 1, 1, 1};
        TVector<float> weight{1, 1, 1, 1};

        NPar::TLocalExecutor executor;
        const auto metric = std::move(CreateSingleTargetMetric(ELossFunction::BrierScore, TLossParams(), /*approxDimension=*/1)[0]);
        TMetricHolder score = metric->Eval(approx, target, weight, {}, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0, 1e-5);
    }
    {
        TVector<TVector<double>> approx{{-100, -100, -100, -100}};
        TVector<float> target{1, 1, 1, 1};
        TVector<float> weight{1, 1, 1, 1};

        NPar::TLocalExecutor executor;
        const auto metric = std::move(CreateSingleTargetMetric(ELossFunction::BrierScore, TLossParams(), /*approxDimension=*/1)[0]);
        TMetricHolder score = metric->Eval(approx, target, weight, {}, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 1, 1e-5);
    }
}
}
