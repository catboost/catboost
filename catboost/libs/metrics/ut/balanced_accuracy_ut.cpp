#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/metrics/metric_holder.h>

#include <library/cpp/testing/unittest/registar.h>

// use balanced_accuracy_score from sklearn to compute benchmark value
Y_UNIT_TEST_SUITE(BalancedAccuracyMetricTest) {
Y_UNIT_TEST(BalancedAccuracyTest) {
    {
        TVector<TVector<double>> approx{{0, 1, 0, 0, 1, 0}};
        TVector<float> target{0, 1, 0, 0, 0, 1};
        TVector<float> weight{1, 1, 1, 1, 1, 1};

        NPar::TLocalExecutor executor;
        auto metric = std::move(CreateSingleTargetMetric(ELossFunction::BalancedAccuracy,
                                             TLossParams(), /*approxDimension=*/1)[0]);
        TMetricHolder score = metric->Eval(approx, target, weight, {}, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0.625, 1e-3);
    }
    {
        TVector<TVector<double>> approx{{0, 0, 1}};
        TVector<float> target{0, 1, 1};
        TVector<float> weight{1, 1, 1};

        NPar::TLocalExecutor executor;
        auto metric = std::move(CreateSingleTargetMetric(ELossFunction::BalancedAccuracy,
                                             TLossParams(), /*approxDimension=*/1)[0]);
        TMetricHolder score = metric->Eval(approx, target, weight, {}, 0, target.size(), executor);;

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0.75, 1e-2);
    }
    {
        TVector<TVector<double>> approx{{1, 1, 1, 0}};
        TVector<float> target{1, 1, 1, 0};
        TVector<float> weight{1, 1, 1, 1};

        NPar::TLocalExecutor executor;
        auto metric = std::move(CreateSingleTargetMetric(ELossFunction::BalancedAccuracy,
                                             TLossParams(), /*approxDimension=*/1)[0]);
        TMetricHolder score = metric->Eval(approx, target, weight, {}, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 1, 1e-1);
    }
    {
        TVector<TVector<double>> approx{{1, 1, 1, 1}};
        TVector<float> target{1, 1, 1, 1};
        TVector<float> weight{1, 1, 1, 1};

        NPar::TLocalExecutor executor;
        auto metric = std::move(CreateSingleTargetMetric(ELossFunction::BalancedAccuracy,
                                             TLossParams(), /*approxDimension=*/1)[0]);
        TMetricHolder score = metric->Eval(approx, target, weight, {}, 1, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 1, 1e-1);
    }
    {
        TVector<TVector<double>> approx{{0, 0, 0, 0}};
        TVector<float> target{0, 0, 0, 0};
        TVector<float> weight{1, 1, 1, 1};

        NPar::TLocalExecutor executor;
        auto metric = std::move(CreateSingleTargetMetric(ELossFunction::BalancedAccuracy,
                                             TLossParams(), /*approxDimension=*/1)[0]);
        TMetricHolder score = metric->Eval(approx, target, weight, {}, 1, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 1, 1e-1);
    }
}
}
