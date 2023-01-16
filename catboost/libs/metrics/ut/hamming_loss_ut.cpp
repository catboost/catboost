#include <library/cpp/testing/unittest/registar.h>
#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/metrics/metric_holder.h>

//The benchmark value was calculated by sklearn.metrics.hamming_loss
Y_UNIT_TEST_SUITE(HammingLossMetricTest) {
Y_UNIT_TEST(HammingLossTest) {
    {
        TVector<TVector<double>> approx{{0, 0, 0, 0}};
        TVector<float> target{0, 0, 0, 0};
        TVector<float> weight{0.26705f, 0.666578f, 0.6702279f, 0.3976618f};

        NPar::TLocalExecutor executor;
        const auto metric = std::move(CreateMetric(ELossFunction::HammingLoss, TLossParams(), approx.size())[0]);
        TMetricHolder score = dynamic_cast<const ISingleTargetEval*>(metric.Get())->Eval(
            approx, target, weight, {}, 0, target.size(), executor
        );

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0, 1e-1);
    }
    {
        TVector<TVector<double>> approx{{1, 1, 1, 1}, {0, 0, 0, 0}};
        TVector<float> target{0, 0, 0, 0};
        TVector<float> weight{0.26705f, 0.666578f, 0.6702279f, 0.3976618f};

        NPar::TLocalExecutor executor;
        const auto metric = std::move(CreateMetric(ELossFunction::HammingLoss, TLossParams(), approx.size())[0]);
        TMetricHolder score = dynamic_cast<const ISingleTargetEval*>(metric.Get())->Eval(
            approx, target, weight, {}, 0, target.size(), executor
        );

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0, 1e-1);
    }
    {
        TVector<TVector<double>> approx{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
        TVector<float> target{0, 0, 2, 3};
        TVector<float> weight{1, 1, 1, 1};

        NPar::TLocalExecutor executor;
        const auto metric = std::move(CreateMetric(ELossFunction::HammingLoss, TLossParams(), approx.size())[0]);
        TMetricHolder score = dynamic_cast<const ISingleTargetEval*>(metric.Get())->Eval(
            approx, target, weight, {}, 0, target.size(), executor
        );

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0.25, 1e-1);
    }
    {
        TVector<TVector<double>> approx{{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1}};
        TVector<float> target{1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1};
        TVector<float> weight{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

        NPar::TLocalExecutor executor;
        const auto metric = std::move(CreateMetric(ELossFunction::HammingLoss, TLossParams(), approx.size())[0]);
        TMetricHolder score = dynamic_cast<const ISingleTargetEval*>(metric.Get())->Eval(
            approx, target, weight, {}, 0, target.size(), executor
        );

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0.192308, 1e-6);
    }
    {
        TVector<TVector<double>> approx{{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}};
        TVector<float> target{1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1};
        TVector<float> weight{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

        NPar::TLocalExecutor executor;
        const auto metric = std::move(CreateMetric(ELossFunction::HammingLoss, TLossParams(), approx.size())[0]);
        TMetricHolder score = dynamic_cast<const ISingleTargetEval*>(metric.Get())->Eval(
            approx, target, weight, {}, 0, target.size(), executor
        );

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0.153846, 1e-6);
    }
    {
        TVector<TVector<double>> approx{{1, 0, 1, 1}};
        TVector<float> target{1, 0, 1, 1};
        TVector<float> weight{1, 1, 1, 1};

        NPar::TLocalExecutor executor;
        const auto metric = std::move(CreateMetric(ELossFunction::HammingLoss, TLossParams(), approx.size())[0]);
        TMetricHolder score = dynamic_cast<const ISingleTargetEval*>(metric.Get())->Eval(
            approx, target, weight, {}, 0, target.size(), executor
        );

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0, 1e-1);
    }
}
}
