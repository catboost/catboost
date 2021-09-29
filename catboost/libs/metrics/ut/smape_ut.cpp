#include <library/cpp/testing/unittest/registar.h>

#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/metrics/metric_holder.h>

#include <util/generic/array_ref.h>

// The benchmark value was calculated by
// https://github.com/T-002/pycast/blob/master/pycast/tests/symmetricmeanabsolutepercentageerrortest.py
Y_UNIT_TEST_SUITE(SMAPEMetricTest) {
Y_UNIT_TEST(SMAPETest) {
    {
        TVector<TVector<double>> approx{{1.0f, 2.3f, 0.1f, -2.0f, -1.0f, 0.0f, -0.2f, -0.3f, 0.15f, -0.2f, 0}};
        TVector<float> target{1.2f, 2.0f, -0.3f, -1.5f, -1.5f, 0.3f, 0.0f, 0.3f, -0.15f, 0.3f, 0};

        NPar::TLocalExecutor executor;
        const auto metric = std::move(CreateSingleTargetMetric(ELossFunction::SMAPE, TLossParams(), /*approxDimension=*/1)[0]);
        TMetricHolder score = metric->Eval(approx, target, {}, {}, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 118.24f, 1e-2);
    }
    {
        TVector<TVector<double>> approx{{1.2f, 2.0f, -0.3f, -1.5f, -1.5f, 0.3f, 0.0f, 0.3f, -0.15f, 0.3f, 0}};
        TVector<float> target{1.2f, 2.0f, -0.3f, -1.5f, -1.5f, 0.3f, 0.0f, 0.3f, -0.15f, 0.3f, 0};

        NPar::TLocalExecutor executor;
        const auto metric = std::move(CreateSingleTargetMetric(ELossFunction::SMAPE, TLossParams(), /*approxDimension=*/1)[0]);
        TMetricHolder score = metric->Eval(approx, target, {}, {}, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0, 1e-4);
    }
}
}
