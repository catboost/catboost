#include <library/cpp/testing/unittest/registar.h>

#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/metrics/metric_holder.h>

#include <util/generic/array_ref.h>

//The benchmark value was calculated by sklearn.metrics.mean_squared_log_error
Y_UNIT_TEST_SUITE(MSLEMetricTest) {
Y_UNIT_TEST(MSLETest) {
    {
        TVector<TVector<double>> approx{{3, 5, 2.5, 7}};
        TVector<float> target{2.5, 5, 4, 8};
        TVector<float> weight{1, 1, 1, 1};

        NPar::TLocalExecutor executor;
        const auto metric = std::move(CreateSingleTargetMetric(ELossFunction::MSLE, TLossParams(), 1)[0]);
        TMetricHolder score = metric->Eval(approx, target, weight, {}, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0.03973, 1e-5);
    }
    {
        TVector<TVector<double>> approx{{0.003333, 0.003333, 0.008571}};
        TVector<float> target{1, 0, 1};
        TVector<float> weight{1, 1, 1};

        NPar::TLocalExecutor executor;
        const auto metric = std::move(CreateSingleTargetMetric(ELossFunction::MSLE, TLossParams(), 1)[0]);
        TMetricHolder score = metric->Eval(approx, target, weight, {}, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0.31485, 1e-5);
    }
}
}
