#include <library/cpp/testing/unittest/registar.h>
#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/metrics/metric_holder.h>

//The benchmark value was calculated by sklearn.metrics.median_absolute_error
Y_UNIT_TEST_SUITE(MedianAbsoluteErrorMetricTest) {
Y_UNIT_TEST(MedianAbsoluteErrorTest) {
    {
        TVector<TVector<double>> approx{{2.5, 0.0, 2, 8}};
        TVector<float> target{3, -0.5, 2, 7};
        TVector<float> weight{1, 1, 1, 1};
        NPar::TLocalExecutor executor;

        const auto metric = std::move(CreateSingleTargetMetric(ELossFunction::MedianAbsoluteError, TLossParams(),
                                                   /*approxDimensions=*/1)[0]);
        TMetricHolder score = metric->Eval(approx, target, weight, {}, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0.5, 1e-1);
    }
    {
        TVector<TVector<double>> approx{{0.00333, 0.00333, 0.00857}};
        TVector<float> target{1, 0, 1};
        TVector<float> weight{1, 1, 1};
        NPar::TLocalExecutor executor;

        const auto metric = std::move(CreateSingleTargetMetric(ELossFunction::MedianAbsoluteError, TLossParams(),
                                                   /*approxDimensions=*/1)[0]);
        TMetricHolder score = metric->Eval(approx, target, weight, {}, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0.99143, 1e-4);
    }
    {
        TVector<TVector<double>> approx{{18.6198, 9.8278}};
        TVector<float> target{21.8598f, 15.1074f};
        TVector<float> weight{1, 1};
        NPar::TLocalExecutor executor;

        const auto metric = std::move(CreateSingleTargetMetric(ELossFunction::MedianAbsoluteError, TLossParams(),
                                                   /*approxDimensions=*/1)[0]);
        TMetricHolder score = metric->Eval(approx, target, weight, {}, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 4.2598, 1e-4);
    }
}
}
