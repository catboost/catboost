#include <library/cpp/testing/unittest/registar.h>

#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/metrics/metric_holder.h>

#include <util/generic/array_ref.h>

// The benchmark value was calculated by python implementation from https://gist.github.com/ivan339339/4a9ad34b6b4af7cb8a1cbb2b1b766d51
Y_UNIT_TEST_SUITE(RMSPEMetricTest) {
Y_UNIT_TEST(RMSPETest) {
    {
        TVector<TVector<double>> approx{{29.2f, 40.2f, -18.5f, 29.1f, 16.5f, -42.6f, 14.2f, 30.7f, 0.9f, -29.5f}};
        TVector<float> target{24.2f, 41.4f, -21.5f, 26.9f, 15.3f, -44.0f, 13.7f, 27.4f, 2.4f, -27.5f};

        NPar::TLocalExecutor executor;
        const auto metric = std::move(CreateSingleTargetMetric(ELossFunction::RMSPE, TLossParams(), /*approxDimension=*/1)[0]);
        TMetricHolder score = metric->Eval(approx, target, {}, {}, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0.221f, 1e-3);
    }
    {
        TVector<TVector<double>> approx{{-34.1f, -30.1f, -36.5f, -17.7f, 26.8f, 11.1f, 19.8f, 18.1f, -15.4f, -32.6f}};
        TVector<float> target{-34.1f, -30.1f, -36.5f, -17.7f, 26.8f, 11.1f, 19.8f, 18.1f, -15.4f, -32.6f};

        NPar::TLocalExecutor executor;
        const auto metric = std::move(CreateSingleTargetMetric(ELossFunction::RMSPE, TLossParams(), /*approxDimension=*/1)[0]);
        TMetricHolder score = metric->Eval(approx, target, {}, {}, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0, 1e-4);
    }
    {
        // test for values in interval (-1,1), metric should clip it 1
        TVector<TVector<double>> approx{{1.0f, 2.3f, 0.1f, -2.0f, -1.0f, 0.0f, -0.2f, -0.3f, 0.15f, -0.2f, 0}};
        TVector<float> target{1.2f, 2.0f, -0.3f, -1.5f, -1.5f, 0.3f, 0.0f, 0.3f, -0.15f, 0.3f, 0};

        NPar::TLocalExecutor executor;
        const auto metric = std::move(CreateSingleTargetMetric(ELossFunction::RMSPE, TLossParams(), /*approxDimension=*/1)[0]);
        TMetricHolder score = metric->Eval(approx, target, {}, {}, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0.3387, 1e-2);
    }
    {
        TVector<TVector<double>> approx{{29.2f, 40.2f, -18.5f, 29.1f, 16.5f, -42.6f, 14.2f, 30.7f, 0.9f, -29.5f}};
        TVector<float> target{24.2f, 41.4f, -21.5f, 26.9f, 15.3f, -44.0f, 13.7f, 27.4f, 2.4f, -27.5f};
        TVector<float> weights{0.8f, 0.5f, 0.84f, 0.11f, 0.03f, 0.16f, 0.15f, 0.08f, 0.84f, 0.88f};

        NPar::TLocalExecutor executor;
        const auto metric = std::move(CreateSingleTargetMetric(ELossFunction::RMSPE, TLossParams(), /*approxDimension=*/1)[0]);
        TMetricHolder score = metric->Eval(approx, target, weights, {}, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0.2965, 1e-3);
    }
}
}
