#include <library/unittest/registar.h>

#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/metrics/metric_holder.h>

#include <util/generic/array_ref.h>

Y_UNIT_TEST_SUITE(LLPMetricTest) {
Y_UNIT_TEST(LLPTest) {
    {
        TVector<TVector<double>> approx{{2.19722458, -2.19722458, -2.19722458}};
        TVector<float> target{1, 0, 0};
        TVector<float> weight{1, 1, 1};

        NPar::TLocalExecutor executor;
        const auto metric = MakeLLPMetric();
        TMetricHolder score = metric->Eval(approx, target, weight, {}, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 1.59346f, 1e-5);
    }
    {
        TVector<TVector<double>> approx{{2.19722458, -2.19722458, -2.19722458}};
        TVector<float> target{0, 0, 0};
        TVector<float> weight{1, 1, 1};

        NPar::TLocalExecutor executor;
        const auto metric = MakeLLPMetric();
        TMetricHolder score = metric->Eval(approx, target, weight, {}, 0, target.size(), executor);

        UNIT_ASSERT_EQUAL(metric->GetFinalError(score), 0);
    }
    {
        TVector<TVector<double>> approx{{2.19722458, -2.19722458, -2.19722458}};
        TVector<float> target{1, 1, 1};
        TVector<float> weight{1, 1, 1};

        NPar::TLocalExecutor executor;
        const auto metric = MakeLLPMetric();
        TMetricHolder score = metric->Eval(approx, target, weight, {}, 0, target.size(), executor);

        UNIT_ASSERT_EQUAL(metric->GetFinalError(score), 0);
    }
    {
        TVector<TVector<double>> approx{{2.19722458, -2.19722458, -2.19722458}};
        TVector<float> target{1, 0, 0};
        TVector<float> weight{0.5f, 0.5f, 0.5f};

        NPar::TLocalExecutor executor;
        const auto metric = MakeLLPMetric();
        TMetricHolder score = metric->Eval(approx, target, weight, {}, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0.403889f, 1e-5);
    }
    {
        TVector<TVector<double>> approx{{2.19722458, -2.19722458, -2.19722458}};
        TVector<float> target{1, 0, 0};
        TVector<float> weight{0.1f, 0.1f, 0.1f};

        NPar::TLocalExecutor executor;
        const auto metric = MakeLLPMetric();
        TMetricHolder score = metric->Eval(approx, target, weight, {}, 0, target.size(), executor);

        UNIT_ASSERT_EQUAL(metric->GetFinalError(score), 0);
    }
    {
        TVector<TVector<double>> approx{{2.19722458, -2.19722458, -2.19722458}};
        TVector<float> target{1, 0, 0};
        TVector<float> weight{0.9f, 0.9f, 0.9f};

        NPar::TLocalExecutor executor;
        const auto metric = MakeLLPMetric();
        TMetricHolder score = metric->Eval(approx, target, weight, {}, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 1.56530f, 1e-5);
    }
}
}
