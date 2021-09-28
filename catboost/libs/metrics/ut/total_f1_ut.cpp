#include <catboost/libs/helpers/vector_helpers.h>
#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/metrics/metric_holder.h>

#include <library/cpp/testing/unittest/registar.h>

Y_UNIT_TEST_SUITE(TotalF1MetricTest) {
    Y_UNIT_TEST(TotalF1WeightedTest) {
        {
            TVector<TVector<double>> approx{{1, 0, 1, 0, 1, 0, 1, 0, 0},
                                            {0, 0, 0, 0, 0, 1, 0, 1, 1},
                                            {0, 1, 0, 1, 0, 0, 0, 0, 0}};
            TVector<float> target{2, 1, 1, 2, 1, 2, 0, 1, 1};
            TVector<float> weight{1, 1, 1, 1, 1, 1, 1, 1, 1};

            NPar::TLocalExecutor executor;
            auto metric = MakeTotalF1Metric(TLossParams(), 3, EF1AverageType::Weighted);
            TMetricHolder score = dynamic_cast<const ISingleTargetEval*>(metric.Get())->Eval(To2DConstArrayRef<double>(approx), {}, false, target, weight, {}, 0, target.size(), executor);

            UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0.4555555555555556, 1e-6);
        }
    }
    Y_UNIT_TEST(TotalF1MicroTest) {
        {
            TVector<TVector<double>> approx{{1, 0, 1, 0, 1, 0, 1, 0, 0},
                                            {0, 0, 0, 0, 0, 1, 0, 1, 1},
                                            {0, 1, 0, 1, 0, 0, 0, 0, 0}};
            TVector<float> target{2, 1, 1, 2, 1, 2, 0, 1, 1};
            TVector<float> weight{1, 1, 1, 1, 1, 1, 1, 1, 1};

            NPar::TLocalExecutor executor;
            auto metric = MakeTotalF1Metric(TLossParams(), 3, EF1AverageType::Micro);
            TMetricHolder score = dynamic_cast<const ISingleTargetEval*>(metric.Get())->Eval(To2DConstArrayRef<double>(approx), {}, false, target, weight, {}, 0, target.size(), executor);

            UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0.4444444444444444, 1e-6);
        }
    }
    Y_UNIT_TEST(TotalF1MacroTest) {
        {
            TVector<TVector<double>> approx{{1, 0, 1, 0, 1, 0, 1, 0, 0},
                                            {0, 0, 0, 0, 0, 1, 0, 1, 1},
                                            {0, 1, 0, 1, 0, 0, 0, 0, 0}};
            TVector<float> target{2, 1, 1, 2, 1, 2, 0, 1, 1};
            TVector<float> weight{1, 1, 1, 1, 1, 1, 1, 1, 1};

            NPar::TLocalExecutor executor;
            auto metric = MakeTotalF1Metric(TLossParams(), 3, EF1AverageType::Macro);
            TMetricHolder score = dynamic_cast<const ISingleTargetEval*>(metric.Get())->Eval(To2DConstArrayRef<double>(approx), {}, false, target, weight, {}, 0, target.size(), executor);

            UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0.43333333333333335, 1e-6);
        }
    }
    Y_UNIT_TEST(TotalF1BinClassTest) {
        {
            TVector<TVector<double>> approx{{0, 1, 1, 1, 0, 1, 1, 1, 0}};
            TVector<float> target{1, 0, 1, 0, 0, 1, 1, 1, 0};
            TVector<float> weight{1, 1, 1, 1, 1, 1, 1, 1, 1};

            NPar::TLocalExecutor executor;
            auto metric = MakeTotalF1Metric(TLossParams());
            TMetricHolder score = dynamic_cast<const ISingleTargetEval*>(metric.Get())->Eval(To2DConstArrayRef<double>(approx), {}, false, target, weight, {}, 0, target.size(), executor);

            UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0.6580086580086579, 1e-6);
        }
    }
}
