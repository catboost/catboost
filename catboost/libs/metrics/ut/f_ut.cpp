#include <catboost/libs/helpers/vector_helpers.h>
#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/metrics/metric_holder.h>

#include <library/cpp/testing/unittest/registar.h>

Y_UNIT_TEST_SUITE(FMetricTests) {
        Y_UNIT_TEST(FSingleLabelTest) {
            {
                TVector<TVector<double>> approx{{0, 1, 1, 1, 0, 1, 1, 1, 0}};
                TVector<float> target{1, 0, 1, 0, 0, 1, 1, 1, 0};
                TVector<float> weight{1, 1, 1, 1, 1, 1, 1, 1, 1};

                NPar::TLocalExecutor executor;
                auto metric = MakeBinClassFMetric(TLossParams(), 0.5, GetDefaultPredictionBorder());
                TMetricHolder score = dynamic_cast<const ISingleTargetEval*>(metric.Get())->Eval(To2DConstArrayRef<double>(approx), {}, false, target, weight, {}, 0, target.size(), executor);

                UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0.689655172413793, 1e-6);
            }
        }
        Y_UNIT_TEST(FWeightedTest) {
            {
                TVector<TVector<double>> approx{{0, 1, 1}};
                TVector<float> target{1, 0, 1};
                TVector<float> weight{0.26705f, 0.666578f, 0.6702279f};

                NPar::TLocalExecutor executor;
                auto metric = MakeBinClassFMetric(TLossParams(), 0.5, GetDefaultPredictionBorder());
                TMetricHolder score = dynamic_cast<const ISingleTargetEval*>(metric.Get())->Eval(To2DConstArrayRef<double>(approx), {}, false, target, weight, {}, 0, target.size(), executor);

                UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0.533238714319664, 1e-6);
            }
        }
        Y_UNIT_TEST(FMultiClassTest) {
            {
                TVector<TVector<double>> approx{{1, 0, 1, 0, 1, 0, 1, 0, 0},
                                                {0, 0, 0, 0, 0, 1, 0, 1, 1},
                                                {0, 1, 0, 1, 0, 0, 0, 0, 0}};
                TVector<float> target{2, 1, 1, 2, 1, 2, 0, 1, 1};
                TVector<float> weight{1, 1, 1, 1, 1, 1, 1, 1, 1};

                NPar::TLocalExecutor executor;
                auto metric = MakeMultiClassFMetric(TLossParams(), 2, 3, 1);
                TMetricHolder score = dynamic_cast<const ISingleTargetEval*>(metric.Get())->Eval(To2DConstArrayRef<double>(approx), {}, false, target, weight, {}, 0, target.size(), executor);

                UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0.43478260869565216, 1e-6);
            }
        }
        Y_UNIT_TEST(FMultiLabelTest) {
            {
                TVector<TVector<double>> approx{{1, 1, 1, 1},
                                                {1, 1, 0, 0},
                                                {0, 1, 0, 1}};
                TVector<TVector<float>> target{{1, 1, 1, 0},
                                               {1, 0, 0, 0},
                                               {0, 1, 0, 1}};
                TVector<float> weight{1, 1, 1, 1};

                NPar::TLocalExecutor executor;
                auto metric = MakeMultiClassFMetric(TLossParams(), 2, 3, 1);
                TMetricHolder score = dynamic_cast<const IMultiTargetEval*>(metric.Get())->Eval(To2DConstArrayRef<double>(approx), {}, To2DConstArrayRef<float>(target), weight, 0, target.size(), executor);

                UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0.8333333333333334, 1e-6);
            }
        }
}
