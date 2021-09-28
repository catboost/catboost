#include <library/cpp/testing/unittest/registar.h>
#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/metrics/metric_holder.h>

//The benchmark value was calculated by sklearn.metrics.auc
Y_UNIT_TEST_SUITE(PRAUCMetricTest) {
    Y_UNIT_TEST(BinaryClassification) {

        TVector<TVector<double>> approx{{-2, -1, -0.5, 0, 0.5, 1, 2}};
        TVector<float> target{0, 1, 0, 1, 0, 1, 1};
        TVector<float> weight;
        NPar::TLocalExecutor executor;

        const auto metric = MakeBinClassPRAUCMetric(TLossParams());
        TMetricHolder score = dynamic_cast<const ISingleTargetEval*>(metric.Get())->Eval(approx, target, weight, {}, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0.8354167, 1e-5);
    }
    Y_UNIT_TEST(BinaryClassification_EqualApproxes) {

        TVector<TVector<double>> approx{{0.1, 0.1, 0.1, 0.2, 0.2, 0.3, 0.4, 0.5, 0.5, 0.6}};
        TVector<float> target{1, 0, 0, 1, 0, 0, 0, 0, 1, 1};
        TVector<float> weight;
        NPar::TLocalExecutor executor;

        const auto metric = MakeBinClassPRAUCMetric(TLossParams());
        TMetricHolder score = dynamic_cast<const ISingleTargetEval*>(metric.Get())->Eval(approx, target, weight, {}, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0.665476, 1e-5);
    }
    Y_UNIT_TEST(MultiClassification) {

        TVector<TVector<double>> approx{
            {0.0, 0.1, 0.0, 0.0, 0.3, 0.0, 0.5, 0.0, 0.8},
            {0.4, 0.2, 0.0, 0.1, 0.3, 0.1, 0.5, 0.1, 0.1},
            {0.6, 0.7, 0.1, 0.0, 0.4, 0.0, 0.0, 0.9, 0.1},
        };
        TVector<float> target{0, 0, 1, 1, 0, 2, 0, 1, 2};
        TVector<float> weight;
        NPar::TLocalExecutor executor;

        const auto metric = MakeMultiClassPRAUCMetric(TLossParams(), 2);
        TMetricHolder score = dynamic_cast<const ISingleTargetEval*>(metric.Get())->Eval(approx, target, weight, {}, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0.138889, 1e-5);
    }
    Y_UNIT_TEST(WeightedMultiClassification) {

        TVector<TVector<double>> approx{
            {0.0, 0.1, 0.0, 0.0, 0.3, 0.0, 0.5, 0.0, 0.8},
            {0.4, 0.2, 0.0, 0.1, 0.3, 0.1, 0.5, 0.1, 0.1},
            {0.6, 0.7, 0.1, 0.0, 0.4, 0.0, 0.0, 0.9, 0.1},
        };
        TVector<float> target{0, 2, 1, 1, 0, 2, 0, 1, 2};
        TVector<float> weight{0.5, 0.3, 1, 0.9, 1, 2, 0, 0.1, 1};
        NPar::TLocalExecutor executor;

        const auto metric = MakeMultiClassPRAUCMetric(TLossParams(), 2);
        metric->UseWeights = true;
        TMetricHolder score = dynamic_cast<const ISingleTargetEval*>(metric.Get())->Eval(approx, target, weight, {}, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0.356588, 1e-5);
    }
}
