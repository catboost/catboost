#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/metrics/metric_holder.h>

#include <library/unittest/registar.h>

Y_UNIT_TEST_SUITE(KappaMetricTest) {
    // use balanced_accuracy_score from sklearn to compute benchmark value
    Y_UNIT_TEST(KappaTest) {
        {
            TVector<TVector<double>> approx{{1, 0, 0, 1, 0, 0, 1, 0, 0, 0},
                                            {0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
                                            {0, 1, 1, 0, 0, 1, 0, 1, 1, 1}};
            TVector<float> target{0, 2, 2, 0, 1, 2, 0, 2, 2, 2};
            TVector<float> weight{1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

            auto metric = TKappaMetric::CreateMultiClassMetric(3);
            TMetricHolder score = metric->EvalSingleThread(approx, target, weight, {}, 0, target.size());

            UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 1, 1e-1);
        }
        {
            TVector<TVector<double>> approx{{0, 1, 0, 1}};
            TVector<float> target{0, 0, 1, 1};
            TVector<float> weight{1, 1, 1, 1};

            auto metric = TKappaMetric::CreateBinClassMetric();
            TMetricHolder score = metric->EvalSingleThread(approx, target, weight, {}, 0, target.size());

            UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0, 1e-1);
        }
        {
            TVector<TVector<double>> approx{{1, 0, 0, 0},
                                            {0, 1, 0, 1},
                                            {0, 0, 1, 0}};
            TVector<float> target{2, 0, 2, 1};
            TVector<float> weight{1, 1, 1, 1};

            auto metric = TKappaMetric::CreateMultiClassMetric(3);
            TMetricHolder score = metric->EvalSingleThread(approx, target, weight, {}, 0, target.size());

            UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0.2727, 1e-4);
        }
    }
    // use cohen_kappa_score with weights='quadratic' param from sklearn to compute benchmark value
    Y_UNIT_TEST(WKappaTest) {
        {
            TVector<TVector<double>> approx{{0, 1, 1, 0}};
            TVector<float> target{1, 0, 1, 0};
            TVector<float> weight{1, 1, 1, 1};

            auto metric = TWKappaMatric::CreateBinClassMetric();
            TMetricHolder score = metric->EvalSingleThread(approx, target, weight, {}, 0, target.size());

            UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0, 1e-1);
        }
        {
            TVector<TVector<double>> approx{{0, 1, 1, 0}};
            TVector<float> target{0, 1, 1, 0};
            TVector<float> weight{1, 1, 1, 1};

            auto metric = TWKappaMatric::CreateBinClassMetric();
            TMetricHolder score = metric->EvalSingleThread(approx, target, weight, {}, 0, target.size());

            UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 1, 1e-1);
        }
        {
            TVector<TVector<double>> approx{{1, 0, 0, 1},
                                            {0, 1, 0, 0},
                                            {0, 0, 1, 0}};
            TVector<float> target{0, 2, 1, 0};
            TVector<float> weight{1, 1, 1, 1};

            auto metric = TWKappaMatric::CreateMultiClassMetric(3);
            TMetricHolder score = metric->EvalSingleThread(approx, target, weight, {}, 0, target.size());

            UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0.6363, 1e-4);
        }
    }
}
