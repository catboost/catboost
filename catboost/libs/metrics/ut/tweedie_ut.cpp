#include <library/cpp/testing/unittest/registar.h>

#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/metrics/metric_holder.h>

#include <util/generic/array_ref.h>

Y_UNIT_TEST_SUITE(TweedieMetricTest) {
    Y_UNIT_TEST(TweedieTest) {
        {
            TVector<TVector<double>> approx{{1, 2, 3, 4}};
            TVector<float> target{3, 4, 5, 1};
            TVector<float> weight{1, 1, 1, 1};

            NPar::TLocalExecutor executor;
            const auto metric = std::move(CreateSingleTargetMetric(ELossFunction::Tweedie,
                                                                   TLossParams::FromVector({{"variance_power", "1.5"}}),
                                                                   /*approxDimension=*/1)[0]);
            TMetricHolder score = metric->Eval(approx, target, weight, {}, 0, target.size(), executor);

            UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 10.38992204811516, 1e-5);
        }
    }
}
