#include <library/cpp/unittest/registar.h>
#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/metrics/metric_holder.h>

Y_UNIT_TEST_SUITE(PrecisionAtKMetricTest) {

// using tests from xgboost as benchmark
// source - https://github.com/dmlc/xgboost/blob/a1ec7b1716f78d333ead277cc17b3c04097a2b7b/tests/cpp/metric/test_rank_metric.cc
Y_UNIT_TEST(PrecisionAtKTest) {
    {
        TVector<TVector<double>> approx{{0, 1}};
        TVector<float> target{0, 1};
        TVector<float> weight{1, 1};
        TVector<TQueryInfo> queries{TQueryInfo(0, 2)};

        NPar::TLocalExecutor executor;
        const auto metric = std::move(CreateMetric(ELossFunction::PrecisionAt, TLossParams::FromVector({{"top", "2"}}), /*approxDimension=*/1)[0]);
        TMetricHolder score = metric->Eval(approx, target, weight, queries, 0, 1, executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0.5, 1e-6);
    }
    {
        TVector<TVector<double>> approx{{0.1, 0.9, 0.1, 0.9}};
        TVector<float> target{0, 0, 1, 1};
        TVector<float> weight{1, 1, 1, 1};
        TVector<TQueryInfo> queries{TQueryInfo(0, 4)};

        NPar::TLocalExecutor executor;
        const auto metric = std::move(CreateMetric(ELossFunction::PrecisionAt, TLossParams::FromVector({{"top", "2"}}), /*approxDimension=*/1)[0]);
        TMetricHolder score = metric->Eval(approx, target, weight, queries, 0, 1, executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0.5, 1e-6);
    }
}

Y_UNIT_TEST(RecallAtKTest) {
    {
        TVector<TVector<double>> approx{{0, 1}};
        TVector<float> target{0, 1};
        TVector<float> weight{1, 1};
        TVector<TQueryInfo> queries{TQueryInfo(0, 2)};

        NPar::TLocalExecutor executor;
        const auto metric = std::move(CreateMetric(ELossFunction::RecallAt, TLossParams::FromVector({{"top", "2"}}), /*approxDimension=*/1)[0]);
        TMetricHolder score = metric->Eval(approx, target, weight, queries, 0, 1, executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 1, 1e-6);
    }
    {
        TVector<TVector<double>> approx{{0.1, 0.9, 0.1, 0.9}};
        TVector<float> target{0, 0, 1, 1};
        TVector<float> weight{1, 1, 1, 1};
        TVector<TQueryInfo> queries{TQueryInfo(0, 4)};

        NPar::TLocalExecutor executor;
        const auto metric = std::move(CreateMetric(ELossFunction::RecallAt, TLossParams::FromVector({{"top", "2"}}), /*approxDimension=*/1)[0]);
        TMetricHolder score = metric->Eval(approx, target, weight, queries, 0, 1, executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0.5, 1e-6);
    }
}

Y_UNIT_TEST(MAPAtKTest) {
    {
        TVector<TVector<double>> approx{{0, 1}};
        TVector<float> target{0, 1};
        TVector<float> weight{1, 1};
        TVector<TQueryInfo> queries{TQueryInfo(0, 2)};

        NPar::TLocalExecutor executor;
        const auto metric = std::move(CreateMetric(ELossFunction::MAP, TLossParams(), /*approxDimension=*/1)[0]);
        TMetricHolder score = metric->Eval(approx, target, weight, queries, 0, 1, executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 1, 1e-1);
    }
    {
        TVector<TVector<double>> approx{{0.1, 0.9, 0.1, 0.9}};
        TVector<float> target{0, 0, 1, 1};
        TVector<float> weight{1, 1, 1, 1};
        TVector<TQueryInfo> queries{TQueryInfo(0, 4)};

        NPar::TLocalExecutor executor;
        const auto metric = std::move(CreateMetric(ELossFunction::MAP, TLossParams(), /*approxDimension=*/1)[0]);
        TMetricHolder score = metric->Eval(approx, target, weight, queries, 0, 1, executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0.5, 1e-2);
    }
    {
        TVector<TVector<double>> approx{{0.1, 0.9, 0.1, 0.9}};
        TVector<float> target{0, 0, 1, 1};
        TVector<float> weight{1, 1, 1, 1};
        TVector<TQueryInfo> queries{TQueryInfo(0, 4)};

        NPar::TLocalExecutor executor;
        const auto metric = std::move(CreateMetric(ELossFunction::MAP, /*params=*/TLossParams::FromVector({{"top", "2"}}), /*approxDimension=*/1)[0]);
        TMetricHolder score = metric->Eval(approx, target, weight, queries, 0, 1, executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0.25, 1e-3);
    }
}
}
