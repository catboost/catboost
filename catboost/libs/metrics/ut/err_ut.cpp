#include <catboost/libs/metrics/metric.h>
#include <catboost/private/libs/data_types/query.h>

#include <library/cpp/testing/unittest/registar.h>


Y_UNIT_TEST_SUITE(ERRTests) {
    Y_UNIT_TEST(ERRTest) {
        {
            TVector<TVector<double>> approx{{1, 0, -2, 5}};
            TVector<float> target{0, 0, 0.4, 0.8};
            TVector<TQueryInfo> queries;
            queries.push_back(TQueryInfo(0, 4));

            NPar::TLocalExecutor executor;
            const auto metric = std::move(CreateSingleTargetMetric(ELossFunction::ERR, TLossParams(), 1)[0]);
            TMetricHolder score = metric->Eval(approx, target, {}, queries, 0, queries.size(), executor);

            UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0.82, 1e-5);
        }
        {
            TVector<TVector<double>> approx{{1, 0, -2, 5}};
            TVector<float> target{0, 0, 0, 0};
            TVector<TQueryInfo> queries;
            queries.push_back(TQueryInfo(0, 4));

            NPar::TLocalExecutor executor;
            const auto metric = std::move(CreateSingleTargetMetric(ELossFunction::ERR, TLossParams(), 1)[0]);
            TMetricHolder score = metric->Eval(approx, target, {}, queries, 0, queries.size(), executor);

            UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0.0, 1e-5);
        }
        {
            TVector<TVector<double>> approx{{5, 5, 5, 5}};
            TVector<float> target{0.1, 0.3, 0.5, 0.7};
            TVector<TQueryInfo> queries;
            queries.push_back(TQueryInfo(0, 4));

            NPar::TLocalExecutor executor;
            const auto metric = std::move(CreateSingleTargetMetric(ELossFunction::ERR, TLossParams(), 1)[0]);
            TMetricHolder score = metric->Eval(approx, target, {}, queries, 0, queries.size(), executor);

            UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0.395125, 1e-5);
        }
        {
            TVector<TVector<double>> approx{{
                1, 0, -2,
                2, 1, 1, 2, 5,
                -1, 0, -2, 0,
                0.5, 0.5, 0.5
            }};
            TVector<float> target{
                0.8, 0.3, 0.3,
                0, 0.2, 0.2, 0.2, 0,
                1, 1, 1, 1,
                1, 0, 0.5
            };
            TVector<TQueryInfo> queries;
            queries.push_back(TQueryInfo(0, 3));
            queries.back().Weight = 1;
            queries.push_back(TQueryInfo(3, 8));
            queries.back().Weight = 2;
            queries.push_back(TQueryInfo(8, 12));
            queries.back().Weight = 3;
            queries.push_back(TQueryInfo(12, 15));
            queries.back().Weight = 4;

            NPar::TLocalExecutor executor;
            const auto metric = std::move(CreateSingleTargetMetric(ELossFunction::ERR, TLossParams(), 1)[0]);
            TMetricHolder score = metric->Eval(approx, target, {}, queries, 0, queries.size(), executor);

            UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0.57752, 1e-5);
        }
    }

    Y_UNIT_TEST(ERRTestWithTopSize) {
        {
            TVector<TVector<double>> approx{{1, 0, -2, 5}};
            TVector<float> target{0, 0, 0.4, 0.8};
            TVector<TQueryInfo> queries;
            queries.push_back(TQueryInfo(0, 4));

            NPar::TLocalExecutor executor;
            const auto metric = std::move(CreateSingleTargetMetric(ELossFunction::ERR, TLossParams::FromVector({{"top", "10"}}), 1)[0]);
            TMetricHolder score = metric->Eval(approx, target, {}, queries, 0, queries.size(), executor);

            UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0.82, 1e-5);
        }
        {
            TVector<TVector<double>> approx{{1, 0, -2, 5}};
            TVector<float> target{0, 0, 0.4, 0.8};
            TVector<TQueryInfo> queries;
            queries.push_back(TQueryInfo(0, 4));

            NPar::TLocalExecutor executor;
            const auto metric = std::move(CreateSingleTargetMetric(ELossFunction::ERR, TLossParams::FromVector({{"top", "1"}}), 1)[0]);
            TMetricHolder score = metric->Eval(approx, target, {}, queries, 0, queries.size(), executor);

            UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0.8, 1e-5);
        }
        {
            TVector<TVector<double>> approx{{4, 3, 2, 1}};
            TVector<float> target{0, 0, 0.4, 0.8};
            TVector<TQueryInfo> queries;
            queries.push_back(TQueryInfo(0, 4));

            NPar::TLocalExecutor executor;
            const auto metric = std::move(CreateSingleTargetMetric(ELossFunction::ERR, TLossParams::FromVector({{"top", "2"}}), 1)[0]);
            TMetricHolder score = metric->Eval(approx, target, {}, queries, 0, queries.size(), executor);

            UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0.0, 1e-5);
        }
    }
}
