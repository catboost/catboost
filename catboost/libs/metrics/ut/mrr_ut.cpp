#include <catboost/libs/metrics/metric.h>
#include <catboost/private/libs/data_types/query.h>

#include <library/cpp/testing/unittest/registar.h>


Y_UNIT_TEST_SUITE(MRRTests) {
    Y_UNIT_TEST(MRRTest) {
        {
            TVector<TVector<double>> approx{{1, 0, -2, 5}};
            TVector<float> target{0, 0, 1, 1};
            TVector<TQueryInfo> queries;
            queries.push_back(TQueryInfo(0, 4));

            NPar::TLocalExecutor executor;
            const auto metric = std::move(CreateSingleTargetMetric(ELossFunction::MRR, TLossParams(), 1)[0]);
            TMetricHolder score = metric->Eval(approx, target, {}, queries, 0, queries.size(), executor);

            UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 1.0, 1e-5);
        }
        {
            TVector<TVector<double>> approx{{1, 0, -2, 5}};
            TVector<float> target{0, 0, 0, 0};
            TVector<TQueryInfo> queries;
            queries.push_back(TQueryInfo(0, 4));

            NPar::TLocalExecutor executor;
            const auto metric = std::move(CreateSingleTargetMetric(ELossFunction::MRR, TLossParams(), 1)[0]);
            TMetricHolder score = metric->Eval(approx, target, {}, queries, 0, queries.size(), executor);

            UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0.0, 1e-5);
        }
        {
            TVector<TVector<double>> approx{{5, 5, 5, 5}};
            TVector<float> target{0, 1, 1, 0};
            TVector<TQueryInfo> queries;
            queries.push_back(TQueryInfo(0, 4));

            NPar::TLocalExecutor executor;
            const auto metric = std::move(CreateSingleTargetMetric(ELossFunction::MRR, TLossParams(), 1)[0]);
            TMetricHolder score = metric->Eval(approx, target, {}, queries, 0, queries.size(), executor);

            UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 1./3., 1e-5);
        }
        {
            TVector<TVector<double>> approx{{
                1, 0, -2,
                2, 1, 1, 2, 5,
                -1, 0, -2, 0,
                0.5, 0.5, 0.5
            }};
            TVector<float> target{
                0, 0, 0,
                0, 1, 1, 1, 0,
                1, 1, 1, 1,
                0, 1, 1
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
            const auto metric = std::move(CreateSingleTargetMetric(ELossFunction::MRR, TLossParams(), 1)[0]);
            TMetricHolder score = metric->Eval(approx, target, {}, queries, 0, queries.size(), executor);

            UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 17./30., 1e-5);
        }
    }

    Y_UNIT_TEST(MRRTestWithTopSize) {
        {
            TVector<TVector<double>> approx{{1, 0, -2, 5}};
            TVector<float> target{0, 0, 1, 1};
            TVector<TQueryInfo> queries;
            queries.push_back(TQueryInfo(0, 4));

            NPar::TLocalExecutor executor;
            const auto metric = std::move(CreateSingleTargetMetric(ELossFunction::MRR, TLossParams::FromVector({{"top", "1"}}), 1)[0]);
            TMetricHolder score = metric->Eval(approx, target, {}, queries, 0, queries.size(), executor);

            UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 1.0, 1e-5);
        }
        {
            TVector<TVector<double>> approx{{1, 0, -2, 5}};
            TVector<float> target{0, 0, 1, 0};
            TVector<TQueryInfo> queries;
            queries.push_back(TQueryInfo(0, 4));

            NPar::TLocalExecutor executor;
            const auto metric = std::move(CreateSingleTargetMetric(ELossFunction::MRR, TLossParams::FromVector({{"top", "1"}}), 1)[0]);
            TMetricHolder score = metric->Eval(approx, target, {}, queries, 0, queries.size(), executor);

            UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0.0, 1e-5);
        }
        {
            TVector<TVector<double>> approx{{1, 0, -2, 5}};
            TVector<float> target{0, 0, 1, 0};
            TVector<TQueryInfo> queries;
            queries.push_back(TQueryInfo(0, 4));

            NPar::TLocalExecutor executor;
            const auto metric = std::move(CreateSingleTargetMetric(ELossFunction::MRR, TLossParams::FromVector({{"top", "10"}}), 1)[0]);
            TMetricHolder score = metric->Eval(approx, target, {}, queries, 0, queries.size(), executor);

            UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0.25, 1e-5);
        }
    }

    Y_UNIT_TEST(MRRTestWithBorder) {
        {
            TVector<TVector<double>> approx{{1, 0, -2, 5}};
            TVector<float> target{0.3, 0.1, 0.4, 0.6};
            TVector<TQueryInfo> queries;
            queries.push_back(TQueryInfo(0, 4));

            NPar::TLocalExecutor executor;
            const auto metric = std::move(CreateSingleTargetMetric(ELossFunction::MRR, TLossParams::FromVector({{"border", "0.8"}}), 1)[0]);
            TMetricHolder score = metric->Eval(approx, target, {}, queries, 0, queries.size(), executor);

            UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0.0, 1e-5);
        }
        {
            TVector<TVector<double>> approx{{1, 2, 2, 2}};
            TVector<float> target{0.4, 0.1, 0.4, 0.1};
            TVector<TQueryInfo> queries;
            queries.push_back(TQueryInfo(0, 4));

            NPar::TLocalExecutor executor;
            const auto metric = std::move(CreateSingleTargetMetric(ELossFunction::MRR, TLossParams::FromVector({{"border", "0.3"}}), 1)[0]);
            TMetricHolder score = metric->Eval(approx, target, {}, queries, 0, queries.size(), executor);

            UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 1./3., 1e-5);
        }
    }
}
