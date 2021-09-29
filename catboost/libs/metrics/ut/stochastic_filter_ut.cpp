#include <catboost/libs/metrics/metric.h>

#include <library/cpp/testing/unittest/registar.h>

#include <util/generic/algorithm.h>
#include <util/generic/vector.h>
#include <util/random/fast.h>

Y_UNIT_TEST_SUITE(StochasticFilterMetricTests) {
    static void StochasticFilterCheck(const TVector<double>& approx, const TVector<float>& target, const TVector<TQueryInfo>& queries,
                                      THolder<TSingleTargetMetric>& stochasticFilter, double expectedMetricValue, double epsilon, NPar::ILocalExecutor& executor) {
        TMetricHolder metricHolder = stochasticFilter->Eval({approx}, {}, false, target, {}, queries, 0, queries.ysize(), executor);
        UNIT_ASSERT_DOUBLES_EQUAL(stochasticFilter->GetFinalError(metricHolder), expectedMetricValue, epsilon);
    }

    Y_UNIT_TEST(StochasticFilterTest) {
        THolder<TSingleTargetMetric> stochasticFilter = std::move(CreateSingleTargetMetric(ELossFunction::FilteredDCG, TLossParams(), /*approxDimension=*/1)[0]);
        double epsilon = 1e-6;
        NPar::TLocalExecutor executor;

        {//One query test
            TVector<TQueryInfo> queries = {TQueryInfo{0, 3}};

            StochasticFilterCheck({0, 0, 0}, {1, 2, 3}, queries, stochasticFilter, 3, epsilon, executor);
            StochasticFilterCheck({5, 10, 25}, {1, 2, 3}, queries, stochasticFilter, 3, epsilon, executor);
            StochasticFilterCheck({-1, -1, -1}, {1, 2, 3}, queries, stochasticFilter, 0, epsilon, executor);
            StochasticFilterCheck({-1, -1, -1}, {10, 20, 30}, queries, stochasticFilter, 0, epsilon, executor);
            StochasticFilterCheck({ 0, -1, -1}, {10, 20, 30}, queries, stochasticFilter, 10, epsilon, executor);
            StochasticFilterCheck({-1,  0, -1}, {10, 20, 30}, queries, stochasticFilter, 20, epsilon, executor);
            StochasticFilterCheck({-1, -1,  0}, {10, 20, 30}, queries, stochasticFilter, 30, epsilon, executor);
        }
        {
            ui32 numQueries = 20;
            ui32 numSamples = 1000;
            TFastRng<ui64> rng(42);
            TVector<double> approx(numSamples, 0.);
            TVector<float> target(numSamples, 1.f);

            ui32 querySize = numSamples / numQueries;
            TVector<TQueryInfo> queries(numQueries);
            for (ui32 i : xrange(numQueries)) {
                queries[i] = TQueryInfo{i * querySize, (i + 1) * querySize};
            }

            TVector<float> partSum(numSamples / numQueries + 1);
            partSum[0] = 0.f;
            for (ui32 i = 1; i < partSum.size(); i++) {
                partSum[i] = partSum[i - 1] + 1.f / i;
            }

            for (ui32 i = 0; i < 30; i++) {
                Generate(approx.begin(), approx.end(), [&rng](){return (int)(rng.GenRandReal1() > 0.5) - 0.5;});
                double expectedMetricValue = 0.;
                for (ui32 j = 0; j < numQueries; j++) {
                    long numFiltered = Count(approx.begin() + queries[j].Begin, approx.begin() + queries[j].End, 0.5);
                    expectedMetricValue += partSum[numFiltered];
                }
                expectedMetricValue /= numQueries;
                StochasticFilterCheck(approx, target, queries, stochasticFilter, expectedMetricValue, epsilon, executor);
            }
        }
    }
}
