#include <catboost/libs/metrics/auc_mu.h>
#include <catboost/libs/helpers/cpu_random.h>
#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/metrics/metric_holder.h>

#include <library/cpp/testing/unittest/registar.h>

#include <util/random/fast.h>
#include <util/random/shuffle.h>
#include <util/generic/array_ref.h>
#include <catboost/private/libs/data_types/query.h>
#include <catboost/libs/metrics/sample.h>



constexpr double EPS = 1e-8;

static TVector<double> RandomProbabilitiesDistribution(ui32 count, TFastRng<ui64>& rng) {
    TVector<double> result;
    result.reserve(count);
    for (ui32 i = 0; i < count; ++i) {
        result.emplace_back(rng.GenRandReal3());
    }
    double sum = 0;
    for (ui32 i = 0; i < count; ++i) {
        sum += result[i];
    }
    for (ui32 i = 0; i < count; ++i) {
        result[i] /= sum;
    }
    return result;
}

static TVector<TVector<double>> GenRandomApprox(ui32 classCount, ui32 samplesCount, TFastRng<ui64>& rng) {
    TVector<TVector<double>> result(classCount);
    for (ui32 i = 0; i < classCount; ++i) {
        result[i].reserve(samplesCount);
    }
    for (ui32 i = 0; i < samplesCount; ++i) {
        TVector<double> probabilities = RandomProbabilitiesDistribution(classCount, rng);
        for (ui32 classId = 0; classId < classCount; ++classId) {
            result[classId].emplace_back(probabilities[classId]);
        }
    }
    return result;
}

static TVector<double> GenRandomTarget(ui32 classCount, ui32 samplesCount, TRandom& rnd) {
    TVector<double> target;
    target.reserve(samplesCount);
    for (ui32 i = 0; i < samplesCount; ++i) {
        target.emplace_back(rnd(classCount));
    }
    return target;
}

static TVector<double> GenOptimalTarget(const TVector<TVector<double>>& approx, bool isBest = true) {
    TVector<double> target;
    target.reserve(approx[0].size());
    for (ui32 i = 0; i < approx[0].size(); ++i) {
        ui32 opt = 0;
        for (ui32 classId = 0; classId < approx.size(); ++classId) {
            if ((approx[classId][i] < approx[opt][i]) ^ isBest) {
                opt = classId;
            }
        }
        target.emplace_back(opt);
    }
    return target;
}

static double RandomSignedNumber(TFastRng<ui64>& rng) {
    return 2 * rng.GenRandReal3() - 1;
}

static TVector<TVector<double>> GenRandomMisclassCostMatrix(ui32 classCount, TFastRng<ui64>& rng, bool twoValues = false) {
    TVector<TVector<double>> result(classCount);
    for (ui32 i = 0; i < classCount; ++i) {
        result[i].reserve(classCount);
        for (ui32 j = 0; j < classCount; ++j) {
            result[i].emplace_back(RandomSignedNumber(rng));
            if (twoValues) {
                if (result[i].back() < 0) {
                    result[i].back() = -1;
                } else {
                    result[i].back() = 1;
                }
            }
        }
        result[i][i] = 0;
    }
    return result;
}

static TVector<double> GenRandomWeight(ui32 samplesCount, TFastRng<ui64>& rng) {
    TVector<double> result;
    result.reserve(samplesCount);
    for (ui32 i = 0; i < samplesCount; ++i) {
        result.emplace_back(rng.GenRandReal3());
    }
    return result;
}

static double DotProduct(
    const TVector<double>& first,
    const TVector<double>& second
) {
    double result = 0;
    for (ui32 i = 0; i < first.size(); ++i) {
        result += first[i] * second[i];
    }
    return result;
}

static double MyMuAuc(
    const TVector<TVector<double>>& approx,
    const TVector<double>& target,
    const TVector<double>& weight,
    const TMaybe<TVector<TVector<double>>>& misclassCostMatrixMaybe
) {
    ui32 classCount = approx.size();
    ui32 samplesCount = target.size();
    TVector<TVector<double>> misclassCostMatrix;
    if (misclassCostMatrixMaybe) {
        misclassCostMatrix = *misclassCostMatrixMaybe;
    } else {
        misclassCostMatrix.resize(classCount);
        for (ui32 i = 0; i < classCount; ++i) {
            misclassCostMatrix[i].assign(classCount, 1);
            misclassCostMatrix[i][i] = 0;
        }
    }
    double result = 0;
    for (ui32 firstClassId = 0; firstClassId < classCount; ++firstClassId) {
        for (ui32 secondClassId = firstClassId + 1; secondClassId < classCount; ++secondClassId) {
            TVector<double> currentVector(classCount);
            for (ui32 column = 0; column < classCount; ++column) {
                currentVector[column] = misclassCostMatrix[firstClassId][column] - misclassCostMatrix[secondClassId][column];
            }
            TVector<double> indicator(classCount, 0);
            indicator[firstClassId] = 1;
            indicator[secondClassId] = -1;
            double pairWeightSum = 0;
            bool isEmpty = true;
            double currentResult = 0;
            const auto realWeight = [&](ui32 index) {
                return weight.empty() ? 1.0 : weight[index];
            };
            for (ui32 i = 0; i < samplesCount; ++i) {
                for (ui32 j = 0; j < samplesCount; ++j) {
                    if (target[i] != firstClassId || target[j] != secondClassId) {
                        continue;
                    }
                    isEmpty = false;
                    pairWeightSum += realWeight(i) * realWeight(j);
                    TVector<double> currentDiff(classCount, 0);
                    for (ui32 row = 0; row < classCount; ++row) {
                        currentDiff[row] = approx[row][i] - approx[row][j];
                    }
                    double multiply = DotProduct(currentVector, indicator) * DotProduct(currentVector, currentDiff);
                    if (multiply == 0) {
                        currentResult += (realWeight(i) * realWeight(j)) / 2;
                    } else if (multiply > 0) {
                        currentResult += realWeight(i) * realWeight(j);
                    }
                }
            }
            if (!isEmpty) {
                result += currentResult / pairWeightSum;
            }
        }
    }
    return (2 * result) / (classCount * (classCount - 1));
}

Y_UNIT_TEST_SUITE(AUCMuMetricTests) {
    static void TestMuAuc(
        const TVector<TVector<double>>& approx,
        const TVector<double>& target,
        const TVector<double>& weight,
        const TMaybe<TVector<TVector<double>>>& misclassCostMatrix = Nothing()
    ) {
        TVector<float> currentTarget(target.begin(), target.end());
        const auto convertedTarget = TConstArrayRef<float>(currentTarget);
        TVector<float> currentWeight(weight.begin(), weight.end());
        const auto convertedWeight = TConstArrayRef<float>(currentWeight);
        double scoreParallel = CalcMuAuc(approx, convertedTarget, convertedWeight, 32, misclassCostMatrix);
        double scoreOneThread = CalcMuAuc(approx, convertedTarget, convertedWeight, 1, misclassCostMatrix);
        double scoreSlowFunction = MyMuAuc(approx, target, weight, misclassCostMatrix);
        UNIT_ASSERT_DOUBLES_EQUAL(scoreParallel, scoreOneThread, EPS);
        UNIT_ASSERT_DOUBLES_EQUAL(scoreParallel, scoreSlowFunction, EPS);
    }

    static void StressTest(ui32 classCount, ui32 samplesCount, ui32 iterations, bool useWeights = false, bool useArgmax = true) {
        TFastRng<ui64> rng(239);
        TRandom rnd(239);
        for (ui32 iterationId = 0; iterationId < iterations; ++iterationId) {
            TVector<TVector<double>> approx = GenRandomApprox(classCount, samplesCount, rng);
            TVector<double> weight;
            if (useWeights) {
                weight = GenRandomWeight(samplesCount, rng);
            }
            TMaybe<TVector<TVector<double>>> misclassCostMatrix = Nothing();
            if (!useArgmax) {
                misclassCostMatrix.ConstructInPlace(GenRandomMisclassCostMatrix(classCount, rng, rnd(2)));
            }
            TVector<double> target = GenRandomTarget(classCount, samplesCount, rnd);
            TestMuAuc(approx, target, weight, misclassCostMatrix);
            target = GenOptimalTarget(approx, true);
            TestMuAuc(approx, target, weight, misclassCostMatrix);
            target = GenOptimalTarget(approx, false);
            TestMuAuc(approx, target, weight, misclassCostMatrix);
        }
    }

    static void TestQueryMuAuc(
        TString description, // description of mu auc metric
        const TVector<TVector<double>>& prediction,
        const TVector<float>& targets,
        const TVector<float>& weights,
        const TVector<TQueryInfo> queryInfos,
        const TMaybe<TVector<TVector<double>>>& misclassCostMatrix = Nothing()
    ) {
        NPar::TLocalExecutor executor;
        executor.RunAdditionalThreads(31);

        const auto queryAUC = std::move(CreateMetricsFromDescription({description}, 2).front());

        TMetricHolder metricHm(2);

        for (const auto& info: queryInfos) {
            TVector<NMetrics::TSample> samples;
            auto startIdx = info.Begin;
            auto endIdx = info.End;

            TVector<float> currentTarget(targets.begin() + startIdx, targets.begin() + endIdx);
            const auto convertedTarget = TConstArrayRef<float>(currentTarget);
            TVector<float> currentWeight(weights.begin() + startIdx, weights.begin() + endIdx);
            const auto convertedWeight = TConstArrayRef<float>(currentWeight);

            TVector<TVector<double>> currentApprox(prediction.begin() + startIdx, prediction.begin() + endIdx);

            metricHm.Stats[0] += CalcMuAuc(currentApprox, convertedTarget, convertedWeight, 1, misclassCostMatrix);
            metricHm.Stats[1] += 1;

        }

        const auto metric = dynamic_cast<const ISingleTargetEval*>(queryAUC.Get())->Eval(
            prediction,
            targets,
            weights,
            queryInfos,
            0,
            queryInfos.size(),
            executor);


        UNIT_ASSERT_VALUES_EQUAL(metric.Stats.size(), 2);
        UNIT_ASSERT_VALUES_EQUAL(metric.Stats.size(), metricHm.Stats.size());
        UNIT_ASSERT_DOUBLES_EQUAL(metric.Stats[1], metricHm.Stats[1], EPS);
        UNIT_ASSERT_DOUBLES_EQUAL(metric.Stats[0], metricHm.Stats[0], EPS);
    }

    Y_UNIT_TEST(SimpleTest) {
        TestMuAuc(
            {{0.5, 0.3, 0.05}, {0.45, 0.4, 0.45}, {0.05, 0.3, 0.5}},
            {0, 1, 2},
            {1, 1, 1}
        );
    }

    Y_UNIT_TEST(SimpleQueryTest) {
        TQueryInfo info(0, 3);

        TestQueryMuAuc(
            "QueryAUC:type=Mu",
            {{0.5, 0.3, 0.05}, {0.45, 0.4, 0.45}, {0.05, 0.3, 0.5}}, // predicts
            {0, 1, 2}, // targets
            {1, 1, 1}, // weights
            {info}
        );
    }

    Y_UNIT_TEST(RandomStressTest) {
        StressTest(5, 50, 50);
    }

    Y_UNIT_TEST(RandomStressTestTwoClasses) {
        StressTest(2, 100, 50);
    }

    Y_UNIT_TEST(RandomStressTestWeights) {
        StressTest(5, 50, 50, true);
    }

    Y_UNIT_TEST(RandomStressTestOwnMatrix) {
        StressTest(5, 50, 50, false, false);
    }

    Y_UNIT_TEST(RandomStressTestOwnMatrixAndWeights) {
        StressTest(5, 50, 50, true, false);
    }

    Y_UNIT_TEST(RandomStressTestOwnMatrixAndWeightsTwoClasses) {
        StressTest(2, 100, 50, true, false);
    }
}
