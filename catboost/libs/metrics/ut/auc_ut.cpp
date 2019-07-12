#include <catboost/libs/metrics/auc.h>
#include <catboost/libs/metrics/metric_holder.h>
#include <catboost/libs/helpers/cpu_random.h>

#include <library/unittest/registar.h>

#include <util/random/fast.h>
#include <util/random/shuffle.h>

constexpr double EPS = 1e-12;

static TVector<ui32> RandomSubset(ui32 size, ui32 num, TRandom& rnd) {
    TVector<ui32> result(num);
    for (size_t i = 0; i < num; ++i) {
        result[i] = rnd(size - num + 1);
    }
    Sort(result.begin(), result.end());
    for (size_t i = 0; i < num; ++i) {
        result[i] += i;
    }
    return result;
}

static TVector<ui32> RandomlyDivide(ui32 size, ui32 blocks, TRandom& rnd) {
    if (blocks == 1u) {
        return {size};
    }
    TVector<ui32> result = RandomSubset(size - 1, blocks - 1, rnd);
    result.push_back(size - result.back() - 1);
    for (ui32 i = result.size() - 2; i >= 1; i--) {
        result[i] -= result[i - 1];
    }
    result[0]++;
    return result;
}

static TVector<double> RandomVector(ui32 size, ui32 differentCount, TRandom& rnd, TFastRng<ui64>& rng) {
    TVector<double> elements(differentCount);
    for (ui32 i = 0; i < differentCount; ++i) {
        elements[i] = rng.GenRandReal3();
    }
    TVector<ui32> counts = RandomlyDivide(size, differentCount, rnd);
    TVector<double> result;
    for (ui32 i = 0; i < differentCount; ++i) {
        for (ui32 j = 0; j < counts[i]; ++j) {
            result.push_back(elements[i]);
        }
    }
    Shuffle(result.begin(), result.end(), rnd);
    return result;
}

static double MyAUC(
    const TVector<double>& prediction,
    const TVector<double>& target,
    const TVector<double>& weight
) {
    size_t size = prediction.size();
    double pairWeightSum = 0;
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = i + 1; j < size; ++j) {
            if (target[i] != target[j]) {
                pairWeightSum += weight[i] * weight[j];
            }
        }
    }
    if (pairWeightSum == 0) {
        return 0;
    }
    double wrongPairsSum = 0;
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            if (target[i] < target[j]) {
                if (prediction[i] > prediction[j]) {
                    wrongPairsSum += weight[i] * weight[j];
                } else if (prediction[i] == prediction[j]) {
                    wrongPairsSum += weight[i] * weight[j] / 2.0;
                }
            }
        }
    }
    return 1 - (wrongPairsSum / pairWeightSum);
}

Y_UNIT_TEST_SUITE(AUCMetricTests) {
    static void TestAuc(
        const TVector<double>& prediction,
        const TVector<double>& target,
        const TVector<double>& weight,
        const double eps
    ) {
        NPar::TLocalExecutor executor;
        executor.RunAdditionalThreads(31);
        TVector<NMetrics::TSample> samples;
        samples.reserve(prediction.size());
        for (ui32 i = 0; i < prediction.size(); ++i) {
            samples.emplace_back(target[i], prediction[i], weight[i]);
        }
        double score = CalcAUC(&samples, &executor);
        UNIT_ASSERT_DOUBLES_EQUAL(score, MyAUC(prediction, target, weight), eps);
    }

    static void TestAucRandom(
        ui32 size,
        ui32 differentPredictions,
        ui32 differentTargets,
        double eps
    ) {
        TFastRng<ui64> rng(239);
        TRandom rnd(239);
        TVector<double> prediction = RandomVector(size, differentPredictions, rnd, rng);
        TVector<double> target = RandomVector(size, differentTargets, rnd, rng);
        TVector<double> weight = RandomVector(size, size, rnd, rng);
        TestAuc(
            prediction,
            target,
            weight,
            eps
        );
    }

    static void TestAucEqualOrders(
        ui32 size,
        ui32 differentCount,
        bool isDifferentOrders,
        double eps
    ) {
        TFastRng<ui64> rng(239);
        TRandom rnd(239);
        TVector<double> prediction = RandomVector(size, differentCount, rnd, rng);
        Sort(prediction.begin(), prediction.end());
        Reverse(prediction.begin(), prediction.end());
        TVector<double> target = RandomVector(size, differentCount, rnd, rng);
        Sort(target.begin(), target.end());
        if (!isDifferentOrders) {
            Reverse(target.begin(), target.end());
        }
        TVector<double> weight = RandomVector(size, size, rnd, rng);
        TestAuc(
            prediction,
            target,
            weight,
            eps
        );
    }

    static void AucRandomStressTest(
        ui32 size,
        ui32 iterations,
        double eps
    ) {
        TRandom rnd(239);
        for (ui32 it = 0; it < iterations; ++it) {
            ui32 differentPredictions = rnd(size) + 1;
            ui32 differentTargets = rnd(size) + 1;
            TestAucRandom(size, differentPredictions, differentTargets, eps);
        }
    }

    static void AucEqualOrdersStressTest(
        ui32 size,
        double eps
    ) {
        for (ui32 i = 1; i <= size; i++) {
            TestAucEqualOrders(size, i, false, eps);
            TestAucEqualOrders(size, i, true, eps);
        }
    }

    Y_UNIT_TEST(SimpleTest) {
        TestAuc(
            {1, 1, 2, 3, 1, 4, 1, 2, 3, 1},
            {1, 2, 3, 4, 5, 5, 4, 3, 2, 1},
            {1, 0.5, 2, 1, 1, 0.75, 1, 1.5, 1.25, 3},
            EPS
        );
    }

    Y_UNIT_TEST(BigRandomTest) {
        TFastRng<ui64> rng(239);
        ui32 size = 2000;
        TVector<double> prediction(size), target(size), weight(size);
        for (ui32 i = 0; i < size; ++i) {
            prediction[i] = rng.GenRandReal3();
            target[i] = rng.GenRandReal3();
            weight[i] = rng.GenRandReal3();
        }
        TestAuc(
            prediction,
            target,
            weight,
            EPS
        );
    }

    Y_UNIT_TEST(EqualPredictionsAndTargetsTest) {
        TestAucRandom(2000, 1, 1, EPS);
    }

    Y_UNIT_TEST(EqualOrdersTest) {
        TestAucEqualOrders(2000, 2000, true, EPS);
    }

    Y_UNIT_TEST(DifferentOrdersTest) {
        TestAucEqualOrders(2000, 2000, false, EPS);
    }

    Y_UNIT_TEST(EqualOrdersTest_100_Test) {
        TestAucEqualOrders(2000, 100, true, EPS);
    }

    Y_UNIT_TEST(DifferentOrdersTest_100_Test) {
        TestAucEqualOrders(2000, 100, false, EPS);
    }

    Y_UNIT_TEST(EqualOrdersTest_10_Test) {
        TestAucEqualOrders(2000, 10, true, EPS);
    }

    Y_UNIT_TEST(DifferentOrdersTest_10_Test) {
        TestAucEqualOrders(2000, 10, false, EPS);
    }

    Y_UNIT_TEST(SmallNumberOfDifferent_1_10_Test) {
        TestAucRandom(2000, 1, 10, EPS);
    }

    Y_UNIT_TEST(SmallNumberOfDifferent_10_1_Test) {
        TestAucRandom(2000, 10, 1, EPS);
    }

    Y_UNIT_TEST(SmallNumberOfDifferent_2_2_Test) {
        TestAucRandom(2000, 2, 2, EPS);
    }

    Y_UNIT_TEST(SmallNumberOfDifferent_10_10_Test) {
        TestAucRandom(2000, 10, 10, EPS);
    }

    Y_UNIT_TEST(SmallNumberOfDifferent_100_100_Test) {
        TestAucRandom(2000, 100, 100, EPS);
    }

    Y_UNIT_TEST(SmallNumberOfDifferent_100_10_Test) {
        TestAucRandom(2000, 100, 10, EPS);
    }

    Y_UNIT_TEST(SmallNumberOfDifferent_10_100_Test) {
        TestAucRandom(2000, 10, 100, EPS);
    }

    Y_UNIT_TEST(SmallNumberOfDifferent_2000_10_Test) {
        TestAucRandom(2000, 2000, 10, EPS);
    }

    Y_UNIT_TEST(SmallNumberOfDifferent_10_2000_Test) {
        TestAucRandom(2000, 10, 2000, EPS);
    }

    Y_UNIT_TEST(SmallNumberOfDifferentStress_100_Test) {
        AucRandomStressTest(100, 500, EPS);
    }

    Y_UNIT_TEST(SmallNumberOfDifferentStress_1000_Test) {
        AucRandomStressTest(1000, 30, EPS);
    }

    Y_UNIT_TEST(AucEqualOrdersStressTest_200_Test) {
        AucEqualOrdersStressTest(200, EPS);
    }
}
