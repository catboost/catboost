#include <catboost/libs/metrics/auc.h>
#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/metrics/metric_holder.h>
#include <catboost/libs/helpers/cpu_random.h>
#include <catboost/private/libs/data_types/query.h>

#include <library/cpp/testing/unittest/registar.h>

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

using NMetrics::TSample;

static double MergeAndCountInversions(TVector<TSample>* samples, TVector<TSample>* aux, ui32 lo, ui32 hi, ui32 mid) {
    double result = 0;
    ui32 left = lo;
    ui32 right = mid;
    auto& input = *samples;
    auto& output = *aux;
    ui32 outputIndex = lo;
    double accumulatedWeight = 0;
    while (outputIndex < hi) {
        if (left == mid || right < hi && input[right].Target < input[left].Target) {
            accumulatedWeight += input[right].Weight;
            output[outputIndex] = input[right];
            ++outputIndex;
            ++right;
        } else {
            result += input[left].Weight * accumulatedWeight;
            output[outputIndex] = input[left];
            ++outputIndex;
            ++left;
        }
    }
    return result;
}

static double SortAndCountInversions(TVector<TSample>* samples, TVector<TSample>* aux, ui32 lo, ui32 hi) {
    if (lo + 1 >= hi) return 0;
    ui32 mid = lo + (hi - lo) / 2;
    auto leftCount = SortAndCountInversions(samples, aux, lo, mid);
    auto rightCount = SortAndCountInversions(samples, aux, mid, hi);
    auto mergeCount = MergeAndCountInversions(samples, aux, lo, hi, mid);
    std::copy(aux->begin() + lo, aux->begin() + hi, samples->begin() + lo);
    return leftCount + rightCount + mergeCount;
}

static double CalcAucSingleThread(TVector<TSample>* samples, double* outWeightSum = nullptr, double* outPairWeightSum = nullptr) {
    double weightSum = 0;
    double pairWeightSum = 0;
    StableSort(samples->begin(), samples->end(), [](const TSample& left, const TSample& right) {
        return left.Target < right.Target;
    });
    double accumulatedWeight = 0;
    for (ui32 i = 0; i < samples->size(); ++i) {
        auto& sample = (*samples)[i];
        if (i > 0 && (*samples)[i - 1].Target != sample.Target) {
            accumulatedWeight = weightSum;
        }
        weightSum += sample.Weight;
        pairWeightSum += accumulatedWeight * sample.Weight;
    }
    if (outWeightSum != nullptr) {
        *outWeightSum = weightSum;
    }
    if (outPairWeightSum != nullptr) {
        *outPairWeightSum = pairWeightSum;
    }
    if (pairWeightSum == 0) {
        return 0;
    }
    TVector<TSample> aux(samples->begin(), samples->end());
    StableSort(samples->begin(), samples->end(), [](const TSample& left, const TSample& right) {
        return left.Prediction < right.Prediction ||
               left.Prediction == right.Prediction && left.Target < right.Target;
    });
    auto optimisticAUC = 1 - SortAndCountInversions(samples, &aux, 0, samples->size()) / pairWeightSum;
    StableSort(samples->begin(), samples->end(), [](const TSample& left, const TSample& right) {
        return left.Prediction < right.Prediction ||
               left.Prediction == right.Prediction && left.Target > right.Target;
    });
    auto pessimisticAUC = 1 - SortAndCountInversions(samples, &aux, 0, samples->size()) / pairWeightSum;
    return (optimisticAUC + pessimisticAUC) / 2.0;
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
        double scoreParallel = CalcAUC(&samples, nullptr, nullptr, &executor);
        Shuffle(samples.begin(), samples.end());
        double score = CalcAUC(&samples);
        Shuffle(samples.begin(), samples.end());
        UNIT_ASSERT_DOUBLES_EQUAL(scoreParallel, MyAUC(prediction, target, weight), eps);
        Shuffle(samples.begin(), samples.end());
        UNIT_ASSERT_DOUBLES_EQUAL(scoreParallel, CalcAucSingleThread(&samples), eps);
        UNIT_ASSERT_DOUBLES_EQUAL(scoreParallel, score, eps);
    }

    static void TestQueryAuc(
        TString description,
        TVector<TVector<double>> prediction,
        const TVector<float>& targets,
        const TVector<float>& weights,
        const TVector<TQueryInfo> queryInfos,
        const double eps)
    {
        NPar::TLocalExecutor executor;
        executor.RunAdditionalThreads(31);

        TMetricHolder metricHm(2);

        for (const auto& info: queryInfos) {
            TVector<NMetrics::TSample> samples;
            auto startIdx = info.Begin;
            auto querySize = info.End - info.Begin;

            for (ui32 i = 0; i < querySize; ++i) {
                samples.emplace_back(targets[startIdx + i], prediction[0][startIdx + i], weights[startIdx + i]);
            }

            metricHm.Stats[0] += CalcAUC(&samples);
            metricHm.Stats[1] += 1;
        }

        const auto queryAUC = std::move(CreateMetricsFromDescription({description}, 1).front());
        const auto metric = dynamic_cast<const ISingleTargetEval*>(queryAUC.Get())->Eval(
            prediction,
            targets,
            weights,
            queryInfos,
            0,
            queryInfos.size(),
            executor);

        UNIT_ASSERT_VALUES_EQUAL(metricHm.Stats.size(), 2);
        UNIT_ASSERT_VALUES_EQUAL(metricHm.Stats.size(), metric.Stats.size());
        UNIT_ASSERT_DOUBLES_EQUAL(metricHm.Stats[1], metric.Stats[1], eps);
        UNIT_ASSERT_DOUBLES_EQUAL(metricHm.Stats[0], metric.Stats[0], eps);
    }

    static void TestBinClassAuc(
        const TVector<double>& prediction,
        const TVector<bool>& target,
        const TVector<double>& weight,
        const double eps
    ) {
        NPar::TLocalExecutor executor;
        executor.RunAdditionalThreads(31);
        TVector<NMetrics::TSample> samples;
        samples.reserve(target.size());
        TVector<double> doubleTarget;
        doubleTarget.reserve(target.size());
        TVector<NMetrics::TBinClassSample> positiveSamples, negativeSamples;
        for (ui32 i = 0; i < prediction.size(); ++i) {
            doubleTarget.emplace_back((double)target[i]);
            samples.emplace_back((double)target[i], prediction[i], weight[i]);
            if (target[i]) {
                positiveSamples.emplace_back(prediction[i], weight[i]);
            } else {
                negativeSamples.emplace_back(prediction[i], weight[i]);
            }
        }
        double scoreParallel = CalcBinClassAuc(&positiveSamples, &negativeSamples, &executor);
        Shuffle(positiveSamples.begin(), positiveSamples.end());
        Shuffle(negativeSamples.begin(), negativeSamples.end());
        double scoreOneThread = CalcBinClassAuc(&positiveSamples, &negativeSamples);
        Shuffle(positiveSamples.begin(), positiveSamples.end());
        Shuffle(negativeSamples.begin(), negativeSamples.end());
        double scoreManyThreads = CalcBinClassAuc(&positiveSamples, &negativeSamples, 32);
        double usualAucScore = CalcAUC(&samples, nullptr, nullptr, &executor);
        Shuffle(samples.begin(), samples.end());
        double usualAucScoreOneThread = CalcAUC(&samples);
        double naiveRealizationScore = MyAUC(prediction, doubleTarget, weight);
        Shuffle(samples.begin(), samples.end());
        double singleThreadScore = CalcAucSingleThread(&samples);
        UNIT_ASSERT_DOUBLES_EQUAL(scoreParallel, scoreOneThread, eps);
        UNIT_ASSERT_DOUBLES_EQUAL(scoreParallel, scoreManyThreads, eps);
        UNIT_ASSERT_DOUBLES_EQUAL(scoreParallel, usualAucScore, eps);
        UNIT_ASSERT_DOUBLES_EQUAL(scoreParallel, usualAucScoreOneThread, eps);
        UNIT_ASSERT_DOUBLES_EQUAL(scoreParallel, naiveRealizationScore, eps);
        UNIT_ASSERT_DOUBLES_EQUAL(scoreParallel, singleThreadScore, eps);
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

    static void TestBinClassAucRandom(
        ui32 size,
        ui32 differentPredictions,
        bool isEqualTargets,
        double eps
    ) {
        TFastRng<ui64> rng(239);
        TRandom rnd(239);
        TVector<double> prediction = RandomVector(size, differentPredictions, rnd, rng);
        TVector<double> weight = RandomVector(size, size, rnd, rng);
        TVector<bool> target;
        target.reserve(size);
        bool defaultValue = rnd(2);
        for (ui32 i = 0; i < size; ++i) {
            target.emplace_back(isEqualTargets ? defaultValue : rnd(2));
        }
        TestBinClassAuc(
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
        StableSort(prediction.begin(), prediction.end());
        Reverse(prediction.begin(), prediction.end());
        TVector<double> target = RandomVector(size, differentCount, rnd, rng);
        StableSort(target.begin(), target.end());
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

    Y_UNIT_TEST(SimpleQueryAucTest) {
        {

            TQueryInfo info(0, 10);
            TestQueryAuc(
                "QueryAUC:type=Ranking;use_weights=true",
                {{1, 1, 2, 3, 1, 4, 1, 2, 3, 1}}, // predicts
                {1, 2, 3, 4, 5, 5, 4, 3, 2, 1}, // targets
                {1, 0.5, 2, 1, 1, 0.75, 1, 1.5, 1.25, 3}, // weights
                {info},
                EPS
            );
        }

        {
            TQueryInfo info(0, 3);
            TestQueryAuc(
                "QueryAUC:type=Ranking;use_weights=true",
                {{3, 2, 1}}, // predicts
                {0, 1, 0}, // targets
                {1, 1, 1}, // weights
                {info},
                EPS
            );
        }
            TQueryInfo info1(0, 10);
            TQueryInfo info2(10, 13);
            TestQueryAuc(
                "QueryAUC:type=Ranking;use_weights=true",
                {{1, 1, 2, 3, 1, 4, 1, 2, 3, 1, 3, 2, 1}}, // predicts
                {1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 0, 1, 0}, // targets
                {1, 0.5, 2, 1, 1, 0.75, 1, 1.5, 1.25, 3, 1, 1, 1}, // weights
                {info1, info2},
                EPS
            );
        {

        }
    }

    Y_UNIT_TEST(ParallelizationOnTest) {
        TVector<double> approx{3, 2, 1};
        TVector<double> target{0, 1, 0};
        TVector<double> weight{1, 1, 1};

        NPar::TLocalExecutor executor;
        executor.RunAdditionalThreads(31);

        TVector<NMetrics::TSample> samples;
        for (ui32 i = 0; i < target.size(); ++i) {
            samples.emplace_back(target[i], approx[i], weight[i]);
        }

        double score = CalcAUC(&samples, nullptr, nullptr, &executor);
        UNIT_ASSERT_DOUBLES_EQUAL(score, 0.5, 1e-6);
    }

    Y_UNIT_TEST(ParallelizationOffTest) {
        TVector<double> approx{3, 2, 1};
        TVector<double> target{0, 1, 0};
        TVector<double> weight{1, 1, 1};

        TVector<NMetrics::TSample> samples;
        for (ui32 i = 0; i < target.size(); ++i) {
            samples.emplace_back(target[i], approx[i], weight[i]);
        }

        double score = CalcAUC(&samples);
        UNIT_ASSERT_DOUBLES_EQUAL(score, 0.5, 1e-6);
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

    Y_UNIT_TEST(FedorAucTest) {
        TVector<double> approx{3, 2, 1};
        TVector<double> target{0, 1, 0};
        TVector<double> weight{1, 1, 1};
        TestAuc(approx, target, weight, EPS);
    }

    Y_UNIT_TEST(BinClassAucEqualTargetsTest) {
        TRandom rnd(239);
        for (ui32 iter = 0; iter < 20; ++iter) {
            TestBinClassAucRandom(1000, rnd(1000) + 1, true, EPS);
        }
    }

    Y_UNIT_TEST(BinClassAucTest) {
        ui32 size = 239;
        for (ui32 i = 1; i <= size; ++i) {
            TestBinClassAucRandom(size, i, false, EPS);
        }
    }

    Y_UNIT_TEST(BigBinClassAucTest) {
        TestBinClassAucRandom(2000, 10, false, EPS);
        TestBinClassAucRandom(2000, 239, false, EPS);
        TestBinClassAucRandom(2000, 1000, false, EPS);
        TestBinClassAucRandom(2000, 2000, false, EPS);
    }
}
