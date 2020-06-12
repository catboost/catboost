#include <catboost/private/libs/algo/monotonic_constraint_utils.h>

#include <library/cpp/testing/unittest/registar.h>
#include <util/generic/xrange.h>
#include <util/generic/algorithm.h>
#include <util/string/vector.h>
#include <util/random/shuffle.h>


Y_UNIT_TEST_SUITE(BuildOrderTest) {
    Y_UNIT_TEST(SmallTrivial) {
        const TVector<int> treeMonotonicConstraints{0};
        UNIT_ASSERT_EQUAL(
            BuildLinearOrderOnLeafsOfMonotonicSubtree(treeMonotonicConstraints, 0u),
            TVector<ui32>{0u}
        );
        UNIT_ASSERT_EQUAL(
            BuildLinearOrderOnLeafsOfMonotonicSubtree(treeMonotonicConstraints, 1u),
            TVector<ui32>{1u}
        );
    }

    Y_UNIT_TEST(LargeTrivial) {
        const size_t treeDepth = 8u;
        const TVector<int> treeMonotonicConstraints(treeDepth, 0);
        for (
            ui32 monotonicSubtreeIndex = 0;
            monotonicSubtreeIndex < (1u << treeDepth);
            ++monotonicSubtreeIndex
        ) {
            UNIT_ASSERT_EQUAL(
                BuildLinearOrderOnLeafsOfMonotonicSubtree(treeMonotonicConstraints, monotonicSubtreeIndex),
                TVector<ui32>{monotonicSubtreeIndex}
            );
        }
    }

    Y_UNIT_TEST(Simple) {
        {
            const TVector<int> treeMonotonicConstraints{1, -1};
            UNIT_ASSERT_EQUAL(
                BuildLinearOrderOnLeafsOfMonotonicSubtree(treeMonotonicConstraints, 0u),
                (TVector<ui32>{2u, 0u, 3u, 1u})
            );
        }
        {
            const TVector<int> treeMonotonicConstraints{-1, 0, 1};
            UNIT_ASSERT_EQUAL(
                BuildLinearOrderOnLeafsOfMonotonicSubtree(treeMonotonicConstraints, 0u),
                (TVector<ui32>{1u, 5u, 0u, 4u})
            );
            UNIT_ASSERT_EQUAL(
                BuildLinearOrderOnLeafsOfMonotonicSubtree(treeMonotonicConstraints, 1u),
                (TVector<ui32>{3u, 7u, 2u, 6u})
            );
        }
    }

    Y_UNIT_TEST(AllOnes) {
        const size_t treeDepth = 6u;
        const size_t leafCount = 1u << treeDepth;
        TVector<ui32> expectedOrder(leafCount, 0u);
        {
            const TVector<int> treeMonotonicConstraints(treeDepth, 1);
            for (ui32 i : xrange<ui32>(leafCount)) {
                expectedOrder[i] = ReverseBits(i, treeDepth);
            }
            UNIT_ASSERT_EQUAL(
                BuildLinearOrderOnLeafsOfMonotonicSubtree(treeMonotonicConstraints, 0u),
                expectedOrder
            );
        }
        {
            const TVector<int> treeMonotonicConstraints(treeDepth, -1);
            Reverse(expectedOrder.begin(), expectedOrder.end());
            UNIT_ASSERT_EQUAL(
                BuildLinearOrderOnLeafsOfMonotonicSubtree(treeMonotonicConstraints, 0u),
                expectedOrder
            );
        }
    }
}

void CheckIsotonicRegressionResult(
    const TVector<double>& values,
    const TVector<ui32>& order,
    const TVector<double>& expectedResult,
    TVector<double> weights = {}
) {
    TVector<double> result(values.size(), 0);
    if (weights.empty()) {
        weights.resize(values.size(), 1.0);
    }
    CalcOneDimensionalIsotonicRegression(values, weights, order, &result);
    UNIT_ASSERT_EQUAL_C(result, expectedResult, JoinVectorIntoString(result, ", "));
}

template <typename T = double, class TRandGen = TFastRng64>
TVector<T> GenerateRandomValues(size_t valueCount, TRandGen&& generator) {
    TVector<T> values;
    while (values.size() < valueCount) {
        values.push_back(generator.GenRandReal3());
    }
    return values;
}

double CalcAverage(const TVector<double>& values, const TVector<double>& weights) {
    double weightedSum = 0.0;
    double totalWeight = 0.0;
    for (auto i : xrange(values.size())) {
        weightedSum += values[i] * weights[i];
        totalWeight += weights[i];
        Y_ASSERT(weights[i] >= 0);
    }
    Y_ASSERT(totalWeight > 0);
    return weightedSum / totalWeight;
}

Y_UNIT_TEST_SUITE(OneDimensionalIsotonicRegressionTest) {
    Y_UNIT_TEST(Single) {
        CheckIsotonicRegressionResult({3.14}, {0u}, {3.14});
    }

    Y_UNIT_TEST(TwoValues) {
        CheckIsotonicRegressionResult({1, 2}, {0, 1}, {1, 2});
        CheckIsotonicRegressionResult({1, 2}, {1, 0}, {1.5, 1.5});
        CheckIsotonicRegressionResult({2, 1}, {0, 1}, {1.5, 1.5});
        CheckIsotonicRegressionResult({2, 1}, {1, 0}, {2, 1});
    }

    Y_UNIT_TEST(Reversed) {
        const size_t testSize = 10;
        TVector<double> values(testSize);
        TVector<ui32> order(testSize);
        for (auto i : xrange(testSize)) {
            values[i] = i;
            order[i] = i;
        }
        Reverse(order.begin(), order.end());
        const double mean = 0.5 * (testSize - 1);
        const TVector<double> expectedResult(testSize, mean);
        CheckIsotonicRegressionResult(values, order, expectedResult);
        Reverse(values.begin(), values.end());
        Reverse(order.begin(), order.end());
        CheckIsotonicRegressionResult(values, order, expectedResult);
    }

    Y_UNIT_TEST(Random) {
        const size_t testSize = 10;
        const size_t testCount = 100;
        TFastRng64 generator(0);
        TVector<ui32> order(testSize, 0);
        for (auto i : xrange(testSize)) {
            order[i] = i;
        }
        for (size_t testNum = 0; testNum < testCount; ++testNum) {
            const auto values = GenerateRandomValues(testSize, generator);
            const auto weights = GenerateRandomValues(testSize, generator);
            Shuffle(order.begin(), order.end(), generator);
            TVector<double> result(values.size(), 0.0);
            CalcOneDimensionalIsotonicRegression(values, weights, order, &result);
            UNIT_ASSERT(CheckMonotonicity(order, result));
            UNIT_ASSERT_DOUBLES_EQUAL(CalcAverage(values, weights), CalcAverage(result, weights), 1e-10);
        }
    }
}
