#include <library/unittest/registar.h>

#include <library/grid_creator/binarization.h>

#include <util/generic/array_ref.h>
#include <util/generic/hash_set.h>
#include <util/generic/serialized_enum.h>
#include <util/generic/vector.h>
#include <util/generic/algorithm.h>
#include <util/generic/ymath.h>


#include <random>
#include <algorithm>

static const TConstArrayRef<size_t> MAX_BORDER_COUNT_VALUES = {1, 10, 128};
static const TConstArrayRef<bool> NAN_IS_INFINITY_VALUES = {true, false};
static const TVector<EBorderSelectionType> BORDER_SELECTION_TYPES = (
    GetEnumAllValues<EBorderSelectionType>().Materialize());

void TestAll(const TVector<float>& values, const THashSet<float>& expected_borders,
                const TVector<EBorderSelectionType>& borderSelectionTypes = BORDER_SELECTION_TYPES,
                const TConstArrayRef<size_t>& borderCounts = MAX_BORDER_COUNT_VALUES,
                const TConstArrayRef<bool>& nanIsInfinityValues = NAN_IS_INFINITY_VALUES) {
    for (const auto& borderSelectionAlgorithm : borderSelectionTypes) {
        for (const auto& nanIsInfinity : nanIsInfinityValues) {
            for (const auto& maxBorderCount : borderCounts) {
                TVector<float> values_copy(values);
                auto borders = BestSplit(values_copy, maxBorderCount, borderSelectionAlgorithm, nanIsInfinity);
                UNIT_ASSERT_EQUAL_C(borders, expected_borders,
                    GetEnumNames<EBorderSelectionType>().at(borderSelectionAlgorithm));
            }
        }
    }
}

TVector<float> Arange(int start, int end) {
    TVector<float> values;
    auto curr = start;
    while (curr < end) {
        values.push_back(curr++);
    }
    return values;
}

TVector<float> Arange(int end) {
    return Arange(0, end);
}

THashSet<float> GetAllBorders(TVector<float> values) {
    Sort(values.begin(), values.end());
    THashSet<float> result;
    for (auto valueIterator = values.begin(); valueIterator + 1 != values.end(); ++valueIterator) {
        if (*valueIterator != *(valueIterator + 1)) {
            result.insert(0.5f * (*valueIterator + *(valueIterator + 1)));
        }
    }
    return result;
}

Y_UNIT_TEST_SUITE(BinarizationTests) {
    Y_UNIT_TEST(TestEmpty) {
        TVector<float> values;
        TestAll(values, {});
    }

    Y_UNIT_TEST(TestSingleValue) {
        TVector<EBorderSelectionType> borderSelectionAlgorithms = {
            EBorderSelectionType::GreedyLogSum, EBorderSelectionType::Median,
            EBorderSelectionType::Uniform, EBorderSelectionType::UniformAndQuantiles
        }; // test causes error for EBorderSelectionType::MinEntropy, EBorderSelectionType::MaxLogSum
        TestAll({0.0}, {}, borderSelectionAlgorithms);
        TVector<float> values(5, 0.0);
        TestAll(values, {}, borderSelectionAlgorithms);
    }

    Y_UNIT_TEST(TestFullSplits) {
        TVector<float> values = {1, 3, 5, 7, 9, 100};
        TestAll(values, GetAllBorders(values), {
            EBorderSelectionType::MaxLogSum, EBorderSelectionType::MinEntropy,
            EBorderSelectionType::GreedyLogSum, EBorderSelectionType::Median},
            {5, 6, 10, 128});
    }
}

Y_UNIT_TEST_SUITE(WeightedBinarizationTests) {
    Y_UNIT_TEST(TestEmpty) {
        THashSet<float> expected_borders;
        UNIT_ASSERT_EQUAL(expected_borders, BestWeightedSplit({}, {}, 1, false, true));
        UNIT_ASSERT_EQUAL(expected_borders,
            BestWeightedSplit({1, 2, 3, 4}, {+0.0f, +0.0f, -0.0f, 1.0f}, 1, true, false));
        UNIT_ASSERT_EQUAL(expected_borders,
                          BestWeightedSplit({1, 2, 3, 4}, {-1.0f, -1.0f, -1.0f, 1.0f}, 1, true, false));
    }

    Y_UNIT_TEST(TestSmall) {
        TVector<float> featureValues = {1.0f, 2.0f};
        {
            TVector<float> weights = {1.0f, 1.0f};
            auto borders = BestWeightedSplit(featureValues, weights, 1, true, true);
            UNIT_ASSERT_EQUAL(borders, GetAllBorders(featureValues));
        }
        {
            TVector<float> weights = {10.0f, 0.1f};
            auto borders = BestWeightedSplit(featureValues, weights, 1, true, true);
            UNIT_ASSERT_EQUAL(borders, GetAllBorders(featureValues));
        }
        {
            TVector<float> weights = {10.0f, std::numeric_limits<float>::min()};
            auto borders = BestWeightedSplit(featureValues, weights, 1, true, true);
            UNIT_ASSERT_EQUAL(borders, GetAllBorders(featureValues));
        }
    }

    Y_UNIT_TEST(TestFullSplits) {
        TVector<float> values = {1, 3, 5, 7, 9, 100};
        TVector<int> borderCounts = {5, 6, 10, 128};
        for (auto borderCount : borderCounts) {
            {
                TVector weights(6, 1.0f);
                auto borders = BestWeightedSplit(values, weights, borderCount, true, true);
                UNIT_ASSERT_EQUAL(borders, GetAllBorders(values));
            }
            {
                auto borders = BestWeightedSplit(
                    values, {1.0, 2.0, 4.0, 32.0, 16.0, 8.0}, borderCount, true, true);
                UNIT_ASSERT_EQUAL(borders, GetAllBorders(values));
            }
            {
                TVector weights(6, std::numeric_limits<float>::min());
                auto borders = BestWeightedSplit(values, weights, borderCount, true, true);
                UNIT_ASSERT_EQUAL(borders, GetAllBorders(values));
            }
        }
    }

    Y_UNIT_TEST(TestConsistency) {
        const size_t test_size = 100;
        for (int seed : Arange(10)) {
            std::mt19937 generator(seed);
            std::uniform_int_distribution<int> rand_int(0, test_size);
            TVector<float> values;
            for (size_t i = 0; i < test_size; ++i) {
                values.push_back(rand_int(generator));
            }
            const TVector<float> weights(100, 1);
            for (int bordersCount : {3, 10, 50, 256}) {
                auto values_copy = values;
                const auto usual_borders = BestSplit(values_copy, bordersCount,
                                                     EBorderSelectionType::GreedyLogSum);
                const auto weighthed_borders = BestWeightedSplit(values, weights, bordersCount);
                UNIT_ASSERT_EQUAL(usual_borders, weighthed_borders);
            }
        }
    }

    Y_UNIT_TEST(TestRuntime) {
        const TVector<float> featureValues = Arange(-50, 50);
        const size_t test_size = featureValues.size();
        TVector<float> weights;
        const float weightLogStep = log(test_size) / test_size;
        for (int val = 0; val < static_cast<int>(test_size); ++val) {
            weights.push_back(sqrt(test_size) * exp(- val * weightLogStep));
        }

        for (int seed : Arange(10)) {
            std::mt19937 generator(seed);
            std::shuffle(weights.begin(), weights.end(), generator);
            for (size_t bordersCount : {3, 10, 50, 256}) {
                UNIT_ASSERT_EQUAL(
                    BestWeightedSplit(featureValues, weights, bordersCount, true, false).size(),
                    Min(bordersCount, test_size - 1));
                UNIT_ASSERT_EQUAL(
                    BestWeightedSplit(featureValues, weights, bordersCount, true, true).size(),
                    Min(bordersCount, test_size - 1));
            }
        }
    }
}
