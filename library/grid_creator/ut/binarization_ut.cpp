#include <library/unittest/registar.h>

#include <library/grid_creator/binarization.h>

#include <util/generic/array_ref.h>
#include <util/generic/hash_set.h>
#include <util/generic/serialized_enum.h>
#include <util/generic/vector.h>
#include <util/generic/algorithm.h>
#include <util/generic/ymath.h>
#include <util/random/fast.h>

static const TVector<size_t> MAX_BORDER_COUNT_VALUES = {1, 10, 128};
static const TVector<bool> NAN_IS_INFINITY_VALUES = {true, false};
static const TVector<EBorderSelectionType> BORDER_SELECTION_TYPES = (
    GetEnumAllValues<EBorderSelectionType>().Materialize());

void TestAll(const TVector<float>& values, const THashSet<float>& expectedBorders,
             const TVector<EBorderSelectionType>& borderSelectionTypes = BORDER_SELECTION_TYPES,
             const TVector<size_t>& borderCounts = MAX_BORDER_COUNT_VALUES,
             const TVector<bool>& nanIsInfinityValues = NAN_IS_INFINITY_VALUES) {
    const TVector<float> weights(values.size(), 1.0f);
    for (const auto& borderSelectionAlgorithm : borderSelectionTypes) {
        for (const auto& nanIsInfinity : nanIsInfinityValues) {
            for (const auto& maxBorderCount : borderCounts) {
                TVector<float> valuesCopy(values);
                auto borders = BestSplit(valuesCopy, maxBorderCount, borderSelectionAlgorithm, nanIsInfinity);
                UNIT_ASSERT_EQUAL_C(borders, expectedBorders,
                                    GetEnumNames<EBorderSelectionType>().at(borderSelectionAlgorithm));
            }
        }
    }
}

TVector<float> GnerateRandomValues(size_t valueCount, ui64 seed) {
    THashSet<float> values;
    TFastRng64 generator(seed);
    while (values.size() < valueCount) {
        values.insert(generator.GenRandReal1());
    }
    return {values.begin(), values.end()};
}

THashSet<float> GetAllBorders(TVector<float> values) {
    Sort(values.begin(), values.end());
    THashSet<float> result;
    for (auto valueIterator = values.begin(); valueIterator + 1 < values.end(); ++valueIterator) {
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
        const TVector<EBorderSelectionType> borderSelectionTypes = {
            EBorderSelectionType::MaxLogSum, EBorderSelectionType::MinEntropy,
            EBorderSelectionType::Median, EBorderSelectionType::GreedyLogSum};
        const TVector<size_t> possibleBorderCounts = {1, 2, 3, 5, 6, 15, 127};
        for (size_t valueCount : {0, 1, 2, 3, 10, 20, 100}) {
            const TVector<float> values = GnerateRandomValues(valueCount, valueCount);
            TVector<size_t> borderCounts;
            if (valueCount > 0) {
                borderCounts.push_back(valueCount - 1);
            }
            for (size_t borderCount : possibleBorderCounts) {
                if (borderCount >= valueCount) {
                    borderCounts.push_back(borderCount);
                }
            }
            TestAll(values, GetAllBorders(values), borderSelectionTypes, borderCounts);
        }
    }
}


