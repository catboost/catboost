#include <library/cpp/grid_creator/binarization.h>

#include <library/cpp/unittest/registar.h>
#include <library/cpp/unittest/gtest.h>

#include <util/generic/hash_set.h>
#include <util/generic/serialized_enum.h>
#include <util/generic/vector.h>
#include <util/generic/algorithm.h>
#include <util/generic/utility.h>
#include <util/generic/ymath.h>
#include <util/generic/xrange.h>
#include <util/random/fast.h>
#include <util/random/shuffle.h>
#include <util/system/compiler.h>


using namespace NSplitSelection;
using namespace NSplitSelection::NImpl;


static const TVector<size_t> MAX_BORDER_COUNT_VALUES = {1, 10, 128};
static const TVector<bool> NAN_IS_INFINITY_VALUES = {true, false};
static const TVector<EPenaltyType> PENALTY_TYPES = GetEnumAllValues<EPenaltyType>().Materialize();
static const TVector<EBorderSelectionType> BORDER_SELECTION_TYPES = (
    GetEnumAllValues<EBorderSelectionType>().Materialize());
static const THashSet<EBorderSelectionType> WEIGHTED_BORDER_SELECTION_TYPES = {
    EBorderSelectionType::MinEntropy, EBorderSelectionType::MaxLogSum,
    EBorderSelectionType::GreedyLogSum, EBorderSelectionType ::GreedyMinEntropy
};
static double EPSILON = 1e-8;

void TestAll(
    const TFeatureValues& features,
    const THashSet<float>& expectedBorders,
    const TVector<EBorderSelectionType>& borderSelectionTypes = BORDER_SELECTION_TYPES,
    const TVector<size_t>& borderCounts = MAX_BORDER_COUNT_VALUES,
    const TVector<bool>& nanIsInfinityValues = NAN_IS_INFINITY_VALUES
) {
    const TVector<float> weights(features.Values.size(), 1.0f);
    for (const auto& borderSelectionType : borderSelectionTypes) {
        for (const auto& nanIsInfinity : nanIsInfinityValues) {
            for (const auto& maxBorderCount : borderCounts) {
                TFeatureValues featuresCopy(features);
                TQuantization quantization = BestSplit(
                    std::move(featuresCopy), nanIsInfinity, maxBorderCount, borderSelectionType);
                THashSet<float> borders(quantization.Borders.begin(), quantization.Borders.end());
                UNIT_ASSERT_EQUAL_C(borders, expectedBorders,
                    GetEnumNames<EBorderSelectionType>().at(borderSelectionType));
                if (WEIGHTED_BORDER_SELECTION_TYPES.contains(borderSelectionType) && !features.DefaultValue) {
                    borders = BestWeightedSplit(TVector<float>(features.Values), weights, maxBorderCount, borderSelectionType,
                        nanIsInfinity);
                    UNIT_ASSERT_EQUAL_C(borders, expectedBorders,
                        GetEnumNames<EBorderSelectionType>().at(borderSelectionType));
                }
            }
        }
    }
}

template <EPenaltyType penaltyType>
double CalcScore(
    const THashSet<float>& borders,
    const TVector<float>& featureValues,
    const TVector<float>& weights
) {
    if (borders.empty()) {
        double totalWeight = 0;
        for (auto weight : weights) {
            totalWeight += weight;
        }
        return -Penalty<penaltyType>(totalWeight);
    }
    TVector<float> sortedBorders(borders.begin(), borders.end());
    Sort(sortedBorders.begin(), sortedBorders.end());
    TVector<float> binWeights(borders.size() + 1, 0.0);
    for (size_t i : xrange(featureValues.size())) {
        auto upperBorder = LowerBound(sortedBorders.begin(), sortedBorders.end(), featureValues[i]);
        size_t binIndex = upperBorder - sortedBorders.begin();
        binWeights.at(binIndex) += weights[i];
    }
    double result = 0;
    for (auto binWeight : binWeights) {
        Y_ASSERT(binWeight > 0);
        result -= Penalty<penaltyType>(binWeight);
    }
    return result;
}

void TestAllWeighted(
    const TVector<float> values,
    const TVector<float> weights,
    size_t maxBorderCount,
    const THashSet<float>& expectedBorders,
    const THashSet<EBorderSelectionType>& borderSelectionTypes = WEIGHTED_BORDER_SELECTION_TYPES
) {
    TVector<bool> isSortedValues = {false};
    if (IsSorted(values.begin(), values.end())) {
        isSortedValues.push_back(true);
    }
    for (auto borderSelectionType : borderSelectionTypes) {
        for (bool filterNans : {false, true}) {
            for (bool isSorted : isSortedValues) {
                const auto borders = BestWeightedSplit(
                    TVector<float>(values), weights, maxBorderCount, borderSelectionType, filterNans, isSorted);
                UNIT_ASSERT_EQUAL(borders, expectedBorders);
            }
        }
    }
}

template <class TRandGen = TFastRng64>
TVector<float> GenerateDistinctRandomValues(size_t valueCount, TRandGen&& generator) {
    THashSet<float> values;
    while (values.size() < valueCount) {
        values.insert(generator.GenRandReal3());
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

template <EPenaltyType penaltyType>
void TestScoreInequalities() {
    const THashSet<size_t> borderCountsWhenGreedyIsOptimal = {3, 7, 15}; // 2^n -1
    for (ui64 seed : xrange(10)) {
        for (size_t valueCount : {10, 100}) {
            TFastRng64 generator(seed);
            const auto values = GenerateDistinctRandomValues(valueCount, generator);
            const auto randomWeights = GenerateDistinctRandomValues(valueCount, generator);
            const TVector<float> usualWeights(valueCount, 1.0);
            for (bool useRandomWeights : {false, true}) {
                const auto& weights = useRandomWeights ? randomWeights : usualWeights;
                for (size_t maxBordersCount : {3, 7, 10, 15, 50}) {
                    size_t expectedBordersCount = Min(maxBordersCount, valueCount - 1);
                    auto valuesCopy = values;
                    const auto medianBorders = BestSplit(valuesCopy, maxBordersCount,
                        EBorderSelectionType::Median, false, false);
                    const auto optimalBorders = BestWeightedSplit<penaltyType>(TVector<float>(values), weights,
                        maxBordersCount, EOptimizationType::Exact, false, false);
                    const auto greedyBorders = BestWeightedSplit<penaltyType>(TVector<float>(values), weights,
                        maxBordersCount, EOptimizationType::Greedy, false, false);

                    UNIT_ASSERT_EQUAL(medianBorders.size(), expectedBordersCount);
                    UNIT_ASSERT_EQUAL(optimalBorders.size(), expectedBordersCount);
                    UNIT_ASSERT_EQUAL(greedyBorders.size(), expectedBordersCount);

                    const double medianScore = CalcScore<penaltyType>(medianBorders, values, weights);
                    const double optimalScore = CalcScore<penaltyType>(optimalBorders, values, weights);
                    const double greedyScore = CalcScore<penaltyType>(greedyBorders, values, weights);

                    UNIT_ASSERT_LE(medianScore, optimalScore + EPSILON);
                    UNIT_ASSERT_LE(greedyScore, optimalScore + EPSILON);

                    if (!useRandomWeights) {
                        UNIT_ASSERT_LE(optimalScore, medianScore + EPSILON);
                        if (borderCountsWhenGreedyIsOptimal.contains(expectedBordersCount)) {
                            UNIT_ASSERT_LE(optimalScore, greedyScore + EPSILON);
                        }
                    }
                }
            }
        }
    }
}

void AssertApproximateEquality(const TVector<float>& lhs, const TVector<float>& rhs) {
    UNIT_ASSERT_EQUAL(lhs.size(), rhs.size());
    auto rhsIter = rhs.begin();
    for (auto lhsValue : lhs) {
        EXPECT_DOUBLE_EQ(lhsValue, *rhsIter++);
    }
}

static TVector<ui32> getDivisionByBorders(const TVector<float>& features, const THashSet<float>& borders) {
    TVector<float> sortedFeatures = features;
    Sort(sortedFeatures.begin(), sortedFeatures.end());
    TVector<ui32> grounds;
    for (const auto& border : borders) {
        grounds.push_back(UpperBound(sortedFeatures.begin(), sortedFeatures.end(), border) - sortedFeatures.begin());
    }
    Sort(grounds.begin(), grounds.end());
    grounds.resize(Unique(grounds.begin(), grounds.end()) - grounds.begin());
    return grounds;
}

Y_UNIT_TEST_SUITE(ValuePreprocessingTests) {
    Y_UNIT_TEST(SmallCustomTest) {
        const TVector<float> featureValues = {-1.0f, -1.0f, 0.0f, 1.1f, 1.1f, 1.1f};
        const TVector<float> weights = {1, -1, 2, 1, 1, 1};
        const TVector<float> expectedValues = {-1.0f, 0.0f, 1.1f};
        const float multiplier = 5.0 / 6;
        const TVector<float> expectedWeights = {1 * multiplier, 2 * multiplier, 3 * multiplier};

        for (bool isSorted : {false, true}) {
            auto[resultValues, resultWeights] = GroupAndSortWeighedValues(
                TVector<float>(featureValues), TVector<float>(weights), false, isSorted);
            UNIT_ASSERT_EQUAL(resultValues, expectedValues);
            AssertApproximateEquality(resultWeights, expectedWeights);
        }
    }

    Y_UNIT_TEST(TestWeightFilter) {
        for (bool isSorted : {false, true}) {
            std::pair<TVector<float>, TVector<float>> expectedResult = {{}, {}};
            UNIT_ASSERT_EQUAL(expectedResult, GroupAndSortWeighedValues({}, {}, false, isSorted) );
            expectedResult = {{4}, {1.0f}};
            UNIT_ASSERT_EQUAL(expectedResult,
                GroupAndSortWeighedValues({1, 2, 3, 4}, {+0.0f, +0.0f, -0.0f, 1.0f}, false, isSorted));
            UNIT_ASSERT_EQUAL(expectedResult,
                GroupAndSortWeighedValues({1, 2, 3, 4}, {-1.0f, -1.0f, -1.0f, 1.0f}, false, isSorted));
        }
    }

    Y_UNIT_TEST(TestNanFiltering) {
        float nan = std::numeric_limits<float>::quiet_NaN();
        for (bool isSorted : {false, true}) {
            std::pair<TVector<float>, TVector<float>> expectedResult = {{}, {}};
            UNIT_ASSERT_EQUAL(expectedResult, GroupAndSortWeighedValues({}, {}, true, isSorted));
            UNIT_ASSERT_EQUAL(expectedResult, GroupAndSortValues(TFeatureValues({}, isSorted), true));
            UNIT_ASSERT_EQUAL(expectedResult, GroupAndSortWeighedValues({nan}, {1.0f}, true, isSorted));
            UNIT_ASSERT_EQUAL(expectedResult, GroupAndSortValues(TFeatureValues({nan}, isSorted), true));
            expectedResult = {{4}, {1.0f}};
            UNIT_ASSERT_EQUAL(expectedResult, GroupAndSortWeighedValues({1, nan, 3, 4},
                {+0.0f, 1.0f, -1.0f, 1.0f}, true, isSorted));
            UNIT_ASSERT_EQUAL(expectedResult, GroupAndSortValues(TFeatureValues({nan, nan, nan, 4}, isSorted), true));
        }
    }
}

Y_UNIT_TEST_SUITE(BinarizationTests) {
    Y_UNIT_TEST(TestEmpty) {
        TestAll(TFeatureValues(TVector<float>()), {});
    }

    Y_UNIT_TEST(TestSingleValue) {
        TestAll(TFeatureValues(TVector<float>{0.0}), {});
        TestAll(TFeatureValues(TVector<float>(5, 0.0)), {});
    }

    Y_UNIT_TEST(TestFullSplits) {
        const TVector<EBorderSelectionType> borderSelectionTypes = {
            EBorderSelectionType::MaxLogSum, EBorderSelectionType::MinEntropy,
            EBorderSelectionType::GreedyLogSum, EBorderSelectionType ::GreedyMinEntropy,
            EBorderSelectionType::Median
        };
        const TVector<size_t> possibleBorderCounts = {1, 2, 3, 5, 6, 15, 127};
        for (size_t valueCount : {0, 1, 2, 3, 10, 20, 100}) {
            TVector<float> values = GenerateDistinctRandomValues(valueCount, TFastRng64(valueCount));
            TVector<size_t> borderCounts;
            if (valueCount > 0) {
                borderCounts.push_back(valueCount - 1);
            }
            for (size_t borderCount : possibleBorderCounts) {
                if (borderCount >= valueCount) {
                    borderCounts.push_back(borderCount);
                }
            }
            auto allBorders = GetAllBorders(values);
            TestAll(TFeatureValues(std::move(values)), allBorders, borderSelectionTypes, borderCounts);
        }
    }

    Y_UNIT_TEST(TestWithDefaultValue) {
        constexpr size_t NON_DEFAULT_VALUE_COUNT = 12;
        constexpr int MAX_BORDERS_COUNT = 6;

        for (auto onlyNonNegativeValues : {false, true}) {
            TVector<float> nonDefaultValues;

            TFastRng64 randGen(NON_DEFAULT_VALUE_COUNT);

            for (auto i : xrange(NON_DEFAULT_VALUE_COUNT)) {
                Y_UNUSED(i);
                nonDefaultValues.push_back(
                    (onlyNonNegativeValues ? 0.f : -5.f) + 10.f * randGen.GenRandReal1()
                );
            }

            const TVector<float> defaultValues = {
                0.0f,
                nonDefaultValues.front(),
                0.5f * (nonDefaultValues.front() + nonDefaultValues.back())
            };

            for (auto defaultValue : defaultValues) {
                for (ui64 defaultValueCount : {1, 5, 30, 250}) {
                    TVector<float> denseValues = nonDefaultValues;

                    for (auto i : xrange(defaultValueCount)) {
                        Y_UNUSED(i);
                        denseValues.push_back(defaultValue);
                    }

                    Shuffle(denseValues.begin(), denseValues.end(), TReallyFastRng32(defaultValueCount));

                    for (auto borderSelectionType : BORDER_SELECTION_TYPES) {
                        const NSplitSelection::TQuantization quantizationFromDenseData
                            = NSplitSelection::BestSplit(
                                TFeatureValues(TVector<float>(denseValues)),
                                /*featureMayContainNans*/ false,
                                MAX_BORDERS_COUNT,
                                borderSelectionType
                            );

                        TFeatureValues featuresWithDefaultValue(
                            TVector<float>(nonDefaultValues),
                            /*valuesSorted*/ false,
                            TDefaultValue(defaultValue, defaultValueCount)
                        );

                        const NSplitSelection::TQuantization quantizationFromDataWithDefaultValue
                            = NSplitSelection::BestSplit(
                                std::move(featuresWithDefaultValue),
                                /*featureMayContainNans*/ false,
                                MAX_BORDERS_COUNT,
                                borderSelectionType
                            );

                        UNIT_ASSERT_EQUAL(quantizationFromDenseData, quantizationFromDataWithDefaultValue);
                    }
                }
            }
        }
    }

    Y_UNIT_TEST(TestDefaultQuantizedBinDoesNotAffectBorders) {
        const TVector<size_t> valueCounts = {10, 50, 100, 1000};
        const TVector<size_t> maxBorderCounts = {5, 10, 20, 254};
        for (auto valueCount : valueCounts) {
            TFastRng64 generator(0);
            const auto values = GenerateDistinctRandomValues(valueCount, generator);

            for (auto maxBorderCount : maxBorderCounts) {
                for (auto borderSelectionType : BORDER_SELECTION_TYPES) {
                    TQuantization quantizationWoQuantizedDefaultBin = BestSplit(
                        TFeatureValues(TVector<float>(values)),
                        /*featureValuesMayContainNans*/ false,
                        maxBorderCount,
                        borderSelectionType,
                        /*quantizedDefaultBinFraction*/ Nothing()
                    );

                    TQuantization quantizationWithQuantizedDefaultBin = BestSplit(
                        TFeatureValues(TVector<float>(values)),
                        /*featureValuesMayContainNans*/ false,
                        maxBorderCount,
                        borderSelectionType,
                        /*quantizedDefaultBinFraction*/ 0.7f
                    );

                    UNIT_ASSERT_VALUES_EQUAL(
                        quantizationWoQuantizedDefaultBin.Borders,
                        quantizationWithQuantizedDefaultBin.Borders
                    );
                }
            }
        }
    }

    Y_UNIT_TEST(TestDefaultQuantizedBin) {
        constexpr float quantizedDefaultBinFraction = 0.7f;

        const TVector<size_t> valueCounts = {12, 100, 1000};
        const TVector<size_t> maxBorderCounts = {5, 10, 20, 254};
        for (auto valueCount : valueCounts) {
            TFastRng64 generator(0);

            size_t nonDefaultValueCount
                = Max<size_t>(
                    3,
                    float(valueCount) * (1.0f - quantizedDefaultBinFraction) * 0.95f * generator.GenRandReal2()
                );

            TVector<float> values = GenerateDistinctRandomValues(nonDefaultValueCount, generator);
            const float defaultValue = values[generator.Uniform(values.size())];

            auto defaultValueCount = valueCount - nonDefaultValueCount;
            for (auto i : xrange(defaultValueCount)) {
                Y_UNUSED(i);
                values.push_back(defaultValue);
            }

            Shuffle(values.begin(), values.end(), generator);

            for (auto maxBorderCount : maxBorderCounts) {
                for (auto borderSelectionType : BORDER_SELECTION_TYPES) {
                    TQuantization quantization = BestSplit(
                        TFeatureValues(TVector<float>(values)),
                        /*featureValuesMayContainNans*/ false,
                        maxBorderCount,
                        borderSelectionType,
                        /*quantizedDefaultBinFraction*/ quantizedDefaultBinFraction
                    );

                    UNIT_ASSERT(quantization.DefaultQuantizedBin.Defined());

                    ui32 defaultBinIdx = quantization.DefaultQuantizedBin->Idx;
                    if (defaultBinIdx == 0) {
                        UNIT_ASSERT(defaultValue < quantization.Borders.front());
                    } else if (defaultBinIdx == quantization.Borders.size()) {
                        UNIT_ASSERT(defaultValue >= quantization.Borders.back());
                    } else {
                        UNIT_ASSERT(
                            (defaultValue >= quantization.Borders[defaultBinIdx - 1]) &&
                            (defaultValue < quantization.Borders[defaultBinIdx])
                        );
                    }

                    UNIT_ASSERT(quantization.DefaultQuantizedBin->Fraction >= quantizedDefaultBinFraction);
                }
            }
        }
    }
}

Y_UNIT_TEST_SUITE(WeightedBinarizationTests) {
    Y_UNIT_TEST(TestEmpty) {
        TestAllWeighted({}, {}, 1, {});
        TestAllWeighted({1, 2, 3, 4}, {+0.0f, +0.0f, -0.0f, 1.0f}, 1, {});
        TestAllWeighted({1, 2, 3, 4}, {-1.0f, -1.0f, -1.0f, 1.0f}, 1, {});
    }

    Y_UNIT_TEST(TestSmall) {
        {
            const TVector<float> featureValues = {1.0f, 2.0f};
            const auto expectedBorders = GetAllBorders(featureValues);
            TestAllWeighted(featureValues, {1.0f, 1.0f}, 1, expectedBorders);
            TestAllWeighted(featureValues, {10.0f, 0.1f}, 1, expectedBorders);
            TestAllWeighted(featureValues, {10.0f, std::numeric_limits<float>::min()}, 1, expectedBorders);
        }
        {
            const TVector<float> values = {0, 1, 2, 3, 4};
            const TVector<float> weights = {16.0f, 8.0f, 4.0f, 2.0f, 1.0f};
            for (size_t borderCount : {1, 2, 3}) {
                const auto expectedBorders = GetAllBorders(xrange<float>(borderCount + 1));
                TestAllWeighted(values, weights, borderCount, expectedBorders);
            }
        }
    }

    Y_UNIT_TEST(TestFullSplits) {
        const TVector<float> values = {1, 3, 5, 7, 9, 100};
        const TVector<size_t> borderCounts = {5, 6, 10, 128};
        const auto expectedBorders = GetAllBorders(values);
        for (auto borderCount : borderCounts) {
            {
                TVector weights(6, 1.0f);
                TestAllWeighted(values, weights, borderCount, expectedBorders);
            }
            TestAllWeighted(values, {1.0, 2.0, 4.0, 32.0, 16.0, 8.0}, borderCount, expectedBorders);
            {
                TVector weights(6, std::numeric_limits<float>::min());
                TestAllWeighted(values, weights, borderCount, expectedBorders);
            }
        }
    }

    Y_UNIT_TEST(TestConsistency) {
        const size_t test_size = 100;
        for (int seed : xrange(10)) {
            TFastRng64 generator(seed);
            TVector<float> values;
            for (size_t i = 0; i < test_size; ++i) {
                values.push_back(generator.Uniform(test_size));
            }
            const TVector<float> weights(100, 1);
            for (int bordersCount : {3, 10, 50, 256}) {
                for (auto borderSelectionType : WEIGHTED_BORDER_SELECTION_TYPES) {
                    auto values_copy = values;
                    const auto usualBorders = BestSplit(values_copy, bordersCount, borderSelectionType);
                    TestAllWeighted(values, weights, bordersCount, usualBorders, {borderSelectionType});
                }
            }
        }
    }

    Y_UNIT_TEST(TestScaleInvariance) {
        const TVector<float> weightMultipliers = {0.1, 3.1415, 1e6}; //, 1e6};
        const TVector<size_t> valueCounts = {10, 50, 100};
        const TVector<size_t> maxBorderCounts = {5, 10, 20};
        for (auto seed : xrange<size_t>(10)) {
            for (auto valueCount : valueCounts) {
                for (auto maxBorderCount : maxBorderCounts) {
                    TFastRng64 generator(seed);
                    const auto values = GenerateDistinctRandomValues(valueCount, generator);
                    const auto weights = GenerateDistinctRandomValues(valueCount, generator);
                    for (auto borderSelectionType : WEIGHTED_BORDER_SELECTION_TYPES) {
                        const auto expectedBorders = BestWeightedSplit(
                            TVector<float>(values), weights, maxBorderCount, borderSelectionType);
                        for (auto weightMultiplier : weightMultipliers) {
                            TVector<float> scaledWeights(weights);
                            for (auto &weight : scaledWeights) {
                                weight *= weightMultiplier;
                            }
                            const auto borders = BestWeightedSplit(
                                TVector<float>(values), scaledWeights, maxBorderCount, borderSelectionType);
                            UNIT_ASSERT_EQUAL(borders, expectedBorders);
                        }
                    }
                }
            }
        }
    }

    Y_UNIT_TEST(TestScoreInequalties) {
        TestScoreInequalities<EPenaltyType::MaxSumLog>();
        TestScoreInequalities<EPenaltyType::MinEntropy>();
        TestScoreInequalities<EPenaltyType::W2>();
    }
}

Y_UNIT_TEST_SUITE(InitialBordersTest) {
    Y_UNIT_TEST(RandomTest) {
        TVector<ui32> featuresCounts = {0, 1, 2, 5, 10, 50, 100};
        TVector<ui32> initialBordersCounts = {0, 1, 2, 5, 10, 50, 100};
        TVector<ui32> maxBorderCounts = {0, 1, 2, 5, 10, 50, 100};
        float maxCoordinate = 1e6;
        for (ui32 seed : xrange<ui32>(10)) {
            TFastRng64 gen(seed);
            for (ui32 featuresCount : featuresCounts) {
                for (ui32 initialBordersCount : initialBordersCounts) {
                    for (ui32 maxBorderCount : maxBorderCounts) {
                        TVector<float> features;
                        for (ui32 i = 0; i < featuresCount; ++i) {
                            features.push_back(gen.GenRandReal1() * maxCoordinate);
                        }
                        TVector<float> initialBorders;
                        for (ui32 i = 0; i < initialBordersCount; i++) {
                            initialBorders.push_back(gen.GenRandReal1() * maxCoordinate);
                        }
                        for (auto borderSelectionType : BORDER_SELECTION_TYPES) {
                            TVector<float> firstFeatures(features);
                            const auto usualBorders = BestSplit(firstFeatures, maxBorderCount, borderSelectionType, false, false);
                            TVector<float> secondFeatures(features);
                            const auto bordersWithInitial = BestSplit(secondFeatures, maxBorderCount, borderSelectionType, false, false, initialBorders);
                            UNIT_ASSERT_EQUAL(getDivisionByBorders(features, usualBorders), getDivisionByBorders(features, bordersWithInitial));
                        }
                    }
                }
            }
        }
    }
}
