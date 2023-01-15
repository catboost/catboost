#include <library/statistics/statistics.h>

#include <library/cpp/accurate_accumulate/accurate_accumulate.h>
#include <library/cpp/unittest/registar.h>

#include <util/generic/vector.h>
#include <util/generic/ymath.h>

extern double normal_0_1[];
extern double normal_1_5[];
extern double normal_1_5_test[];

extern double exp_1[];
extern double exp_5[];

// generated using numpy.random.normal(loc=0, scale=1, size=1000)
#include "normal_0_1"

// generated using numpy.random.normal(loc=1, scale=5, size=1000)
#include "normal_1_5"
#include "normal_1_5_test"

// generated using numpy.random.exponential(scale=1, size=1000)
#include "exp_1"

// generated using numpy.random.exponential(scale=5, size=1000)
#include "exp_5"

// real data web.web_ctr15
#include "small_obj_cont.inc"
#include "small_obj_test.inc"

#define SIZE_OF(distribution) (sizeof(distribution) / sizeof(distribution[0]))
#define RANGE(distribution) distribution, distribution + SIZE_OF(distribution)
#define PART_RANGE(distribution, begin, length) distribution + begin, distribution + begin + length
#define POINTER_AND_LENGTH(distribution) distribution, SIZE_OF(distribution)

Y_UNIT_TEST_SUITE(TStatisticsTest) {
    Y_UNIT_TEST(AverageTest) {
        UNIT_ASSERT_DOUBLES_EQUAL(NStatistics::Average(RANGE(normal_0_1)), 0, 0.1);
        UNIT_ASSERT_DOUBLES_EQUAL(NStatistics::Average(RANGE(normal_1_5)), 1, 0.1);
        UNIT_ASSERT_DOUBLES_EQUAL(NStatistics::Average(RANGE(normal_1_5_test)), 1, 0.1);
        UNIT_ASSERT_DOUBLES_EQUAL(NStatistics::Average(RANGE(exp_1)), 1, 0.1);
        UNIT_ASSERT_DOUBLES_EQUAL(NStatistics::Average(RANGE(exp_5)), 5, 0.2);
    }

    Y_UNIT_TEST(MeanTest) {
        UNIT_ASSERT_DOUBLES_EQUAL(NStatistics::MeanAndStandardDeviation(RANGE(normal_0_1)).Mean, 0, 0.1);
        UNIT_ASSERT_DOUBLES_EQUAL(NStatistics::MeanAndStandardDeviation(RANGE(normal_1_5)).Mean, 1, 0.1);
        UNIT_ASSERT_DOUBLES_EQUAL(NStatistics::MeanAndStandardDeviation(RANGE(normal_1_5_test)).Mean, 1, 0.1);
        UNIT_ASSERT_DOUBLES_EQUAL(NStatistics::MeanAndStandardDeviation(RANGE(exp_1)).Mean, 1, 0.1);
        UNIT_ASSERT_DOUBLES_EQUAL(NStatistics::MeanAndStandardDeviation(RANGE(exp_5)).Mean, 5, 0.1);
    }

    Y_UNIT_TEST(StandardDeviationTest) {
        UNIT_ASSERT_DOUBLES_EQUAL(NStatistics::MeanAndStandardDeviation(RANGE(normal_0_1)).Std, 1, 0.1);
        UNIT_ASSERT_DOUBLES_EQUAL(NStatistics::MeanAndStandardDeviation(RANGE(normal_1_5)).Std, 5, 0.1);
        UNIT_ASSERT_DOUBLES_EQUAL(NStatistics::MeanAndStandardDeviation(RANGE(normal_1_5_test)).Std, 5, 0.1);
        UNIT_ASSERT_DOUBLES_EQUAL(NStatistics::MeanAndStandardDeviation(RANGE(exp_1)).Std, 1, 0.1);
        UNIT_ASSERT_DOUBLES_EQUAL(NStatistics::MeanAndStandardDeviation(RANGE(exp_5)).Std, 5, 0.2);
    }

    template <typename InputIterator>
    NStatistics::TMeanStd<typename std::iterator_traits<InputIterator>::value_type>
    KahanMeanAndStandardDeviation(InputIterator begin, InputIterator end) {
        using TValueType = typename std::iterator_traits<InputIterator>::value_type;
        NStatistics::TStatisticsCalculator<TKahanAccumulator<TValueType>> calculator;
        for (InputIterator iter = begin; iter != end; ++iter)
            calculator.Push(*iter);

        return {calculator.Mean(), calculator.StandardDeviation()};
    }

    Y_UNIT_TEST(KahanMeanTest) {
        UNIT_ASSERT_DOUBLES_EQUAL(KahanMeanAndStandardDeviation(RANGE(normal_0_1)).Mean, 0, 0.1);
        UNIT_ASSERT_DOUBLES_EQUAL(KahanMeanAndStandardDeviation(RANGE(normal_1_5)).Mean, 1, 0.1);
        UNIT_ASSERT_DOUBLES_EQUAL(KahanMeanAndStandardDeviation(RANGE(normal_1_5_test)).Mean, 1, 0.1);
        UNIT_ASSERT_DOUBLES_EQUAL(KahanMeanAndStandardDeviation(RANGE(exp_1)).Mean, 1, 0.1);
        UNIT_ASSERT_DOUBLES_EQUAL(KahanMeanAndStandardDeviation(RANGE(exp_5)).Mean, 5, 0.1);
    }

    Y_UNIT_TEST(KahanStandardDeviationTest) {
        UNIT_ASSERT_DOUBLES_EQUAL(KahanMeanAndStandardDeviation(RANGE(normal_0_1)).Std, 1, 0.1);
        UNIT_ASSERT_DOUBLES_EQUAL(KahanMeanAndStandardDeviation(RANGE(normal_1_5)).Std, 5, 0.1);
        UNIT_ASSERT_DOUBLES_EQUAL(KahanMeanAndStandardDeviation(RANGE(normal_1_5_test)).Std, 5, 0.1);
        UNIT_ASSERT_DOUBLES_EQUAL(KahanMeanAndStandardDeviation(RANGE(exp_1)).Std, 1, 0.1);
        UNIT_ASSERT_DOUBLES_EQUAL(KahanMeanAndStandardDeviation(RANGE(exp_5)).Std, 5, 0.2);
    }

    NStatistics::TMeanStd<double> SplitAndCalcualteMeanAndStandardDeviation(const double* a,
                                                                            const size_t length,
                                                                            const size_t parts) {
        const size_t partLength = length / parts;
        NStatistics::TStatisticsCalculator<double> calculatorTotal;
        size_t partStart = 0;
        size_t partStop = partLength;
        for (size_t partIndex = 0; partIndex < parts; ++partIndex) {
            NStatistics::TStatisticsCalculator<double> calculator;
            for (size_t index = partStart; index < partStop; ++index)
                calculator.Push(*(a + index));

            calculatorTotal = calculatorTotal + calculator;
            partStart = partStop;
            partStop += partLength;
            if (partIndex + 1 == parts)
                partStop = length;
        }

        return {calculatorTotal.Mean(), calculatorTotal.StandardDeviation()};
    }

    Y_UNIT_TEST(SummatorMeanTest) {
        for (size_t parts = 2; parts < 5; ++parts) {
            UNIT_ASSERT_DOUBLES_EQUAL(SplitAndCalcualteMeanAndStandardDeviation(POINTER_AND_LENGTH(normal_0_1), parts).Mean, 0, 0.1);
            UNIT_ASSERT_DOUBLES_EQUAL(SplitAndCalcualteMeanAndStandardDeviation(POINTER_AND_LENGTH(normal_1_5), parts).Mean, 1, 0.1);
            UNIT_ASSERT_DOUBLES_EQUAL(SplitAndCalcualteMeanAndStandardDeviation(POINTER_AND_LENGTH(normal_1_5_test), parts).Mean, 1, 0.1);
            UNIT_ASSERT_DOUBLES_EQUAL(SplitAndCalcualteMeanAndStandardDeviation(POINTER_AND_LENGTH(exp_1), parts).Mean, 1, 0.1);
            UNIT_ASSERT_DOUBLES_EQUAL(SplitAndCalcualteMeanAndStandardDeviation(POINTER_AND_LENGTH(exp_5), parts).Mean, 5, 0.1);
        }
    }

    Y_UNIT_TEST(SummatorStandardDeviationTest) {
        for (size_t parts = 2; parts < 5; ++parts) {
            UNIT_ASSERT_DOUBLES_EQUAL(SplitAndCalcualteMeanAndStandardDeviation(POINTER_AND_LENGTH(normal_0_1), parts).Std, 1, 0.1);
            UNIT_ASSERT_DOUBLES_EQUAL(SplitAndCalcualteMeanAndStandardDeviation(POINTER_AND_LENGTH(normal_1_5), parts).Std, 5, 0.1);
            UNIT_ASSERT_DOUBLES_EQUAL(SplitAndCalcualteMeanAndStandardDeviation(POINTER_AND_LENGTH(normal_1_5_test), parts).Std, 5, 0.1);
            UNIT_ASSERT_DOUBLES_EQUAL(SplitAndCalcualteMeanAndStandardDeviation(POINTER_AND_LENGTH(exp_1), parts).Std, 1, 0.1);
            UNIT_ASSERT_DOUBLES_EQUAL(SplitAndCalcualteMeanAndStandardDeviation(POINTER_AND_LENGTH(exp_5), parts).Std, 5, 0.2);
        }
    }

    double MannWhitneyPartTest(const size_t length, const double* a, const double* b = normal_1_5_test) {
        return NStatistics::MannWhitney(PART_RANGE(a, 0, length), PART_RANGE(b, 0, length));
    }

    double WilcoxonPartTest(const size_t length, const double* a, const double* b = normal_1_5_test) {
        TVector<float> v(a, a + length);
        for (size_t i = 0; i < length; ++i) {
            v[i] -= b[i];
        }

        const double wilcoxonTwoSamples = NStatistics::Wilcoxon(PART_RANGE(a, 0, length), PART_RANGE(b, 0, length));
        const double wilcoxonOneSamples = NStatistics::Wilcoxon(v.begin(), v.end());
        UNIT_ASSERT_DOUBLES_EQUAL(wilcoxonTwoSamples, wilcoxonOneSamples, 0.001);
        return wilcoxonTwoSamples;
    }

    int MannWhitneySignPartTest(const size_t lengthA, const size_t lengthB, const double* a, const double* b = normal_1_5_test) {
        return NStatistics::MannWhitneyWithSign(PART_RANGE(a, 0, lengthA), PART_RANGE(b, 0, lengthB)).Sign;
    }

    double MannWhitneyWithSignPartTest(const size_t length, const double* a, const double* b = normal_1_5_test) {
        return NStatistics::MannWhitneyWithSign(PART_RANGE(a, 0, length), PART_RANGE(b, 0, length)).PValue;
    }

    int WilcoxonSignPartTest(const size_t length, const double* a, const double* b = normal_1_5_test) {
        TVector<float> v(a, a + length);
        for (size_t i = 0; i < length; ++i) {
            v[i] -= b[i];
        }

        const int wilcoxonTwoSamplesSign = NStatistics::WilcoxonWithSign(PART_RANGE(a, 0, length), PART_RANGE(b, 0, length)).Sign;
        const int wilcoxonOneSamplesSign = NStatistics::WilcoxonWithSign(v.begin(), v.end()).Sign;
        UNIT_ASSERT_EQUAL(wilcoxonTwoSamplesSign, wilcoxonOneSamplesSign);

        return wilcoxonTwoSamplesSign;
    }

    double WilcoxonWithSignPartTest(const size_t length, const double* a, const double* b = normal_1_5_test) {
        TVector<float> v(a, a + length);
        for (size_t i = 0; i < length; ++i) {
            v[i] -= b[i];
        }

        const double wilcoxonTwoSamples = NStatistics::WilcoxonWithSign(PART_RANGE(a, 0, length), PART_RANGE(b, 0, length)).PValue;
        const double wilcoxonOneSamples = NStatistics::WilcoxonWithSign(v.begin(), v.end()).PValue;
        UNIT_ASSERT_DOUBLES_EQUAL(wilcoxonTwoSamples, wilcoxonOneSamples, 0.001);
        return wilcoxonTwoSamples;
    }

    double OneSampleTPartTest(const size_t length, const double* a, double expectedMean, const bool isTailed = false, const bool isLeftTailed = true) {
        return NStatistics::TTest(PART_RANGE(a, 0, length), expectedMean, isTailed, isLeftTailed);
    }

    double TwoSamplesTPartTest(const size_t aLength, const size_t bLength, const double* a, const double* b = normal_1_5_test,
                               const bool isTailed = false, const bool isLeftTailed = true) {
        return NStatistics::TTest(PART_RANGE(a, 0, aLength), PART_RANGE(b, 0, bLength), isTailed, isLeftTailed);
    }

    Y_UNIT_TEST(MannWhitneyTest) {
        TVector<double> a(100, 0);
        TVector<double> b(100, 0);
        UNIT_ASSERT_DOUBLES_EQUAL(NStatistics::MannWhitneyWithSign(a.begin(), a.end(), b.begin(), b.end()).PValue, 1., 0.01);
        a[0] = 3;
        b[0] = 2;
        UNIT_ASSERT_DOUBLES_EQUAL(NStatistics::MannWhitneyWithSign(a.begin(), a.end(), b.begin(), b.end()).PValue, 1., 0.01);

        UNIT_ASSERT_DOUBLES_EQUAL(MannWhitneyPartTest(1, normal_0_1), 1., 0.01);

        UNIT_ASSERT_DOUBLES_EQUAL(MannWhitneyPartTest(10, normal_0_1), 1., 0.01);
        UNIT_ASSERT_DOUBLES_EQUAL(MannWhitneyPartTest(10, normal_1_5), 1., 0.01);

        UNIT_ASSERT_DOUBLES_EQUAL(MannWhitneyPartTest(30, normal_0_1), 0, 0.07);
        UNIT_ASSERT_DOUBLES_EQUAL(MannWhitneyPartTest(30, normal_1_5), 1, 0.95);

        UNIT_ASSERT_DOUBLES_EQUAL(MannWhitneyPartTest(100, normal_0_1), 0, 0.05);
        UNIT_ASSERT_DOUBLES_EQUAL(MannWhitneyPartTest(100, normal_1_5), 1, 0.95);

        UNIT_ASSERT_DOUBLES_EQUAL(MannWhitneyPartTest(500, normal_0_1), 0, 0.05);
        UNIT_ASSERT_DOUBLES_EQUAL(MannWhitneyPartTest(500, normal_1_5), 1, 0.95);

        UNIT_ASSERT_DOUBLES_EQUAL(MannWhitneyPartTest(1000, normal_0_1), 0, 0.05);
        UNIT_ASSERT_DOUBLES_EQUAL(MannWhitneyPartTest(1000, normal_1_5), 1, 0.95);
        UNIT_ASSERT_DOUBLES_EQUAL(NStatistics::MannWhitney(RANGE(SmallObjCont), RANGE(SmallObjTest)), 0.08, 0.01);
    }

    Y_UNIT_TEST(WilcoxonTest) {
        UNIT_ASSERT_DOUBLES_EQUAL(WilcoxonPartTest(10, normal_0_1), 0, 0.05);
        UNIT_ASSERT_DOUBLES_EQUAL(WilcoxonPartTest(10, normal_1_5), 1, 0.95);

        UNIT_ASSERT_DOUBLES_EQUAL(WilcoxonPartTest(30, normal_0_1), 0, 0.1);
        UNIT_ASSERT_DOUBLES_EQUAL(WilcoxonPartTest(30, normal_1_5), 1, 0.95);

        UNIT_ASSERT_DOUBLES_EQUAL(WilcoxonPartTest(100, normal_0_1), 0, 0.05);
        UNIT_ASSERT_DOUBLES_EQUAL(WilcoxonPartTest(100, normal_1_5), 1, 0.95);

        UNIT_ASSERT_DOUBLES_EQUAL(WilcoxonPartTest(500, normal_0_1), 0, 0.05);
        UNIT_ASSERT_DOUBLES_EQUAL(WilcoxonPartTest(500, normal_1_5), 1, 0.95);

        UNIT_ASSERT_DOUBLES_EQUAL(WilcoxonPartTest(1000, normal_0_1), 0, 0.05);
        UNIT_ASSERT_DOUBLES_EQUAL(WilcoxonPartTest(1000, normal_1_5), 1, 0.95);

        // In the 500 and 1000 tests it's ok that the pvalue is near 0.5.
        // normal_1_5 and normal_1_5_test are a little different in the calculated mean value.
        // About 5%-10% and it makes them look quite different for Wilcoxon test.
    }

    Y_UNIT_TEST(MannWhitneyWithSignTest) {
        UNIT_ASSERT_DOUBLES_EQUAL(MannWhitneyWithSignPartTest(10, normal_0_1), 1, 0.01);
        UNIT_ASSERT_DOUBLES_EQUAL(MannWhitneyWithSignPartTest(10, normal_1_5), 1, 0.01);

        UNIT_ASSERT_DOUBLES_EQUAL(MannWhitneyWithSignPartTest(30, normal_0_1), 0, 0.07);
        UNIT_ASSERT_DOUBLES_EQUAL(MannWhitneyWithSignPartTest(30, normal_1_5), 1, 0.95);

        UNIT_ASSERT_DOUBLES_EQUAL(MannWhitneyWithSignPartTest(100, normal_0_1), 0, 0.05);
        UNIT_ASSERT_DOUBLES_EQUAL(MannWhitneyWithSignPartTest(100, normal_1_5), 1, 0.95);

        UNIT_ASSERT_DOUBLES_EQUAL(MannWhitneyWithSignPartTest(500, normal_0_1), 0, 0.05);
        UNIT_ASSERT_DOUBLES_EQUAL(MannWhitneyWithSignPartTest(500, normal_1_5), 1, 0.95);

        UNIT_ASSERT_DOUBLES_EQUAL(MannWhitneyWithSignPartTest(1000, normal_0_1), 0, 0.05);
        UNIT_ASSERT_DOUBLES_EQUAL(MannWhitneyWithSignPartTest(1000, normal_1_5), 1, 0.95);
        UNIT_ASSERT_DOUBLES_EQUAL(NStatistics::MannWhitneyWithSign(RANGE(SmallObjCont), RANGE(SmallObjTest)).PValue, 0.08, 0.01);

        UNIT_ASSERT_EQUAL(MannWhitneySignPartTest(500, 500, normal_0_1, normal_1_5), -1);
        UNIT_ASSERT_EQUAL(MannWhitneySignPartTest(500, 500, normal_1_5, normal_0_1), 1);

        UNIT_ASSERT_EQUAL(MannWhitneySignPartTest(1000, 1000, normal_0_1, normal_1_5), -1);
        UNIT_ASSERT_EQUAL(MannWhitneySignPartTest(1000, 100, normal_0_1, normal_1_5), -1);


        UNIT_ASSERT_EQUAL(MannWhitneySignPartTest(500, 500, normal_1_5, normal_0_1), 1);
    }

    Y_UNIT_TEST(WilcoxonWithSignTest) {
        UNIT_ASSERT_DOUBLES_EQUAL(WilcoxonWithSignPartTest(10, normal_0_1), 0, 0.05);
        UNIT_ASSERT_DOUBLES_EQUAL(WilcoxonWithSignPartTest(10, normal_1_5), 1, 0.95);

        UNIT_ASSERT_DOUBLES_EQUAL(WilcoxonWithSignPartTest(30, normal_0_1), 0, 0.1);
        UNIT_ASSERT_DOUBLES_EQUAL(WilcoxonWithSignPartTest(30, normal_1_5), 1, 0.95);

        UNIT_ASSERT_DOUBLES_EQUAL(WilcoxonWithSignPartTest(100, normal_0_1), 0, 0.05);
        UNIT_ASSERT_DOUBLES_EQUAL(WilcoxonWithSignPartTest(100, normal_1_5), 1, 0.95);

        UNIT_ASSERT_DOUBLES_EQUAL(WilcoxonWithSignPartTest(500, normal_0_1), 0, 0.05);
        UNIT_ASSERT_DOUBLES_EQUAL(WilcoxonWithSignPartTest(500, normal_1_5), 1, 0.95);

        UNIT_ASSERT_DOUBLES_EQUAL(WilcoxonWithSignPartTest(1000, normal_0_1), 0, 0.05);
        UNIT_ASSERT_DOUBLES_EQUAL(WilcoxonWithSignPartTest(1000, normal_1_5), 1, 0.95);

        UNIT_ASSERT_EQUAL(WilcoxonSignPartTest(500, normal_0_1, normal_1_5), -1);
        UNIT_ASSERT_EQUAL(WilcoxonSignPartTest(500, normal_1_5, normal_0_1), 1);

        UNIT_ASSERT_EQUAL(WilcoxonSignPartTest(1000, normal_0_1, normal_1_5), -1);
        UNIT_ASSERT_EQUAL(WilcoxonSignPartTest(1000, normal_1_5, normal_0_1), 1);

        // In the 500 and 1000 tests it's ok that the pvalue is near 0.5.
        // normal_1_5 and normal_1_5_test are a little different in the calculated mean value.
        // About 5%-10% and it makes them look quite different for Wilcoxon test.
    }

    Y_UNIT_TEST(OneSampleTTest) {
        UNIT_ASSERT_DOUBLES_EQUAL(OneSampleTPartTest(10, normal_0_1, 0), 1, 0.95);
        UNIT_ASSERT_DOUBLES_EQUAL(OneSampleTPartTest(10, normal_1_5_test, 1), 1, 0.95);

        UNIT_ASSERT_DOUBLES_EQUAL(OneSampleTPartTest(30, normal_0_1, 0), 1, 0.95);
        UNIT_ASSERT_DOUBLES_EQUAL(OneSampleTPartTest(30, normal_1_5_test, 1), 1, 0.95);

        UNIT_ASSERT_DOUBLES_EQUAL(OneSampleTPartTest(100, normal_0_1, 0), 1, 0.95);
        UNIT_ASSERT_DOUBLES_EQUAL(OneSampleTPartTest(100, normal_1_5_test, 1), 1, 0.95);

        UNIT_ASSERT_DOUBLES_EQUAL(OneSampleTPartTest(500, normal_0_1, 0), 1, 0.95);
        UNIT_ASSERT_DOUBLES_EQUAL(OneSampleTPartTest(500, normal_1_5_test, 1), 1, 0.95);

        UNIT_ASSERT_DOUBLES_EQUAL(OneSampleTPartTest(1000, normal_0_1, 0), 1, 0.95);
        UNIT_ASSERT_DOUBLES_EQUAL(OneSampleTPartTest(1000, normal_1_5_test, 1), 1, 0.95);

        UNIT_ASSERT_DOUBLES_EQUAL(OneSampleTPartTest(10, normal_0_1, 0, true, true), 1, 0.95);
        UNIT_ASSERT_DOUBLES_EQUAL(OneSampleTPartTest(10, normal_1_5_test, 0, true, true), 1, 0.95);

        UNIT_ASSERT_DOUBLES_EQUAL(OneSampleTPartTest(10, normal_0_1, 0, true, false), 1, 0.95);
        UNIT_ASSERT_DOUBLES_EQUAL(OneSampleTPartTest(10, normal_1_5_test, 0, true, false), 0, 0.05);

        UNIT_ASSERT_DOUBLES_EQUAL(OneSampleTPartTest(10, normal_0_1, 0.1, true, true), 0, 0.05);
        UNIT_ASSERT_DOUBLES_EQUAL(OneSampleTPartTest(10, normal_1_5_test, 0.1, true, true), 1, 0.95);

        UNIT_ASSERT_DOUBLES_EQUAL(OneSampleTPartTest(10, normal_0_1, 0.1, true, false), 1, 0.95);
        UNIT_ASSERT_DOUBLES_EQUAL(OneSampleTPartTest(10, normal_1_5_test, 0.1, true, false), 0, 0.05);
    }

    Y_UNIT_TEST(TwoSamplesTTest) {
        UNIT_ASSERT_DOUBLES_EQUAL(TwoSamplesTPartTest(10, 10, normal_0_1), 0, 0.05);
        UNIT_ASSERT_DOUBLES_EQUAL(TwoSamplesTPartTest(10, 10, normal_1_5), 1, 0.95);

        UNIT_ASSERT_DOUBLES_EQUAL(TwoSamplesTPartTest(20, 10, normal_0_1), 0, 0.05);
        UNIT_ASSERT_DOUBLES_EQUAL(TwoSamplesTPartTest(10, 20, normal_1_5), 1, 0.95);

        UNIT_ASSERT_DOUBLES_EQUAL(TwoSamplesTPartTest(30, 30, normal_0_1), 0, 0.1);
        UNIT_ASSERT_DOUBLES_EQUAL(TwoSamplesTPartTest(30, 30, normal_1_5), 1, 0.95);

        UNIT_ASSERT_DOUBLES_EQUAL(TwoSamplesTPartTest(60, 30, normal_0_1), 0, 0.12);
        UNIT_ASSERT_DOUBLES_EQUAL(TwoSamplesTPartTest(30, 60, normal_1_5), 1, 0.95);

        UNIT_ASSERT_DOUBLES_EQUAL(TwoSamplesTPartTest(100, 100, normal_0_1), 0, 0.05);
        UNIT_ASSERT_DOUBLES_EQUAL(TwoSamplesTPartTest(100, 100, normal_1_5), 1, 0.95);

        UNIT_ASSERT_DOUBLES_EQUAL(TwoSamplesTPartTest(200, 190, normal_0_1), 0, 0.1);
        UNIT_ASSERT_DOUBLES_EQUAL(TwoSamplesTPartTest(190, 200, normal_1_5), 1, 0.95);

        UNIT_ASSERT_DOUBLES_EQUAL(TwoSamplesTPartTest(500, 500, normal_0_1), 0, 0.05);
        UNIT_ASSERT_DOUBLES_EQUAL(TwoSamplesTPartTest(500, 500, normal_1_5), 1, 0.95);

        UNIT_ASSERT_DOUBLES_EQUAL(TwoSamplesTPartTest(1000, 500, normal_0_1), 0, 0.05);
        UNIT_ASSERT_DOUBLES_EQUAL(TwoSamplesTPartTest(500, 1000, normal_1_5), 1, 0.95);

        UNIT_ASSERT_DOUBLES_EQUAL(TwoSamplesTPartTest(1000, 1000, normal_0_1), 0, 0.05);
        UNIT_ASSERT_DOUBLES_EQUAL(TwoSamplesTPartTest(1000, 1000, normal_1_5), 1, 0.95);

        UNIT_ASSERT_DOUBLES_EQUAL(TwoSamplesTPartTest(10, 10, normal_0_1, normal_1_5_test, true, true), 0, 0.05);
        UNIT_ASSERT_DOUBLES_EQUAL(TwoSamplesTPartTest(10, 10, normal_1_5_test, normal_0_1, true, true), 1, 0.95);

        UNIT_ASSERT_DOUBLES_EQUAL(TwoSamplesTPartTest(10, 10, normal_0_1, normal_1_5_test, true, false), 1, 0.95);
        UNIT_ASSERT_DOUBLES_EQUAL(TwoSamplesTPartTest(10, 10, normal_1_5_test, normal_0_1, true, false), 0, 0.05);

        UNIT_ASSERT_DOUBLES_EQUAL(TwoSamplesTPartTest(10, 10, normal_1_5, normal_1_5_test, true, true), 1, 0.95);
        UNIT_ASSERT_DOUBLES_EQUAL(TwoSamplesTPartTest(10, 10, normal_1_5, normal_1_5_test, true, false), 1, 0.95);

        UNIT_ASSERT_DOUBLES_EQUAL(TwoSamplesTPartTest(20, 10, normal_0_1, normal_1_5_test, true, true), 0, 0.05);
        UNIT_ASSERT_DOUBLES_EQUAL(TwoSamplesTPartTest(10, 20, normal_1_5_test, normal_0_1, true, true), 1, 0.95);

        UNIT_ASSERT_DOUBLES_EQUAL(TwoSamplesTPartTest(20, 10, normal_0_1, normal_1_5_test, true, false), 1, 0.95);
        UNIT_ASSERT_DOUBLES_EQUAL(TwoSamplesTPartTest(10, 20, normal_1_5_test, normal_0_1, true, false), 0, 0.05);

        UNIT_ASSERT_DOUBLES_EQUAL(TwoSamplesTPartTest(20, 10, normal_1_5, normal_1_5_test, true, true), 1, 0.95);
        UNIT_ASSERT_DOUBLES_EQUAL(TwoSamplesTPartTest(10, 20, normal_1_5, normal_1_5_test, true, false), 1, 0.95);

        UNIT_ASSERT_DOUBLES_EQUAL(TwoSamplesTPartTest(500, 1000, normal_0_1, normal_1_5_test, true, true), 0, 0.05);
        UNIT_ASSERT_DOUBLES_EQUAL(TwoSamplesTPartTest(1000, 500, normal_1_5_test, normal_0_1, true, true), 1, 0.95);

        UNIT_ASSERT_DOUBLES_EQUAL(TwoSamplesTPartTest(500, 1000, normal_0_1, normal_1_5_test, true, false), 1, 0.95);
        UNIT_ASSERT_DOUBLES_EQUAL(TwoSamplesTPartTest(1000, 500, normal_1_5_test, normal_0_1, true, false), 0, 0.05);

        UNIT_ASSERT_DOUBLES_EQUAL(TwoSamplesTPartTest(500, 1000, normal_1_5, normal_1_5_test, true, true), 1, 0.95);
        UNIT_ASSERT_DOUBLES_EQUAL(TwoSamplesTPartTest(1000, 500, normal_1_5, normal_1_5_test, true, false), 1, 0.95);
    }

    Y_UNIT_TEST(ProbitTest) {
        UNIT_ASSERT_DOUBLES_EQUAL(NStatistics::Probit<double>(0.1), -1.28155, 1e-5);
        UNIT_ASSERT_DOUBLES_EQUAL(NStatistics::Probit<double>(0.5), 0, 1e-5);
        UNIT_ASSERT_DOUBLES_EQUAL(NStatistics::Probit<double>(0.9), 1.28155, 1e-5);
        UNIT_ASSERT_DOUBLES_EQUAL(NStatistics::Probit<double>(0.95), 1.64485, 1e-5);
        UNIT_ASSERT_DOUBLES_EQUAL(NStatistics::Probit<double>(0.975), 1.95996, 1e-5);
    }

    Y_UNIT_TEST(KLDivergenceTest) {
        TVector<double> qNormal = {0.000229231405911, 0.00597703624674, 0.0605975359431, 0.241730337457, 0.382924922548, 0.241730337457, 0.0605975359431, 0.00597703624674, 0.000229231405911};
        TVector<double> pNormal = {1, 9, 54, 230, 402, 247, 48, 9, 0};
        UNIT_ASSERT_DOUBLES_EQUAL(NStatistics::KLDivergence(pNormal.begin(), pNormal.end(), qNormal.begin(), qNormal.end()), 0.0048574438, 1e-5);
        TVector<int> qSimple = {1, 1, 0};
        TVector<int> pSimple = {0, 2, 0};
        UNIT_ASSERT_DOUBLES_EQUAL(NStatistics::KLDivergence(pSimple.begin(), pSimple.end(), qSimple.begin(), qSimple.end()), 0.69314781, 1e-5);
        TVector<int> pBinomial = {2, 8};
        TVector<double> qBinomial = {8, 2};
        UNIT_ASSERT_DOUBLES_EQUAL(NStatistics::KLDivergence(pBinomial.begin(), pBinomial.end(), qBinomial.begin(), qBinomial.end()), 0.8317766167, 1e-5);
        TVector<int> qBadDataBinomial = {0, 1};
        TVector<int> pBadDataBinomial = {1, 1};
        bool exceptionCaught = false;
        try {
            NStatistics::KLDivergence(pBadDataBinomial.begin(), pBadDataBinomial.end(), qBadDataBinomial.begin(), qBadDataBinomial.end());
        } catch (const yexception&) {
            exceptionCaught = true;
        }
        UNIT_ASSERT(exceptionCaught);
    }

    Y_UNIT_TEST(KolmogorovSmirnovHistogramStatistics) {
        TVector<double> p;
        TVector<double> q;
        p = {1, 2, 1};
        q = {1, 0, 3};
        UNIT_ASSERT_DOUBLES_EQUAL(NStatistics::KolmogorovSmirnovHistogramStatistics(p.begin(), p.end(), q.begin(), q.end()), 0.5, 1e-5);
        p = {3, 5, 12, 8, 7};
        q = {2, 3, 7, 15, 2};
        UNIT_ASSERT_DOUBLES_EQUAL(NStatistics::KolmogorovSmirnovHistogramStatistics(p.begin(), p.end(), q.begin(), q.end()), 0.157635468, 1e-5);
        UNIT_ASSERT_DOUBLES_EQUAL(NStatistics::KolmogorovSmirnovHistogramStatistics(q.begin(), q.end(), p.begin(), p.end()), 0.157635468, 1e-5);
    }

    NStatistics::TStatisticsCalculator<double> QuadStatistics(size_t start, size_t end) {
        NStatistics::TStatisticsCalculator<double> result;
        for (; start != end; ++start) {
            result.Push(start * start);
        }
        return result;
    }

    NStatistics::TStatisticsCalculator<double> RemoveQuadStatistics(NStatistics::TStatisticsCalculator<double> statisticsCalculator,
                                                                    size_t start,
                                                                    size_t end) {
        for (; start != end; ++start) {
            statisticsCalculator.Remove(start * start);
        }
        return statisticsCalculator;
    }
}

#undef SIZE_OF
#undef RANGE
#undef PART_RANGE
