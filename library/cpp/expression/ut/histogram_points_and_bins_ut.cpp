#include <library/cpp/testing/unittest/registar.h>

#include <utility>

#include "histogram_points_and_bins.h"


Y_UNIT_TEST_SUITE(THistogramBinAndPointsTest) {
    Y_UNIT_TEST(TestEqualOperator) {
        TVector<double> points = {1, 2, 3, 4};
        TVector<double> bins = {10, 20, 30, 40, 50};
        THistogramPointsAndBins firstEqualHistogramData = THistogramPointsAndBins(points, bins);
        THistogramPointsAndBins secondEqualHistogramData = THistogramPointsAndBins(points, bins);
        THistogramPointsAndBins emptyHistogramData = THistogramPointsAndBins();

        UNIT_ASSERT_EQUAL(firstEqualHistogramData == secondEqualHistogramData, true);
        UNIT_ASSERT_EQUAL(firstEqualHistogramData == firstEqualHistogramData, true);
        UNIT_ASSERT_EQUAL(firstEqualHistogramData == emptyHistogramData, false);
    }
    Y_UNIT_TEST(TestSettersAndGetters) {
        TVector<double> points = {1, 2, 3, 4};
        TVector<double> bins = {10, 20, 30, 40, 50};
        THistogramPointsAndBins filledHistogramData = THistogramPointsAndBins(points, bins);

        UNIT_ASSERT_EQUAL(filledHistogramData.GetPoints(), points);
        UNIT_ASSERT_EQUAL(filledHistogramData.GetBins(), bins);

        THistogramPointsAndBins emptyHistogramData = THistogramPointsAndBins();
        UNIT_ASSERT_EQUAL(emptyHistogramData.GetPoints().size(), 0);
        UNIT_ASSERT_EQUAL(emptyHistogramData.GetBins().size(), 0);

        TVector<double> badBins = {10, 20};
        emptyHistogramData.SetPointsAndBins(points, badBins);
        UNIT_ASSERT_EQUAL(emptyHistogramData.GetPoints().size(), 0);
        UNIT_ASSERT_EQUAL(emptyHistogramData.GetBins().size(), 0);

        emptyHistogramData.SetPointsAndBins(points, bins);
        UNIT_ASSERT_EQUAL(emptyHistogramData.GetPoints(), points);
        UNIT_ASSERT_EQUAL(emptyHistogramData.GetBins(), bins);
    }
    Y_UNIT_TEST(TestFindBinAndPArtion) {
        TVector<double> points = {1, 2, 3, 4};
        TVector<double> bins = {10, 20, 30, 40, 50};
        THistogramPointsAndBins histogramData = THistogramPointsAndBins(points, bins);

        UNIT_ASSERT_EQUAL(histogramData.FindBinAndPartion(0), std::make_pair(0, 0.0));
        UNIT_ASSERT_EQUAL(histogramData.FindBinAndPartion(20), std::make_pair(1, 1.0));
        UNIT_ASSERT_EQUAL(histogramData.FindBinAndPartion(30), std::make_pair(2, 0.5));
        UNIT_ASSERT_EQUAL(histogramData.FindBinAndPartion(60), std::make_pair(3, 0.75));
        UNIT_ASSERT_EQUAL(histogramData.FindBinAndPartion(95), std::make_pair(4, 0.85));
        UNIT_ASSERT_EQUAL(histogramData.FindBinAndPartion(100), std::make_pair(4, 1.0));
    }
    Y_UNIT_TEST(TestValidationMethods) {
        THistogramPointsAndBins emptyHistogramData = THistogramPointsAndBins();

        UNIT_ASSERT_EQUAL(emptyHistogramData.IsBinsFilledWithZeros(), true); // 0 bins -> 0 not zero bins
        UNIT_ASSERT_EQUAL(emptyHistogramData.IsEmptyData(), true);
        UNIT_ASSERT_EQUAL(emptyHistogramData.IsInvalidData(95), true);

        UNIT_ASSERT_EQUAL(emptyHistogramData.IsInvalidPercentile(-10), true);
        UNIT_ASSERT_EQUAL(emptyHistogramData.IsInvalidPercentile(105), true);
        UNIT_ASSERT_EQUAL(emptyHistogramData.IsInvalidPercentile(100), true);
        UNIT_ASSERT_EQUAL(emptyHistogramData.IsInvalidPercentile(0), true);

        TVector<double> points = {1, 2, 3, 4};
        TVector<double> bins = {10, 20, 30, 40, 50};
        THistogramPointsAndBins filledHistogramData = THistogramPointsAndBins(points, bins);

        UNIT_ASSERT_EQUAL(filledHistogramData.IsBinsFilledWithZeros(), false);
        UNIT_ASSERT_EQUAL(filledHistogramData.IsEmptyData(), false);
        UNIT_ASSERT_EQUAL(filledHistogramData.IsInvalidData(95), false);

        UNIT_ASSERT_EQUAL(filledHistogramData.IsInvalidData(-95), true);

        UNIT_ASSERT_EQUAL(filledHistogramData.IsInvalidPercentile(93), false);
        UNIT_ASSERT_EQUAL(filledHistogramData.IsInvalidPercentile(10.3), false);
    }
    Y_UNIT_TEST(TestOutput) {
        TVector<double> points = {1, 2};
        TVector<double> bins = {10, 20, 30};
        THistogramPointsAndBins filledHistogramData = THistogramPointsAndBins(points, bins);

        UNIT_ASSERT_EQUAL(ToString(filledHistogramData), "1.000000,2.000000,;10.000000,20.000000,30.000000,");
    }

} // THistogramBinAndPointsTest
