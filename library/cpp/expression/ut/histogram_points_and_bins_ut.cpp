#include <library/cpp/testing/unittest/registar.h>

#include <utility>

#include "histogram_points_and_bins.h"


Y_UNIT_TEST_SUITE(THistogramBinAndPointsTest) {
    Y_UNIT_TEST(TestEqualOperator) {
        TVector<double> firstPoints = {1, 2, 3, 4.000002};
        TVector<double> firstBins = {10, 20, 30, 40, 50};
        THistogramPointsAndBins firstEqualHistogramData = THistogramPointsAndBins(firstPoints, firstBins);
        TVector<double> secondPoints = {1, 2, 3, 4.000001};
        TVector<double> secondBins = {10, 20, 30, 40, 50};
        THistogramPointsAndBins secondEqualHistogramData = THistogramPointsAndBins(secondPoints, secondBins);
        THistogramPointsAndBins emptyHistogramData = THistogramPointsAndBins();

        UNIT_ASSERT_EQUAL(firstEqualHistogramData.IsEqual(secondEqualHistogramData, 0.0), false);
        UNIT_ASSERT_EQUAL(firstEqualHistogramData.IsEqual(secondEqualHistogramData, 1e-5), true);
        UNIT_ASSERT_EQUAL(firstEqualHistogramData.IsEqual(firstEqualHistogramData, 1e-5), true);
        UNIT_ASSERT_EQUAL(firstEqualHistogramData.IsEqual(emptyHistogramData, 1e-5), false);
    }
    Y_UNIT_TEST(TestSettersAndGetters) {
        TVector<double> points = {1, 2, 3, 4};
        TVector<double> bins = {10, 20, 30, 40, 50};
        THistogramPointsAndBins filledHistogramData = THistogramPointsAndBins(points, bins);

        UNIT_ASSERT_EQUAL(filledHistogramData.GetPoints(), points);
        UNIT_ASSERT_EQUAL(filledHistogramData.GetBins(), bins);

        TVector<double> badBins = {10, 20};
        THistogramPointsAndBins badHistogramData = THistogramPointsAndBins(points, badBins);
        UNIT_ASSERT_EQUAL(badHistogramData.GetPoints().size(), 0);
        UNIT_ASSERT_EQUAL(badHistogramData.GetBins().size(), 0);

        THistogramPointsAndBins emptyHistogramData = THistogramPointsAndBins();
        UNIT_ASSERT_EQUAL(emptyHistogramData.GetPoints().size(), 0);
        UNIT_ASSERT_EQUAL(emptyHistogramData.GetBins().size(), 0);

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

        UNIT_ASSERT_EQUAL(emptyHistogramData.IsValidBins(), false);
        UNIT_ASSERT_EQUAL(emptyHistogramData.IsValidPoints(), false);
        UNIT_ASSERT_EQUAL(emptyHistogramData.IsValidData(95), false);

        UNIT_ASSERT_EQUAL(emptyHistogramData.IsValidPercentile(-10), false);
        UNIT_ASSERT_EQUAL(emptyHistogramData.IsValidPercentile(105), false);
        UNIT_ASSERT_EQUAL(emptyHistogramData.IsValidPercentile(100), false);
        UNIT_ASSERT_EQUAL(emptyHistogramData.IsValidPercentile(0), false);

        TVector<double> zeroPoints = {0, 0, 0, 0};
        TVector<double> zeroBins = {0, 0, 0, 0, 0};
        THistogramPointsAndBins zeroHistogramData = THistogramPointsAndBins(zeroPoints, zeroBins);

        UNIT_ASSERT_EQUAL(zeroHistogramData.IsValidBins(), false);
        UNIT_ASSERT_EQUAL(zeroHistogramData.IsValidPoints(), false);
        UNIT_ASSERT_EQUAL(zeroHistogramData.IsValidData(95), false);

        TVector<double> negativePoints = {-1, 0, 1, 2};
        TVector<double> negativeBins = {0, -5.21, 0, 10.1, 0};
        THistogramPointsAndBins negativeHistogramData = THistogramPointsAndBins(negativePoints, negativeBins);

        UNIT_ASSERT_EQUAL(negativeHistogramData.IsValidBins(), false);
        UNIT_ASSERT_EQUAL(negativeHistogramData.IsValidPoints(), true);
        UNIT_ASSERT_EQUAL(negativeHistogramData.IsValidData(73), false);

        TVector<double> points = {1, 2, 3, 4};
        TVector<double> bins = {10, 20, 30, 40, 50};
        THistogramPointsAndBins filledHistogramData = THistogramPointsAndBins(points, bins);

        UNIT_ASSERT_EQUAL(filledHistogramData.IsValidBins(), true);
        UNIT_ASSERT_EQUAL(filledHistogramData.IsValidPoints(), true);
        UNIT_ASSERT_EQUAL(filledHistogramData.IsValidData(95), true);

        UNIT_ASSERT_EQUAL(filledHistogramData.IsValidData(-95), false);

        UNIT_ASSERT_EQUAL(filledHistogramData.IsValidPercentile(93), true);
        UNIT_ASSERT_EQUAL(filledHistogramData.IsValidPercentile(10.3), true);
    }
    Y_UNIT_TEST(TestOutput) {
        TVector<double> points = {1, 2};
        TVector<double> bins = {10, 20, 30};
        THistogramPointsAndBins filledHistogramData = THistogramPointsAndBins(points, bins);

        UNIT_ASSERT_EQUAL(ToString(filledHistogramData), "1.000000,2.000000,;10.000000,20.000000,30.000000,");
    }

} // THistogramBinAndPointsTest
