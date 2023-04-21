#include <catboost/libs/dataset_statistics/histograms.h>

#include <library/cpp/testing/unittest/registar.h>
#include <util/generic/ymath.h>

using namespace std;
using namespace NCB;

Y_UNIT_TEST_SUITE(THistograms) {
    Y_UNIT_TEST(TestUniformFloatFeatureHistograms) {
        ui32 borderCount = 5;
        TVector<float> features = {-0.2, -0.01, 0.1, 0.2, 0.1, 0.5, 1., 0.6, -5, 5};
        TVector<float> trueBorders = {-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1};
        // -5, -0.2,  0. |  0.1, 0.1, 0.2 | | 0.5, 0.6 | |, 1., 5}
        // -5, OOD \-0.2\  -0.2, -0.01, \0\ 0.1, 0.1,  0.2, \0.2\ \0.4\ 0.5,  0.6, \0.6\  \0.8\ 0.9 1\1\  OOD   5}
        TVector<ui32> trueHist = {2, 3, 0, 2, 0, 1,};
        TFloatFeatureHistogram floatFeatureHistogram = TFloatFeatureHistogram(TBorders(5, -0.2, 1.));
        floatFeatureHistogram.ConvertBitToUniform();
        floatFeatureHistogram.CalcUniformHistogram(features);
        auto floatFeatureHistogramBorders = floatFeatureHistogram.Borders.GetBorders();

        UNIT_ASSERT_EQUAL(floatFeatureHistogramBorders.size(), trueBorders.size());
        UNIT_ASSERT_EQUAL(floatFeatureHistogramBorders.size(), borderCount + 2);
        for (ui32 i = 0; i < borderCount; ++i) {
            UNIT_ASSERT(Abs(floatFeatureHistogramBorders[i] - trueBorders[i]) < 1e-6);
        }

        UNIT_ASSERT_EQUAL(floatFeatureHistogram.Histogram.size(), trueHist.size());
        UNIT_ASSERT_EQUAL(floatFeatureHistogram.Histogram.size(), borderCount + 1);

        for (ui32 i = 0; i <= borderCount; ++i) {
            UNIT_ASSERT_EQUAL(floatFeatureHistogram.Histogram[i], trueHist[i]);
        }
        UNIT_ASSERT_EQUAL(floatFeatureHistogram.Nans, 0);
        UNIT_ASSERT_EQUAL(floatFeatureHistogram.MinusInf, 0);
        UNIT_ASSERT_EQUAL(floatFeatureHistogram.PlusInf, 0);
        UNIT_ASSERT_EQUAL(floatFeatureHistogram.Borders.OutOfDomainValuesCount, 2);
    }

    Y_UNIT_TEST(TestFloatFeatureHistograms) {
        ui32 borderCount = 6;
        TVector<float> features = {-0.2, 0., 0.1, 0.2, 0.1, 0.5, 1., 0.6};
        TVector<float> trueBorders = {-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1};
        TVector<ui32> trueHist = {1, 1, 3, 0, 2, 0, 1};
        TFloatFeatureHistogram floatFeatureHistogram = TFloatFeatureHistogram(TBorders(trueBorders));
        floatFeatureHistogram.CalcHistogramWithBorders(features);
        UNIT_ASSERT_EQUAL(floatFeatureHistogram.Histogram.size(), trueHist.size());
        UNIT_ASSERT_EQUAL(floatFeatureHistogram.Histogram.size(), borderCount + 1);
        for (ui32 i = 0; i <= borderCount; ++i) {
            UNIT_ASSERT_EQUAL(floatFeatureHistogram.Histogram[i], trueHist[i]);
        }
        UNIT_ASSERT_EQUAL(floatFeatureHistogram.Nans, 0);
        UNIT_ASSERT_EQUAL(floatFeatureHistogram.MinusInf, 0);
        UNIT_ASSERT_EQUAL(floatFeatureHistogram.PlusInf, 0);
    }
}
