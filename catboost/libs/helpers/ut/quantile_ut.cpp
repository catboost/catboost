#include <library/cpp/testing/unittest/registar.h>

#include <catboost/libs/helpers/quantile.h>
#include <catboost/private/libs/options/restrictions.h>

#include <util/generic/array_ref.h>
#include <util/generic/fwd.h>
#include <util/generic/vector.h>

#include <cfloat>

Y_UNIT_TEST_SUITE(TCalcQuantile) {

    const TVector<float> weightsNoWeights = {1, 1, 1, 1, 1, 1, 1, 1};
    const TVector<float> weightsHasWeights = {0.2, 0.4, 0.13, 0.43, 0.74, 0.3, 0.44, 0.37};
    const TVector<float> sampleOrderedNoWeights = {0, 2, 10, 37, 40, 500, 501, 600};
    const TVector<float> sampleUnorderedNoWeights = {600, 40, 2, 500, 0, 37, 10, 501};

    Y_UNIT_TEST(TCalcQuantileSampleOrderedNoWeights1) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalcSampleQuantile(sampleOrderedNoWeights, weightsNoWeights, 0), 0, 1e-6);
    }

    Y_UNIT_TEST(TCalcQuantileSampleOrderedNoWeights2) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalcSampleQuantile(sampleOrderedNoWeights, weightsNoWeights, 0.125 - 5 * DBL_EPSILON), 0, 1e-6);
    }

    Y_UNIT_TEST(TCalcQuantileSampleOrderedNoWeights3) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalcSampleQuantile(sampleOrderedNoWeights, weightsNoWeights, 0.25 - 5 * DBL_EPSILON), 2, 1e-6);
    }

    Y_UNIT_TEST(TCalcQuantileSampleOrderedNoWeights4) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalcSampleQuantile(sampleOrderedNoWeights, weightsNoWeights, 0.375 - 5 * DBL_EPSILON), 10, 1e-6);
    }

    Y_UNIT_TEST(TCalcQuantileSampleOrderedNoWeights5) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalcSampleQuantile(sampleOrderedNoWeights, weightsNoWeights, 0.5 - 5 * DBL_EPSILON), 37, 1e-6);
    }

    Y_UNIT_TEST(TCalcQuantileSampleOrderedNoWeights6) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalcSampleQuantile(sampleOrderedNoWeights, weightsNoWeights, 0.625 - 5 * DBL_EPSILON), 40, 1e-6);
    }

    Y_UNIT_TEST(TCalcQuantileSampleOrderedNoWeights7) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalcSampleQuantile(sampleOrderedNoWeights, weightsNoWeights, 0.75 - 5 * DBL_EPSILON), 500, 1e-6);
    }

    Y_UNIT_TEST(TCalcQuantileSampleOrderedNoWeights8) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalcSampleQuantile(sampleOrderedNoWeights, weightsNoWeights, 0.875 - 5 * DBL_EPSILON), 501, 1e-6);
    }

    Y_UNIT_TEST(TCalcQuantileSampleOrderedNoWeights9) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalcSampleQuantile(sampleOrderedNoWeights, weightsNoWeights, 1 - 5 * DBL_EPSILON), 600, 1e-6);
    }

    Y_UNIT_TEST(TCalcQuantileSampleOrderedNoWeights10) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalcSampleQuantile(sampleOrderedNoWeights, weightsNoWeights, 5 * DBL_EPSILON), 0, 1e-6);
    }

    Y_UNIT_TEST(TCalcQuantileSampleOrderedNoWeights11) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalcSampleQuantile(sampleOrderedNoWeights, weightsNoWeights, 0.125 + 5 * DBL_EPSILON), 2, 1e-6);
    }

    Y_UNIT_TEST(TCalcQuantileSampleOrderedNoWeights12) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalcSampleQuantile(sampleOrderedNoWeights, weightsNoWeights, 0.25 + 5 * DBL_EPSILON), 10, 1e-6);
    }

    Y_UNIT_TEST(TCalcQuantileSampleOrderedNoWeights13) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalcSampleQuantile(sampleOrderedNoWeights, weightsNoWeights, 0.375 + 5 * DBL_EPSILON), 37, 1e-6);
    }

    Y_UNIT_TEST(TCalcQuantileSampleOrderedNoWeights14) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalcSampleQuantile(sampleOrderedNoWeights, weightsNoWeights, 0.5 + 5 * DBL_EPSILON), 40, 1e-6);
    }

    Y_UNIT_TEST(TCalcQuantileSampleOrderedNoWeights15) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalcSampleQuantile(sampleOrderedNoWeights, weightsNoWeights, 0.625 + 5 * DBL_EPSILON), 500, 1e-6);
    }

    Y_UNIT_TEST(TCalcQuantileSampleOrderedNoWeights16) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalcSampleQuantile(sampleOrderedNoWeights, weightsNoWeights, 0.75 + 5 * DBL_EPSILON), 501, 1e-6);
    }

    Y_UNIT_TEST(TCalcQuantileSampleOrderedNoWeights17) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalcSampleQuantile(sampleOrderedNoWeights, weightsNoWeights, 0.875 + 5 * DBL_EPSILON), 600, 1e-6);
    }

    Y_UNIT_TEST(TCalcQuantileSampleUnrderedNoWeights1) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalcSampleQuantile(sampleUnorderedNoWeights, weightsNoWeights, 0), 0, 1e-6);
    }

    Y_UNIT_TEST(TCalcQuantileSampleUnrderedNoWeights2) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalcSampleQuantile(sampleUnorderedNoWeights, weightsNoWeights, 0.125 - 5 * DBL_EPSILON), 0, 1e-6);
    }

    Y_UNIT_TEST(TCalcQuantileSampleUnrderedNoWeights3) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalcSampleQuantile(sampleUnorderedNoWeights, weightsNoWeights, 0.25 - 5 * DBL_EPSILON), 2, 1e-6);
    }

    Y_UNIT_TEST(TCalcQuantileSampleUnrderedNoWeights4) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalcSampleQuantile(sampleUnorderedNoWeights, weightsNoWeights, 0.375 - 5 * DBL_EPSILON), 10, 1e-6);
    }

    Y_UNIT_TEST(TCalcQuantileSampleUnrderedNoWeights5) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalcSampleQuantile(sampleUnorderedNoWeights, weightsNoWeights, 0.5 - 5 * DBL_EPSILON), 37, 1e-6);
    }

    Y_UNIT_TEST(TCalcQuantileSampleUnrderedNoWeights6) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalcSampleQuantile(sampleUnorderedNoWeights, weightsNoWeights, 0.625 - 5 * DBL_EPSILON), 40, 1e-6);
    }

    Y_UNIT_TEST(TCalcQuantileSampleUnrderedNoWeights7) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalcSampleQuantile(sampleUnorderedNoWeights, weightsNoWeights, 0.75 - 5 * DBL_EPSILON), 500, 1e-6);
    }

    Y_UNIT_TEST(TCalcQuantileSampleUnrderedNoWeights8) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalcSampleQuantile(sampleUnorderedNoWeights, weightsNoWeights, 0.875 - 5 * DBL_EPSILON), 501 , 1e-6);
    }

    Y_UNIT_TEST(TCalcQuantileSampleUnrderedNoWeights9) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalcSampleQuantile(sampleUnorderedNoWeights, weightsNoWeights, 1 - 5 * DBL_EPSILON), 600 , 1e-6);
    }

    Y_UNIT_TEST(TCalcQuantileSampleUnrderedNoWeights10) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalcSampleQuantile(sampleUnorderedNoWeights, weightsNoWeights, 5 * DBL_EPSILON), 0, 1e-6);
    }

    Y_UNIT_TEST(TCalcQuantileSampleUnrderedNoWeights11) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalcSampleQuantile(sampleUnorderedNoWeights, weightsNoWeights, 0.125 + 5 * DBL_EPSILON), 2 , 1e-6);
    }

    Y_UNIT_TEST(TCalcQuantileSampleUnrderedNoWeights12) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalcSampleQuantile(sampleUnorderedNoWeights, weightsNoWeights, 0.25 + 5 * DBL_EPSILON), 10 , 1e-6);
    }

    Y_UNIT_TEST(TCalcQuantileSampleUnrderedNoWeights13) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalcSampleQuantile(sampleUnorderedNoWeights, weightsNoWeights, 0.375 + 5 * DBL_EPSILON), 37 , 1e-6);
    }

    Y_UNIT_TEST(TCalcQuantileSampleUnrderedNoWeights14) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalcSampleQuantile(sampleUnorderedNoWeights, weightsNoWeights, 0.5 + 5 * DBL_EPSILON), 40 , 1e-6);
    }

    Y_UNIT_TEST(TCalcQuantileSampleUnrderedNoWeights15) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalcSampleQuantile(sampleUnorderedNoWeights, weightsNoWeights, 0.625 + 5 * DBL_EPSILON), 500 , 1e-6);
    }

    Y_UNIT_TEST(TCalcQuantileSampleUnrderedNoWeights16) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalcSampleQuantile(sampleUnorderedNoWeights, weightsNoWeights, 0.75 + 5 * DBL_EPSILON), 501 , 1e-6);
    }

    Y_UNIT_TEST(TCalcQuantileSampleUnrderedNoWeights17) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalcSampleQuantile(sampleUnorderedNoWeights, weightsNoWeights, 0.875 + 5 * DBL_EPSILON), 600 , 1e-6);
    }

    Y_UNIT_TEST(TCalcQuantileSampleOrderedNoWeightsRepeating1) {
        TVector<float> sample = {0, 1, 1, 1, 2, 2, 2, 2};
        UNIT_ASSERT_DOUBLES_EQUAL(CalcSampleQuantile(sample, weightsNoWeights, 0), 0, 1e-6);
    }

    Y_UNIT_TEST(TCalcQuantileSampleOrderedNoWeightsRepeating2) {
        TVector<float> sample = {0, 1, 1, 1, 2, 2, 2, 2};
        UNIT_ASSERT_DOUBLES_EQUAL(CalcSampleQuantile(sample, weightsNoWeights, 0.125 - 5 * DBL_EPSILON), 0, 1e-6);
    }

    Y_UNIT_TEST(TCalcQuantileSampleOrderedNoWeightsRepeating3) {
        TVector<float> sample = {0, 1, 1, 1, 2, 2, 2, 2};
        UNIT_ASSERT_DOUBLES_EQUAL(CalcSampleQuantile(sample, weightsNoWeights, 0.125 + 5 * DBL_EPSILON), 1 , 1e-6);
    }

    Y_UNIT_TEST(TCalcQuantileSampleOrderedNoWeightsRepeating4) {
        TVector<float> sample = {0, 1, 1, 1, 2, 2, 2, 2};
        UNIT_ASSERT_DOUBLES_EQUAL(CalcSampleQuantile(sample, weightsNoWeights, 0.2 - 5 * DBL_EPSILON), 1 , 1e-6);
    }

    Y_UNIT_TEST(TCalcQuantileSampleOrderedNoWeightsRepeating5) {
        TVector<float> sample = {0, 1, 1, 1, 2, 2, 2, 2};
        UNIT_ASSERT_DOUBLES_EQUAL(CalcSampleQuantile(sample, weightsNoWeights, 0.2 + 5 * DBL_EPSILON), 1 , 1e-6);
    }

    Y_UNIT_TEST(TCalcQuantileSampleOrderedNoWeightsRepeating6) {
        TVector<float> sample = {0, 1, 1, 1, 2, 2, 2, 2};
        UNIT_ASSERT_DOUBLES_EQUAL(CalcSampleQuantile(sample, weightsNoWeights, 0.5 - 5 * DBL_EPSILON), 1  , 1e-6);
    }

    Y_UNIT_TEST(TCalcQuantileSampleOrderedNoWeightsRepeating7) {
        TVector<float> sample = {0, 1, 1, 1, 2, 2, 2, 2};
        UNIT_ASSERT_DOUBLES_EQUAL(CalcSampleQuantile(sample, weightsNoWeights, 0.5 + 5 * DBL_EPSILON), 2 , 1e-6);
    }

    Y_UNIT_TEST(TCalcQuantileSampleOrderedWeights1) {
        TVector<float> sample =    {0,     1,      2,      3,      4,      5,      6,      7};
        UNIT_ASSERT_DOUBLES_EQUAL(CalcSampleQuantile(sample, weightsHasWeights, 0.38), 3 , 1e-6);
    }

    Y_UNIT_TEST(TCalcQuantileSampleOrderedWeights2) {
        TVector<float> sample =    {0,     1,      2,      3,      4,      5,      6,      7};
        UNIT_ASSERT_DOUBLES_EQUAL(CalcSampleQuantile(sample, weightsHasWeights, 0.5), 4 , 1e-6);
    }

    Y_UNIT_TEST(TCalcQuantileSampleOrderedWeights3) {
        TVector<float> sample =    {0,     1,      2,      3,      4,      5,      6,      7};
        UNIT_ASSERT_DOUBLES_EQUAL(CalcSampleQuantile(sample, weightsHasWeights, 0.52), 4 , 1e-6);
    }
}
