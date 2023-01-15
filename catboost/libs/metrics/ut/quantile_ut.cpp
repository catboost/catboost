#include <library/cpp/testing/unittest/registar.h>

#include <catboost/libs/metrics/optimal_const_for_loss.h>
#include <catboost/private/libs/options/restrictions.h>

#include <util/generic/array_ref.h>
#include <util/generic/fwd.h>
#include <util/generic/vector.h>

#include <cfloat>

using namespace NCB;

Y_UNIT_TEST_SUITE(TCalculateWeightedTargetQuantile) {

    const TVector<float> weightsNoWeights = {1, 1, 1, 1, 1, 1, 1, 1};
    const TVector<float> weightsHasWeights = {0.2, 0.4, 0.13, 0.43, 0.74, 0.3, 0.44, 0.37};
    const TVector<float> sampleOrderedNoWeights = {0, 2, 10, 37, 40, 500, 501, 600};
    const TVector<float> sampleUnorderedNoWeights = {600, 40, 2, 500, 0, 37, 10, 501};
    const double eps = 1e-6;

    Y_UNIT_TEST(TCalculateWeightedTargetQuantileSampleOrderedNoWeights1) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalculateWeightedTargetQuantile(sampleOrderedNoWeights, weightsNoWeights, 0, eps), -eps, 1e-6);
    }

    Y_UNIT_TEST(TCalculateWeightedTargetQuantileSampleOrderedNoWeights2) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalculateWeightedTargetQuantile(sampleOrderedNoWeights, weightsNoWeights, 0.125 - 5 * DBL_EPSILON, eps), eps, 1e-6);
    }

    Y_UNIT_TEST(TCalculateWeightedTargetQuantileSampleOrderedNoWeights3) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalculateWeightedTargetQuantile(sampleOrderedNoWeights, weightsNoWeights, 0.25 - 5 * DBL_EPSILON, eps), 2 + eps, 1e-6);
    }

    Y_UNIT_TEST(TCalculateWeightedTargetQuantileSampleOrderedNoWeights4) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalculateWeightedTargetQuantile(sampleOrderedNoWeights, weightsNoWeights, 0.375 - 5 * DBL_EPSILON, eps), 10 + eps, 1e-6);
    }

    Y_UNIT_TEST(TCalculateWeightedTargetQuantileSampleOrderedNoWeights5) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalculateWeightedTargetQuantile(sampleOrderedNoWeights, weightsNoWeights, 0.5 - 5 * DBL_EPSILON, eps), 37 + eps, 1e-6);
    }

    Y_UNIT_TEST(TCalculateWeightedTargetQuantileSampleOrderedNoWeights6) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalculateWeightedTargetQuantile(sampleOrderedNoWeights, weightsNoWeights, 0.625 - 5 * DBL_EPSILON, eps), 40 + eps, 1e-6);
    }

    Y_UNIT_TEST(TCalculateWeightedTargetQuantileSampleOrderedNoWeights7) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalculateWeightedTargetQuantile(sampleOrderedNoWeights, weightsNoWeights, 0.75 - 5 * DBL_EPSILON, eps), 500 + eps, 1e-6);
    }

    Y_UNIT_TEST(TCalculateWeightedTargetQuantileSampleOrderedNoWeights8) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalculateWeightedTargetQuantile(sampleOrderedNoWeights, weightsNoWeights, 0.875 - 5 * DBL_EPSILON, eps), 501 + eps, 1e-6);
    }

    Y_UNIT_TEST(TCalculateWeightedTargetQuantileSampleOrderedNoWeights9) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalculateWeightedTargetQuantile(sampleOrderedNoWeights, weightsNoWeights, 1 - 5 * DBL_EPSILON, eps), 600 - eps, 1e-6);
    }

    Y_UNIT_TEST(TCalculateWeightedTargetQuantileSampleOrderedNoWeights10) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalculateWeightedTargetQuantile(sampleOrderedNoWeights, weightsNoWeights, 5 * DBL_EPSILON, eps), eps, 1e-6);
    }

    Y_UNIT_TEST(TCalculateWeightedTargetQuantileSampleOrderedNoWeights11) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalculateWeightedTargetQuantile(sampleOrderedNoWeights, weightsNoWeights, 0.125 + 5 * DBL_EPSILON, eps), 2 - eps, 1e-6);
    }

    Y_UNIT_TEST(TCalculateWeightedTargetQuantileSampleOrderedNoWeights12) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalculateWeightedTargetQuantile(sampleOrderedNoWeights, weightsNoWeights, 0.25 + 5 * DBL_EPSILON, eps), 10 - eps, 1e-6);
    }

    Y_UNIT_TEST(TCalculateWeightedTargetQuantileSampleOrderedNoWeights13) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalculateWeightedTargetQuantile(sampleOrderedNoWeights, weightsNoWeights, 0.375 + 5 * DBL_EPSILON, eps), 37 - eps, 1e-6);
    }

    Y_UNIT_TEST(TCalculateWeightedTargetQuantileSampleOrderedNoWeights14) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalculateWeightedTargetQuantile(sampleOrderedNoWeights, weightsNoWeights, 0.5 + 5 * DBL_EPSILON, eps), 40 - eps, 1e-6);
    }

    Y_UNIT_TEST(TCalculateWeightedTargetQuantileSampleOrderedNoWeights15) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalculateWeightedTargetQuantile(sampleOrderedNoWeights, weightsNoWeights, 0.625 + 5 * DBL_EPSILON, eps), 500 - eps, 1e-6);
    }

    Y_UNIT_TEST(TCalculateWeightedTargetQuantileSampleOrderedNoWeights16) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalculateWeightedTargetQuantile(sampleOrderedNoWeights, weightsNoWeights, 0.75 + 5 * DBL_EPSILON, eps), 501 - eps, 1e-6);
    }

    Y_UNIT_TEST(TCalculateWeightedTargetQuantileSampleOrderedNoWeights17) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalculateWeightedTargetQuantile(sampleOrderedNoWeights, weightsNoWeights, 0.875 + 5 * DBL_EPSILON, eps), 600 - eps, 1e-6);
    }

    Y_UNIT_TEST(TCalculateWeightedTargetQuantileSampleUnrderedNoWeights1) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalculateWeightedTargetQuantile(sampleUnorderedNoWeights, weightsNoWeights, 0, eps), -eps, 1e-6);
    }

    Y_UNIT_TEST(TCalculateWeightedTargetQuantileSampleUnrderedNoWeights2) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalculateWeightedTargetQuantile(sampleUnorderedNoWeights, weightsNoWeights, 0.125 - 5 * DBL_EPSILON, eps), eps, 1e-6);
    }

    Y_UNIT_TEST(TCalculateWeightedTargetQuantileSampleUnrderedNoWeights3) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalculateWeightedTargetQuantile(sampleUnorderedNoWeights, weightsNoWeights, 0.25 - 5 * DBL_EPSILON, eps), 2 + eps, 1e-6);
    }

    Y_UNIT_TEST(TCalculateWeightedTargetQuantileSampleUnrderedNoWeights4) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalculateWeightedTargetQuantile(sampleUnorderedNoWeights, weightsNoWeights, 0.375 - 5 * DBL_EPSILON, eps), 10 + eps, 1e-6);
    }

    Y_UNIT_TEST(TCalculateWeightedTargetQuantileSampleUnrderedNoWeights5) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalculateWeightedTargetQuantile(sampleUnorderedNoWeights, weightsNoWeights, 0.5 - 5 * DBL_EPSILON, eps), 37 + eps, 1e-6);
    }

    Y_UNIT_TEST(TCalculateWeightedTargetQuantileSampleUnrderedNoWeights6) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalculateWeightedTargetQuantile(sampleUnorderedNoWeights, weightsNoWeights, 0.625 - 5 * DBL_EPSILON, eps), 40 + eps, 1e-6);
    }

    Y_UNIT_TEST(TCalculateWeightedTargetQuantileSampleUnrderedNoWeights7) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalculateWeightedTargetQuantile(sampleUnorderedNoWeights, weightsNoWeights, 0.75 - 5 * DBL_EPSILON, eps), 500 + eps, 1e-6);
    }

    Y_UNIT_TEST(TCalculateWeightedTargetQuantileSampleUnrderedNoWeights8) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalculateWeightedTargetQuantile(sampleUnorderedNoWeights, weightsNoWeights, 0.875 - 5 * DBL_EPSILON, eps), 501 + eps, 1e-6);
    }

    Y_UNIT_TEST(TCalculateWeightedTargetQuantileSampleUnrderedNoWeights9) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalculateWeightedTargetQuantile(sampleUnorderedNoWeights, weightsNoWeights, 1 - 5 * DBL_EPSILON, eps), 600 - eps, 1e-6);
    }

    Y_UNIT_TEST(TCalculateWeightedTargetQuantileSampleUnrderedNoWeights10) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalculateWeightedTargetQuantile(sampleUnorderedNoWeights, weightsNoWeights, 5 * DBL_EPSILON, eps), eps, 1e-6);
    }

    Y_UNIT_TEST(TCalculateWeightedTargetQuantileSampleUnrderedNoWeights11) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalculateWeightedTargetQuantile(sampleUnorderedNoWeights, weightsNoWeights, 0.125 + 5 * DBL_EPSILON, eps), 2 - eps, 1e-6);
    }

    Y_UNIT_TEST(TCalculateWeightedTargetQuantileSampleUnrderedNoWeights12) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalculateWeightedTargetQuantile(sampleUnorderedNoWeights, weightsNoWeights, 0.25 + 5 * DBL_EPSILON, eps), 10 - eps, 1e-6);
    }

    Y_UNIT_TEST(TCalculateWeightedTargetQuantileSampleUnrderedNoWeights13) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalculateWeightedTargetQuantile(sampleUnorderedNoWeights, weightsNoWeights, 0.375 + 5 * DBL_EPSILON, eps), 37 - eps, 1e-6);
    }

    Y_UNIT_TEST(TCalculateWeightedTargetQuantileSampleUnrderedNoWeights14) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalculateWeightedTargetQuantile(sampleUnorderedNoWeights, weightsNoWeights, 0.5 + 5 * DBL_EPSILON, eps), 40 - eps, 1e-6);
    }

    Y_UNIT_TEST(TCalculateWeightedTargetQuantileSampleUnrderedNoWeights15) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalculateWeightedTargetQuantile(sampleUnorderedNoWeights, weightsNoWeights, 0.625 + 5 * DBL_EPSILON, eps), 500 - eps, 1e-6);
    }

    Y_UNIT_TEST(TCalculateWeightedTargetQuantileSampleUnrderedNoWeights16) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalculateWeightedTargetQuantile(sampleUnorderedNoWeights, weightsNoWeights, 0.75 + 5 * DBL_EPSILON, eps), 501 - eps, 1e-6);
    }

    Y_UNIT_TEST(TCalculateWeightedTargetQuantileSampleUnrderedNoWeights17) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalculateWeightedTargetQuantile(sampleUnorderedNoWeights, weightsNoWeights, 0.875 + 5 * DBL_EPSILON, eps), 600 - eps, 1e-6);
    }

    Y_UNIT_TEST(TCalculateWeightedTargetQuantileSampleOrderedNoWeightsRepeating1) {
        TVector<float> sample = {0, 1, 1, 1, 2, 2, 2, 2};
        UNIT_ASSERT_DOUBLES_EQUAL(CalculateWeightedTargetQuantile(sample, weightsNoWeights, 0, eps), -eps, 1e-6);
    }

    Y_UNIT_TEST(TCalculateWeightedTargetQuantileSampleOrderedNoWeightsRepeating2) {
        TVector<float> sample = {0, 1, 1, 1, 2, 2, 2, 2};
        UNIT_ASSERT_DOUBLES_EQUAL(CalculateWeightedTargetQuantile(sample, weightsNoWeights, 0.125 - 5 * DBL_EPSILON, eps), eps, 1e-6);
    }

    Y_UNIT_TEST(TCalculateWeightedTargetQuantileSampleOrderedNoWeightsRepeating3) {
        TVector<float> sample = {0, 1, 1, 1, 2, 2, 2, 2};
        UNIT_ASSERT_DOUBLES_EQUAL(CalculateWeightedTargetQuantile(sample, weightsNoWeights, 0.125 + 5 * DBL_EPSILON, eps), 1 - eps, 1e-6);
    }

    Y_UNIT_TEST(TCalculateWeightedTargetQuantileSampleOrderedNoWeightsRepeating4) {
        TVector<float> sample = {0, 1, 1, 1, 2, 2, 2, 2};
        UNIT_ASSERT_DOUBLES_EQUAL(CalculateWeightedTargetQuantile(sample, weightsNoWeights, 0.2 - 5 * DBL_EPSILON, eps), 1 - eps, 1e-6);
    }

    Y_UNIT_TEST(TCalculateWeightedTargetQuantileSampleOrderedNoWeightsRepeating5) {
        TVector<float> sample = {0, 1, 1, 1, 2, 2, 2, 2};
        UNIT_ASSERT_DOUBLES_EQUAL(CalculateWeightedTargetQuantile(sample, weightsNoWeights, 0.2 + 5 * DBL_EPSILON, eps), 1 + eps, 1e-6);
    }

    Y_UNIT_TEST(TCalculateWeightedTargetQuantileSampleOrderedNoWeightsRepeating6) {
        TVector<float> sample = {0, 1, 1, 1, 2, 2, 2, 2};
        UNIT_ASSERT_DOUBLES_EQUAL(CalculateWeightedTargetQuantile(sample, weightsNoWeights, 0.5 - 5 * DBL_EPSILON, eps), 1 + eps , 1e-6);
    }

    Y_UNIT_TEST(TCalculateWeightedTargetQuantileSampleOrderedNoWeightsRepeating7) {
        TVector<float> sample = {0, 1, 1, 1, 2, 2, 2, 2};
        UNIT_ASSERT_DOUBLES_EQUAL(CalculateWeightedTargetQuantile(sample, weightsNoWeights, 0.5 + 5 * DBL_EPSILON, eps), 2 - eps, 1e-6);
    }

    Y_UNIT_TEST(TCalculateWeightedTargetQuantileSampleOrderedWeights1) {
        TVector<float> sample =    {0,     1,      2,      3,      4,      5,      6,      7};
        UNIT_ASSERT_DOUBLES_EQUAL(CalculateWeightedTargetQuantile(sample, weightsHasWeights, 0.38, eps), 3 + eps, 1e-6);
    }

    Y_UNIT_TEST(TCalculateWeightedTargetQuantileSampleOrderedWeights2) {
        TVector<float> sample =    {0,     1,      2,      3,      4,      5,      6,      7};
        UNIT_ASSERT_DOUBLES_EQUAL(CalculateWeightedTargetQuantile(sample, weightsHasWeights, 0.5, eps), 4 - eps, 1e-6);
    }

    Y_UNIT_TEST(TCalculateWeightedTargetQuantileSampleOrderedWeights3) {
        TVector<float> sample =    {0,     1,      2,      3,      4,      5,      6,      7};
        UNIT_ASSERT_DOUBLES_EQUAL(CalculateWeightedTargetQuantile(sample, weightsHasWeights, 0.52, eps), 4 + eps, 1e-6);
    }
}
