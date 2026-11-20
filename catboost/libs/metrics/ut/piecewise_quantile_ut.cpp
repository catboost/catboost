#include <library/cpp/testing/unittest/registar.h>

#include <catboost/libs/metrics/optimal_const_for_loss.h>
#include <catboost/private/libs/options/restrictions.h>

#include <util/generic/array_ref.h>
#include <util/generic/fwd.h>
#include <util/generic/vector.h>

#include <cfloat>

using namespace NCB;

Y_UNIT_TEST_SUITE(TCalculateWeightedPiecewiseQuantile) {

    const TVector<float> weightsNoWeights = {1, 1, 1, 1, 1};
    const TVector<float> weightsHasWeights = {0.1, 0.1, 0.5, 0.5, 0.8};
    const TVector<float> sampleOrderedNoWeights = {1, 2, 3, 4, 5};
    const TVector<float> sampleUnorderedNoWeights = {2, 1, 5, 3, 4};
    const TVector<float> diffs = {0.1, 0.2, 0.3, 0.4, 0.5};
    const TVector<float> diffsUnordered = {0.2, 0.1, 0.5, 0.3, 0.4};


    // no weights, original and new sample are the same, everything is ordered.

    Y_UNIT_TEST(TCalcPiecewiseQuantileMinimumAllEqualNoWeightsLowBound) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalcPiecewiseQuantileMinimum(sampleOrderedNoWeights, weightsNoWeights, sampleOrderedNoWeights,
            TVector<double>{3, 10}, TVector<double>{0.01, 0.01, 0.01}), 1, 1e-6);
    }

    Y_UNIT_TEST(TCalcPiecewiseQuantileMinimumAllEqualNoWeightsHighBound) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalcPiecewiseQuantileMinimum(sampleOrderedNoWeights, weightsNoWeights, sampleOrderedNoWeights,
            TVector<double>{3, 10}, TVector<double>{1.0, 1.0, 1.0}), 5 , 1e-6);
    }

    Y_UNIT_TEST(TCalcPiecewiseQuantileMinimumAllEqualNoWeightsMid) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalcPiecewiseQuantileMinimum(sampleOrderedNoWeights, weightsNoWeights, sampleOrderedNoWeights,
            TVector<double>{3, 10}, TVector<double>{0.5, 0.5, 0.5}), 3 , 1e-6);
    }

    // no weights, original and new sample are the same, check that ordering works

    Y_UNIT_TEST(TCalcPiecewiseQuantileMinimumAllEqualNoWeightsDisorderedLowBound) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalcPiecewiseQuantileMinimum(sampleUnorderedNoWeights, weightsNoWeights, sampleUnorderedNoWeights,
            TVector<double>{3, 10}, TVector<double>{0.01, 0.01, 0.01}), 1, 1e-6);
    }

    Y_UNIT_TEST(TCalcPiecewiseQuantileMinimumAllEqualNoWeightsDisorderedHighBound) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalcPiecewiseQuantileMinimum(sampleUnorderedNoWeights, weightsNoWeights, sampleOrderedNoWeights,
            TVector<double>{3, 10}, TVector<double>{1.0, 1.0, 1.0}), 5 , 1e-6);
    }

    Y_UNIT_TEST(TCalcPiecewiseQuantileMinimumAllEqualNoWeightsDisorderedMid) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalcPiecewiseQuantileMinimum(sampleUnorderedNoWeights, weightsNoWeights, sampleUnorderedNoWeights,
            TVector<double>{3, 10}, TVector<double>{0.5, 0.5, 0.5}), 3 , 1e-6);
    }

    // no weights, original and new sample differ, everything is ordered

    Y_UNIT_TEST(TCalcPiecewiseQuantileMinimumAllEqualNoWeightsDiffLowBound) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalcPiecewiseQuantileMinimum(diffs, weightsNoWeights, sampleOrderedNoWeights,
            TVector<double>{3, 10}, TVector<double>{0.01, 0.01, 0.01}), 0.1, 1e-6);
    }

    Y_UNIT_TEST(TCalcPiecewiseQuantileMinimumAllEqualNoWeightsDiffHighBound) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalcPiecewiseQuantileMinimum(diffs, weightsNoWeights, sampleOrderedNoWeights,
            TVector<double>{3, 10}, TVector<double>{1.0, 1.0, 1.0}), 0.5 , 1e-6);
    }

    Y_UNIT_TEST(TCalcPiecewiseQuantileMinimumAllEqualNoWeightsDiffMid) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalcPiecewiseQuantileMinimum(diffs, weightsNoWeights, sampleOrderedNoWeights,
            TVector<double>{3, 10}, TVector<double>{0.5, 0.5, 0.5}), 0.3 , 1e-6);
    }

    // no weights, original and new sample differ, but things aren't ordered any more (checks that both arrays are sorted together)

    Y_UNIT_TEST(TCalcPiecewiseQuantileMinimumAllEqualNoWeightsDiffDisorderedLowBound) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalcPiecewiseQuantileMinimum(diffsUnordered, weightsNoWeights, sampleUnorderedNoWeights,
            TVector<double>{3, 10}, TVector<double>{0.01, 0.01, 0.01}), 0.1, 1e-6);
    }

    Y_UNIT_TEST(TCalcPiecewiseQuantileMinimumAllEqualNoWeightsDiffDisorderedHighBound) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalcPiecewiseQuantileMinimum(diffsUnordered, weightsNoWeights, sampleUnorderedNoWeights,
            TVector<double>{3, 10}, TVector<double>{1.0, 1.0, 1.0}), 0.5 , 1e-6);
    }

    Y_UNIT_TEST(TCalcPiecewiseQuantileMinimumAllEqualNoWeightsDiffDisorderedMid) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalcPiecewiseQuantileMinimum(diffsUnordered, weightsNoWeights, sampleUnorderedNoWeights,
            TVector<double>{3, 10}, TVector<double>{0.5, 0.5, 0.5}), 0.3 , 1e-6);
    }


    // test multiple quantiles individually in for each window
    Y_UNIT_TEST(TCalcPiecewiseQuantileMinimumMultifirstWindow) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalcPiecewiseQuantileMinimum(sampleOrderedNoWeights, weightsNoWeights, sampleOrderedNoWeights,
        TVector<double>{10, 20}, TVector<double>{0.5, 0.1, 0.1}), 3 , 1e-6);
    }
    Y_UNIT_TEST(TCalcPiecewiseQuantileMinimumMultisecondWindow) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalcPiecewiseQuantileMinimum(sampleOrderedNoWeights, weightsNoWeights, sampleOrderedNoWeights,
        TVector<double>{0.1, 10}, TVector<double>{0.1, 0.5, 0.1}), 3 , 1e-6);
    }
    Y_UNIT_TEST(TCalcPiecewiseQuantileMinimumMultithirdWindow) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalcPiecewiseQuantileMinimum(sampleOrderedNoWeights, weightsNoWeights, sampleOrderedNoWeights,
        TVector<double>{0.1, 0.2}, TVector<double>{0.1, 0.1, 0.5}), 3 , 1e-6);
    }

    // test simple case, but with weights
    Y_UNIT_TEST(TCalcPiecewiseQuantileMinimumAllEqualWeights) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalcPiecewiseQuantileMinimum(sampleOrderedNoWeights, weightsHasWeights, sampleOrderedNoWeights,
            TVector<double>{3, 10}, TVector<double>{0.5, 0.5, 0.5}), 4 , 1e-6);
    }

    // as above, but unordered (check that weights are ordered correctly)
    Y_UNIT_TEST(TCalcPiecewiseQuantileMinimumAllEqualDisorderedWeights) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalcPiecewiseQuantileMinimum(sampleUnorderedNoWeights, weightsHasWeights, sampleUnorderedNoWeights,
            TVector<double>{3, 10}, TVector<double>{0.5, 0.5, 0.5}), 4 , 1e-6);
    }

    // test simple case, but there are entries for each different quantile
    Y_UNIT_TEST(TCalcPiecewiseQuantileMinimumMultiAllWindows) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalcPiecewiseQuantileMinimum(diffs, weightsNoWeights, sampleOrderedNoWeights,
            TVector<double>{1.5, 3.5}, TVector<double>{0.1, 0.5, 0.9}), 0.3, 1e-6);
    }

}
