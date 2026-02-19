#include <library/cpp/testing/unittest/registar.h>

#include <catboost/libs/metrics/optimal_const_for_loss.h>
#include <catboost/private/libs/options/restrictions.h>

#include <util/generic/array_ref.h>
#include <util/generic/fwd.h>
#include <util/generic/vector.h>

#include <cfloat>

using namespace NCB;

Y_UNIT_TEST_SUITE(TCalculateWeightedTargetDependentQuantile) {

    const TVector<float> weightsNoWeights = {1, 1, 1, 1, 1};
    const TVector<float> weightsHasWeights = {0.1, 0.1, 0.5, 0.5, 0.8};
    const TVector<float> sampleOrderedNoWeights = {1, 2, 3, 4, 5};
    const TVector<float> sampleUnorderedNoWeights = {2, 1, 5, 3, 4};
    const TVector<float> diffs = {0.1, 0.2, 0.3, 0.4, 0.5};
    const TVector<float> diffsUnordered = {0.2, 0.1, 0.5, 0.3, 0.4};


    // no weights, original and new sample are the same, everything is ordered.

    Y_UNIT_TEST(TCalcTargetDependentMinimumAllEqualNoWeightsLowBound) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalcTargetDependentMinimum(sampleOrderedNoWeights, weightsNoWeights, sampleOrderedNoWeights,
            TVector<double>{3, 10}, TVector<double>{0.01, 0.01, 0.01}), 1, 1e-6);
    }

    Y_UNIT_TEST(TCalcTargetDependentMinimumAllEqualNoWeightsHighBound) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalcTargetDependentMinimum(sampleOrderedNoWeights, weightsNoWeights, sampleOrderedNoWeights,
            TVector<double>{3, 10}, TVector<double>{1.0, 1.0, 1.0}), 5 , 1e-6);
    }

    Y_UNIT_TEST(TCalcTargetDependentMinimumAllEqualNoWeightsMid) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalcTargetDependentMinimum(sampleOrderedNoWeights, weightsNoWeights, sampleOrderedNoWeights,
            TVector<double>{3, 10}, TVector<double>{0.5, 0.5, 0.5}), 3 , 1e-6);
    }

    // no weights, original and new sample are the same, check that ordering works

    Y_UNIT_TEST(TCalcTargetDependentMinimumAllEqualNoWeightsDisorderedLowBound) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalcTargetDependentMinimum(sampleUnorderedNoWeights, weightsNoWeights, sampleUnorderedNoWeights,
            TVector<double>{3, 10}, TVector<double>{0.01, 0.01, 0.01}), 1, 1e-6);
    }

    Y_UNIT_TEST(TCalcTargetDependentMinimumAllEqualNoWeightsDisorderedHighBound) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalcTargetDependentMinimum(sampleUnorderedNoWeights, weightsNoWeights, sampleOrderedNoWeights,
            TVector<double>{3, 10}, TVector<double>{1.0, 1.0, 1.0}), 5 , 1e-6);
    }

    Y_UNIT_TEST(TCalcTargetDependentMinimumAllEqualNoWeightsDisorderedMid) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalcTargetDependentMinimum(sampleUnorderedNoWeights, weightsNoWeights, sampleUnorderedNoWeights,
            TVector<double>{3, 10}, TVector<double>{0.5, 0.5, 0.5}), 3 , 1e-6);
    }

    // no weights, original and new sample differ, everything is ordered

    Y_UNIT_TEST(TCalcTargetDependentMinimumAllEqualNoWeightsDiffLowBound) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalcTargetDependentMinimum(diffs, weightsNoWeights, sampleOrderedNoWeights,
            TVector<double>{3, 10}, TVector<double>{0.01, 0.01, 0.01}), 0.1, 1e-6);
    }

    Y_UNIT_TEST(TCalcTargetDependentMinimumAllEqualNoWeightsDiffHighBound) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalcTargetDependentMinimum(diffs, weightsNoWeights, sampleOrderedNoWeights,
            TVector<double>{3, 10}, TVector<double>{1.0, 1.0, 1.0}), 0.5 , 1e-6);
    }

    Y_UNIT_TEST(TCalcTargetDependentMinimumAllEqualNoWeightsDiffMid) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalcTargetDependentMinimum(diffs, weightsNoWeights, sampleOrderedNoWeights,
            TVector<double>{3, 10}, TVector<double>{0.5, 0.5, 0.5}), 0.3 , 1e-6);
    }

    // no weights, original and new sample differ, but things aren't ordered any more (checks that both arrays are sorted together)

    Y_UNIT_TEST(TCalcTargetDependentMinimumAllEqualNoWeightsDiffDisorderedLowBound) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalcTargetDependentMinimum(diffsUnordered, weightsNoWeights, sampleUnorderedNoWeights,
            TVector<double>{3, 10}, TVector<double>{0.01, 0.01, 0.01}), 0.1, 1e-6);
    }

    Y_UNIT_TEST(TCalcTargetDependentMinimumAllEqualNoWeightsDiffDisorderedHighBound) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalcTargetDependentMinimum(diffsUnordered, weightsNoWeights, sampleUnorderedNoWeights,
            TVector<double>{3, 10}, TVector<double>{1.0, 1.0, 1.0}), 0.5 , 1e-6);
    }

    Y_UNIT_TEST(TCalcTargetDependentMinimumAllEqualNoWeightsDiffDisorderedMid) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalcTargetDependentMinimum(diffsUnordered, weightsNoWeights, sampleUnorderedNoWeights,
            TVector<double>{3, 10}, TVector<double>{0.5, 0.5, 0.5}), 0.3 , 1e-6);
    }


    // test multiple quantiles individually in for each window
    Y_UNIT_TEST(TCalcTargetDependentMinimumMultifirstWindow) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalcTargetDependentMinimum(sampleOrderedNoWeights, weightsNoWeights, sampleOrderedNoWeights,
        TVector<double>{10, 20}, TVector<double>{0.5, 0.1, 0.1}), 3 , 1e-6);
    }
    Y_UNIT_TEST(TCalcTargetDependentMinimumMultisecondWindow) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalcTargetDependentMinimum(sampleOrderedNoWeights, weightsNoWeights, sampleOrderedNoWeights,
        TVector<double>{0.1, 10}, TVector<double>{0.1, 0.5, 0.1}), 3 , 1e-6);
    }
    Y_UNIT_TEST(TCalcTargetDependentMinimumMultithirdWindow) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalcTargetDependentMinimum(sampleOrderedNoWeights, weightsNoWeights, sampleOrderedNoWeights,
        TVector<double>{0.1, 0.2}, TVector<double>{0.1, 0.1, 0.5}), 3 , 1e-6);
    }

    // test simple case, but with weights
    Y_UNIT_TEST(TCalcTargetDependentMinimumAllEqualWeights) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalcTargetDependentMinimum(sampleOrderedNoWeights, weightsHasWeights, sampleOrderedNoWeights,
            TVector<double>{3, 10}, TVector<double>{0.5, 0.5, 0.5}), 4 , 1e-6);
    }

    // as above, but unordered (check that weights are ordered correctly)
    Y_UNIT_TEST(TCalcTargetDependentMinimumAllEqualDisorderedWeights) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalcTargetDependentMinimum(sampleUnorderedNoWeights, weightsHasWeights, sampleUnorderedNoWeights,
            TVector<double>{3, 10}, TVector<double>{0.5, 0.5, 0.5}), 4 , 1e-6);
    }

    // test simple case, but there are entries for each different quantile
    Y_UNIT_TEST(TCalcTargetDependentMinimumMultiAllWindows) {
        UNIT_ASSERT_DOUBLES_EQUAL(CalcTargetDependentMinimum(diffs, weightsNoWeights, sampleOrderedNoWeights,
            TVector<double>{1.5, 3.5}, TVector<double>{0.1, 0.5, 0.9}), 0.3, 1e-6);
    }

}
