#include <catboost/libs/data/weights.h>

#include <catboost/libs/helpers/vector_helpers.h>

#include <util/generic/xrange.h>

#include <library/cpp/testing/unittest/registar.h>


using namespace NCB;


Y_UNIT_TEST_SUITE(TWeights) {
    Y_UNIT_TEST(Test) {
        // trivial
        {
            TWeights<float> weights(10);

            UNIT_ASSERT_VALUES_EQUAL(weights.GetSize(), 10);
            UNIT_ASSERT(weights.IsTrivial());

            for (auto i : xrange(10)) {
                UNIT_ASSERT_VALUES_EQUAL(weights[i], 1.f);
            }

            UNIT_ASSERT_EXCEPTION(weights.GetNonTrivialData(), TCatBoostException);
        }

        // non-trivial
        {
            TVector<float> weightsData = {0.0f, 1.f, 2.0f};
            TWeights<float> weights{TVector<float>(weightsData)};

            UNIT_ASSERT_VALUES_EQUAL(weights.GetSize(), 3);
            UNIT_ASSERT(!weights.IsTrivial());

            for (auto i : xrange(weightsData.size())) {
                UNIT_ASSERT_VALUES_EQUAL(weights[i], weightsData[i]);
            }

            UNIT_ASSERT(Equal(weights.GetNonTrivialData(), weightsData));
        }

        // bad data

        UNIT_ASSERT_EXCEPTION(
            TWeights<float>(10, TMaybeOwningArrayHolder<float>::CreateOwning(TVector<float>{0.0f, 1.0f})),
            TCatBoostException
        );

        UNIT_ASSERT_EXCEPTION(
            TWeights<float>(TVector<float>{0.0f, -2.0f}),
            TCatBoostException
        );
    }

    Y_UNIT_TEST(Subset) {
        {
            TArraySubsetIndexing<ui32> subset( TIndexedSubset<ui32>{4, 2, 3} );

            TWeights<float> weights(10);

            NPar::TLocalExecutor localExecutor;
            localExecutor.RunAdditionalThreads(2);

            TWeights<float> subsetWeights = weights.GetSubset(subset, &localExecutor);

            UNIT_ASSERT_VALUES_EQUAL(subsetWeights.GetSize(), 3);
            UNIT_ASSERT(subsetWeights.IsTrivial());

            for (auto i : xrange(3)) {
                UNIT_ASSERT_VALUES_EQUAL(subsetWeights[i], 1.f);
            }
        }

        {
            TArraySubsetIndexing<ui32> subset( TIndexedSubset<ui32>{4, 2, 3} );

            TVector<float> weightsData = {0.0f, 1.0f, 2.0f, 0.25f, 7.0f, 0.12f};
            TVector<float> expectedSubsetWeightsData = {7.0f, 2.0f, 0.25f};

            TWeights<float> weights{TVector<float>(weightsData)};

            NPar::TLocalExecutor localExecutor;
            localExecutor.RunAdditionalThreads(2);

            TWeights<float> subsetWeights = weights.GetSubset(subset, &localExecutor);

            UNIT_ASSERT_VALUES_EQUAL(subsetWeights.GetSize(), 3);
            UNIT_ASSERT(!subsetWeights.IsTrivial());

            for (auto i : xrange(3)) {
                UNIT_ASSERT_VALUES_EQUAL(subsetWeights[i], expectedSubsetWeightsData[i]);
            }
        }
    }
}
