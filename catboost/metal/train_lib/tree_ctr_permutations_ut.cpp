#include "tree_ctr_permutations.h"

#include <catboost/cuda/data/data_utils.h>
#include <library/cpp/testing/unittest/registar.h>

#include <limits>

Y_UNIT_TEST_SUITE(TMetalFeatureParallelPermutations) {
    Y_UNIT_TEST(TestBlockPolicyThresholdsAndCeilLog) {
        for (ui32 configured : {0u, 1u, 3u, 64u, 65u, 1024u})
            UNIT_ASSERT_VALUES_EQUAL(NCB::GetMetalFeatureParallelBlockSize(49999, configured), 1);
        UNIT_ASSERT_VALUES_EQUAL(NCB::GetMetalFeatureParallelBlockSize(50000, 0), 64);
        UNIT_ASSERT_VALUES_EQUAL(NCB::GetMetalFeatureParallelBlockSize(50000, 1), 1);
        UNIT_ASSERT_VALUES_EQUAL(NCB::GetMetalFeatureParallelBlockSize(50000, 3), 4);
        UNIT_ASSERT_VALUES_EQUAL(NCB::GetMetalFeatureParallelBlockSize(50000, 65), 128);
        UNIT_ASSERT_VALUES_EQUAL(NCB::GetMetalFeatureParallelBlockSize(50000, 1024), 256);
        UNIT_ASSERT_VALUES_EQUAL(NCB::GetMetalFeatureParallelBlockSize(65535, 1024), 256);
        UNIT_ASSERT_VALUES_EQUAL(NCB::GetMetalFeatureParallelBlockSize(65536, 1024), 512);
        UNIT_ASSERT_VALUES_EQUAL(NCB::GetMetalFeatureParallelBlockSize(1u << 24, std::numeric_limits<ui32>::max()), 131072);
    }

    Y_UNIT_TEST(TestOrdersMatchActualCudaShuffle) {
        // Compare to the original CUDA host implementation, including ragged
        // blocks and overflow of its uint32 permutation seed expression.
        for (ui32 rows : {1u, 2u, 17u, 259u, 50003u, 65539u}) {
            for (ui32 block : {1u, 3u, 64u, 128u}) {
                for (ui32 permutation : {1u, 2u, 3u, 2581u, std::numeric_limits<ui32>::max()}) {
                    TVector<ui32> expected;
                    const ui32 seed = 1664525u * permutation + 1013904223u + block;
                    NCatboostCuda::Shuffle(seed, block, rows, &expected);
                    UNIT_ASSERT(NCB::MakeMetalFeatureParallelBlockOrder(rows, permutation, block) == expected);
                }
                const auto identity = NCB::MakeMetalFeatureParallelBlockOrder(rows, 0, block);
                for (ui32 row = 0; row < rows; ++row) UNIT_ASSERT_VALUES_EQUAL(identity[row], row);
            }
        }
    }

    Y_UNIT_TEST(TestGroupsMatchActualCudaAndPreserveMembers) {
        const TVector<TGroupBounds> groups = {{0, 3}, {3, 4}, {4, 9}, {9, 11}, {11, 18}, {18, 20}};
        for (ui32 block : {1u, 2u, 4u, 64u}) {
            for (ui32 permutation : {1u, 2u, 3u, 2581u}) {
                TVector<ui32> expected;
                const ui32 seed = 1664525u * permutation + 1013904223u + block;
                NCatboostCuda::GenerateQueryDocsOrder(seed, block, groups, &expected);
                UNIT_ASSERT(NCB::MakeMetalFeatureParallelGroupedHistoryOrder(20, groups, permutation, block) == expected);
            }
        }
    }

    Y_UNIT_TEST(TestChooserConsumesSharedStreamExactly) {
        for (ui32 count : {1u, 2u, 3u, 4u, 5u, 9u}) {
            TRandom actual(239), expected(239);
            for (ui32 iteration = 0; iteration < 40; ++iteration) {
                const ui32 learnCount = count > 1 ? count - 1 : 1;
                const ui32 selected = learnCount > 1 ? expected.NextUniformL() % (learnCount - 1) : 0;
                UNIT_ASSERT_VALUES_EQUAL(NCB::ChooseMetalFeatureParallelPermutation(count, actual), selected);
                // Other host consumers must remain interleaved. P3 consumes a
                // draw even though modulo one always returns permutation zero.
                for (ui32 draw = 0; draw < iteration % 5; ++draw)
                    UNIT_ASSERT_VALUES_EQUAL(actual.NextUniformL(), expected.NextUniformL());
            }
            UNIT_ASSERT_VALUES_EQUAL(actual.NextUniformL(), expected.NextUniformL());
        }
    }

    Y_UNIT_TEST(TestRejectsInvalidCountsAndGroups) {
        TRandom random(0);
        UNIT_ASSERT_EXCEPTION(NCB::ChooseMetalFeatureParallelPermutation(0, random), TCatBoostException);
        UNIT_ASSERT_EXCEPTION(NCB::MakeMetalFeatureParallelBlockOrder(10, 1, 0), TCatBoostException);
        const TVector<TGroupBounds> invalid = {{0, 11}};
        UNIT_ASSERT_EXCEPTION(NCB::MakeMetalFeatureParallelGroupedHistoryOrder(10, invalid, 1, 1), TCatBoostException);
    }
}
