#include "ordered_random.h"

#include <library/cpp/testing/unittest/registar.h>

namespace {
    // Replay CUDA's host call sites, independently of the production helper:
    // dynamic_boosting.h chooses from P-2; TGpuAwareRandom initializes mirror
    // bootstrap seeds once; oblivious_tree_structure_searcher.cpp draws each
    // compressed-dataset seed, then one seed for the complete tree-CTR visitor.
    // Its nested seeds deliberately use a separate stream as in CUDA.
    class TCudaHostDrawOracle {
    public:
        explicit TCudaHostDrawOracle(ui64 seed)
            : Random(seed)
        {}

        ui32 Choose(ui32 permutations) {
            const ui32 learnCount = permutations > 1 ? permutations - 1 : 1;
            return learnCount > 1 ? Next() % (learnCount - 1) : 0;
        }

        void Bootstrap() {
            if (!Initialized) {
                for (ui32 i = 0; i < 65537; ++i) Next();
                Initialized = true;
            }
        }

        void Attempt(bool dependentSimple, bool treeCtrs, ui32 packCount) {
            TRandom independent(Next());
            independent.NextUniformL();
            if (dependentSimple) {
                TRandom dependent(Next());
                dependent.NextUniformL();
            }
            if (treeCtrs) {
                TRandom devices(Next());
                const ui64 deviceSeed = devices.NextUniformL();
                for (ui32 pack = 0; pack < packCount; ++pack) {
                    TRandom tensor(deviceSeed + pack);
                    tensor.NextUniformL();
                }
            }
        }

        ui64 Draws = 0;

    private:
        ui64 Next() {
            ++Draws;
            return Random.NextUniformL();
        }

        TRandom Random;
        bool Initialized = false;
    };
}

Y_UNIT_TEST_SUITE(TMetalFeatureParallelTreeCtrRandom) {
    Y_UNIT_TEST(TestDynamicMixedScorersFollowCudaHostStreamAndResume) {
        for (ui32 permutations : {1u, 2u, 3u, 4u, 7u}) {
            for (bool hasDependentCtr : {false, true}) {
                constexpr ui32 depth = 4;
                NCB::TMetalOrderedRandom actual(239, permutations, depth, 8, true);
                TCudaHostDrawOracle oracle(239);
                for (ui32 iteration = 0; iteration < 25; ++iteration) {
                    UNIT_ASSERT_VALUES_EQUAL(actual.SelectPermutation(), oracle.Choose(permutations));
                    oracle.Bootstrap();
                    // Includes duplicate-winner early termination and a full
                    // tree. Numeric, FeatureFreq-only and target-CTR mixtures
                    // have different static dataset counts, including P1's
                    // collapse of all simple columns into one dataset.
                    const ui32 actualDepth = iteration % (depth + 1);
                    const ui32 attempts = actualDepth == depth ? depth : actualDepth + 1;
                    ui32 draws = 0;
                    for (ui32 attempt = 0; attempt < attempts; ++attempt) {
                        const bool dependent = permutations > 1 && hasDependentCtr;
                        // Replacing pure-tree packs can deactivate all dynamic
                        // tensors. Candidate count and number of packs do not
                        // change the one host draw for an active visitor.
                        const bool dynamic = attempt != 0 && (attempt + iteration) % 3 != 0;
                        oracle.Attempt(dependent, dynamic, 1 + iteration % 5);
                        draws += 1 + dependent + dynamic;
                    }
                    actual.FinishIterationWithDraws(actualDepth, draws);
                    const auto state = actual.GetState();
                    UNIT_ASSERT_VALUES_EQUAL(state.DrawCount, oracle.Draws);
                    UNIT_ASSERT_VALUES_EQUAL(state.CompletedIterations, iteration + 1);
                    UNIT_ASSERT(state.BootstrapInitialized);

                    NCB::TMetalOrderedRandom restored(239, permutations, depth, 8, true);
                    restored.RestoreState(state, iteration + 1);
                    // Resume is checked against a separately advanced CUDA
                    // oracle, without asking the original helper for an answer.
                    auto continuedOracle = oracle;
                    UNIT_ASSERT_VALUES_EQUAL(restored.SelectPermutation(), continuedOracle.Choose(permutations));
                }
            }
        }
    }

    Y_UNIT_TEST(TestExistingSimpleModeRetainsItsExactDrawSchedule) {
        for (ui32 permutations : {1u, 3u, 7u}) {
            for (ui32 candidateCount : {0u, 8u}) {
                NCB::TMetalOrderedRandom actual(17, permutations, 4, candidateCount);
                TCudaHostDrawOracle oracle(17);
                for (ui32 iteration = 0; iteration < 15; ++iteration) {
                    UNIT_ASSERT_VALUES_EQUAL(actual.SelectPermutation(), oracle.Choose(permutations));
                    oracle.Bootstrap();
                    const ui32 actualDepth = candidateCount ? iteration % 5 : 0;
                    const ui32 attempts = candidateCount ? (actualDepth == 4 ? 4 : actualDepth + 1) : 0;
                    for (ui32 attempt = 0; attempt < attempts; ++attempt) oracle.Attempt(false, false, 0);
                    actual.FinishIteration(actualDepth);
                    UNIT_ASSERT_VALUES_EQUAL(actual.GetState().DrawCount, oracle.Draws);
                }
            }
        }
    }

    Y_UNIT_TEST(TestDynamicDepthZeroAndAbsentCandidates) {
        for (ui32 depth : {0u, 4u}) {
            NCB::TMetalOrderedRandom actual(17, 7, depth, 0, true);
            TCudaHostDrawOracle oracle(17);
            for (ui32 iteration = 0; iteration < 3; ++iteration) {
                UNIT_ASSERT_VALUES_EQUAL(actual.SelectPermutation(), oracle.Choose(7));
                oracle.Bootstrap();
                actual.FinishIterationWithDraws(0, 0);
                UNIT_ASSERT_VALUES_EQUAL(actual.GetState().DrawCount, oracle.Draws);
            }
        }
    }

    Y_UNIT_TEST(TestRejectsWrongModeAndImpossibleDrawCountsWithoutFinishing) {
        NCB::TMetalOrderedRandom simple(17, 7, 4, 8);
        simple.SelectPermutation();
        UNIT_ASSERT_EXCEPTION(simple.FinishIterationWithDraws(1, 4), TCatBoostException);
        simple.FinishIteration(1);

        NCB::TMetalOrderedRandom dynamic(17, 7, 4, 8, true);
        UNIT_ASSERT_EXCEPTION(dynamic.FinishIterationWithDraws(1, 4), TCatBoostException);
        dynamic.SelectPermutation();
        UNIT_ASSERT_EXCEPTION(dynamic.FinishIteration(1), TCatBoostException);
        UNIT_ASSERT_EXCEPTION(dynamic.FinishIterationWithDraws(1, 1), TCatBoostException);
        UNIT_ASSERT_EXCEPTION(dynamic.FinishIterationWithDraws(1, 7), TCatBoostException);
        UNIT_ASSERT_EXCEPTION(dynamic.GetState(), TCatBoostException);
        dynamic.FinishIterationWithDraws(1, 4);
        UNIT_ASSERT_VALUES_EQUAL(dynamic.GetState().CompletedIterations, 1);
    }

    Y_UNIT_TEST(TestRestoreAcceptsDynamicBoundsAndRejectsCorruption) {
        NCB::TMetalOrderedRandom source(17, 7, 4, 8, true);
        source.SelectPermutation();
        source.FinishIterationWithDraws(4, 11);
        const auto valid = source.GetState();
        NCB::TMetalOrderedRandom legacy(17, 7, 4, 8);
        UNIT_ASSERT_EXCEPTION(legacy.RestoreState(valid, 1), TCatBoostException);

        for (ui32 kind = 0; kind < 4; ++kind) {
            auto corrupt = valid;
            if (kind == 0) corrupt.DrawCount = 65537 + 1;
            if (kind == 1) corrupt.DrawCount = 65537 + 1 + 13;
            if (kind == 2) corrupt.BootstrapInitialized = false;
            if (kind == 3) corrupt.CompletedIterations = 2;
            NCB::TMetalOrderedRandom restored(17, 7, 4, 8, true);
            UNIT_ASSERT_EXCEPTION(restored.RestoreState(corrupt, 1), TCatBoostException);
            restored.RestoreState(valid, 1);
            UNIT_ASSERT_VALUES_EQUAL(restored.GetState().DrawCount, valid.DrawCount);
        }
    }
}
