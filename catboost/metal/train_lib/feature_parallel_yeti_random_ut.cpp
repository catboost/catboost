#include "feature_parallel_yeti_random.h"

#include <library/cpp/testing/unittest/registar.h>

namespace {
    class TCudaFeatureParallelYetiOracle {
    public:
        explicit TCudaFeatureParallelYetiOracle(ui64 seed)
            : Random(seed)
        {}

        ui32 Choose(ui32 permutations) {
            const ui32 learnCount = permutations > 1 ? permutations - 1 : 1;
            return learnCount > 1 ? Next() % (learnCount - 1) : 0;
        }

        TVector<uint64_t> Weak(ui32 selectedFoldCount, bool ordered) {
            TVector<uint64_t> result;
            if (ordered) {
                for (ui32 fold = 0; fold < selectedFoldCount; ++fold) {
                    result.push_back(Next()); // LearnTarget->NewtonAtZero.
                    result.push_back(Next()); // TestTarget->NewtonAtZero, even empty.
                }
            } else {
                result.push_back(Next()); // SingleTaskTarget->NewtonAtZero.
            }
            if (!BootstrapInitialized) {
                Next(); // TGpuAwareRandom mirror seed allocation base.
                for (ui32 thread = 0; thread < 65536; ++thread) Next();
                BootstrapInitialized = true;
            }
            return result;
        }

        ui32 Search(ui32 attempts, bool dependent, ui32 iteration) {
            ui32 draws = 0;
            for (ui32 depth = 0; depth < attempts; ++depth) {
                Next(); // Independent dataset exists even without columns.
                ++draws;
                if (dependent) {
                    Next();
                    ++draws;
                }
                if (depth && (depth + iteration) % 3 != 0) {
                    TRandom localVisitor(Next());
                    localVisitor.NextUniformL(); // Device seed is not shared.
                    ++draws;
                }
            }
            return draws;
        }

        TVector<uint64_t> Leaves(ui32 tasks, ui32 iterations) {
            TVector<uint64_t> result;
            // TNewtonLikeWalker initializes once. With I>1 it evaluates the
            // candidate after each of I moves, even the last unused gradient.
            for (ui32 task = 0; task < tasks; ++task) result.push_back(Next());
            if (iterations != 1) {
                for (ui32 step = 0; step < iterations; ++step)
                    for (ui32 task = 0; task < tasks; ++task) result.push_back(Next());
            }
            return result;
        }

        ui64 DrawCount = 0;

    private:
        ui64 Next() {
            ++DrawCount;
            return Random.NextUniformL();
        }

        TRandom Random;
        bool BootstrapInitialized = false;
    };

    template <class TAccept>
    ui32 WalkerEvaluations(ui32 iterations, TAccept accept) {
        // Host control flow from TNewtonLikeWalker, without target math. A
        // rejection consumes an evaluation; it cannot be replaced by drawing
        // a maximum-size packet and discarding an unused suffix.
        ui32 evaluations = 1;
        if (iterations == 1) return evaluations;
        bool updated = false;
        for (ui32 iteration = 0; iteration < iterations;) {
            for (; iteration < iterations || (!updated && iteration < 100); ++iteration) {
                ++evaluations;
                if (accept(evaluations - 2)) {
                    ++iteration;
                    updated = true;
                    break;
                }
            }
        }
        return evaluations;
    }
}

Y_UNIT_TEST_SUITE(TMetalFeatureParallelYetiRandom) {
    Y_UNIT_TEST(TestSourceCallOrderPlainOrderedAndExactRestoration) {
        for (bool ordered : {false, true}) {
            for (ui32 permutations : {1u, 2u, 3u, 4u, 7u}) {
                for (ui32 iterations : {1u, 3u}) {
                    TVector<ui32> weakCounts(permutations, 1);
                    if (ordered) {
                        for (ui32 p = 0; p < permutations; ++p) weakCounts[p] = 2 * (p + 1);
                    }
                    // Prefix tasks can be omitted for all-singleton groups.
                    // This configured count represents only registered tasks.
                    const ui32 leafTasks = ordered ? 9 : permutations;
                    NCB::TMetalFeatureParallelYetiRandom actual(239, permutations, 4, 8,
                        iterations, weakCounts, leafTasks);
                    TCudaFeatureParallelYetiOracle oracle(239);
                    for (ui32 tree = 0; tree < 15; ++tree) {
                        const ui32 selected = oracle.Choose(permutations);
                        UNIT_ASSERT_VALUES_EQUAL(actual.SelectPermutation(), selected);
                        UNIT_ASSERT(actual.WeakSeeds() == oracle.Weak(selected + 1, ordered));
                        const ui32 attempts = 1 + tree % 4;
                        const ui32 searchDraws = oracle.Search(attempts, permutations > 1, tree);
                        UNIT_ASSERT(actual.LeafSeeds(searchDraws) == oracle.Leaves(leafTasks, iterations));
                        actual.Complete();
                        const auto state = actual.GetState();
                        UNIT_ASSERT_VALUES_EQUAL(state.DrawCount, oracle.DrawCount);
                        UNIT_ASSERT_VALUES_EQUAL(state.CompletedIterations, tree + 1);
                        UNIT_ASSERT(state.BootstrapInitialized);

                        NCB::TMetalFeatureParallelYetiRandom restored(239, permutations, 4, 8,
                            iterations, weakCounts, leafTasks);
                        restored.Restore(state, tree + 1);
                        auto continuedOracle = oracle;
                        const ui32 next = continuedOracle.Choose(permutations);
                        UNIT_ASSERT_VALUES_EQUAL(restored.SelectPermutation(), next);
                        UNIT_ASSERT(restored.WeakSeeds() == continuedOracle.Weak(next + 1, ordered));
                        const ui32 nextSearchDraws = continuedOracle.Search(2, permutations > 1, tree + 1);
                        UNIT_ASSERT(restored.LeafSeeds(nextSearchDraws) == continuedOracle.Leaves(leafTasks, iterations));
                        restored.Complete();
                        UNIT_ASSERT_VALUES_EQUAL(restored.GetState().DrawCount, continuedOracle.DrawCount);
                    }
                }
            }
        }
    }

    Y_UNIT_TEST(TestPlainSinglePermutationHasOneLeafTaskAndNoBaseSeedDraw) {
        const TVector<ui32> weakCounts{1};
        NCB::TMetalFeatureParallelYetiRandom actual(17, 1, 0, 0, 1, weakCounts, 1);
        TRandom source(17);
        UNIT_ASSERT_VALUES_EQUAL(actual.SelectPermutation(), 0);
        const auto weak = actual.WeakSeeds();
        UNIT_ASSERT_VALUES_EQUAL(weak.size(), 1);
        UNIT_ASSERT_VALUES_EQUAL(weak[0], source.NextUniformL());
        source.Advance(65537);
        const auto leaf = actual.LeafSeeds(0);
        UNIT_ASSERT_VALUES_EQUAL(leaf.size(), 1);
        UNIT_ASSERT_VALUES_EQUAL(leaf[0], source.NextUniformL());
        actual.Complete();
        UNIT_ASSERT_VALUES_EQUAL(actual.GetState().DrawCount, 65539);
    }

    Y_UNIT_TEST(TestAbsentCandidatesStillComputesWeakAndLeafTargets) {
        const TVector<ui32> weakCounts{2, 4, 6, 8};
        for (ui32 depth : {0u, 4u}) {
            NCB::TMetalFeatureParallelYetiRandom actual(17, 4, depth, 0, 2, weakCounts, 3);
            TCudaFeatureParallelYetiOracle oracle(17);
            for (ui32 tree = 0; tree < 3; ++tree) {
                const ui32 selected = oracle.Choose(4);
                UNIT_ASSERT_VALUES_EQUAL(actual.SelectPermutation(), selected);
                UNIT_ASSERT(actual.WeakSeeds() == oracle.Weak(selected + 1, true));
                UNIT_ASSERT(actual.LeafSeeds(0) == oracle.Leaves(3, 2));
                actual.Complete();
                UNIT_ASSERT_VALUES_EQUAL(actual.GetState().DrawCount, oracle.DrawCount);
            }
        }
    }

    Y_UNIT_TEST(TestInvalidCallSequenceAndSearchCountsDoNotAdvance) {
        const TVector<ui32> weakCounts{2};
        NCB::TMetalFeatureParallelYetiRandom actual(17, 1, 4, 8, 1, weakCounts, 1);
        TCudaFeatureParallelYetiOracle oracle(17);
        UNIT_ASSERT_EXCEPTION(actual.WeakSeeds(), TCatBoostException);
        UNIT_ASSERT_EXCEPTION(actual.LeafSeeds(1), TCatBoostException);
        UNIT_ASSERT_EXCEPTION(actual.Complete(), TCatBoostException);
        UNIT_ASSERT_VALUES_EQUAL(actual.GetState().DrawCount, 0);
        UNIT_ASSERT_VALUES_EQUAL(actual.SelectPermutation(), oracle.Choose(1));
        UNIT_ASSERT_EXCEPTION(actual.SelectPermutation(), TCatBoostException);
        UNIT_ASSERT_EXCEPTION(actual.GetState(), TCatBoostException);
        UNIT_ASSERT(actual.WeakSeeds() == oracle.Weak(1, true));
        UNIT_ASSERT_EXCEPTION(actual.WeakSeeds(), TCatBoostException);
        UNIT_ASSERT_EXCEPTION(actual.LeafSeeds(0), TCatBoostException);
        UNIT_ASSERT_EXCEPTION(actual.LeafSeeds(13), TCatBoostException);
        oracle.Search(1, false, 0);
        UNIT_ASSERT(actual.LeafSeeds(1) == oracle.Leaves(1, 1));
        UNIT_ASSERT_EXCEPTION(actual.LeafSeeds(1), TCatBoostException);
        actual.Complete();
        UNIT_ASSERT_VALUES_EQUAL(actual.GetState().DrawCount, oracle.DrawCount);
    }

    Y_UNIT_TEST(TestSnapshotRejectsBoundsAndMetadataCorruption) {
        const TVector<ui32> weakCounts{2};
        NCB::TMetalFeatureParallelYetiRandom source(17, 1, 4, 8, 1, weakCounts, 1);
        source.SelectPermutation();
        source.WeakSeeds();
        source.LeafSeeds(3);
        source.Complete();
        const auto valid = source.GetState();
        for (ui32 kind = 0; kind < 4; ++kind) {
            auto corrupt = valid;
            if (kind == 0) corrupt.DrawCount = 65537 + 2 + 1;
            if (kind == 1) corrupt.DrawCount = 65537 + 2 + 1 + 13;
            if (kind == 2) corrupt.CompletedIterations = 2;
            if (kind == 3) corrupt.BootstrapInitialized = false;
            NCB::TMetalFeatureParallelYetiRandom restored(17, 1, 4, 8, 1, weakCounts, 1);
            UNIT_ASSERT_EXCEPTION(restored.Restore(corrupt, 1), TCatBoostException);
            restored.Restore(valid, 1);
            UNIT_ASSERT_VALUES_EQUAL(restored.GetState().DrawCount, valid.DrawCount);
        }
    }

    Y_UNIT_TEST(TestVariableTrialsConsumeOnlyExecutedTaskComponentCallsAndResume) {
        const TVector<ui32> weakCounts{2, 2, 2, 2};
        // Three leaf tasks with two Yeti components each. Runtime callback
        // order is evaluation, task, component; no unused seeds are drawn.
        constexpr ui32 groups = 6;
        for (ui32 iterations : {1u, 3u, 101u}) {
            NCB::TMetalFeatureParallelYetiRandom actual(239, 4, 4, 8,
                iterations, weakCounts, groups, true);
            TCudaFeatureParallelYetiOracle oracle(239);
            for (ui32 tree = 0; tree < 6; ++tree) {
                UNIT_ASSERT_VALUES_EQUAL(actual.SelectPermutation(), oracle.Choose(4));
                UNIT_ASSERT(actual.WeakSeeds() == oracle.Weak(1, true));
                const ui32 searchDraws = oracle.Search(1 + tree % 4, true, tree);
                actual.BeginLeafCalls(searchDraws);
                const ui32 evaluations = WalkerEvaluations(iterations, [&](ui32 trial) {
                    if (tree % 3 == 0) return true;
                    if (tree % 3 == 1) return trial >= iterations + 5;
                    return false;
                });
                for (ui32 evaluation = 0; evaluation < evaluations; ++evaluation) {
                    const auto expected = oracle.Leaves(groups, 1);
                    for (ui32 task = 0; task < 3; ++task)
                        for (ui32 component = 0; component < 2; ++component)
                            UNIT_ASSERT_VALUES_EQUAL(actual.NextLeafSeed(), expected[2 * task + component]);
                }
                actual.Complete();
                const auto state = actual.GetState();
                UNIT_ASSERT_VALUES_EQUAL(state.DrawCount, oracle.DrawCount);

                NCB::TMetalFeatureParallelYetiRandom restored(239, 4, 4, 8,
                    iterations, weakCounts, groups, true);
                restored.Restore(state, tree + 1);
                auto expectedContinuation = oracle;
                UNIT_ASSERT_VALUES_EQUAL(restored.SelectPermutation(), expectedContinuation.Choose(4));
                UNIT_ASSERT(restored.WeakSeeds() == expectedContinuation.Weak(1, true));
                const ui32 nextSearchDraws = expectedContinuation.Search(1, true, 0);
                restored.BeginLeafCalls(nextSearchDraws);
                const auto expectedFirst = expectedContinuation.Leaves(groups, 1);
                for (uint64_t seed : expectedFirst) UNIT_ASSERT_VALUES_EQUAL(restored.NextLeafSeed(), seed);
            }
        }
    }

    Y_UNIT_TEST(TestVariableCallsRejectPartialGroupsAndExceededWalkerBudget) {
        const TVector<ui32> weakCounts{2};
        NCB::TMetalFeatureParallelYetiRandom actual(17, 1, 4, 8, 2, weakCounts, 2, true);
        UNIT_ASSERT_EXCEPTION(actual.BeginLeafCalls(1), TCatBoostException);
        UNIT_ASSERT_EXCEPTION(actual.NextLeafSeed(), TCatBoostException);
        actual.SelectPermutation();
        actual.WeakSeeds();
        UNIT_ASSERT_EXCEPTION(actual.LeafSeeds(1), TCatBoostException);
        UNIT_ASSERT_EXCEPTION(actual.BeginLeafCalls(0), TCatBoostException);
        actual.BeginLeafCalls(1);
        for (ui32 seed = 0; seed < 5; ++seed) actual.NextLeafSeed();
        UNIT_ASSERT_EXCEPTION(actual.Complete(), TCatBoostException);
        actual.NextLeafSeed(); // Minimum three complete evaluations.
        actual.NextLeafSeed(); // A fourth evaluation has only half its group.
        UNIT_ASSERT_EXCEPTION(actual.Complete(), TCatBoostException);
        actual.NextLeafSeed();
        actual.Complete();
        UNIT_ASSERT_VALUES_EQUAL(actual.GetState().DrawCount, 65537 + 2 + 1 + 8);

        // I2 can reject 100 trial points, plus the initial evaluation, but no
        // additional callback may consume RNG beyond that exact source bound.
        actual.SelectPermutation();
        actual.WeakSeeds();
        actual.BeginLeafCalls(1);
        for (ui32 seed = 0; seed < 2 * 101; ++seed) actual.NextLeafSeed();
        UNIT_ASSERT_EXCEPTION(actual.NextLeafSeed(), TCatBoostException);
        actual.Complete();
        UNIT_ASSERT_VALUES_EQUAL(actual.GetState().CompletedIterations, 2);
    }

    Y_UNIT_TEST(TestVariableSnapshotDrawBoundsRejectCorruptCounts) {
        const TVector<ui32> weakCounts{2};
        NCB::TMetalFeatureParallelYetiRandom source(17, 1, 4, 8, 2, weakCounts, 2, true);
        source.SelectPermutation();
        source.WeakSeeds();
        source.BeginLeafCalls(1);
        for (ui32 seed = 0; seed < 2 * 101; ++seed) source.NextLeafSeed();
        source.Complete();
        const auto valid = source.GetState();
        // The fixed pure-Yeti protocol cannot accept these additional trials.
        NCB::TMetalFeatureParallelYetiRandom fixed(17, 1, 4, 8, 2, weakCounts, 2);
        UNIT_ASSERT_EXCEPTION(fixed.Restore(valid, 1), TCatBoostException);
        for (ui64 badDrawCount : {ui64(65537 + 2 + 1 + 5), ui64(65537 + 2 + 12 + 203)}) {
            auto corrupt = valid;
            corrupt.DrawCount = badDrawCount;
            NCB::TMetalFeatureParallelYetiRandom restored(17, 1, 4, 8, 2, weakCounts, 2, true);
            UNIT_ASSERT_EXCEPTION(restored.Restore(corrupt, 1), TCatBoostException);
            restored.Restore(valid, 1);
            UNIT_ASSERT_VALUES_EQUAL(restored.GetState().DrawCount, valid.DrawCount);
        }
    }
}
