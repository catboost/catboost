#include "yeti_random.h"

#include <library/cpp/testing/unittest/registar.h>

Y_UNIT_TEST_SUITE(TMetalDocParallelYetiRandom) {
    Y_UNIT_TEST(TestSymmetricAndGreedySourceOrder) {
        for (bool greedy : {false, true}) for (bool bootstrap : {false, true}) {
            NCB::TMetalYetiRandom actual(239, bootstrap, 3, 4, 8, 4, greedy, 32);
            TRandom oracle(239);
            oracle.NextUniformL(); // TDocParallelBoosting::BaseIterationSeed.
            for (ui32 tree = 0; tree < 4; ++tree) {
                if (!tree && greedy && bootstrap) oracle.Advance(65537);
                UNIT_ASSERT_VALUES_EQUAL(actual.Begin(), oracle.NextUniformL());
                if (!tree && !greedy && bootstrap) oracle.Advance(65537);
                const ui32 attempts = greedy ? 2 + tree : 3;
                oracle.Advance(attempts);
                const auto packet = actual.LeafSeeds(attempts);
                ui32 index = 0;
                // DocParallel estimates every permutation's full walker
                // before starting the following permutation.
                for (ui32 permutation = 0; permutation < 4; ++permutation)
                    for (ui32 evaluation = 0; evaluation < 4; ++evaluation)
                        UNIT_ASSERT_VALUES_EQUAL(packet[index++], oracle.NextUniformL());
                actual.Complete();
            }
        }
    }

    Y_UNIT_TEST(TestGreedyRestoresActualSearchCalls) {
        NCB::TMetalYetiRandom actual(123, true, 2, 12, 8, 3, true, 64);
        const TVector<ui32> depths{2, 4, 3};
        const TVector<ui32> attempts{4, 8, 5};
        for (ui32 search : attempts) {
            actual.Begin(); actual.LeafSeeds(search); actual.Complete();
        }
        const auto state = actual.GetState();
        NCB::TMetalYetiRandom restored(123, true, 2, 12, 8, 3, true, 64);
        restored.Restore(state, depths, attempts);
        UNIT_ASSERT_VALUES_EQUAL(actual.Begin(), restored.Begin());
        NCB::TMetalYetiRandom invalid(123, true, 2, 12, 8, 3, true, 64);
        UNIT_ASSERT_EXCEPTION(invalid.Restore(state, depths, TVector<ui32>{4, 7, 5}), TCatBoostException);
    }

    Y_UNIT_TEST(TestCombinationActualTrialSeedsAndResume) {
        NCB::TMetalYetiRandom actual(9, true, 2, 4, 8, 3, false, 0, 2, true);
        TRandom oracle(9);
        oracle.NextUniformL();
        TVector<ui32> depths;
        for (ui32 tree = 0; tree < 3; ++tree) {
            const auto weak = actual.BeginSeeds();
            for (ui64 seed : weak) UNIT_ASSERT_VALUES_EQUAL(seed, oracle.NextUniformL());
            if (!tree) oracle.Advance(65537);
            oracle.Advance(3);
            actual.BeginLeafCalls(3);
            // Separate permutation walkers can reject different trial counts.
            for (ui32 permutation = 0; permutation < 3; ++permutation)
                for (ui32 evaluation = 0; evaluation < 3 + tree + permutation; ++evaluation)
                    for (ui32 component = 0; component < 2; ++component)
                        UNIT_ASSERT_VALUES_EQUAL(actual.NextLeafSeed(), oracle.NextUniformL());
            actual.Complete();
            depths.push_back(2);
        }
        NCB::TMetalYetiRandom restored(9, true, 2, 4, 8, 3, false, 0, 2, true);
        restored.Restore(actual.GetState(), depths);
        UNIT_ASSERT(actual.BeginSeeds() == restored.BeginSeeds());
    }

    Y_UNIT_TEST(TestCombinationSimpleSkipsLeafOracle) {
        NCB::TMetalYetiRandom actual(99, false, 1, 3, 8, 4, false, 0, 2, true, true);
        TRandom oracle(99);
        oracle.NextUniformL();
        const auto weak = actual.BeginSeeds();
        for (ui64 seed : weak) UNIT_ASSERT_VALUES_EQUAL(seed, oracle.NextUniformL());
        actual.BeginLeafCalls(2); oracle.Advance(2);
        UNIT_ASSERT_EXCEPTION(actual.NextLeafSeed(), TCatBoostException);
        actual.Complete();
        NCB::TMetalYetiRandom restored(99, false, 1, 3, 8, 4, false, 0, 2, true, true);
        restored.Restore(actual.GetState(), TVector<ui32>{1});
        UNIT_ASSERT_VALUES_EQUAL(restored.BeginSeeds().front(), oracle.NextUniformL());
    }

    Y_UNIT_TEST(TestCombinationRejectsIncompleteAndExcessiveCalls) {
        NCB::TMetalYetiRandom actual(1, false, 1, 3, 8, 2, false, 0, 2, true);
        actual.BeginSeeds(); actual.BeginLeafCalls(1);
        for (ui32 i = 0; i < 3; ++i) actual.NextLeafSeed();
        UNIT_ASSERT_EXCEPTION(actual.Complete(), TCatBoostException);
        actual.NextLeafSeed();
        UNIT_ASSERT_EXCEPTION(actual.NextLeafSeed(), TCatBoostException);
        actual.Complete();
        auto state = actual.GetState();
        ++state.DrawCount;
        NCB::TMetalYetiRandom restored(1, false, 1, 3, 8, 2, false, 0, 2, true);
        UNIT_ASSERT_EXCEPTION(restored.Restore(state, TVector<ui32>{0}), TCatBoostException);
    }
}
