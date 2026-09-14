#include "langevin_random.h"

#include <library/cpp/testing/unittest/registar.h>
#include <library/cpp/threading/local_executor/local_executor.h>

#include <util/random/fast.h>

#include <cmath>

namespace {
    // Independent transcription of the source contract, deliberately without
    // calling AddLangevinNoiseToDerivatives or StdNormalDistribution. CUDA leaf
    // callbacks use the flat CPU vector helper: blocks of 128 elements, a fresh
    // TFastRng64(seed + block), and uncached polar Box-Muller doubles.
    TVector<double> SourceLeafNoise(ui64 seed, ui32 count, float temperature, float learningRate) {
        TVector<double> result(count, 0.0);
        if (temperature == 0) return result;
        const double coefficient = std::sqrt(2.0 / learningRate / temperature);
        for (ui32 begin = 0; begin < count; begin += 128) {
            TFastRng64 rng(seed + begin / 128);
            for (ui32 i = begin; i < Min(begin + 128, count); ++i) {
                double x, y, radius;
                do {
                    x = rng.GenRandReal1() * 2.0 - 1.0;
                    y = rng.GenRandReal1() * 2.0 - 1.0;
                    radius = x * x + y * y;
                } while (radius > 1.0 || radius <= 0.0);
                result[i] = coefficient * (x * std::sqrt(-2.0 * std::log(radius) / radius));
            }
        }
        return result;
    }

    void AssertNoise(NCB::TMetalLangevinRandom& actual, TRandom& source, uint32_t event,
                     ui32 count, float temperature, float learningRate) {
        TVector<double> result(count);
        UNIT_ASSERT_VALUES_EQUAL(NCB::TMetalLangevinRandom::NoiseCallback(
            &actual, event, count, result.data()), 0);
        const auto expected = SourceLeafNoise(source.NextUniformL(), count, temperature, learningRate);
        UNIT_ASSERT_VALUES_EQUAL(result.size(), expected.size());
        for (ui32 i = 0; i < count; ++i) UNIT_ASSERT_VALUES_EQUAL(result[i], expected[i]);
    }

    void AssertSeed(NCB::TMetalLangevinRandom& actual, TRandom& source, uint32_t event) {
        uint64_t value = 0;
        UNIT_ASSERT_VALUES_EQUAL(NCB::TMetalLangevinRandom::SeedCallback(&actual, event, &value), 0);
        UNIT_ASSERT_VALUES_EQUAL(value, source.NextUniformL());
    }
}

Y_UNIT_TEST_SUITE(TMetalLangevinHostRandom) {
    Y_UNIT_TEST(TestExactDoubleNoiseAtSourceBlockBoundaries) {
        NPar::TLocalExecutor executor;
        executor.RunAdditionalThreads(2);
        for (ui64 seed : {0ull, 239ull, 0xffffffffffffffffull}) {
            for (ui32 count : {1u, 127u, 128u, 129u, 255u, 256u, 257u, 1025u}) {
                NCB::TMetalLangevinRandom actual(seed, true, 1, 271.25f, 0.03125f, &executor);
                TRandom source(seed);
                actual.BeginIteration();
                AssertNoise(actual, source, CBM_LANGEVIN_INITIAL_GRADIENT, count, 271.25f, 0.03125f);
                AssertNoise(actual, source, CBM_LANGEVIN_INITIAL_HESSIAN, count, 271.25f, 0.03125f);
                actual.FinishIteration();
                UNIT_ASSERT_VALUES_EQUAL(actual.GetState().DrawCount, 2);
            }
        }
    }

    Y_UNIT_TEST(TestZeroTemperatureStillAdvancesEveryLeafCallback) {
        NPar::TLocalExecutor executor;
        NCB::TMetalLangevinRandom actual(991, true, 1, 0, 0.03f, &executor);
        TRandom source(991);
        actual.BeginIteration();
        for (uint32_t event : {CBM_LANGEVIN_INITIAL_GRADIENT, CBM_LANGEVIN_INITIAL_HESSIAN,
                              CBM_LANGEVIN_TRIAL_GRADIENT, CBM_LANGEVIN_ACCEPTED_GRADIENT}) {
            AssertNoise(actual, source, event, 257, 0, 0.03f);
        }
        AssertSeed(actual, source, CBM_LANGEVIN_YETI_LEAF);
        actual.FinishIteration();
        UNIT_ASSERT_VALUES_EQUAL(actual.GetState().DrawCount, 5);
    }

    Y_UNIT_TEST(TestWeakCacheInitializationPacketAndIdempotentCallback) {
        NPar::TLocalExecutor executor;
        NCB::TMetalLangevinRandom actual(23, true, 1, 10, 0.03f, &executor);
        TRandom source(23);
        actual.BeginIteration();
        AssertSeed(actual, source, CBM_LANGEVIN_YETI_WEAK);
        TVector<uint64_t> packet;
        UNIT_ASSERT(actual.EnsureWeakSeedCache(&packet));
        source.NextUniformL(); // CreateSeeds allocation seed is not a fill seed.
        UNIT_ASSERT_VALUES_EQUAL(packet.size(), 65536);
        for (const auto seed : packet) UNIT_ASSERT_VALUES_EQUAL(seed, source.NextUniformL());
        uint64_t cacheResult = 99;
        UNIT_ASSERT_VALUES_EQUAL(NCB::TMetalLangevinRandom::SeedCallback(
            &actual, CBM_LANGEVIN_WEAK_SEED_CACHE, &cacheResult), 0);
        UNIT_ASSERT_VALUES_EQUAL(cacheResult, 0);
        UNIT_ASSERT(!actual.EnsureWeakSeedCache(&packet));
        UNIT_ASSERT(packet.empty());
        AssertSeed(actual, source, CBM_LANGEVIN_YETI_WEAK);
        actual.FinishIteration();
        UNIT_ASSERT_VALUES_EQUAL(actual.GetState().DrawCount, 65539);
        UNIT_ASSERT(actual.GetState().WeakSeedCacheInitialized);
    }

    Y_UNIT_TEST(TestInterleavedYetiRejectedAndAcceptedTrialsResumeExactly) {
        NPar::TLocalExecutor executor;
        for (ui32 permutations : {1u, 2u, 3u, 4u, 7u}) {
            NCB::TMetalLangevinRandom actual(99, true, permutations, 500, 0.04f, &executor);
            TRandom source(99);
            ui64 expectedDraws = 0;
            for (ui32 tree = 0; tree < 4; ++tree) {
                const ui32 selected = permutations > 2 ? source.NextUniformL() % (permutations - 2) : 0;
                expectedDraws += permutations > 2;
                UNIT_ASSERT_VALUES_EQUAL(actual.SelectPermutation(), selected);
                AssertSeed(actual, source, CBM_LANGEVIN_YETI_WEAK);
                ++expectedDraws;
                uint64_t unused = 99;
                UNIT_ASSERT_VALUES_EQUAL(NCB::TMetalLangevinRandom::SeedCallback(
                    &actual, CBM_LANGEVIN_WEAK_SEED_CACHE, &unused), 0);
                if (!tree) { source.Advance(65537); expectedDraws += 65537; }
                AssertSeed(actual, source, CBM_LANGEVIN_YETI_WEAK);
                AssertSeed(actual, source, CBM_LANGEVIN_SEARCH);
                actual.AdvanceSearch(2);
                source.Advance(2);
                expectedDraws += 4;

                // A batched oracle evaluates all Yeti tasks, then noises the
                // entire active-task vector. Later accepted trials noise the
                // gradient twice and do not noise their freshly read Hessian.
                for (ui32 evaluation = 0; evaluation < 3; ++evaluation) {
                    AssertSeed(actual, source, CBM_LANGEVIN_YETI_LEAF);
                    AssertSeed(actual, source, CBM_LANGEVIN_YETI_LEAF);
                    expectedDraws += 2;
                    AssertNoise(actual, source, evaluation ? CBM_LANGEVIN_TRIAL_GRADIENT :
                        CBM_LANGEVIN_INITIAL_GRADIENT, 192, 500, 0.04f);
                    ++expectedDraws;
                    if (evaluation == 0 || evaluation == 2) {
                        AssertNoise(actual, source, evaluation ? CBM_LANGEVIN_ACCEPTED_GRADIENT :
                            CBM_LANGEVIN_INITIAL_HESSIAN, 192, 500, 0.04f);
                        ++expectedDraws;
                    }
                }
                actual.FinishIteration();
                const auto state = actual.GetState();
                UNIT_ASSERT_VALUES_EQUAL(state.DrawCount, expectedDraws);
                UNIT_ASSERT_VALUES_EQUAL(state.CompletedIterations, tree + 1);

                NCB::TMetalLangevinRandom resumed(99, true, permutations, 500, 0.04f, &executor);
                resumed.RestoreState(state, tree + 1, expectedDraws);
                auto continuedSource = source;
                resumed.BeginIteration();
                AssertSeed(resumed, continuedSource, CBM_LANGEVIN_YETI_WEAK);
                AssertNoise(resumed, continuedSource, CBM_LANGEVIN_INITIAL_GRADIENT, 257, 500, 0.04f);
                resumed.FinishIteration();
                UNIT_ASSERT_VALUES_EQUAL(resumed.GetState().DrawCount, expectedDraws + 2);
            }
        }
    }

    Y_UNIT_TEST(TestDocParallelConstructorAndAbsoluteIterationChooser) {
        NPar::TLocalExecutor executor;
        NCB::TMetalLangevinRandom actual(123, false, 7, 100, 0.02f, &executor);
        TRandom source(123);
        const ui64 base = source.NextUniformL();
        UNIT_ASSERT_VALUES_EQUAL(actual.GetBaseIterationSeed(), base);
        UNIT_ASSERT_VALUES_EQUAL(actual.GetState().DrawCount, 1);
        for (ui64 iteration : {17ull, 18ull, 19ull}) {
            TRandom chooser(base + iteration);
            chooser.Advance(10);
            UNIT_ASSERT_VALUES_EQUAL(actual.SelectPermutation(iteration), chooser.NextUniformL() % 5);
            AssertNoise(actual, source, CBM_LANGEVIN_INITIAL_GRADIENT, 16, 100, 0.02f);
            AssertNoise(actual, source, CBM_LANGEVIN_INITIAL_HESSIAN, 16, 100, 0.02f);
            actual.FinishIteration();
        }
        UNIT_ASSERT_VALUES_EQUAL(actual.GetState().DrawCount, 7);
        NCB::TMetalLangevinRandom resumed(123, false, 7, 100, 0.02f, &executor);
        resumed.RestoreState(actual.GetState(), 3, 7);
        UNIT_ASSERT_VALUES_EQUAL(resumed.GetBaseIterationSeed(), base);
        resumed.BeginIteration();
        AssertSeed(resumed, source, CBM_LANGEVIN_YETI_LEAF);
        resumed.FinishIteration();
    }

    Y_UNIT_TEST(TestTaskConcatenationDoesNotRestartNoiseAtTaskBoundaries) {
        NPar::TLocalExecutor executor;
        NCB::TMetalLangevinRandom actual(29, true, 1, 100, 0.1f, &executor);
        actual.BeginIteration();
        TVector<double> noise(3 * 64);
        actual.FillLeafNoise(CBM_LANGEVIN_INITIAL_GRADIENT, noise.size(), noise.data());
        TRandom source(29);
        const ui64 seed = source.NextUniformL();
        const auto expected = SourceLeafNoise(seed, noise.size(), 100, 0.1f);
        UNIT_ASSERT(noise == expected);
        const auto independentlyRestartedTask = SourceLeafNoise(seed + 1, 64, 100, 0.1f);
        UNIT_ASSERT(noise[64] != independentlyRestartedTask[0]);
        UNIT_ASSERT_VALUES_EQUAL(noise[128], independentlyRestartedTask[0]);
        actual.FinishIteration();
        UNIT_ASSERT_VALUES_EQUAL(actual.GetState().DrawCount, 1);
    }

    Y_UNIT_TEST(TestInvalidCallbacksDoNotConsumeSeedsOrThrowThroughCAbi) {
        NPar::TLocalExecutor executor;
        NCB::TMetalLangevinRandom actual(9, true, 1, 100, 0.1f, &executor);
        double noise = 42;
        uint64_t seed = 42;
        UNIT_ASSERT(NCB::TMetalLangevinRandom::NoiseCallback(nullptr, 1, 1, &noise));
        UNIT_ASSERT(NCB::TMetalLangevinRandom::SeedCallback(nullptr, 0, &seed));
        UNIT_ASSERT(NCB::TMetalLangevinRandom::NoiseCallback(&actual, 1, 1, &noise));
        UNIT_ASSERT(!actual.GetCallbackError().empty());
        actual.BeginIteration();
        UNIT_ASSERT(NCB::TMetalLangevinRandom::NoiseCallback(&actual, CBM_LANGEVIN_SEARCH, 1, &noise));
        UNIT_ASSERT(NCB::TMetalLangevinRandom::NoiseCallback(&actual, 1, 0, &noise));
        UNIT_ASSERT(NCB::TMetalLangevinRandom::NoiseCallback(&actual, 1, 1, nullptr));
        UNIT_ASSERT(NCB::TMetalLangevinRandom::SeedCallback(&actual, CBM_LANGEVIN_BASE_ITERATION, &seed));
        UNIT_ASSERT(NCB::TMetalLangevinRandom::SeedCallback(&actual, CBM_LANGEVIN_PERMUTATION, &seed));
        UNIT_ASSERT(NCB::TMetalLangevinRandom::SeedCallback(&actual, CBM_LANGEVIN_SEARCH, nullptr));
        UNIT_ASSERT_VALUES_EQUAL(noise, 42);
        UNIT_ASSERT_VALUES_EQUAL(seed, 42);
        actual.FinishIteration();
        UNIT_ASSERT_VALUES_EQUAL(actual.GetState().DrawCount, 0);
    }

    Y_UNIT_TEST(TestSnapshotRequiresCompletedIterationAndExplicitReplayBound) {
        NPar::TLocalExecutor executor;
        NCB::TMetalLangevinRandom actual(77, false, 4, 100, 0.1f, &executor);
        actual.BeginIteration();
        UNIT_ASSERT_EXCEPTION(actual.GetState(), TCatBoostException);
        actual.EnsureWeakSeedCache();
        actual.NextSeed(CBM_LANGEVIN_YETI_LEAF);
        actual.FinishIteration();
        const auto state = actual.GetState();
        NCB::TMetalLangevinRandom resumed(77, false, 4, 100, 0.1f, &executor);
        UNIT_ASSERT_EXCEPTION(resumed.RestoreState(state, 2, state.DrawCount), TCatBoostException);
        UNIT_ASSERT_EXCEPTION(resumed.RestoreState(state, 1, state.DrawCount - 1), TCatBoostException);
        auto invalid = state;
        invalid.DrawCount = 1;
        UNIT_ASSERT_EXCEPTION(resumed.RestoreState(invalid, 1, state.DrawCount), TCatBoostException);
        resumed.RestoreState(state, 1, state.DrawCount);
        UNIT_ASSERT_VALUES_EQUAL(resumed.GetState().DrawCount, state.DrawCount);
        UNIT_ASSERT(resumed.GetState().WeakSeedCacheInitialized);
        UNIT_ASSERT_EXCEPTION(resumed.RestoreState(state, 1, state.DrawCount), TCatBoostException);
    }
}
