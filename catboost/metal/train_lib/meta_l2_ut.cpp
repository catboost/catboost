#include "meta_l2.h"
#include "meta_l2_random.h"
#include "ordered_random.h"
#include "feature_parallel_yeti_random.h"
#include "yeti_random.h"

#include <library/cpp/testing/unittest/registar.h>

#include <limits>

namespace {
    double CudaUniform(ui64 seed) {
        // Independent literal host launch arithmetic from random_gen.cuh.
        const ui32 high = 36969u * (ui32(seed >> 32) % 65536u) + ui32(seed >> 48);
        const ui32 low = 18000u * (ui32(seed) % 65536u) + (ui32(seed) >> 16);
        return ui32(ui64(high) * 65536u + low) * 2.328306435996595e-10;
    }
}

Y_UNIT_TEST_SUITE(TMetalMetaL2) {
    Y_UNIT_TEST(TestPolicySeedsFollowPresentGridOrder) {
        for (ui64 seed : {0ull, 731ull, 0xffffffffffffffffull}) {
            for (ui8 mask = 1; mask <= 7; ++mask) {
                TRandom source(seed);
                const auto result = NCB::MetalMetaL2PolicyExponents(seed, mask, 0.6, 0.43);
                for (ui32 policy = 0; policy < 3; ++policy) {
                    const float expected = !(mask & (1u << policy)) ? 1.0f
                        : CudaUniform(source.NextUniformL()) >= 0.43 ? 1.0f : 0.6f;
                    UNIT_ASSERT_VALUES_EQUAL(result[policy], expected);
                }
            }
        }
    }

    Y_UNIT_TEST(TestIndependentDatasetsAndDeclaredEmptyPolicies) {
        using namespace NCB;
        TVector<TMetalMetaL2DataSet> datasets{
            {18, 7, {{0, EMetalMetaL2Policy::OneByte}, {2, EMetalMetaL2Policy::HalfByte}}},
            {751, 4, {{1, EMetalMetaL2Policy::OneByte}}},
        };
        const auto result = BuildMetalMetaL2FeatureExponents(4, -0.5, 0.71, datasets);
        TRandom staticSource(18), dependentSource(751);
        staticSource.NextUniformL(); // Declared Binary policy has no native column.
        const auto half = staticSource.NextUniformL(), byte = staticSource.NextUniformL();
        UNIT_ASSERT_VALUES_EQUAL(result[0], CudaUniform(byte) >= 0.71 ? 1.0f : -0.5f);
        UNIT_ASSERT_VALUES_EQUAL(result[2], CudaUniform(half) >= 0.71 ? 1.0f : -0.5f);
        UNIT_ASSERT_VALUES_EQUAL(result[1], CudaUniform(dependentSource.NextUniformL()) >= 0.71 ? 1.0f : -0.5f);
        UNIT_ASSERT_VALUES_EQUAL(result[3], 1.0f);
    }

    Y_UNIT_TEST(TestTreeSeedExpansionDoesNotAdvanceSharedStream) {
        TRandom shared(901), oracle(901);
        const ui64 visitor = shared.NextUniformL();
        UNIT_ASSERT_VALUES_EQUAL(visitor, oracle.NextUniformL());
        TRandom devices(visitor);
        for (ui32 device = 0; device < 4; ++device) {
            const ui64 seed = devices.NextUniformL();
            for (ui64 tensorHash : {0ull, 681ull, 0xffffffffffffffffull})
                UNIT_ASSERT_VALUES_EQUAL(NCB::MetalMetaL2TreeDataSetSeed(visitor, device, tensorHash), seed + tensorHash);
        }
        UNIT_ASSERT_VALUES_EQUAL(shared.NextUniformL(), oracle.NextUniformL());
    }

    Y_UNIT_TEST(TestLiteralFiniteParameterDomain) {
        for (double exponent : {-2.0, 0.0, 1.0, 2.0}) {
            const auto never = NCB::MetalMetaL2PolicyExponents(713, 7, exponent, -1);
            const auto always = NCB::MetalMetaL2PolicyExponents(713, 7, exponent, 2);
            for (ui32 policy = 0; policy < 3; ++policy) {
                UNIT_ASSERT_VALUES_EQUAL(never[policy], 1.0f);
                UNIT_ASSERT_VALUES_EQUAL(always[policy], float(exponent));
            }
        }
        UNIT_ASSERT_VALUES_EQUAL(NCB::MetalMetaL2Uniform(0), 0.0);
        for (ui64 seed : {1ull, 0xffffffffull, 0x12345678abcdef01ull})
            UNIT_ASSERT_VALUES_EQUAL(NCB::MetalMetaL2Uniform(seed), CudaUniform(seed));
    }

    Y_UNIT_TEST(TestRejectsAmbiguousOrCorruptMapping) {
        using namespace NCB;
        TVector<TMetalMetaL2DataSet> duplicate{
            {0, 1, {{0, EMetalMetaL2Policy::Binary}}},
            {1, 1, {{0, EMetalMetaL2Policy::Binary}}},
        };
        UNIT_ASSERT_EXCEPTION(BuildMetalMetaL2FeatureExponents(1, 2, 0.5, duplicate), TCatBoostException);
        duplicate.resize(1);
        duplicate[0].PolicyMask = 2;
        UNIT_ASSERT_EXCEPTION(BuildMetalMetaL2FeatureExponents(1, 2, 0.5, duplicate), TCatBoostException);
        UNIT_ASSERT_EXCEPTION(MetalMetaL2PolicyExponents(0, 8, 2, 0.5), TCatBoostException);
        UNIT_ASSERT_EXCEPTION(MetalMetaL2PolicyExponents(0, 7, 2,
            std::numeric_limits<double>::quiet_NaN()), TCatBoostException);
    }

    Y_UNIT_TEST(TestRepeatedDynamicFeatureKeepsBothCompleteScoreChoices) {
        using namespace NCB;
        ui64 low = 0, high = 0;
        bool haveLow = false, haveHigh = false;
        for (ui64 seed = 0; seed < 1000 && !(haveLow && haveHigh); ++seed) {
            const auto selected = MetalMetaL2PolicyExponents(seed, 2, 0.7, 0.5);
            if (selected[1] == 1) { high = seed; haveHigh = true; }
            else { low = seed; haveLow = true; }
        }
        UNIT_ASSERT(haveLow && haveHigh);
        TVector<TMetalMetaL2DataSet> datasets{
            {low, 2, {{0, EMetalMetaL2Policy::HalfByte}}},
            {high, 2, {{0, EMetalMetaL2Policy::HalfByte}, {1, EMetalMetaL2Policy::HalfByte}}},
        };
        const auto choices = BuildMetalMetaL2FeatureChoices(3, 0.7, 0.5, datasets);
        UNIT_ASSERT_VALUES_EQUAL(choices[0], 3);
        UNIT_ASSERT_VALUES_EQUAL(choices[1], 1);
        UNIT_ASSERT_VALUES_EQUAL(choices[2], 1);
    }

    Y_UNIT_TEST(TestFeatureParallelScorePeekingPreservesFinishedSnapshotState) {
        NCB::TMetalOrderedRandom actual(731, 4, 3, 8, true), untouched(731, 4, 3, 8, true);
        TRandom source(731);
        UNIT_ASSERT_VALUES_EQUAL(actual.SelectPermutation(), untouched.SelectPermutation());
        source.NextUniformL(); source.Advance(65537);
        const auto first = actual.PeekScoreSeeds(0, 2), second = actual.PeekScoreSeeds(2, 1);
        UNIT_ASSERT_VALUES_EQUAL(first[0], source.NextUniformL());
        UNIT_ASSERT_VALUES_EQUAL(first[1], source.NextUniformL());
        UNIT_ASSERT_VALUES_EQUAL(second[0], source.NextUniformL());
        actual.FinishIterationWithDraws(0, 3); untouched.FinishIterationWithDraws(0, 3);
        UNIT_ASSERT_VALUES_EQUAL(actual.GetState().DrawCount, untouched.GetState().DrawCount);
        NCB::TMetalOrderedRandom restored(731, 4, 3, 8, true);
        restored.RestoreState(actual.GetState(), 1);
        UNIT_ASSERT_VALUES_EQUAL(actual.SelectPermutation(), restored.SelectPermutation());
        UNIT_ASSERT(actual.PeekScoreSeeds(0, 3) == restored.PeekScoreSeeds(0, 3));
    }

    Y_UNIT_TEST(TestStochasticScorePeekingPreservesLeafOracleOrder) {
        const TVector<ui32> shape(4, 1);
        NCB::TMetalFeatureParallelYetiRandom actual(413, 4, 3, 8, 2, shape, 4);
        TRandom source(413);
        actual.SelectPermutation(); source.NextUniformL();
        UNIT_ASSERT_VALUES_EQUAL(actual.WeakSeeds()[0], source.NextUniformL());
        source.Advance(65537);
        const auto scores = actual.PeekScoreSeeds(0, 3);
        for (ui64 seed : scores) UNIT_ASSERT_VALUES_EQUAL(seed, source.NextUniformL());
        for (ui64 seed : actual.LeafSeeds(3)) UNIT_ASSERT_VALUES_EQUAL(seed, source.NextUniformL());
        actual.Complete();

        NCB::TMetalYetiRandom doc(413, false, 2, 3, 8, 4, false, 0, 1, false, false, 2);
        TRandom docSource(413); docSource.NextUniformL();
        UNIT_ASSERT_VALUES_EQUAL(doc.Begin(), docSource.NextUniformL());
        for (ui64 seed : doc.PeekScoreSeeds(0, 2)) UNIT_ASSERT_VALUES_EQUAL(seed, docSource.NextUniformL());
        for (ui64 seed : doc.LeafSeeds(1)) UNIT_ASSERT_VALUES_EQUAL(seed, docSource.NextUniformL());
        doc.Complete();
        const TVector<ui32> depths{0};
        NCB::TMetalYetiRandom restored(413, false, 2, 3, 8, 4, false, 0, 1, false, false, 2);
        restored.Restore(doc.GetState(), depths);
        UNIT_ASSERT_VALUES_EQUAL(restored.Begin(), doc.Begin());
    }

    Y_UNIT_TEST(TestDocParallelSourceStreamRestoresFromDepths) {
        NCB::TMetalMetaL2DocRandom actual(714, true, 3, 8, 2);
        TRandom source(714); source.Advance(65538);
        actual.Begin();
        for (ui64 seed : actual.ScoreSeeds(0, 2)) UNIT_ASSERT_VALUES_EQUAL(seed, source.NextUniformL());
        actual.Finish(0);
        actual.Begin();
        for (ui32 offset = 0; offset < 6; offset += 2)
            for (ui64 seed : actual.ScoreSeeds(offset, 2)) UNIT_ASSERT_VALUES_EQUAL(seed, source.NextUniformL());
        actual.Finish(3);
        NCB::TMetalMetaL2DocRandom restored(714, true, 3, 8, 2);
        const TVector<ui32> depths{0, 3};
        restored.Restore(depths);
        actual.Begin(); restored.Begin();
        UNIT_ASSERT(actual.ScoreSeeds(0, 2) == restored.ScoreSeeds(0, 2));
        UNIT_ASSERT_EXCEPTION(actual.ScoreSeeds(0, 2), TCatBoostException);
    }
}
