#include <library/unittest/registar.h>

#include <catboost/libs/algo/short_vector_ops.h>

#include <library/sse/sse.h>

Y_UNIT_TEST_SUITE(ShortVectorOpsTest) {
    Y_UNIT_TEST(CalculateScorePattern) {
#ifdef ARCADIA_SSE
        const double values[] = {0.0, 0.25, 0.5, 0.75};
        auto genericHorizontalSum = NGenericSimdOps::HorizontalAdd(
            NGenericSimdOps::FusedMultiplyAdd(
                values + 0,
                values + 2,
                NGenericSimdOps::MakeZeros()
        ));
        auto sse2HorizontalSum = NSse2SimdOps::HorizontalAdd(
            NSse2SimdOps::FusedMultiplyAdd(
                values + 0,
                values + 2,
                NSse2SimdOps::MakeZeros()
        ));
        UNIT_ASSERT_DOUBLES_EQUAL(genericHorizontalSum, sse2HorizontalSum, 1e-18);
#endif
    }

    Y_UNIT_TEST(CalculatePairwiseScorePattern) {
#ifdef ARCADIA_SSE
        const double values[] = {0.0, 0.25, 0.5, 0.75};
        auto genericHorizontalSum = NGenericSimdOps::HorizontalAdd(
            NGenericSimdOps::ElementwiseAdd(
                NGenericSimdOps::Gather(values + 0, values + 2),
                NGenericSimdOps::Gather(values + 1, values + 3)
        ));
        auto sse2HorizontalSum = NSse2SimdOps::HorizontalAdd(
            NSse2SimdOps::ElementwiseAdd(
                NSse2SimdOps::Gather(values + 0, values + 2),
                NSse2SimdOps::Gather(values + 1, values + 3)
        ));
        UNIT_ASSERT_DOUBLES_EQUAL(genericHorizontalSum, sse2HorizontalSum, 1e-18);
#endif
    }
}
