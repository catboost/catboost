#include <library/cpp/testing/unittest/registar.h>

#include <catboost/libs/helpers/short_vector_ops.h>

#include <library/cpp/sse/sse.h>

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

    Y_UNIT_TEST(UpdateScoreBinKernelPlain) {
#ifdef _sse_
        const double trueStats[] = {0.0, 0.25, 0.5, 0.75};
        const double falseStats[] = {1.0, 0.75, 0.5, 0.25};
        const double scaledL2Regularizer = 1.0;
        double genericScoreBin[2];
        NGenericSimdOps::UpdateScoreBinKernelPlain(
            scaledL2Regularizer,
            &trueStats[0],
            &falseStats[0],
            &genericScoreBin[0]
        );
        double sse2ScoreBin[2];
        NGenericSimdOps::UpdateScoreBinKernelPlain(
            scaledL2Regularizer,
            &trueStats[0],
            &falseStats[0],
            &sse2ScoreBin[0]
        );

        UNIT_ASSERT_DOUBLES_EQUAL(genericScoreBin[0], sse2ScoreBin[0], 1e-18);
        UNIT_ASSERT_DOUBLES_EQUAL(genericScoreBin[1], sse2ScoreBin[1], 1e-18);
#endif
    }

    Y_UNIT_TEST(UpdateScoreBinKernelOrdered) {
#ifdef _sse_
        const double trueStats[] = {0.0, 0.25, 0.5, 0.75};
        const double falseStats[] = {1.0, 0.75, 0.5, 0.25};
        const double scaledL2Regularizer = 1.0;
        double genericScoreBin[2];
        NGenericSimdOps::UpdateScoreBinKernelOrdered(
            scaledL2Regularizer,
            &trueStats[0],
            &falseStats[0],
            &genericScoreBin[0]
        );
        double sse2ScoreBin[2];
        NGenericSimdOps::UpdateScoreBinKernelOrdered(
            scaledL2Regularizer,
            &trueStats[0],
            &falseStats[0],
            &sse2ScoreBin[0]
        );

        UNIT_ASSERT_DOUBLES_EQUAL(genericScoreBin[0], sse2ScoreBin[0], 1e-18);
        UNIT_ASSERT_DOUBLES_EQUAL(genericScoreBin[1], sse2ScoreBin[1], 1e-18);
#endif
    }
}
