#include <library/unittest/registar.h>
#include "catboost/libs/algo/fold.h"
#include "catboost/libs/algo/tensor_search_helpers.h"
#include "catboost/libs/algo/mvs.h"
#include <catboost/libs/helpers/restorable_rng.h>
#include <util/generic/ymath.h>

Y_UNIT_TEST_SUITE(mvs) {
    Y_UNIT_TEST(mvs_GenWeights) {
        const ui32 SampleCount = 2560;
        TFold ff;
        ff.SampleWeights.resize(SampleCount, 1);

        const int SampleCountAsInt = SafeIntegerCast<int>(SampleCount);

        TFold::TBodyTail bt(0, 0, SampleCountAsInt, SampleCountAsInt, (double)SampleCountAsInt);

        bt.WeightedDerivatives.resize(1, TVector<double>(SampleCount));

        for (ui32 j = 0; j < CB_THREAD_LIMIT; ++j) {
            for (ui32 i = 0; i < 20; ++i) {
                bt.WeightedDerivatives[0][20 * j + i] = (double)(i + 1);
            }
        }

        ff.BodyTailArr.emplace_back(std::move(bt));

        const EBoostingType boostingType = Plain;
        NPar::TLocalExecutor executor;
        executor.RunAdditionalThreads(1);

        TMvsSampler sampler(SampleCount, 0.3);

        TRestorableFastRng64 rand(0);
        sampler.GenSampleWeights(ff, boostingType, &rand, &executor);

        for (ui32 j = 0; j < CB_THREAD_LIMIT; ++j) {
            for (ui32 i = 0; i < 20; ++i) {
                if (i>12) {
                    UNIT_ASSERT_DOUBLES_EQUAL(ff.SampleWeights[20 * j + i], 1.0, 1e-6);
                } else {
                    UNIT_ASSERT(Abs(ff.SampleWeights[20 * j + i] - 14.0 / (i + 1)) < 1e-6 || ff.SampleWeights[20 * j + i] < 1e-6);
                }
            }
        }
    }

    Y_UNIT_TEST(mvs_GenWeights_take_all) {
        const ui32 SampleCount = 2560;
        TFold ff;
        ff.SampleWeights.resize(SampleCount, 1);

        const int SampleCountAsInt = SafeIntegerCast<int>(SampleCount);

        TFold::TBodyTail bt(0, 0, SampleCountAsInt, SampleCountAsInt, (double)SampleCountAsInt);

        bt.WeightedDerivatives.resize(1, TVector<double>(SampleCount));

        for (ui32 j = 0; j < CB_THREAD_LIMIT; ++j) {
            for (ui32 i = 1; i < 20; ++i) {
                bt.WeightedDerivatives[0][20 * j + i] = (double)(i + 1);
            }
            bt.WeightedDerivatives[0][20 * j] = (double)(2);
        }

        ff.BodyTailArr.emplace_back(std::move(bt));

        const EBoostingType boostingType = Plain;
        NPar::TLocalExecutor executor;
        executor.RunAdditionalThreads(1);

        TMvsSampler sampler(SampleCount, 0.99);

        TRestorableFastRng64 rand(0);
        sampler.GenSampleWeights(ff, boostingType, &rand, &executor);

        for (ui32 j = 0; j < CB_THREAD_LIMIT; ++j) {
            for (ui32 i = 0; i < 20; ++i) {
                UNIT_ASSERT_DOUBLES_EQUAL(ff.SampleWeights[20 * j + i], 1.0, 1e-6);
            }
        }
    }
}
