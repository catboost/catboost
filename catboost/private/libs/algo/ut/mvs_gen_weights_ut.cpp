#include <library/cpp/testing/unittest/registar.h>
#include "catboost/private/libs/algo/fold.h"
#include "catboost/private/libs/algo/tensor_search_helpers.h"
#include "catboost/private/libs/algo/mvs.h"
#include <catboost/libs/helpers/restorable_rng.h>
#include <util/generic/ymath.h>
#include <util/generic/maybe.h>
#include <util/generic/vector.h>

Y_UNIT_TEST_SUITE(mvs) {
    Y_UNIT_TEST(mvs_GenWeights) {
        const ui32 SampleCount = CB_THREAD_LIMIT * 20;
        TFold ff;
        ff.SampleWeights.resize(SampleCount, 1);

        const int SampleCountAsInt = SafeIntegerCast<int>(SampleCount);

        TFold::TBodyTail bt(0, 0, SampleCountAsInt, SampleCountAsInt, (double)SampleCountAsInt);

        bt.WeightedDerivatives.resize(1, TVector<double>(SampleCount));
        bt.Approx.resize(1, TVector<double>(SampleCount));

        for (ui32 j = 0; j < CB_THREAD_LIMIT; ++j) {
            for (ui32 i = 0; i < 20; ++i) {
                bt.WeightedDerivatives[0][20 * j + i] = sqrt((i + 1) * (i + 1) - 1);
            }
        }

        ff.BodyTailArr.emplace_back(std::move(bt));

        const EBoostingType boostingType = Plain;
        NPar::TLocalExecutor executor;
        executor.RunAdditionalThreads(1);

        TMvsSampler sampler(SampleCount, 0.75, 1);

        TRestorableFastRng64 rand(0);
        sampler.GenSampleWeights(boostingType, {}, &rand, &executor, &ff);

        for (ui32 j = 0; j < CB_THREAD_LIMIT; ++j) {
            for (ui32 i = 0; i < 20; ++i) {
                const double weight = ff.SampleWeights[j * 20 + i];
                if (i + 1 > 11) {
                    UNIT_ASSERT_DOUBLES_EQUAL(weight, 1.0, 1e-6);
                } else {
                    UNIT_ASSERT(Abs(weight - 11. / (i + 1)) < 1e-6 || Abs(weight) < 1e-6);
                }
            }
        }
    }

    Y_UNIT_TEST(mvs_GenWeights_adaptive) {
        const ui32 SampleCount = CB_THREAD_LIMIT * 20;
        TFold ff;
        ff.SampleWeights.resize(SampleCount, 1);

        const int SampleCountAsInt = SafeIntegerCast<int>(SampleCount);

        TFold::TBodyTail bt(0, 0, SampleCountAsInt, SampleCountAsInt, (double)SampleCountAsInt);

        bt.WeightedDerivatives.resize(1, TVector<double>(SampleCount));
        bt.Approx.resize(1, TVector<double>(SampleCount));

        for (ui32 j = 0; j < CB_THREAD_LIMIT; ++j) {
            for (ui32 i = 0; i < 20; ++i) {
                bt.WeightedDerivatives[0][20 * j + i] = sqrt((i + 1) * (i + 1) - 1);
            }
        }

        ff.BodyTailArr.emplace_back(std::move(bt));

        const EBoostingType boostingType = Plain;
        NPar::TLocalExecutor executor;
        executor.RunAdditionalThreads(1);

        TMvsSampler sampler(SampleCount, 0.75, Nothing());

        TRestorableFastRng64 rand(0);
        sampler.GenSampleWeights(boostingType, {{{0, 1, 2},},}, &rand, &executor, &ff);

        for (ui32 j = 0; j < CB_THREAD_LIMIT; ++j) {
            for (ui32 i = 0; i < 20; ++i) {
                const double weight = ff.SampleWeights[j * 20 + i];
                if (i + 1 > 11) {
                    UNIT_ASSERT_DOUBLES_EQUAL(weight, 1.0, 1e-6);
                } else {
                    UNIT_ASSERT(Abs(weight - 11. / (i + 1)) < 1e-6 || Abs(weight) < 1e-6);
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
        bt.Approx.resize(1, TVector<double>(SampleCount));

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

        TMvsSampler sampler(SampleCount, 1, Nothing());

        TRestorableFastRng64 rand(0);
        sampler.GenSampleWeights(boostingType, {}, &rand, &executor, &ff);

        for (ui32 j = 0; j < CB_THREAD_LIMIT; ++j) {
            for (ui32 i = 0; i < 20; ++i) {
                UNIT_ASSERT_DOUBLES_EQUAL(ff.SampleWeights[20 * j + i], 1.0, 1e-6);
            }
        }
    }
}
