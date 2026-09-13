#pragma once

#include "tree_ctr_permutations.h"

#include <catboost/libs/helpers/cpu_random.h>
#include <catboost/libs/helpers/exception.h>

#include <util/generic/ylimits.h>

namespace NCB {
    struct TMetalOrderedRandomState {
        ui64 DrawCount = 0;
        ui32 CompletedIterations = 0;
        bool BootstrapInitialized = false;
    };

    // The numeric, single-device CUDA FeatureParallel host stream is shared by
    // dynamic_boosting.h's permutation chooser, bootstrap's MirrorMapping seed
    // cache, and numeric split searches. GPU bootstrap and score-noise streams
    // retain the separately defined Metal implementation; this helper matches
    // the host draws that determine the next learning permutation.
    class TMetalOrderedRandom {
    public:
        TMetalOrderedRandom(ui64 seed, ui32 permutationCount, ui32 maxDepth, ui32 candidateCount)
            : Seed(seed)
            , PermutationCount(permutationCount)
            , MaxDepth(maxDepth)
            , CandidateCount(candidateCount)
            , Random(seed)
        {
            CB_ENSURE(PermutationCount > 0 && PermutationCount <= 64,
                "Metal Ordered requires between 1 and 64 permutations");
            CB_ENSURE(MaxDepth <= 16, "Metal Ordered random stream supports depth up to 16");
        }

        ui32 SelectPermutation() {
            CB_ENSURE(!Pending, "Finish the pending Metal Ordered iteration before selecting again");
            Pending = true;
            // Preserve CUDA's unusual modulo P-2. P=1/2 consume no chooser
            // draw; P=3 consumes one draw but still selects permutation zero.
            if (PermutationCount > 2) {
                ++State.DrawCount;
            }
            return ChooseMetalFeatureParallelPermutation(PermutationCount, Random);
        }

        void FinishIteration(ui32 actualDepth) {
            CB_ENSURE(Pending, "Select the Metal Ordered permutation before finishing its iteration");
            CB_ENSURE(actualDepth <= MaxDepth && (CandidateCount || actualDepth == 0),
                "Metal Ordered returned an invalid depth for its random stream");
            CB_ENSURE(State.CompletedIterations < Max<ui32>(), "Metal Ordered iteration count overflows");
            // TGpuAwareRandom::GetGpuSeeds<TMirrorMapping> consumes one base
            // draw plus 65,536 FillSeeds draws on first bootstrap, even No.
            if (!State.BootstrapInitialized) {
                Advance(BootstrapDrawCount);
                State.BootstrapInitialized = true;
            }
            // featuresScoreCalcer.ComputeOptimalSplit consumes one shared draw
            // per attempted numeric depth, even when random_strength is zero.
            // A successful native step stops early only after an invalid or
            // repeated winner, which adds a final attempt beyond actualDepth.
            const ui32 attempts = CandidateCount && MaxDepth ? Min(actualDepth + 1, MaxDepth) : 0;
            Advance(attempts);
            ++State.CompletedIterations;
            Pending = false;
        }

        TMetalOrderedRandomState GetState() const {
            CB_ENSURE(!Pending, "Cannot snapshot a pending Metal Ordered random stream");
            return State;
        }

        void RestoreState(const TMetalOrderedRandomState& state, ui32 expectedCompletedIterations) {
            CB_ENSURE(!Pending && State.CompletedIterations == 0 && State.DrawCount == 0,
                "Restore the Metal Ordered random stream before its first iteration");
            CB_ENSURE(state.CompletedIterations == expectedCompletedIterations,
                "Metal Ordered random state iteration count differs from the snapshot");
            CB_ENSURE(state.BootstrapInitialized == (expectedCompletedIterations != 0),
                "Metal Ordered random state has an invalid bootstrap initialization flag");
            const ui64 chooserDraws = PermutationCount > 2 ? expectedCompletedIterations : 0;
            const ui64 fixedDraws = chooserDraws + (state.BootstrapInitialized ? BootstrapDrawCount : 0);
            const ui64 minSearchDraws = CandidateCount && MaxDepth ? expectedCompletedIterations : 0;
            const ui64 maxSearchDraws = CandidateCount ? ui64(expectedCompletedIterations) * MaxDepth : 0;
            CB_ENSURE(state.DrawCount >= fixedDraws + minSearchDraws &&
                      state.DrawCount <= fixedDraws + maxSearchDraws,
                "Metal Ordered random state draw count is inconsistent with its completed iterations");

            // TRandom exposes no serialization. Replaying bounded host draws
            // reconstructs its exact MT19937-64 state without replaying training.
            // An init_model starts a fresh stream: expectedCompletedIterations
            // counts new trees only, separately from the bootstrap model offset.
            Random = TRandom(Seed);
            State = {};
            Advance(state.DrawCount);
            State = state;
        }

    private:
        void Advance(ui64 count) {
            State.DrawCount += count;
            while (count) {
                const ui32 chunk = Min<ui64>(count, Max<ui32>());
                Random.Advance(chunk);
                count -= chunk;
            }
        }

        static constexpr ui64 BootstrapDrawCount = 65537;
        const ui64 Seed;
        const ui32 PermutationCount;
        const ui32 MaxDepth;
        const ui32 CandidateCount;
        TRandom Random;
        TMetalOrderedRandomState State;
        bool Pending = false;
    };
}
