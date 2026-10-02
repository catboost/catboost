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
        TMetalOrderedRandom(ui64 seed, ui32 permutationCount, ui32 maxDepth, ui32 candidateCount,
                            bool dynamicMode = false)
            : Seed(seed)
            , PermutationCount(permutationCount)
            , MaxDepth(maxDepth)
            , CandidateCount(candidateCount)
            , DynamicMode(dynamicMode)
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
            CB_ENSURE(!DynamicMode, "Metal dynamic FeatureParallel search requires explicit random draw counts");
            FinishIterationImpl(actualDepth, SearchAttempts(actualDepth));
        }

        // FeatureParallel always has the independent compressed dataset, even
        // when it has no columns. Each attempted depth consumes its seed, a
        // second seed when a permutation-dependent simple dataset exists, and
        // a third when tree-CTR packs are active BEFORE that attempt. The
        // visitor's device and tensor seeds use local TRandom instances, so
        // they consume no additional draws from this shared stream.
        // Source: oblivious_tree_structure_searcher.cpp lines 165-216 and
        // tree_ctr_datasets_visitor.cpp SetScoreStdDevAndSeed/Accept.
        void FinishIterationWithDraws(ui32 actualDepth, ui32 searchDrawCount) {
            CB_ENSURE(DynamicMode, "Explicit FeatureParallel draw counts require dynamic random mode");
            const ui32 attempts = SearchAttempts(actualDepth);
            CB_ENSURE(searchDrawCount >= attempts && searchDrawCount <= 3u * attempts,
                "Metal FeatureParallel search draw count is inconsistent with its attempted depths");
            FinishIterationImpl(actualDepth, searchDrawCount);
        }

        TVector<ui64> PeekScoreSeeds(ui32 offset, ui32 count) const {
            CB_ENSURE(Pending && ui64(offset) + count <= 3ull * MaxDepth,
                "Metal score seeds require a pending bounded FeatureParallel search");
            TRandom copy = Random;
            if (!State.BootstrapInitialized) copy.Advance(BootstrapDrawCount);
            copy.Advance(offset);
            TVector<ui64> result(count);
            for (auto& seed : result) seed = copy.NextUniformL();
            return result;
        }

    private:
        ui32 SearchAttempts(ui32 actualDepth) const {
            return CandidateCount && MaxDepth ? Min(actualDepth + 1, MaxDepth) : 0;
        }

        void FinishIterationImpl(ui32 actualDepth, ui32 searchDrawCount) {
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
            Advance(searchDrawCount);
            ++State.CompletedIterations;
            Pending = false;
        }

    public:
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
            const ui64 maxSearchDraws = CandidateCount
                ? ui64(expectedCompletedIterations) * MaxDepth * (DynamicMode ? 3 : 1) : 0;
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
        const bool DynamicMode;
        TRandom Random;
        TMetalOrderedRandomState State;
        bool Pending = false;
    };
}
