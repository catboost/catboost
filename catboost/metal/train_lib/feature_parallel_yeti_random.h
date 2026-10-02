#pragma once

#include "ordered_random.h"

#include <cstdint>

#include <util/generic/array_ref.h>
#include <util/generic/vector.h>

namespace NCB {
    // Shared single-device CUDA FeatureParallel host RNG for classic YetiRank
    // and Combination objectives containing YetiRank components.
    // Unlike DocParallel, there is no initial BaseIterationSeed draw. Targets,
    // bootstrap initialization, structure seeds and the batched leaf estimator
    // all consume this stream in dynamic_boosting.h's execution order.
    class TMetalFeatureParallelYetiRandom {
    public:
        TMetalFeatureParallelYetiRandom(ui64 seed, ui32 permutationCount,
                                       ui32 maxDepth, ui32 candidateCount,
                                       ui32 leafIterations,
                                       TConstArrayRef<ui32> weakSeedCounts,
                                       ui32 leafTaskCount,
                                       bool variableLeafCalls = false)
            : Seed(seed)
            , PermutationCount(permutationCount)
            , MaxDepth(maxDepth)
            , CandidateCount(candidateCount)
            , LeafTaskCount(leafTaskCount)
            , LeafEvaluations(leafIterations + ui32(leafIterations > 1))
            , MaxLeafEvaluations(variableLeafCalls && leafIterations > 1
                ? Max(leafIterations, 100u) + 1u : LeafEvaluations)
            , VariableLeafCalls(variableLeafCalls)
            , WeakSeedCounts(weakSeedCounts.begin(), weakSeedCounts.end())
            , Random(seed)
        {
            CB_ENSURE(PermutationCount >= 1 && PermutationCount <= 64 &&
                      MaxDepth <= 16 && leafIterations >= 1 && leafIterations <= 1000 &&
                      WeakSeedCounts.size() == PermutationCount && LeafTaskCount > 0,
                "Invalid Metal FeatureParallel YetiRank random dimensions");
            CB_ENSURE(ui64(LeafTaskCount) * MaxLeafEvaluations <= MaxPacketSeeds,
                "Metal FeatureParallel YetiRank leaf seed packet exceeds the 1 GiB limit");
            for (ui32 count : WeakSeedCounts) {
                CB_ENSURE(count > 0 && count <= MaxPacketSeeds,
                    "Invalid Metal FeatureParallel YetiRank weak seed count");
            }
        }

        ui32 SelectPermutation() {
            CB_ENSURE(Phase == 0, "Complete the pending FeatureParallel YetiRank iteration first");
            CB_ENSURE(State.CompletedIterations < Max<ui32>(),
                "Metal FeatureParallel YetiRank iteration count overflows");
            if (PermutationCount > 2) ++State.DrawCount;
            SelectedPermutation = ChooseMetalFeatureParallelPermutation(PermutationCount, Random);
            Phase = 1;
            return SelectedPermutation;
        }

        // Ordered: two seeds per selected fold, learn then quality, including
        // an empty quality slice. Plain: one full-data weak-target seed.
        // CUDA computes the weak target BEFORE initializing mirror bootstrap
        // seeds. FeatureParallel initializes that cache even for bootstrap No.
        TVector<uint64_t> WeakSeeds() {
            CB_ENSURE(Phase == 1, "Select a FeatureParallel YetiRank permutation before requesting weak seeds");
            auto result = DrawPacket(WeakSeedCounts[SelectedPermutation]);
            if (!State.BootstrapInitialized) {
                Advance(BootstrapDrawCount);
                State.BootstrapInitialized = true;
            }
            Phase = 2;
            return result;
        }

        TVector<ui64> PeekScoreSeeds(ui32 offset, ui32 count) const {
            CB_ENSURE(Phase == 2 && ui64(offset) + count <= 3ull * MaxDepth,
                "Metal score seeds require completed FeatureParallel weak targets");
            TRandom copy = Random;
            copy.Advance(offset);
            TVector<ui64> result(count);
            for (auto& seed : result) seed = copy.NextUniformL();
            return result;
        }

        // searchDrawCount counts each attempted independent/simple/tree-CTR
        // scorer at its source call site. Leaf seeds are evaluation-major and
        // then task-major, then Yeti-component-major. Tasks are each learning
        // permutation's eligible prefixes followed by the full estimation
        // task. Plain P1 already has
        // one full learning task and omits only the duplicate estimation task.
        // I>1 consumes the final derivative evaluation after its last move.
        TVector<uint64_t> LeafSeeds(ui32 searchDrawCount) {
            CB_ENSURE(!VariableLeafCalls,
                "Variable FeatureParallel YetiRank leaves require on-demand seeds");
            CB_ENSURE(Phase == 2, "FeatureParallel YetiRank weak seeds must precede leaf seeds");
            ValidateSearchDraws(searchDrawCount);
            Advance(searchDrawCount);
            auto result = DrawPacket(LeafTaskCount * LeafEvaluations);
            Phase = 3;
            return result;
        }

        // Combination can contain Yeti components while enabling backtracking.
        // Each actual target invocation requests its seed here; unused trial
        // capacity never advances the shared stream. LeafTaskCount includes
        // both active estimation tasks and the Yeti components in each task.
        void BeginLeafCalls(ui32 searchDrawCount) {
            CB_ENSURE(VariableLeafCalls && Phase == 2,
                "Variable FeatureParallel YetiRank leaf calls require completed weak targets");
            ValidateSearchDraws(searchDrawCount);
            Advance(searchDrawCount);
            LeafCallCount = 0;
            Phase = 4;
        }

        uint64_t NextLeafSeed() {
            CB_ENSURE(Phase == 4, "Begin FeatureParallel YetiRank leaf calls before requesting a seed");
            CB_ENSURE(LeafCallCount < LeafTaskCount * MaxLeafEvaluations,
                "FeatureParallel YetiRank leaf calls exceed the CUDA walker bound");
            const uint64_t seed = Random.NextUniformL();
            ++LeafCallCount;
            ++State.DrawCount;
            return seed;
        }

        void Complete() {
            CB_ENSURE(Phase == 3 || Phase == 4, "FeatureParallel YetiRank leaf seeds must precede completion");
            if (Phase == 4) {
                // One full callback group per oracle evaluation. The walker
                // always evaluates initially, then I candidates for I>1. It
                // can attempt up to max(I,100) candidates when its first
                // accepted move is delayed (descent_helpers.cpp); I1 bypasses
                // trials.
                CB_ENSURE(LeafCallCount >= LeafTaskCount * LeafEvaluations &&
                          LeafCallCount % LeafTaskCount == 0,
                    "FeatureParallel YetiRank leaf calls do not cover complete oracle evaluations");
            }
            ++State.CompletedIterations;
            Phase = 0;
        }

        TMetalOrderedRandomState GetState() const {
            CB_ENSURE(Phase == 0, "Cannot snapshot an incomplete FeatureParallel YetiRank iteration");
            return State;
        }

        // The caller validates the complete data/options/task fingerprint.
        // Draw-count bounds account for varying selected fold counts and
        // inactive dynamic packs. Replaying only RNG draws reconstructs the
        // exact stream; no completed training or leaf evaluations are replayed.
        void Restore(const TMetalOrderedRandomState& state, ui32 expectedCompletedIterations) {
            CB_ENSURE(Phase == 0 && State.CompletedIterations == 0 && State.DrawCount == 0,
                "Restore FeatureParallel YetiRank RNG before its first iteration");
            CB_ENSURE(state.CompletedIterations == expectedCompletedIterations &&
                      state.BootstrapInitialized == (expectedCompletedIterations != 0),
                "FeatureParallel YetiRank snapshot iteration state is inconsistent");
            const ui32 reachable = PermutationCount > 2 ? PermutationCount - 2 : 1;
            ui32 minWeak = WeakSeedCounts[0], maxWeak = WeakSeedCounts[0];
            for (ui32 permutation = 1; permutation < reachable; ++permutation) {
                minWeak = Min(minWeak, WeakSeedCounts[permutation]);
                maxWeak = Max(maxWeak, WeakSeedCounts[permutation]);
            }
            const ui64 minLeafSeeds = ui64(LeafTaskCount) * LeafEvaluations;
            const ui64 maxLeafSeeds = ui64(LeafTaskCount) * MaxLeafEvaluations;
            const ui64 fixed = (state.BootstrapInitialized ? BootstrapDrawCount : 0) +
                ui64(expectedCompletedIterations) * ui32(PermutationCount > 2);
            const ui32 minSearch = CandidateCount && MaxDepth ? 1 : 0;
            const ui32 maxSearch = CandidateCount ? 3u * MaxDepth : 0;
            CB_ENSURE(state.DrawCount >= fixed + ui64(expectedCompletedIterations) * (minWeak + minSearch + minLeafSeeds) &&
                      state.DrawCount <= fixed + ui64(expectedCompletedIterations) * (maxWeak + maxSearch + maxLeafSeeds),
                "FeatureParallel YetiRank snapshot draw count is inconsistent");
            Random = TRandom(Seed);
            Advance(state.DrawCount);
            State = state;
        }

    private:
        void ValidateSearchDraws(ui32 searchDrawCount) const {
            const ui32 minimum = CandidateCount && MaxDepth ? 1 : 0;
            const ui32 maximum = CandidateCount ? 3u * MaxDepth : 0;
            CB_ENSURE(searchDrawCount >= minimum && searchDrawCount <= maximum,
                "Invalid Metal FeatureParallel YetiRank search draw count");
        }

        TVector<uint64_t> DrawPacket(ui32 count) {
            TVector<uint64_t> result(count);
            for (auto& seed : result) seed = Random.NextUniformL();
            State.DrawCount += count;
            return result;
        }

        void Advance(ui64 count) {
            State.DrawCount += count;
            while (count) {
                const ui32 chunk = Min<ui64>(count, Max<ui32>());
                Random.Advance(chunk);
                count -= chunk;
            }
        }

        static constexpr ui64 BootstrapDrawCount = 65537;
        static constexpr ui32 MaxPacketSeeds = (ui64(1) << 30) / sizeof(uint64_t);
        const ui64 Seed;
        const ui32 PermutationCount, MaxDepth, CandidateCount, LeafTaskCount, LeafEvaluations, MaxLeafEvaluations;
        const bool VariableLeafCalls;
        const TVector<ui32> WeakSeedCounts;
        TRandom Random;
        TMetalOrderedRandomState State;
        ui32 Phase = 0;
        ui32 SelectedPermutation = 0;
        ui32 LeafCallCount = 0;
    };
}
