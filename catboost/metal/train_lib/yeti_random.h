#pragma once

#include <cstdint>

#include <catboost/libs/helpers/cpu_random.h>
#include <catboost/libs/helpers/exception.h>
#include <util/generic/array_ref.h>
#include <util/generic/vector.h>

namespace NCB {
    struct TMetalYetiRandomState {
        ui64 DrawCount = 0;
        ui32 CompletedIterations = 0;
        bool BootstrapInitialized = false;
    };

    // Single-device DocParallel: the constructor, weak target,
    // bootstrap cache, attempted splits, and leaf oracle share a host stream.
    // GPU bootstrap and score noise retain their documented Metal adaptation.
    class TMetalYetiRandom {
    public:
        TMetalYetiRandom(ui64 seed, bool bootstrap, ui32 leafIterations, ui32 depth, ui32 candidates,
                        ui32 datasetPermutations = 1)
            : Seed(seed), Bootstrap(bootstrap), LeafIterations(leafIterations), Depth(depth), Candidates(candidates),
              DatasetPermutations(datasetPermutations), Random(seed)
        {
            CB_ENSURE(LeafIterations >= 1 && LeafIterations <= 1000 && Depth <= 16 &&
                DatasetPermutations >= 1 && DatasetPermutations <= 64,
                "Invalid Metal YetiRank random stream dimensions");
            Advance(1); // TDocParallelBoosting::BaseIterationSeed.
        }

        ui64 Begin() {
            CB_ENSURE(Phase == 0, "Complete the pending YetiRank random iteration first");
            Phase = 1;
            ++State.DrawCount;
            return Random.NextUniformL();
        }

        TVector<uint64_t> LeafSeeds(ui32 attempts) {
            CB_ENSURE(Phase == 1 && attempts <= Depth && (Candidates || !attempts),
                "Invalid YetiRank split attempt count or random phase");
            if (Bootstrap && !State.BootstrapInitialized) {
                Advance(65537); // GetGpuSeeds<StripeMapping> base plus FillSeeds.
                State.BootstrapInitialized = true;
            }
            Advance(attempts);
            TVector<uint64_t> result(LeafSeedCount());
            for (auto& seed : result) { seed = Random.NextUniformL(); ++State.DrawCount; }
            Phase = 2;
            return result;
        }

        void Complete() {
            CB_ENSURE(Phase == 2 && State.CompletedIterations < Max<ui32>(),
                "YetiRank leaf seeds must precede completion");
            ++State.CompletedIterations;
            Phase = 0;
        }

        TMetalYetiRandomState GetState() const {
            CB_ENSURE(Phase == 0, "Cannot snapshot an incomplete YetiRank random iteration");
            return State;
        }

        void Restore(const TMetalYetiRandomState& state, TConstArrayRef<ui32> depths) {
            CB_ENSURE(Phase == 0 && State.CompletedIterations == 0 && State.DrawCount == 1,
                "Restore YetiRank RNG before its first iteration");
            CB_ENSURE(state.CompletedIterations == depths.size() &&
                state.BootstrapInitialized == (Bootstrap && !depths.empty()),
                "YetiRank snapshot random state differs from its completed trees");
            ui64 expected = 1 + ui64(depths.size()) * (1 + LeafSeedCount()) +
                (state.BootstrapInitialized ? 65537 : 0);
            for (ui32 actualDepth : depths) {
                CB_ENSURE(actualDepth <= Depth && (Candidates || !actualDepth),
                    "YetiRank snapshot random state has an invalid tree depth");
                // A duplicate/invalid winner is still an attempted search.
                expected += Candidates && Depth ? Min(actualDepth + 1, Depth) : 0;
            }
            CB_ENSURE(state.DrawCount == expected,
                "YetiRank snapshot random draw count disagrees with its completed trees");
            // TRandom has no serialization API. Replay only the exact bounded
            // host draws, never completed GPU training. init_model starts a new
            // stream; depths here count this fit's new trees only.
            Random = TRandom(Seed);
            State = {};
            Advance(state.DrawCount);
            State = state;
        }

    private:
        ui32 LeafSeedCount() const { return DatasetPermutations * (LeafIterations + ui32(LeafIterations > 1)); }
        void Advance(ui64 count) {
            State.DrawCount += count;
            while (count) {
                const ui32 chunk = Min<ui64>(count, Max<ui32>());
                Random.Advance(chunk);
                count -= chunk;
            }
        }
        const ui64 Seed;
        const bool Bootstrap;
        const ui32 LeafIterations, Depth, Candidates, DatasetPermutations;
        TRandom Random;
        TMetalYetiRandomState State;
        ui32 Phase = 0;
    };
}
