#pragma once

#include "meta_l2.h"
#include <util/generic/ylimits.h>

namespace NCB {
    // Only non-stochastic symmetric DocParallel lacks a saved shared host
    // stream in the existing adapter. Reconstruct its source stream from the
    // completed tree depths. The chooser is a separate iteration-local RNG;
    // ordinary target/leaf derivatives consume no host randomness.
    class TMetalMetaL2DocRandom {
    public:
        TMetalMetaL2DocRandom(ui64 seed, bool bootstrap, ui32 maxDepth, ui32 candidates, ui32 datasets)
            : Seed(seed), Bootstrap(bootstrap), MaxDepth(maxDepth), Candidates(candidates), DataSets(datasets), Random(seed)
        {
            CB_ENSURE(MaxDepth <= 16 && DataSets >= 1 && DataSets <= 2,
                "Invalid Metal DocParallel meta-L2 random dimensions");
            Random.Advance(1); // BaseIterationSeed, once per fit (including init_model).
        }

        void Begin() {
            CB_ENSURE(!Pending, "Finish the preceding Metal meta-L2 tree first");
            if (Bootstrap && !BootstrapInitialized) {
                Random.Advance(65537);
                BootstrapInitialized = true;
            }
            Pending = true;
            SearchDraws = 0;
        }

        TVector<ui64> ScoreSeeds(ui32 offset, ui32 count) {
            CB_ENSURE(Pending && offset == SearchDraws && count == DataSets &&
                ui64(offset) + count <= ui64(MaxDepth) * DataSets,
                "Invalid Metal DocParallel meta-L2 scorer sequence");
            TVector<ui64> result(count);
            for (auto& seed : result) seed = Random.NextUniformL();
            SearchDraws += count;
            return result;
        }

        void Finish(ui32 actualDepth) {
            CB_ENSURE(Pending && actualDepth <= MaxDepth && (!actualDepth || Candidates) &&
                SearchDraws == Attempts(actualDepth) * DataSets,
                "Metal DocParallel meta-L2 tree consumed an inconsistent scorer count");
            Pending = false;
            ++Completed;
        }

        void Restore(TConstArrayRef<ui32> depths) {
            CB_ENSURE(!Pending && Completed == 0 && !BootstrapInitialized,
                "Restore Metal meta-L2 state before its first tree");
            ui64 draws = Bootstrap && !depths.empty() ? 65537 : 0;
            for (ui32 depth : depths) {
                CB_ENSURE(depth <= MaxDepth && (!depth || Candidates), "Invalid Metal meta-L2 snapshot depth");
                draws += ui64(Attempts(depth)) * DataSets;
            }
            Random = TRandom(Seed);
            ++draws;
            while (draws) {
                const ui32 chunk = Min<ui64>(draws, Max<ui32>());
                Random.Advance(chunk); draws -= chunk;
            }
            BootstrapInitialized = Bootstrap && !depths.empty();
            Completed = depths.size();
        }

    private:
        ui32 Attempts(ui32 depth) const { return Candidates && MaxDepth ? Min(depth + 1, MaxDepth) : 0; }
        const ui64 Seed;
        const bool Bootstrap;
        const ui32 MaxDepth, Candidates, DataSets;
        TRandom Random;
        ui32 Completed = 0, SearchDraws = 0;
        bool Pending = false, BootstrapInitialized = false;
    };
}
