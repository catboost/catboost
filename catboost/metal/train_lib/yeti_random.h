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
                        ui32 datasetPermutations = 1, bool greedy = false, ui32 maxSearchAttempts = 0,
                        ui32 oracleCount = 1, bool variableLeafCalls = false, bool simpleLeaves = false)
            : Seed(seed), Bootstrap(bootstrap), LeafIterations(leafIterations), Depth(depth), Candidates(candidates),
              DatasetPermutations(datasetPermutations), Greedy(greedy),
              MaxSearchAttempts(greedy ? maxSearchAttempts : depth), OracleCount(oracleCount),
              VariableLeafCalls(variableLeafCalls), SimpleLeaves(simpleLeaves), Random(seed)
        {
            CB_ENSURE(LeafIterations >= 1 && LeafIterations <= 1000 && Depth <= (Greedy ? 65535u : 16u) &&
                (!Greedy || MaxSearchAttempts > 0) &&
                DatasetPermutations >= 1 && DatasetPermutations <= 64 && OracleCount >= 1 && OracleCount <= 128 &&
                (!SimpleLeaves || (VariableLeafCalls && LeafIterations == 1)),
                "Invalid Metal YetiRank random stream dimensions");
            Advance(1); // TDocParallelBoosting::BaseIterationSeed.
        }

        ui64 Begin() {
            CB_ENSURE(OracleCount == 1, "A multi-component target requires a weak seed packet");
            return BeginSeeds().front();
        }

        TVector<uint64_t> BeginSeeds() {
            CB_ENSURE(Phase == 0, "Complete the pending YetiRank random iteration first");
            Phase = 1;
            // Greedy TWeakObjective::StochasticDer calls BootstrapAndFilter
            // before drawing its Yeti target seed. Symmetric DocParallel keeps
            // the original weak-target-first order below.
            if (Greedy) InitializeBootstrap();
            TVector<uint64_t> result(OracleCount);
            for (auto& seed : result) { ++State.DrawCount; seed = Random.NextUniformL(); }
            return result;
        }

        TVector<uint64_t> LeafSeeds(ui32 attempts) {
            CB_ENSURE(!VariableLeafCalls, "Combination leaf callbacks require BeginLeafCalls");
            CB_ENSURE(Phase == 1 && attempts <= MaxSearchAttempts && (Candidates || !attempts),
                "Invalid YetiRank split attempt count or random phase");
            InitializeBootstrap();
            Advance(attempts);
            TVector<uint64_t> result(LeafSeedCount());
            for (auto& seed : result) { seed = Random.NextUniformL(); ++State.DrawCount; }
            Phase = 2;
            return result;
        }

        void BeginLeafCalls(ui32 attempts) {
            CB_ENSURE(VariableLeafCalls && Phase == 1 && attempts <= MaxSearchAttempts && (Candidates || !attempts),
                "Invalid Combination split attempt count or random phase");
            InitializeBootstrap();
            Advance(attempts);
            CurrentLeafCalls = 0;
            Phase = 2;
        }

        ui64 NextLeafSeed() {
            CB_ENSURE(VariableLeafCalls && !SimpleLeaves && Phase == 2 && CurrentLeafCalls < MaximumLeafSeedCount(),
                "Combination leaf callback exceeds its bounded random stream");
            ++CurrentLeafCalls;
            ++State.DrawCount;
            return Random.NextUniformL();
        }

        void Complete() {
            CB_ENSURE(Phase == 2 && State.CompletedIterations < Max<ui32>(),
                "YetiRank leaf seeds must precede completion");
            if (VariableLeafCalls) CB_ENSURE(CurrentLeafCalls >= (SimpleLeaves ? 0 : LeafSeedCount()) &&
                CurrentLeafCalls % OracleCount == 0 && (!SimpleLeaves || !CurrentLeafCalls),
                "Combination leaf callback count differs from its target components");
            ++State.CompletedIterations;
            Phase = 0;
        }

        TMetalYetiRandomState GetState() const {
            CB_ENSURE(Phase == 0, "Cannot snapshot an incomplete YetiRank random iteration");
            return State;
        }

        void Restore(const TMetalYetiRandomState& state, TConstArrayRef<ui32> depths,
                     TConstArrayRef<ui32> searchAttempts = {}) {
            CB_ENSURE(Phase == 0 && State.CompletedIterations == 0 && State.DrawCount == 1,
                "Restore YetiRank RNG before its first iteration");
            CB_ENSURE(state.CompletedIterations == depths.size() &&
                state.BootstrapInitialized == (Bootstrap && !depths.empty()),
                "YetiRank snapshot random state differs from its completed trees");
            CB_ENSURE(Greedy ? searchAttempts.size() == depths.size() : searchAttempts.empty(),
                "YetiRank snapshot search counters differ from the training policy");
            ui64 expected = 1 + ui64(depths.size()) * (OracleCount + (SimpleLeaves ? 0 : LeafSeedCount())) +
                (state.BootstrapInitialized ? 65537 : 0);
            for (ui32 tree = 0; tree < depths.size(); ++tree) {
                const ui32 actualDepth = depths[tree];
                CB_ENSURE(actualDepth <= Depth && (Candidates || !actualDepth),
                    "YetiRank snapshot random state has an invalid tree depth");
                // A duplicate/invalid winner is still an attempted search.
                if (Greedy) {
                    CB_ENSURE(searchAttempts[tree] <= MaxSearchAttempts && (Candidates || !searchAttempts[tree]),
                        "Greedy YetiRank snapshot has invalid search draw counts");
                    expected += searchAttempts[tree];
                } else expected += Candidates && Depth ? Min(actualDepth + 1, Depth) : 0;
            }
            const ui64 extra = VariableLeafCalls && !SimpleLeaves ? ui64(depths.size()) *
                (MaximumLeafSeedCount() - LeafSeedCount()) : 0;
            CB_ENSURE(state.DrawCount >= expected && state.DrawCount <= expected + extra &&
                (!VariableLeafCalls || (state.DrawCount - expected) % OracleCount == 0),
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
        void InitializeBootstrap() {
            if (Bootstrap && !State.BootstrapInitialized) {
                Advance(65537); // GetGpuSeeds<StripeMapping> base plus FillSeeds.
                State.BootstrapInitialized = true;
            }
        }
        ui32 LeafSeedCount() const { return OracleCount * DatasetPermutations * (LeafIterations + ui32(LeafIterations > 1)); }
        ui32 MaximumLeafSeedCount() const {
            return OracleCount * DatasetPermutations * (LeafIterations == 1 ? 1 : 1 + Max(LeafIterations, 100u));
        }
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
        const bool Greedy;
        const ui32 MaxSearchAttempts;
        const ui32 OracleCount;
        const bool VariableLeafCalls, SimpleLeaves;
        TRandom Random;
        TMetalYetiRandomState State;
        ui32 Phase = 0;
        ui32 CurrentLeafCalls = 0;
    };
}
