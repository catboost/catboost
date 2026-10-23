#pragma once

#include <catboost/metal/native/metal_langevin.h>
#include <catboost/libs/helpers/cpu_random.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/private/libs/algo_helpers/langevin_utils.h>

#include <util/generic/vector.h>
#include <util/generic/ylimits.h>
#include <util/generic/yexception.h>

#include <cmath>

namespace NCB {
    struct TMetalLangevinRandomState {
        ui64 DrawCount = 0;
        ui32 CompletedIterations = 0;
        bool WeakSeedCacheInitialized = false;
    };

    // One source host stream for Langevin, target RNG, and structure search.
    // The caller invokes events where CUDA invokes them; no fixed leaf packet
    // is reserved because rejected trials and target components consume seeds
    // in their actual evaluation order. Instantiate only for langevin=true.
    class TMetalLangevinRandom {
    public:
        TMetalLangevinRandom(ui64 seed, bool featureParallel, ui32 permutationCount,
                            float diffusionTemperature, float learningRate,
                            NPar::ILocalExecutor* localExecutor)
            : Seed(seed)
            , FeatureParallel(featureParallel)
            , PermutationCount(permutationCount)
            , DiffusionTemperature(diffusionTemperature)
            , LearningRate(learningRate)
            , LocalExecutor(localExecutor)
            , Random(seed)
        {
            CB_ENSURE(PermutationCount >= 1 && PermutationCount <= 64,
                "Metal Langevin requires between 1 and 64 permutations");
            CB_ENSURE(std::isfinite(DiffusionTemperature) && DiffusionTemperature >= 0 &&
                      std::isfinite(LearningRate) && LearningRate > 0 && LocalExecutor,
                "Metal Langevin requires finite nonnegative temperature, positive learning rate, and an executor");
            // Only TDocParallelBoosting has this constructor draw. Its chooser
            // later uses an iteration-local RNG and never advances this one.
            if (!FeatureParallel) BaseIterationSeed = Draw();
        }

        ui64 GetBaseIterationSeed() const {
            CB_ENSURE(!FeatureParallel, "FeatureParallel has no base iteration seed");
            return BaseIterationSeed;
        }

        void BeginIteration() {
            CB_ENSURE(!Pending && State.CompletedIterations < Max<ui32>(),
                "Complete the pending Metal Langevin iteration before beginning another");
            Pending = true;
        }

        // absoluteIteration includes init_model trees for DocParallel. The
        // FeatureParallel chooser ignores it and consumes its shared stream.
        ui32 SelectPermutation(ui64 absoluteIteration = 0) {
            BeginIteration();
            const ui32 learnCount = PermutationCount > 1 ? PermutationCount - 1 : 1;
            if (learnCount <= 1) return 0;
            if (FeatureParallel) return Draw() % (learnCount - 1);
            TRandom iterationRandom(BaseIterationSeed + absoluteIteration);
            iterationRandom.Advance(10);
            return iterationRandom.NextUniformL() % (learnCount - 1);
        }

        // The allocation seed is consumed but unused by CUDA CreateSeeds.
        // Optional packet contains the 65,536 subsequent FillSeeds draws only.
        // It initializes device state; a restored device cache must come from
        // the snapshot, not from replaying this initialization packet.
        bool EnsureWeakSeedCache(TVector<uint64_t>* initialSeeds = nullptr) {
            RequirePending();
            if (initialSeeds) initialSeeds->clear();
            if (State.WeakSeedCacheInitialized) return false;
            EnsureDrawCapacity(WeakSeedCacheDrawCount);
            if (initialSeeds) initialSeeds->resize(WeakSeedCount);
            Draw();
            if (initialSeeds) {
                for (auto& seed : *initialSeeds) seed = Draw();
            } else {
                Advance(WeakSeedCount);
            }
            State.WeakSeedCacheInitialized = true;
            return true;
        }

        ui64 NextSeed(uint32_t event) {
            RequirePending();
            if (event == CBM_LANGEVIN_WEAK_SEED_CACHE) {
                EnsureWeakSeedCache();
                return 0;
            }
            CB_ENSURE(IsLeafEvent(event) || event == CBM_LANGEVIN_YETI_WEAK ||
                      event == CBM_LANGEVIN_YETI_LEAF || event == CBM_LANGEVIN_SEARCH,
                "Invalid Metal Langevin shared-seed event");
            return Draw();
        }

        void AdvanceSearch(ui32 count) {
            RequirePending();
            Advance(count);
        }

        void FillLeafNoise(uint32_t event, ui32 count, double* noise) {
            RequirePending();
            CB_ENSURE(IsLeafEvent(event) && count && noise,
                "Metal Langevin leaf noise requires a leaf event and nonempty output");
            TVector<double> values(count, 0.0);
            // The seed draw deliberately precedes the source helper's T=0
            // early return. This is the flat vector helper, not CPU training's
            // leaf-sum helper with weight/Hessian-dependent noise scaling.
            AddLangevinNoiseToDerivatives(DiffusionTemperature, LearningRate,
                NextSeed(event), &values, LocalExecutor);
            for (ui32 i = 0; i < count; ++i) noise[i] = values[i];
        }

        void FinishIteration() {
            RequirePending();
            ++State.CompletedIterations;
            Pending = false;
        }

        TMetalLangevinRandomState GetState() const {
            CB_ENSURE(!Pending, "Cannot snapshot a pending Metal Langevin iteration");
            return State;
        }

        // The adapter supplies its validated maximum from the configured
        // iterations/tasks/walker/search bounds, and validates the data/options
        // fingerprint separately. A malformed snapshot cannot request an
        // unbounded replay. expectedCompletedIterations counts new trees only.
        void RestoreState(const TMetalLangevinRandomState& state,
                          ui32 expectedCompletedIterations, ui64 maximumDrawCount) {
            const ui64 initialDraws = FeatureParallel ? 0 : 1;
            CB_ENSURE(!Pending && State.CompletedIterations == 0 &&
                      State.DrawCount == initialDraws && !State.WeakSeedCacheInitialized,
                "Restore Metal Langevin RNG before its first iteration");
            CB_ENSURE(state.CompletedIterations == expectedCompletedIterations,
                "Metal Langevin snapshot iteration count differs from its random state");
            const ui64 chooserDraws = FeatureParallel && PermutationCount > 2
                ? expectedCompletedIterations : 0;
            const ui64 minimumDraws = initialDraws + chooserDraws +
                (state.WeakSeedCacheInitialized ? WeakSeedCacheDrawCount : 0);
            CB_ENSURE(state.DrawCount >= minimumDraws && state.DrawCount <= maximumDrawCount &&
                      (expectedCompletedIterations ||
                       (!state.WeakSeedCacheInitialized && state.DrawCount == initialDraws)),
                "Metal Langevin snapshot draw count is outside its validated bounds");
            Random = TRandom(Seed);
            State = {};
            Advance(state.DrawCount);
            State = state;
        }

        const TString& GetCallbackError() const { return CallbackError; }

        static int NoiseCallback(void* context, uint32_t event, uint32_t count, double* noise) noexcept {
            if (!context) return 1;
            auto& self = *static_cast<TMetalLangevinRandom*>(context);
            try {
                self.CallbackError.clear();
                self.FillLeafNoise(event, count, noise);
                return 0;
            } catch (...) {
                self.RecordCallbackError();
                return 1;
            }
        }

        static int SeedCallback(void* context, uint32_t event, uint64_t* seed) noexcept {
            if (!context) return 1;
            auto& self = *static_cast<TMetalLangevinRandom*>(context);
            try {
                self.CallbackError.clear();
                CB_ENSURE(seed, "Metal Langevin seed callback requires output storage");
                *seed = self.NextSeed(event);
                return 0;
            } catch (...) {
                self.RecordCallbackError();
                return 1;
            }
        }

        static constexpr ui32 WeakSeedCount = 65536;
        static constexpr ui64 WeakSeedCacheDrawCount = 65537;

    private:
        static bool IsLeafEvent(uint32_t event) {
            return event >= CBM_LANGEVIN_INITIAL_GRADIENT && event <= CBM_LANGEVIN_ACCEPTED_GRADIENT;
        }

        void RequirePending() const {
            CB_ENSURE(Pending, "Begin the Metal Langevin iteration before consuming seeds");
        }

        void EnsureDrawCapacity(ui64 count) const {
            CB_ENSURE(count <= Max<ui64>() - State.DrawCount, "Metal Langevin random draw count overflows");
        }

        ui64 Draw() {
            EnsureDrawCapacity(1);
            ++State.DrawCount;
            return Random.NextUniformL();
        }

        void Advance(ui64 count) {
            EnsureDrawCapacity(count);
            State.DrawCount += count;
            while (count) {
                const ui32 chunk = Min<ui64>(count, Max<ui32>());
                Random.Advance(chunk);
                count -= chunk;
            }
        }

        void RecordCallbackError() noexcept {
            try { CallbackError = CurrentExceptionMessage(); } catch (...) {}
        }

        const ui64 Seed;
        const bool FeatureParallel;
        const ui32 PermutationCount;
        const float DiffusionTemperature, LearningRate;
        NPar::ILocalExecutor* const LocalExecutor;
        TRandom Random;
        ui64 BaseIterationSeed = 0;
        TMetalLangevinRandomState State;
        bool Pending = false;
        TString CallbackError;
    };
}
