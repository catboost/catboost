#pragma once

#include <catboost/libs/helpers/cpu_random.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/private/libs/options/enums.h>

#include <util/generic/vector.h>
#include <util/random/shuffle.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <utility>

namespace NCB {
    struct TMetalRsmFeature {
        ui32 ManagerId = 0;
        ui32 FoldCount = 0;
        ui32 BinCount = 0;
        bool IsCtr = false;
        bool PermutationDependent = false;
        // Equivalent native columns share one registered CUDA feature ID.
        TVector<ui32> RuntimeFeatures;
    };

    class TMetalFullMatrixRsm {
    public:
        TMetalFullMatrixRsm(ui64 seed, double rsm, ELossFunction objective,
                EBootstrapType bootstrap, double subsample, ui32 histories,
                bool simpleLeaves, ui32 runtimeFeatures, TVector<TMetalRsmFeature> features)
            : Rsm(rsm), Objective(objective), Bootstrap(bootstrap), Subsample(subsample),
              Histories(histories), SimpleLeaves(simpleLeaves), RuntimeFeatures(runtimeFeatures), Random(seed)
        {
            CB_ENSURE(std::isfinite(Rsm) && Rsm > 0 && Rsm <= 1,
                "Metal rsm must be finite and in (0, 1]");
            CB_ENSURE(Objective == ELossFunction::PairLogitPairwise ||
                Objective == ELossFunction::QueryCrossEntropy || Objective == ELossFunction::YetiRankPairwise,
                "Metal rsm requires PairLogitPairwise, QueryCrossEntropy or YetiRankPairwise");
            CB_ENSURE(Histories >= 1 && Histories <= 64 && RuntimeFeatures > 0,
                "Invalid Metal rsm feature or history count");
            CB_ENSURE(Bootstrap == EBootstrapType::No || Bootstrap == EBootstrapType::Bayesian ||
                Bootstrap == EBootstrapType::Bernoulli || Bootstrap == EBootstrapType::Poisson,
                "Unsupported full-matrix bootstrap for rsm");
            CB_ENSURE(std::isfinite(Subsample) && Subsample > 0 && Subsample <= 1,
                "Invalid full-matrix rsm subsample");
            CB_ENSURE(Objective != ELossFunction::QueryCrossEntropy ||
                Bootstrap == EBootstrapType::No || Bootstrap == EBootstrapType::Bernoulli,
                "QueryCrossEntropy rsm supports No or Bernoulli bootstrap");
            CB_ENSURE(Objective != ELossFunction::YetiRankPairwise || Bootstrap != EBootstrapType::Poisson,
                "YetiRankPairwise rsm does not support Poisson bootstrap");
            std::sort(features.begin(), features.end(), [](const auto& left, const auto& right) {
                return left.ManagerId < right.ManagerId;
            });
            TVector<ui8> seen(RuntimeFeatures, 0);
            for (auto& feature : features) {
                CB_ENSURE(feature.FoldCount <= 255 && feature.BinCount <= 256 &&
                    feature.FoldCount < feature.BinCount + ui32(feature.IsCtr == false),
                    "Invalid CUDA feature grid for Metal rsm");
                for (ui32 runtimeFeature : feature.RuntimeFeatures) {
                    CB_ENSURE(runtimeFeature < RuntimeFeatures && !seen[runtimeFeature],
                        "Metal rsm runtime feature mapping is invalid or duplicated");
                    seen[runtimeFeature] = 1;
                }
                if (!feature.FoldCount) continue;
                CB_ENSURE(!feature.RuntimeFeatures.empty(), "An rsm feature has no runtime columns");
                const ui32 policy = feature.FoldCount <= 1 ? 0 : feature.FoldCount <= 15 ? 1 : 2;
                const ui32 dataset = Histories > 1 && feature.PermutationDependent ? 1 : 0;
                Grids[dataset * 3 + policy].push_back(std::move(feature));
            }
            // feature_layout_doc_parallel.h first shuffles each OneByte grid
            // with a local seed, then sorts by the original binarization level.
            // This local shuffle consumes no draws from the target's stream.
            for (ui32 dataset = 0; dataset < 2; ++dataset) {
                auto& grid = Grids[dataset * 3 + 2];
                TRandom ordering(0);
                Shuffle(grid.begin(), grid.end(), ordering);
                std::sort(grid.begin(), grid.end(), [](const auto& left, const auto& right) {
                    return GroupingLevel(left) < GroupingLevel(right);
                });
            }
            bool nonBinary = false;
            for (ui32 grid = 0; grid < Grids.size(); ++grid)
                nonBinary |= grid % 3 != 0 && !Grids[grid].empty();
            // CUDA checks this inside SampleFeatures; an all-binary grid is
            // retained without sampling even for smaller positive rsm values.
            CB_ENSURE(!nonBinary || Rsm == 1 || Rsm > 1e-2, "Too low rsm " << Rsm);
            Advance(1); // TDocParallelBoosting::BaseIterationSeed.
        }

        TVector<ui8> NextMask() {
            // Reproduce the single-device shared HOST stream around RSM.
            // Existing Metal bootstrap/PFound device streams stay unchanged.
            const bool needsCache = Objective != ELossFunction::QueryCrossEntropy ||
                (Bootstrap == EBootstrapType::Bernoulli && Subsample < 1);
            if (needsCache && !BootstrapInitialized) {
                Advance(65537); // Stripe GetGpuSeeds: base + 65536 FillSeeds.
                BootstrapInitialized = true;
            }
            if (Objective == ELossFunction::YetiRankPairwise) Advance(1); // Weak PFound DistributedSeed.
            TVector<ui8> active(RuntimeFeatures, 0);
            for (ui32 index = 0; index < Grids.size(); ++index) {
                const auto& grid = Grids[index];
                if (grid.empty()) continue;
                if (index % 3 == 0 || Rsm == 1) {
                    for (const auto& feature : grid) Mark(feature, &active);
                    continue;
                }
                const ui32 perPack = index % 3 == 1 ? 8 : 4;
                double probability = Rsm;
                for (;;) {
                    bool kept = false;
                    for (ui32 first = 0; first < grid.size(); first += perPack) {
                        ++DrawCount;
                        if (Random.NextUniform() > probability) continue;
                        kept = true;
                        for (ui32 feature = first; feature < std::min<ui32>(first + perPack, grid.size()); ++feature)
                            Mark(grid[feature], &active);
                    }
                    if (kept) break;
                    probability = std::min(2 * probability, 1.0);
                }
            }
            // Pair/QCE leaf oracles are deterministic. A non-Simple PFound
            // oracle constructs one fresh target per history, independent of
            // the number of leaf iterations or line-search attempts.
            if (Objective == ELossFunction::YetiRankPairwise && !SimpleLeaves) Advance(Histories);
            ++CompletedIterations;
            return active;
        }

        // The grids and schedule are immutable for this fit. Resume replays
        // only inexpensive host sampling, never completed GPU training.
        void Restore(ui32 completedIterations) {
            CB_ENSURE(CompletedIterations == 0 && DrawCount == 1,
                "Restore Metal rsm before its first sampled tree");
            for (ui32 tree = 0; tree < completedIterations; ++tree) NextMask();
        }
        ui64 GetDrawCount() const { return DrawCount; }
        ui32 GetCompletedIterations() const { return CompletedIterations; }

    private:
        static double GroupingLevel(const TMetalRsmFeature& feature) {
            return feature.BinCount / 256.0 + double(feature.BinCount > 129 && !feature.IsCtr);
        }
        static void Mark(const TMetalRsmFeature& feature, TVector<ui8>* active) {
            for (ui32 index : feature.RuntimeFeatures) (*active)[index] = 1;
        }
        void Advance(ui32 count) { Random.Advance(count); DrawCount += count; }
        const double Rsm;
        const ELossFunction Objective;
        const EBootstrapType Bootstrap;
        const double Subsample;
        const ui32 Histories;
        const bool SimpleLeaves;
        const ui32 RuntimeFeatures;
        std::array<TVector<TMetalRsmFeature>, 6> Grids;
        TRandom Random;
        ui64 DrawCount = 0;
        ui32 CompletedIterations = 0;
        bool BootstrapInitialized = false;
    };
}
