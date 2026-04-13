#pragma once

// Pairwise ranking target functions for CatBoost-MLX.
//
// Two losses are implemented:
//   TPairLogitTarget  — all ordered (winner, loser) pairs within each group,
//                       generated once at construction and reused every iteration.
//   TYetiRankTarget   — pairs are regenerated every iteration from a random
//                       permutation within each group (stochastic ranking).
//
// Both classes compute gradients/hessians on the CPU, using the raw prediction
// cursor evaluated from the GPU, then upload the result back as mx::arrays.
// This mirrors the reference implementation in csv_train.cpp (lines 1436-1545).

#include <catboost/mlx/targets/target_func.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

namespace NCatboostMlx {

    // A directed pair of document indices within a single query group.
    // Winner has higher relevance than Loser.
    struct TPair {
        ui32  Winner;   // doc index with higher relevance
        ui32  Loser;    // doc index with lower relevance
        float Weight;   // contribution weight for this pair
    };

    // ---------------------------------------------------------------------------
    // Free functions — pair generation and gradient scatter
    // ---------------------------------------------------------------------------

    /// Generate all ordered (winner, loser) pairs within each group.
    /// For PairLogit: weight = 1.0 for every pair where targets[i] > targets[j].
    inline std::vector<TPair> GeneratePairLogitPairs(
        const std::vector<float>& targets,
        const std::vector<ui32>& groupOffsets,
        ui32 numGroups
    ) {
        std::vector<TPair> pairs;
        for (ui32 g = 0; g < numGroups; ++g) {
            const ui32 begin = groupOffsets[g];
            const ui32 end   = groupOffsets[g + 1];
            for (ui32 i = begin; i < end; ++i) {
                for (ui32 j = begin; j < end; ++j) {
                    if (targets[i] > targets[j]) {
                        pairs.push_back({i, j, 1.0f});
                    }
                }
            }
        }
        return pairs;
    }

    /// Generate YetiRank pairs: random permutation within each group, then
    /// adjacent-position pairs weighted by relevance difference and log position.
    /// The RNG is advanced in place so callers control the seed sequence.
    inline std::vector<TPair> GenerateYetiRankPairs(
        const std::vector<float>& targets,
        const std::vector<ui32>& groupOffsets,
        ui32 numGroups,
        std::mt19937& rng
    ) {
        std::vector<TPair> pairs;
        for (ui32 g = 0; g < numGroups; ++g) {
            const ui32 begin     = groupOffsets[g];
            const ui32 end       = groupOffsets[g + 1];
            const ui32 groupSize = end - begin;
            if (groupSize < 2) {
                continue;
            }

            // Random permutation of absolute doc indices within this group
            std::vector<ui32> perm(groupSize);
            std::iota(perm.begin(), perm.end(), begin);
            std::shuffle(perm.begin(), perm.end(), rng);

            // Adjacent pairs in the shuffled order
            for (ui32 pos = 1; pos < groupSize; ++pos) {
                const ui32  i         = perm[pos - 1];
                const ui32  j         = perm[pos];
                const float relevDiff = std::fabs(targets[i] - targets[j]);
                if (relevDiff < 1e-8f) {
                    continue;
                }
                const float weight = relevDiff / std::log2(2.0f + static_cast<float>(pos));
                if (targets[i] > targets[j]) {
                    pairs.push_back({i, j, weight});
                } else {
                    pairs.push_back({j, i, weight});
                }
            }
        }
        return pairs;
    }

    /// Scatter pairwise logistic gradients and hessians to per-document arrays.
    ///
    /// For each pair:
    ///   p = sigmoid(pred[winner] - pred[loser])
    ///   grad[winner] += w * (p - 1)   // push winner score up
    ///   grad[loser]  += w * (1 - p)   // push loser score down
    ///   hess[winner] += w * p*(1-p)
    ///   hess[loser]  += w * p*(1-p)
    /// Hessians are floored at 1e-6 to avoid degenerate Newton steps.
    inline void ScatterPairwiseGradients(
        const std::vector<TPair>& pairs,
        const float*              preds,   // raw predictions [numDocs]
        ui32                      numDocs,
        std::vector<float>&       grads,
        std::vector<float>&       hess
    ) {
        grads.assign(numDocs, 0.0f);
        hess.assign(numDocs, 0.0f);

        for (const auto& pair : pairs) {
            const float diff = preds[pair.Winner] - preds[pair.Loser];
            const float p    = 1.0f / (1.0f + std::exp(-diff));  // sigmoid(diff)
            const float w    = pair.Weight;

            grads[pair.Winner] += w * (p - 1.0f);
            grads[pair.Loser]  += w * (1.0f - p);

            const float h = w * p * (1.0f - p);
            hess[pair.Winner] += h;
            hess[pair.Loser]  += h;
        }

        // Floor hessians to avoid zero denominator in Newton step
        for (ui32 d = 0; d < numDocs; ++d) {
            hess[d] = std::max(hess[d], 1e-6f);
        }
    }

    // ---------------------------------------------------------------------------
    // TPairLogitTarget
    // ---------------------------------------------------------------------------

    /// PairLogit ranking loss.
    ///
    /// All ordered (winner, loser) pairs within each query group are generated
    /// once at construction time and reused unchanged across iterations.
    ///
    /// Loss = mean over pairs: w * log(1 + exp(-(pred[winner] - pred[loser])))
    class TPairLogitTarget : public IMLXTargetFunc {
    public:
        /// @param targets      CPU-side target/relevance values [numDocs]
        /// @param groupOffsets Group start offsets (size = numGroups + 1)
        /// @param numGroups    Number of query groups
        TPairLogitTarget(
            const std::vector<float>& targets,
            const std::vector<ui32>&  groupOffsets,
            ui32                      numGroups
        )
            : Targets_(targets)
            , NumDocs_(static_cast<ui32>(targets.size()))
            , Pairs_(GeneratePairLogitPairs(targets, groupOffsets, numGroups))
        {}

        ui32 GetApproxDimension() const override { return 1; }

        void ComputeDerivatives(
            const mx::array& cursor,
            const mx::array& /*targets*/,   // group targets stored in Targets_
            const mx::array& /*weights*/,   // pairwise losses don't use per-doc weights
            mx::array& gradients,
            mx::array& hessians
        ) const override {
            // Evaluate cursor to CPU
            mx::eval(cursor);
            const float* preds = cursor.data<float>();

            std::vector<float> grads, hess;
            ScatterPairwiseGradients(Pairs_, preds, NumDocs_, grads, hess);

            // Upload back to GPU as 1-D [numDocs] arrays then reshape to [1, numDocs]
            // to match the [approxDim, numDocs] layout expected by the training loop.
            auto g1d = mx::array(grads.data(), {static_cast<int>(NumDocs_)}, mx::float32);
            auto h1d = mx::array(hess.data(),  {static_cast<int>(NumDocs_)}, mx::float32);
            gradients = mx::reshape(g1d, {1, static_cast<int>(NumDocs_)});
            hessians  = mx::reshape(h1d, {1, static_cast<int>(NumDocs_)});

            TMLXDevice::EvalNow({gradients, hessians});
        }

        mx::array ComputeLoss(
            const mx::array& cursor,
            const mx::array& /*targets*/,
            const mx::array& /*weights*/
        ) const override {
            if (Pairs_.empty()) {
                return mx::array(0.0f);
            }

            mx::eval(cursor);
            const float* preds = cursor.data<float>();

            double totalLoss = 0.0;
            for (const auto& pair : Pairs_) {
                const float diff = preds[pair.Winner] - preds[pair.Loser];
                totalLoss += static_cast<double>(pair.Weight) *
                             std::log(1.0f + std::exp(-diff));
            }
            const float loss = static_cast<float>(totalLoss / Pairs_.size());
            return mx::array(loss);
        }

    private:
        std::vector<float> Targets_;    // CPU copy of relevance labels
        ui32               NumDocs_;
        std::vector<TPair> Pairs_;      // pre-generated at construction
    };

    // ---------------------------------------------------------------------------
    // TYetiRankTarget
    // ---------------------------------------------------------------------------

    /// YetiRank ranking loss.
    ///
    /// Pairs are regenerated from a fresh random permutation at the start of
    /// every ComputeDerivatives call, making each iteration's gradient stochastic.
    /// This is the key algorithmic difference from TPairLogitTarget.
    ///
    /// The RNG state is mutable so that ComputeDerivatives can advance it while
    /// still satisfying the const interface of IMLXTargetFunc.
    class TYetiRankTarget : public IMLXTargetFunc {
    public:
        /// @param targets      CPU-side target/relevance values [numDocs]
        /// @param groupOffsets Group start offsets (size = numGroups + 1)
        /// @param numGroups    Number of query groups
        /// @param seed         Seed for the internal Mersenne-Twister RNG
        TYetiRankTarget(
            const std::vector<float>& targets,
            const std::vector<ui32>&  groupOffsets,
            ui32                      numGroups,
            ui32                      seed = 42
        )
            : Targets_(targets)
            , GroupOffsets_(groupOffsets)
            , NumGroups_(numGroups)
            , NumDocs_(static_cast<ui32>(targets.size()))
            , Rng_(seed)
        {}

        ui32 GetApproxDimension() const override { return 1; }

        void ComputeDerivatives(
            const mx::array& cursor,
            const mx::array& /*targets*/,
            const mx::array& /*weights*/,
            mx::array& gradients,
            mx::array& hessians
        ) const override {
            // Regenerate pairs for this iteration (stochastic sampling)
            CurrentPairs_ = GenerateYetiRankPairs(Targets_, GroupOffsets_, NumGroups_, Rng_);

            // Evaluate cursor to CPU
            mx::eval(cursor);
            const float* preds = cursor.data<float>();

            std::vector<float> grads, hess;
            ScatterPairwiseGradients(CurrentPairs_, preds, NumDocs_, grads, hess);

            auto g1d = mx::array(grads.data(), {static_cast<int>(NumDocs_)}, mx::float32);
            auto h1d = mx::array(hess.data(),  {static_cast<int>(NumDocs_)}, mx::float32);
            gradients = mx::reshape(g1d, {1, static_cast<int>(NumDocs_)});
            hessians  = mx::reshape(h1d, {1, static_cast<int>(NumDocs_)});

            TMLXDevice::EvalNow({gradients, hessians});
        }

        mx::array ComputeLoss(
            const mx::array& cursor,
            const mx::array& /*targets*/,
            const mx::array& /*weights*/
        ) const override {
            // Report loss from the most recently generated pair set.
            // If ComputeDerivatives hasn't been called yet, return 0.
            if (CurrentPairs_.empty()) {
                return mx::array(0.0f);
            }

            mx::eval(cursor);
            const float* preds = cursor.data<float>();

            double totalLoss = 0.0;
            for (const auto& pair : CurrentPairs_) {
                const float diff = preds[pair.Winner] - preds[pair.Loser];
                totalLoss += static_cast<double>(pair.Weight) *
                             std::log(1.0f + std::exp(-diff));
            }
            const float loss = static_cast<float>(totalLoss / CurrentPairs_.size());
            return mx::array(loss);
        }

    private:
        std::vector<float> Targets_;
        std::vector<ui32>  GroupOffsets_;
        ui32               NumGroups_;
        ui32               NumDocs_;

        mutable std::mt19937       Rng_;          // advances every ComputeDerivatives call
        mutable std::vector<TPair> CurrentPairs_; // pairs from the most recent iteration
    };

}  // namespace NCatboostMlx
