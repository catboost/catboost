#pragma once

#include "greedy_snapshot.h"
#include "yeti_random.h"

#include <catboost/libs/helpers/progress_helper.h>
#include <catboost/libs/train_lib/train_model.h>

namespace NCB {
    // CUDA checkpoints save the training cursor together with trees and host
    // progress. Keep the same boundary: restoring never retrains completed trees.
    struct TMetalSnapshot {
        TString Params;
        ui32 Checksum = 0;
        float Bias = 0;
        float MvsLambda = 0;
        ui32 MvsLambdaIsSet = 0;
        TVector<ui32> Depths;
        TVector<ui32> SplitFeatures;
        TVector<ui32> SplitBins;
        TVector<ui8> SplitTypes;
        TVector<float> Leaves;
        TVector<float> Weights;
        TVector<float> Predictions;
        TVector<float> PermutationPredictions;
        TVector<float> PermutationMvsLambdas;
        TVector<ui8> PermutationMvsValid;
        TVector<ui8> UsedFeatures;
        TVector<float> OptimizationPredictions;
        TVector<ui32> OrderedDescriptors;
        TVector<float> OrderedCursors;
        ui64 OrderedRandomDrawCount = 0;
        ui32 OrderedRandomCompletedIterations = 0;
        bool OrderedBootstrapInitialized = false;
        TMetricsAndTimeLeftHistory History;
        // Symmetric v6 bytes stay unchanged. Non-symmetric training appends a
        // tagged compact-tree payload after the common cursor/history fields.
        bool Greedy = false;
        TMetalGreedySnapshotTrees GreedyTrees;
        bool YetiRank = false;
        TMetalYetiRandomState YetiRandom;
        // Enabled by the current configuration. Keep legacy v6 payloads
        // byte-for-byte unchanged when compound CTRs are disabled.
        bool TreeCtrs = false;
        TString TreeCtrState;
        TVector<ui32> TreeCtrCounts;
        TVector<float> TreeCtrWeights;
        TVector<ui8> TreeCtrFlags;
        TVector<ui8> TreeCtrUsed;
        TVector<ui8> TreeCtrActive;

        bool Load(const TString& path, ITrainingCallbacks* callbacks) {
            if (!TFsPath(path).Exists()) {
                return false;
            }
            const TString expectedParams = Params;
            const ui32 expectedChecksum = Checksum;
            bool loaded = false;
            TProgressHelper("CatBoost Metal snapshot v6").CheckedLoad(path, [&](IInputStream* in) {
                if (callbacks && !callbacks->OnLoadSnapshot(in)) {
                    return;
                }
                ::LoadMany(in, Params, Checksum, Bias, MvsLambda, MvsLambdaIsSet, Depths, SplitFeatures, SplitBins,
                    SplitTypes, Leaves, Weights, Predictions, History,
                    PermutationPredictions, PermutationMvsLambdas, PermutationMvsValid, UsedFeatures,
                    OptimizationPredictions, OrderedDescriptors, OrderedCursors,
                    OrderedRandomDrawCount, OrderedRandomCompletedIterations, OrderedBootstrapInitialized);
                if (Greedy) {
                    TString tag;
                    ::LoadMany(in, tag, GreedyTrees);
                    CB_ENSURE(tag == "Metal greedy trees v1", "Unknown Metal greedy snapshot payload");
                }
                if (YetiRank) {
                    TString tag;
                    ::LoadMany(in, tag, YetiRandom.DrawCount, YetiRandom.CompletedIterations, YetiRandom.BootstrapInitialized);
                    CB_ENSURE(tag == "Metal YetiRank random v1", "Unknown Metal YetiRank snapshot payload");
                }
                if (TreeCtrs) {
                    TString tag;
                    ::LoadMany(in, tag, TreeCtrState, TreeCtrCounts, TreeCtrWeights,
                        TreeCtrFlags, TreeCtrUsed, TreeCtrActive);
                    CB_ENSURE(tag == "Metal tree CTR features v1", "Unknown Metal tree CTR snapshot payload");
                    ValidateTreeCtrMetadata();
                }
                CB_ENSURE(NCatboostOptions::IsParamsCompatible(expectedParams, Params),
                          "Saved Metal snapshot parameters differ from the current parameters");
                CB_ENSURE(expectedChecksum == Checksum,
                          "Saved Metal snapshot training/evaluation data or initial model differ");
                loaded = true;
            });
            Params = expectedParams;
            return loaded;
        }

        void Save(const TString& path, const TString& device, ITrainingCallbacks* callbacks) const {
            TProgressHelper("CatBoost Metal snapshot v6").Write(path, [&](IOutputStream* out) {
                NJson::TJsonValue processors(NJson::JSON_ARRAY);
                processors.AppendValue(device);
                if (callbacks) {
                    callbacks->OnSaveSnapshot(processors, out);
                }
                ::SaveMany(out, Params, Checksum, Bias, MvsLambda, MvsLambdaIsSet, Depths, SplitFeatures, SplitBins,
                    SplitTypes, Leaves, Weights, Predictions, History,
                    PermutationPredictions, PermutationMvsLambdas, PermutationMvsValid, UsedFeatures,
                    OptimizationPredictions, OrderedDescriptors, OrderedCursors,
                    OrderedRandomDrawCount, OrderedRandomCompletedIterations, OrderedBootstrapInitialized);
                if (Greedy) ::SaveMany(out, TString("Metal greedy trees v1"), GreedyTrees);
                if (YetiRank) ::SaveMany(out, TString("Metal YetiRank random v1"),
                    YetiRandom.DrawCount, YetiRandom.CompletedIterations, YetiRandom.BootstrapInitialized);
                if (TreeCtrs) {
                    ValidateTreeCtrMetadata();
                    ::SaveMany(out, TString("Metal tree CTR features v1"), TreeCtrState,
                        TreeCtrCounts, TreeCtrWeights, TreeCtrFlags, TreeCtrUsed, TreeCtrActive);
                }
            });
        }

        void ValidateTreeCtrMetadata() const {
            const ui64 count = TreeCtrCounts.size();
            CB_ENSURE(TreeCtrs && !TreeCtrState.empty() && count &&
                      count <= (1ull << 30) / 11 && TreeCtrWeights.size() == count &&
                      TreeCtrFlags.size() == count && TreeCtrUsed.size() == count &&
                      TreeCtrActive.size() == count,
                      "Saved Metal tree CTR feature metadata has inconsistent dimensions");
            for (ui64 feature = 0; feature < count; ++feature) {
                CB_ENSURE(std::isfinite(TreeCtrWeights[feature]) && TreeCtrWeights[feature] >= 0 &&
                          TreeCtrFlags[feature] <= 3 && TreeCtrUsed[feature] <= 1 &&
                          TreeCtrActive[feature] <= 1 &&
                          (!TreeCtrUsed[feature] || (TreeCtrCounts[feature] && (TreeCtrFlags[feature] & 2))),
                          "Saved Metal tree CTR feature metadata is invalid");
            }
        }

        void ValidateGreedy(ui32 rows, ui32 features, ui32 policy, ui32 depth, ui32 maxLeaves, ui32 iterations,
                           ui32 permutationCount = 1, ui32 approxDimension = 1, ui32 optimizerDimension = 0) const {
            CB_ENSURE(approxDimension >= 1 && approxDimension <= 64 &&
                (approxDimension == 1 ? optimizerDimension == 0 :
                    (optimizerDimension == approxDimension || optimizerDimension + 1 == approxDimension)),
                "Saved Metal greedy snapshot has invalid approximation dimensions");
            CB_ENSURE(Greedy && Predictions.size() == ui64(rows) * approxDimension && History.TimeHistory.size() == Depths.size() &&
                SplitFeatures.empty() && SplitBins.empty() && SplitTypes.empty() && Leaves.empty() && Weights.empty() &&
                UsedFeatures.empty() && OptimizationPredictions.size() == ui64(rows) * permutationCount * optimizerDimension &&
                OrderedDescriptors.empty() && OrderedCursors.empty() && !OrderedRandomDrawCount &&
                !OrderedRandomCompletedIterations && !OrderedBootstrapInitialized,
                "Saved Metal greedy snapshot has inconsistent array sizes");
            const ui64 count = permutationCount > 1 || approxDimension > 1 ? permutationCount : 0;
            CB_ENSURE(PermutationPredictions.size() == count * rows * approxDimension &&
                PermutationMvsLambdas.size() == count && PermutationMvsValid.size() == count,
                "Saved Metal greedy snapshot has inconsistent permutation state");
            for (float value : OptimizationPredictions) CB_ENSURE(std::isfinite(value), "Nonfinite greedy optimizer cursor");
            for (float value : Predictions) CB_ENSURE(std::isfinite(value), "Nonfinite greedy snapshot cursor");
            for (float value : PermutationPredictions) CB_ENSURE(std::isfinite(value), "Nonfinite greedy permutation cursor");
            for (ui32 p = 0; p < count; ++p) CB_ENSURE(PermutationMvsLambdas[p] == 0 && PermutationMvsValid[p] == 0,
                "Saved Metal greedy snapshot contains unsupported MVS state");
            if (count) CB_ENSURE(std::equal(Predictions.begin(), Predictions.end(),
                PermutationPredictions.begin() + (count - 1) * rows * approxDimension), "Greedy snapshot export cursor differs from its last dataset");
            GreedyTrees.Validate(features, policy, depth, maxLeaves, iterations, approxDimension);
            CB_ENSURE(Depths == GreedyTrees.Depths, "Saved Metal greedy snapshot has inconsistent tree depths");
        }

        void Validate(ui32 rows, ui32 maxDepth, ui32 maxIterations, ui32 approxDimension = 1,
                      ui32 permutationCount = 1, ui32 optimizerDimension = 0, bool ordered = false,
                      bool featureParallel = false) const {
            const ui64 trees = Depths.size();
            CB_ENSURE(trees <= maxIterations && SplitFeatures.size() == trees * maxDepth &&
                      SplitBins.size() == trees * maxDepth && SplitTypes.size() == trees * maxDepth &&
                      Leaves.size() == trees * (1u << maxDepth) * approxDimension &&
                      Weights.size() == trees * (1u << maxDepth) &&
                      Predictions.size() == ui64(rows) * approxDimension && History.TimeHistory.size() == trees,
                      "Saved Metal snapshot has inconsistent array sizes");
            if (ordered) {
                CB_ENSURE(approxDimension == 1 && OrderedDescriptors.size() >= 8 &&
                          OrderedDescriptors.size() % 4 == 0 && OrderedCursors.size() >= rows &&
                          PermutationPredictions.empty() && PermutationMvsLambdas.empty() &&
                          PermutationMvsValid.empty() && OptimizationPredictions.empty() && UsedFeatures.empty() &&
                          OrderedRandomCompletedIterations == trees && OrderedBootstrapInitialized == (trees != 0),
                          "Saved Metal snapshot has inconsistent Ordered prefix state");
            } else {
                CB_ENSURE(OrderedDescriptors.empty() && OrderedCursors.empty() &&
                          (featureParallel ? OrderedRandomCompletedIterations == trees &&
                              OrderedBootstrapInitialized == (trees != 0) :
                              !OrderedRandomDrawCount && !OrderedRandomCompletedIterations && !OrderedBootstrapInitialized) &&
                          PermutationPredictions.size() == ui64(rows) * permutationCount * approxDimension &&
                          PermutationMvsLambdas.size() == permutationCount && PermutationMvsValid.size() == permutationCount &&
                          OptimizationPredictions.size() == ui64(rows) * permutationCount * optimizerDimension,
                          "Saved Metal snapshot has inconsistent permutation state");
            }
            for (ui32 depth : Depths) {
                CB_ENSURE(depth <= maxDepth, "Saved Metal snapshot has an invalid tree depth");
            }
        }
    };
}
