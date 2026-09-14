#pragma once

#include "greedy_snapshot.h"
#include "yeti_random.h"
#include "langevin_random.h"

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
        // Optional trailing payload preserves the retained-best online cursor
        // for the standalone native frontend. Older v6 snapshots end before
        // this payload; their training/model recovery remains unchanged.
        TVector<float> BestLearnPredictions;
        i32 BestLearnIteration = -1;
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
        bool CombinationYeti = false;
        TMetalYetiRandomState YetiRandom;
        TVector<ui32> YetiSearchAttempts;
        // Present only for langevin=true. The host stream includes stochastic
        // targets, actual scorer calls, and every accepted/rejected leaf trial.
        // Metal's weak device noise uses stateless item/iteration domains;
        // only the shared host seed-cache initialization needs persistence.
        bool Langevin = false;
        TMetalLangevinRandomState LangevinRandom;
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
                BestLearnPredictions.clear();
                BestLearnIteration = -1;
                LangevinRandom = {};
                ::LoadMany(in, Params, Checksum, Bias, MvsLambda, MvsLambdaIsSet, Depths, SplitFeatures, SplitBins,
                    SplitTypes, Leaves, Weights, Predictions, History,
                    PermutationPredictions, PermutationMvsLambdas, PermutationMvsValid, UsedFeatures,
                    OptimizationPredictions, OrderedDescriptors, OrderedCursors,
                    OrderedRandomDrawCount, OrderedRandomCompletedIterations, OrderedBootstrapInitialized);
                // Optional payload types depend on the current trainer. Check
                // compatibility before reading them when options have changed.
                CB_ENSURE(NCatboostOptions::IsParamsCompatible(expectedParams, Params),
                          "Saved Metal snapshot parameters differ from the current parameters");
                CB_ENSURE(expectedChecksum == Checksum,
                          "Saved Metal snapshot training/evaluation data or initial model differ");
                if (Greedy) {
                    TString tag;
                    ::LoadMany(in, tag, GreedyTrees);
                    CB_ENSURE(tag == "Metal greedy trees v1", "Unknown Metal greedy snapshot payload");
                }
                if (YetiRank) {
                    TString tag;
                    ::LoadMany(in, tag, YetiRandom.DrawCount, YetiRandom.CompletedIterations, YetiRandom.BootstrapInitialized);
                    CB_ENSURE(tag == "Metal YetiRank random v1", "Unknown Metal YetiRank snapshot payload");
                    if (Greedy) {
                        ::LoadMany(in, tag, YetiSearchAttempts);
                        CB_ENSURE(tag == "Metal greedy YetiRank search v1" && YetiSearchAttempts.size() == Depths.size(),
                            "Unknown or inconsistent Metal greedy YetiRank search payload");
                    }
                }
                if (CombinationYeti) {
                    TString tag;
                    ::LoadMany(in, tag, YetiRandom.DrawCount, YetiRandom.CompletedIterations, YetiRandom.BootstrapInitialized);
                    CB_ENSURE(tag == "Metal Combination target random v1", "Unknown Metal Combination random payload");
                }
                if (TreeCtrs) {
                    TString tag;
                    ::LoadMany(in, tag, TreeCtrState, TreeCtrCounts, TreeCtrWeights,
                        TreeCtrFlags, TreeCtrUsed, TreeCtrActive);
                    CB_ENSURE(tag == "Metal tree CTR features v1", "Unknown Metal tree CTR snapshot payload");
                    ValidateTreeCtrMetadata();
                }
                // Older v6 snapshots end here. Optional records have a fixed
                // order: a best cursor, then the Langevin stream. Either may
                // be absent, so Langevin-only checkpoints need no fake cursor.
                bool bestLoaded = false, langevinLoaded = false;
                for (;;) {
                    ui32 tag = 0;
                    const size_t bytes = in->Read(&tag, sizeof(tag));
                    if (!bytes) break;
                    CB_ENSURE(bytes == sizeof(tag), (bestLoaded ? "Saved Metal best-learn cursor has trailing data" :
                        "Unknown Metal best-learn snapshot payload"));
                    if (tag == 0x4D424C31 || tag == 0x4D424C32) {
                        CB_ENSURE(!bestLoaded && !langevinLoaded,
                            "Saved Metal best-learn cursor has trailing data");
                        bestLoaded = true;
                        if (tag == 0x4D424C32) ::Load(in, BestLearnIteration);
                        ::Load(in, BestLearnPredictions);
                        CB_ENSURE(BestLearnPredictions.size() == Predictions.size(),
                            "Saved Metal best-learn cursor has inconsistent dimensions");
                        CB_ENSURE(tag == 0x4D424C31 ||
                            (BestLearnIteration >= 0 && ui64(BestLearnIteration) < Depths.size()),
                            "Saved Metal best-learn cursor has an invalid iteration");
                        for (float value : BestLearnPredictions) {
                            CB_ENSURE(std::isfinite(value), "Saved Metal best-learn cursor is nonfinite");
                        }
                    } else if (tag == 0x4D4C4731) { // Metal Langevin random v1
                        CB_ENSURE(Langevin && !langevinLoaded,
                            "Unexpected or duplicate Metal Langevin snapshot payload");
                        langevinLoaded = true;
                        ::LoadMany(in, LangevinRandom.DrawCount, LangevinRandom.CompletedIterations,
                            LangevinRandom.WeakSeedCacheInitialized);
                        CB_ENSURE(LangevinRandom.CompletedIterations == Depths.size(),
                            "Saved Metal Langevin random iteration count differs from its trees");
                    } else {
                        CB_ENSURE(false, (bestLoaded ? "Saved Metal best-learn cursor has trailing data" :
                            "Unknown Metal best-learn snapshot payload"));
                    }
                }
                CB_ENSURE(langevinLoaded == Langevin, "Saved Metal Langevin random payload is missing");
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
                if (YetiRank && Greedy) ::SaveMany(out, TString("Metal greedy YetiRank search v1"), YetiSearchAttempts);
                if (CombinationYeti) ::SaveMany(out, TString("Metal Combination target random v1"),
                    YetiRandom.DrawCount, YetiRandom.CompletedIterations, YetiRandom.BootstrapInitialized);
                if (TreeCtrs) {
                    ValidateTreeCtrMetadata();
                    ::SaveMany(out, TString("Metal tree CTR features v1"), TreeCtrState,
                        TreeCtrCounts, TreeCtrWeights, TreeCtrFlags, TreeCtrUsed, TreeCtrActive);
                }
                if (!BestLearnPredictions.empty()) {
                    CB_ENSURE(BestLearnPredictions.size() == Predictions.size() &&
                        BestLearnIteration >= 0 && ui64(BestLearnIteration) < Depths.size(),
                        "Metal best-learn snapshot cursor has inconsistent dimensions");
                    ::SaveMany(out, ui32(0x4D424C32), BestLearnIteration, BestLearnPredictions);
                }
                if (Langevin) {
                    CB_ENSURE(LangevinRandom.CompletedIterations == Depths.size(),
                        "Metal Langevin snapshot random iteration count differs from its trees");
                    ::SaveMany(out, ui32(0x4D4C4731), LangevinRandom.DrawCount,
                        LangevinRandom.CompletedIterations, LangevinRandom.WeakSeedCacheInitialized);
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
                           ui32 permutationCount = 1, ui32 approxDimension = 1, ui32 optimizerDimension = 0,
                           bool allowSignedLeafWeights = false) const {
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
            GreedyTrees.Validate(features, policy, depth, maxLeaves, iterations, approxDimension, allowSignedLeafWeights);
            CB_ENSURE(Depths == GreedyTrees.Depths, "Saved Metal greedy snapshot has inconsistent tree depths");
        }

        void Validate(ui32 rows, ui32 maxDepth, ui32 maxIterations, ui32 approxDimension = 1,
                      ui32 permutationCount = 1, ui32 optimizerDimension = 0, bool ordered = false,
                      bool featureParallel = false, bool allowSignedLeafWeights = false) const {
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
                          (Langevin ? !OrderedRandomDrawCount && !OrderedRandomCompletedIterations && !OrderedBootstrapInitialized :
                              OrderedRandomCompletedIterations == trees && OrderedBootstrapInitialized == (trees != 0)),
                          "Saved Metal snapshot has inconsistent Ordered prefix state");
            } else {
                CB_ENSURE(OrderedDescriptors.empty() && OrderedCursors.empty() &&
                          (featureParallel && !Langevin ? OrderedRandomCompletedIterations == trees &&
                              OrderedBootstrapInitialized == (trees != 0) :
                              !OrderedRandomDrawCount && !OrderedRandomCompletedIterations && !OrderedBootstrapInitialized) &&
                          PermutationPredictions.size() == ui64(rows) * permutationCount * approxDimension &&
                          PermutationMvsLambdas.size() == permutationCount && PermutationMvsValid.size() == permutationCount &&
                          OptimizationPredictions.size() == ui64(rows) * permutationCount * optimizerDimension,
                          "Saved Metal snapshot has inconsistent permutation state");
            }
            for (ui64 tree = 0; tree < trees; ++tree) {
                const ui32 depth = Depths[tree];
                CB_ENSURE(depth <= maxDepth, "Saved Metal snapshot has an invalid tree depth");
                // Permission comes from the current objective/estimator, not
                // the snapshot. Ignore unused padding beyond the actual tree.
                const ui64 offset = tree * (1u << maxDepth);
                for (ui32 leaf = 0; leaf < (1u << depth); ++leaf) {
                    CB_ENSURE(std::isfinite(Weights[offset + leaf]) &&
                              (allowSignedLeafWeights || Weights[offset + leaf] >= 0),
                              "Saved Metal snapshot has invalid leaf weights");
                }
            }
        }
    };
}
