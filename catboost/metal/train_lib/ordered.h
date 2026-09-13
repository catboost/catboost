#pragma once

#include "ordered_random.h"

#include <catboost/libs/data/objects_grouping.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/metal/native/metal_ordered_trainer.h>

#include <util/generic/array_ref.h>
#include <util/generic/vector.h>

#include <algorithm>
#include <functional>
#include <numeric>

namespace NCB {
    struct TMetalOrderedState {
        // [task][4]: estimateEnd, qualityEnd, cursorOffset, permutationId.
        // The final task estimates the independent full model.
        TVector<ui32> Descriptors;
        TVector<float> Cursors;
    };

    struct TMetalOrderedBootstrapState {
        ui32 AbsoluteIterations = 0;
        float MvsLambda = 0;
        ui32 MvsLambdaIsSet = 0;
    };

    // Owns the numeric/one-hot Ordered session, prefix cursors and the persistent
    // CUDA FeatureParallel host chooser, with group-preserving permutations
    // and independent group-aligned prefix lists when the Pool is grouped.
    class TMetalOrderedSession {
    public:
        TMetalOrderedSession(
            const CBMOrderedParams& params,
            TConstArrayRef<ui8> bins,
            TConstArrayRef<float> targets,
            TConstArrayRef<float> weights,
            TConstArrayRef<float> initialPredictions,
            TConstArrayRef<ui32> candidateFeatures,
            TConstArrayRef<ui32> candidateBins,
            ui64 randomSeed,
            const TObjectsGrouping& grouping,
            TConstArrayRef<TVector<ui32>> permutationOrders = {},
            TConstArrayRef<ui8> candidateTypes = {},
            double groupFoldGrowth = 0,
            TConstArrayRef<TVector<ui8>> additionalBins = {},
            bool dynamicTreeCtrs = false)
            : Params(params)
            , RandomSeed(randomSeed)
            , Random(randomSeed, params.permutations, params.depth, params.candidates, dynamicTreeCtrs)
        {
            CB_ENSURE(permutationOrders.empty() ? Params.permutations == 1 :
                      permutationOrders.size() == Params.permutations,
                "Metal Ordered history maps must match the actual permutation count");
            CB_ENSURE(grouping.GetGroupCount() >= 4,
                "Metal Ordered requires at least four groups or documents");
            CB_ENSURE(Params.rows >= 4 && Params.rows <= (1u << 24) && grouping.GetObjectCount() == Params.rows,
                "Metal Ordered requires between 4 and 16777216 objects with matching grouping");
            CB_ENSURE(Params.features && bins.size() == ui64(Params.features) * Params.rows,
                "Metal Ordered bins must contain one value per feature and object");
            CB_ENSURE(targets.size() == Params.rows &&
                (weights.empty() || weights.size() == Params.rows) &&
                (initialPredictions.empty() || initialPredictions.size() == Params.rows),
                "Metal Ordered target, weight, and initial prediction dimensions must match the objects");
            CB_ENSURE(candidateFeatures.size() == Params.candidates && candidateBins.size() == Params.candidates,
                "Metal Ordered candidate dimensions differ from the configured count");
            CB_ENSURE(candidateTypes.empty() || candidateTypes.size() == Params.candidates,
                "Metal Ordered candidate type count differs from the configured count");
            CB_ENSURE(Params.leaf_method <= 1,
                "Native Metal Ordered currently supports Newton or Gradient leaf estimation");

            CB_ENSURE(ui64(Params.permutations) * Params.rows * sizeof(ui32) <= (1ull << 30),
                "Metal Ordered history maps exceed the 1 GiB limit");
            TVector<ui32> maps;
            maps.reserve(ui64(Params.permutations) * Params.rows);
            if (permutationOrders.empty()) {
                // CUDA data/permutation.cpp::FillOrder gives P0 identity.
                maps.resize(Params.rows);
                std::iota(maps.begin(), maps.end(), 0);
            } else {
                for (const auto& order : permutationOrders) {
                    CB_ENSURE(order.size() == Params.rows,
                        "Metal Ordered history map dimensions must match the objects");
                    maps.insert(maps.end(), order.begin(), order.end());
                }
            }
            TVector<ui32> groupOffsets;
            if (!grouping.IsTrivial()) {
                groupOffsets.reserve(grouping.GetGroupCount() + 1); groupOffsets.push_back(0);
                for (const auto& group : grouping.GetNonTrivialGroups()) groupOffsets.push_back(group.End);
            }
            CB_ENSURE(additionalBins.empty() || additionalBins.size() + 1 == Params.permutations,
                "Metal Ordered feature bank count must match the history maps");
            CB_ENSURE(ui64(bins.size()) * (additionalBins.size() + 1) <= (1ull << 30),
                "Metal Ordered feature banks exceed 1 GiB");
            TVector<ui8> allBins;
            if (!additionalBins.empty()) {
                allBins.reserve(ui64(bins.size()) * (additionalBins.size() + 1));
                allBins.insert(allBins.end(), bins.begin(), bins.end());
                for (const auto& bank : additionalBins) {
                    CB_ENSURE(bank.size() == bins.size(), "Metal Ordered feature bank dimensions differ");
                    allBins.insert(allBins.end(), bank.begin(), bank.end());
                }
            }
            char error[2048] = {};
            const bool typed = std::any_of(candidateTypes.begin(), candidateTypes.end(), [](ui8 type) { return type != 0; });
            const int status = !allBins.empty()
                ? cbm_ordered_session_create_banked(&Params, Params.permutations, allBins.size(), allBins.data(),
                    targets.data(), weights.empty() ? nullptr : weights.data(), initialPredictions.empty() ? nullptr : initialPredictions.data(),
                    candidateFeatures.data(), candidateBins.data(), candidateTypes.empty() ? nullptr : candidateTypes.data(), maps.data(),
                    groupOffsets.empty() ? 0 : grouping.GetGroupCount(), groupOffsets.empty() ? nullptr : groupOffsets.data(),
                    groupFoldGrowth ? groupFoldGrowth : Params.fold_growth, &Session.Value, error, sizeof(error))
                : !groupOffsets.empty()
                ? cbm_ordered_session_create_grouped(&Params, bins.data(), targets.data(),
                    weights.empty() ? nullptr : weights.data(), initialPredictions.empty() ? nullptr : initialPredictions.data(),
                    candidateFeatures.data(), candidateBins.data(), candidateTypes.empty() ? nullptr : candidateTypes.data(), maps.data(),
                    grouping.GetGroupCount(), groupOffsets.data(), groupFoldGrowth ? groupFoldGrowth : Params.fold_growth,
                    &Session.Value, error, sizeof(error))
                : typed ? cbm_ordered_session_create_typed(&Params, bins.data(), targets.data(),
                    weights.empty() ? nullptr : weights.data(), initialPredictions.empty() ? nullptr : initialPredictions.data(),
                    candidateFeatures.data(), candidateBins.data(), candidateTypes.data(), maps.data(),
                    &Session.Value, error, sizeof(error))
                : cbm_ordered_session_create(&Params, bins.data(), targets.data(),
                    weights.empty() ? nullptr : weights.data(), initialPredictions.empty() ? nullptr : initialPredictions.data(),
                    candidateFeatures.data(), candidateBins.data(), maps.data(), &Session.Value, error, sizeof(error));
            CB_ENSURE(status == 0, "Metal Ordered initialization failed: " << error);

            // The runtime builds each permutation's numeric or group-aligned
            // folds from min_fold_size and fold growth; lengths may differ.
            CB_ENSURE(cbm_ordered_session_state_shape(Session.Value, &TaskCount, &CursorCount,
                error, sizeof(error)) == 0, "Metal Ordered state shape query failed: " << error);
            CB_ENSURE(TaskCount > 1 && CursorCount >= Params.rows,
                "Metal Ordered returned an invalid prefix-state layout");

            CBMBootstrapOptions bootstrap = {};
            bootstrap.random_seed_low = static_cast<ui32>(RandomSeed);
            bootstrap.random_seed_high = static_cast<ui32>(RandomSeed >> 32);
            bootstrap.bagging_temperature = 1;
            bootstrap.subsample = 1;
            SetBootstrap(bootstrap, true);
        }

        TMetalOrderedSession(const TMetalOrderedSession&) = delete;
        TMetalOrderedSession& operator=(const TMetalOrderedSession&) = delete;

        void* GetHandle() const noexcept {
            return Session.Value;
        }

        ui64 GetStateBytes() const noexcept {
            return ui64(TaskCount) * 4 * sizeof(ui32) + ui64(CursorCount) * sizeof(float);
        }

        CBMStepInfo GetInfo() const {
            CBMStepInfo result = {};
            char error[2048] = {};
            CB_ENSURE(cbm_ordered_session_info(Session.Value, &result, error, sizeof(error)) == 0,
                "Metal Ordered info query failed: " << error);
            return result;
        }

        void SetBootstrap(const CBMBootstrapOptions& options, bool testOnly) {
            CB_ENSURE((ui64(options.random_seed_high) << 32 | options.random_seed_low) == RandomSeed,
                "Metal Ordered bootstrap seed must match the session seed");
            char error[2048] = {};
            CB_ENSURE(cbm_ordered_session_set_bootstrap(Session.Value, &options, testOnly,
                error, sizeof(error)) == 0, "Metal Ordered bootstrap configuration failed: " << error);
            IterationOffset = options.iteration_offset;
        }

        void SetScoreNoise(const CBMScoreNoiseOptions& options) {
            char error[2048] = {};
            CB_ENSURE(cbm_ordered_session_set_score_noise(Session.Value, &options, error, sizeof(error)) == 0,
                "Metal Ordered score noise configuration failed: " << error);
        }

        void SetBacktracking(ui32 type) {
            char error[2048] = {};
            CB_ENSURE(cbm_ordered_session_set_backtracking(Session.Value, type, error, sizeof(error)) == 0,
                "Metal Ordered backtracking configuration failed: " << error);
        }

        TMetalOrderedBootstrapState GetBootstrapState() const {
            TMetalOrderedBootstrapState result;
            char error[2048] = {};
            CB_ENSURE(cbm_ordered_session_get_bootstrap_state(Session.Value, &result.AbsoluteIterations,
                &result.MvsLambda, &result.MvsLambdaIsSet, error, sizeof(error)) == 0,
                "Metal Ordered bootstrap state copy failed: " << error);
            return result;
        }

        TMetalOrderedRandomState GetRandomState() const {
            return Random.GetState();
        }

        void RestoreRandomState(const TMetalOrderedRandomState& state, ui32 restoredIterations) {
            CB_ENSURE(CompletedIterations == 0,
                "Restore the Metal Ordered random stream before the first training step");
            Random.RestoreState(state, restoredIterations);
        }

        void Step(
            ui64 absoluteIteration,
            CBMStepInfo* info,
            ui32* depth,
            TArrayRef<ui32> splitFeatures,
            TArrayRef<ui32> splitBins,
            TArrayRef<ui8> splitTypes,
            TArrayRef<float> leafValues,
            TArrayRef<float> leafWeights)
        {
            CB_ENSURE(absoluteIteration == ui64(IterationOffset) + CompletedIterations,
                "Metal Ordered step must use the next absolute iteration");
            CB_ENSURE(info && depth && splitFeatures.size() >= Params.depth &&
                splitBins.size() >= Params.depth && splitTypes.size() >= Params.depth &&
                leafValues.size() >= (1u << Params.depth) && leafWeights.size() >= (1u << Params.depth),
                "Metal Ordered output buffers are smaller than the configured tree strides");
            const ui32 selected = Random.SelectPermutation();
            char error[2048] = {};
            CB_ENSURE(cbm_ordered_session_step(Session.Value, selected, info, depth,
                splitFeatures.data(), splitBins.data(), splitTypes.data(), leafValues.data(), leafWeights.data(),
                error, sizeof(error)) == 0,
                "Metal Ordered iteration " << absoluteIteration << " failed: " << error);
            ++CompletedIterations;
            CB_ENSURE(info->completed_iterations == CompletedIterations && *depth <= Params.depth,
                "Metal Ordered returned inconsistent iteration or depth information");
            Random.FinishIteration(*depth);
        }

        // The controller appends newly generated CTR columns after each
        // selected split. All prefix cursors and the one bootstrap draw stay
        // resident throughout this begin/grow/finish sequence.
        void StepDynamic(
            ui64 absoluteIteration,
            CBMStepInfo* info,
            ui32* depth,
            TArrayRef<ui32> splitFeatures,
            TArrayRef<ui32> splitBins,
            TArrayRef<ui8> splitTypes,
            TArrayRef<float> leafValues,
            TArrayRef<float> leafWeights,
            const std::function<void(ui32)>& begin,
            const std::function<void(ui32, const CBMStructureInfo&)>& split,
            const std::function<ui32()>& scoreDraws)
        {
            CB_ENSURE(absoluteIteration == ui64(IterationOffset) + CompletedIterations,
                "Metal Ordered dynamic step must use the next absolute iteration");
            CB_ENSURE(info && depth && splitFeatures.size() >= Params.depth &&
                splitBins.size() >= Params.depth && splitTypes.size() >= Params.depth &&
                leafValues.size() >= (1u << Params.depth) && leafWeights.size() >= (1u << Params.depth) &&
                begin && split && scoreDraws,
                "Metal Ordered dynamic step requires full output buffers and CTR callbacks");
            const ui32 selected = Random.SelectPermutation();
            begin(selected);
            char error[2048] = {};
            CB_ENSURE(cbm_ordered_session_begin_tree(Session.Value, selected, error, sizeof(error)) == 0,
                "Metal Ordered dynamic tree initialization failed: " << error);
            CBMStructureInfo structure = {};
            ui32 searchDrawCount = 0;
            do {
                if (Params.depth && Params.candidates) searchDrawCount += scoreDraws();
                CB_ENSURE(cbm_ordered_session_grow_tree(Session.Value, &structure, error, sizeof(error)) == 0,
                    "Metal Ordered dynamic split search failed: " << error);
                if (structure.has_split) split(selected, structure);
            } while (!structure.finished);
            CB_ENSURE(cbm_ordered_session_finish_tree(Session.Value, info, depth,
                splitFeatures.data(), splitBins.data(), splitTypes.data(), leafValues.data(), leafWeights.data(),
                error, sizeof(error)) == 0,
                "Metal Ordered dynamic leaf estimation failed: " << error);
            ++CompletedIterations;
            CB_ENSURE(info->completed_iterations == CompletedIterations && *depth <= Params.depth,
                "Metal Ordered dynamic step returned inconsistent iteration or depth information");
            Random.FinishIterationWithDraws(*depth, searchDrawCount);
        }

        void SetFeaturePenalties(TConstArrayRef<ui32> counts, float modelSizeReg) {
            CB_ENSURE(counts.size() == Params.features, "Metal Ordered CTR counts must match features");
            CBMFeaturePenaltyOptions options = {modelSizeReg, 0, 0, 0}; char error[2048] = {};
            CB_ENSURE(cbm_ordered_session_set_feature_penalties(Session.Value, &options, counts.data(), nullptr,
                error, sizeof(error)) == 0, "Metal Ordered CTR penalty setup failed: " << error);
        }

        void CopyPredictions(TArrayRef<float> predictions) const {
            CB_ENSURE(predictions.size() == Params.rows,
                "Metal Ordered prediction output must match the object count");
            char error[2048] = {};
            CB_ENSURE(cbm_ordered_session_copy_predictions(Session.Value, predictions.data(), error, sizeof(error)) == 0,
                "Metal Ordered prediction copy failed: " << error);
        }

        TMetalOrderedState CopyState() const {
            TMetalOrderedState result;
            result.Descriptors.resize(ui64(TaskCount) * 4);
            result.Cursors.resize(CursorCount);
            char error[2048] = {};
            CB_ENSURE(cbm_ordered_session_copy_state(Session.Value, TaskCount, CursorCount,
                result.Descriptors.data(), result.Cursors.data(), error, sizeof(error)) == 0,
                "Metal Ordered prefix state copy failed: " << error);
            return result;
        }

        void RestoreState(const TMetalOrderedState& state) {
            CB_ENSURE(CompletedIterations == 0,
                "Metal Ordered prefix state must be restored before the first training step");
            CB_ENSURE(state.Descriptors.size() == ui64(TaskCount) * 4 && state.Cursors.size() == CursorCount,
                "Saved Metal Ordered prefix state has different dimensions");
            // Dataset/options fingerprints belong to the native snapshot
            // adapter. Check exact fold boundaries and offsets here as well.
            const auto expected = CopyState();
            CB_ENSURE(state.Descriptors == expected.Descriptors,
                "Saved Metal Ordered fold descriptors do not match this session");
            char error[2048] = {};
            CB_ENSURE(cbm_ordered_session_restore_cursors(Session.Value, CursorCount, state.Cursors.data(),
                error, sizeof(error)) == 0, "Metal Ordered prefix state restore failed: " << error);
        }

    private:
        struct TSessionHandle {
            void* Value = nullptr;

            ~TSessionHandle() {
                cbm_ordered_session_close(Value);
            }
        };

        const CBMOrderedParams Params;
        const ui64 RandomSeed;
        TMetalOrderedRandom Random;
        TSessionHandle Session;
        ui32 TaskCount = 0;
        ui32 CursorCount = 0;
        ui32 IterationOffset = 0;
        ui32 CompletedIterations = 0;
    };
}
