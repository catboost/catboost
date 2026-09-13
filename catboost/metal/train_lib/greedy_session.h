#pragma once

#include <catboost/libs/helpers/exception.h>
#include <catboost/metal/native/metal_greedy_trainer.h>

#include <util/generic/array_ref.h>
#include <util/generic/vector.h>

namespace NCB {
    struct TMetalGreedyTree {
        CBMGreedyStepInfo Info = {};
        // Root index0; internal nodes have leaf=Max<ui32>(). Terminal leaf
        // IDs address Values/Weights independently of the flat node order.
        TVector<CBMGreedyNode> Nodes;
        TVector<float> Values;
        TVector<float> Weights;
    };

    struct TMetalGreedyBootstrapState {
        ui32 AbsoluteIterations = 0;
        float MvsLambda = 0;
        ui32 MvsLambdaIsSet = 0;
    };

    // Owning adapter for Plain scalar Depthwise/Lossguide/Region sessions.
    // It starts with one prepared feature-major matrix and original object
    // weights. Histograms, routing, leaf estimation, and cursor updates run
    // on Metal. Ordinary trainer option gates retain CUDA's registrations:
    // native objective6 (Lq) is a private Metal extension, not registered by
    // CUDA's greedy trainer factory, and MVS is unsupported for these policies.
    //
    // TMetalDocParallelPermutations can attach exclusive CTR history banks
    // and independent cursors. Unlike symmetric CUDA search, these policies
    // do not apply changing model_size_reg penalties to selected CTRs.
    // Resume uses the FULL saved raw cursor as initialPredictions and sets
    // bootstrap.iteration_offset to the next absolute tree index. There is
    // no hidden per-tree RNG state or MVS state. Persist the returned numeric
    // node/leaf arrays, model metadata, metric history and all permutation
    // cursors in the outer native snapshot/controller. A trimmed best-model
    // cursor is not a resume cursor.
    class TMetalGreedySession {
    public:
        TMetalGreedySession(
            const CBMGreedyTrainParams& params,
            const CBMObjectiveOptions& objectiveOptions,
            TConstArrayRef<ui8> bins,
            TConstArrayRef<float> targets,
            TConstArrayRef<float> weights,
            TConstArrayRef<float> initialPredictions,
            TConstArrayRef<ui32> candidateFeatures,
            TConstArrayRef<ui32> candidateBins,
            TConstArrayRef<ui8> candidateTypes = {},
            const CBMQueryOptions* queryOptions = nullptr,
            TConstArrayRef<ui32> groupOffsets = {},
            const CBMPairOptions* pairOptions = nullptr,
            TConstArrayRef<ui32> pairWinners = {}, TConstArrayRef<ui32> pairLosers = {},
            TConstArrayRef<float> pairWeights = {})
            : Params(params)
        {
            CB_ENSURE(Params.rows && Params.rows <= (1u << 24) && Params.features &&
                bins.size() == ui64(Params.features) * Params.rows,
                "Metal greedy bins must contain one value per feature and object");
            CB_ENSURE(targets.size() == Params.rows &&
                (weights.empty() || weights.size() == Params.rows) &&
                (initialPredictions.empty() || initialPredictions.size() == Params.rows),
                "Metal greedy target, weight, and initial cursor dimensions must match the objects");
            CB_ENSURE(candidateFeatures.size() == Params.candidates && candidateBins.size() == Params.candidates &&
                (candidateTypes.empty() || candidateTypes.size() == Params.candidates),
                "Metal greedy candidate dimensions differ from their configured count");
            CB_ENSURE(Params.max_leaves && Params.max_leaves <= 65536,
                "Metal greedy max_leaves must be in [1,65536]");
            CB_ENSURE(objectiveOptions.objective == Params.objective &&
                objectiveOptions.leaf_estimation_method == Params.leaf_method,
                "Metal greedy objective configuration must match the session parameters");
            char error[2048] = {};
            if (pairOptions) {
                CB_ENSURE(!queryOptions && pairWinners.size() == pairOptions->pair_count &&
                    pairLosers.size() == pairWinners.size() && pairWeights.size() == pairWinners.size(),
                    "Metal greedy PairLogit requires matching original pair arrays");
                CB_ENSURE(cbm_greedy_session_create_pair(&Params, &objectiveOptions, pairOptions,
                    pairWinners.data(), pairLosers.data(), pairWeights.data(), pairWinners.size(),
                    groupOffsets.empty() ? nullptr : groupOffsets.data(), groupOffsets.size(), bins.data(), targets.data(),
                    initialPredictions.empty() ? nullptr : initialPredictions.data(),
                    candidateFeatures.data(), candidateBins.data(), candidateTypes.empty() ? nullptr : candidateTypes.data(),
                    &Session.Value, error, sizeof(error)) == 0, "Metal greedy PairLogit initialization failed: " << error);
            } else if (queryOptions) {
                CB_ENSURE(groupOffsets.size() == ui64(queryOptions->group_count) + 1,
                    "Metal greedy query offsets count must match group_count + 1");
                CB_ENSURE(cbm_greedy_session_create_query(&Params, &objectiveOptions, queryOptions,
                    groupOffsets.data(), groupOffsets.size(), bins.data(), targets.data(),
                    weights.empty() ? nullptr : weights.data(),
                    initialPredictions.empty() ? nullptr : initialPredictions.data(),
                    candidateFeatures.data(), candidateBins.data(), candidateTypes.empty() ? nullptr : candidateTypes.data(),
                    &Session.Value, error, sizeof(error)) == 0, "Metal greedy query initialization failed: " << error);
            } else CB_ENSURE(cbm_greedy_session_create_configured(&Params, &objectiveOptions,
                bins.data(), targets.data(), weights.empty() ? nullptr : weights.data(),
                initialPredictions.empty() ? nullptr : initialPredictions.data(),
                candidateFeatures.data(), candidateBins.data(),
                candidateTypes.empty() ? nullptr : candidateTypes.data(),
                &Session.Value, error, sizeof(error)) == 0,
                "Metal greedy initialization failed: " << error);
        }

        TMetalGreedySession(const TMetalGreedySession&) = delete;
        TMetalGreedySession& operator=(const TMetalGreedySession&) = delete;

        void* GetHandle() const noexcept {
            return Session.Value;
        }

        const CBMGreedyTrainParams& GetParams() const noexcept {
            return Params;
        }

        ui32 GetCompletedIterations() const noexcept {
            return CompletedIterations;
        }

        CBMGreedyStepInfo GetInfo() const {
            CBMGreedyStepInfo result = {};
            char error[2048] = {};
            CB_ENSURE(cbm_greedy_session_info(Session.Value, &result, error, sizeof(error)) == 0,
                "Metal greedy info query failed: " << error);
            return result;
        }

        void SetBootstrap(const CBMBootstrapOptions& options) {
            char error[2048] = {};
            CB_ENSURE(cbm_greedy_session_set_bootstrap(Session.Value, &options, error, sizeof(error)) == 0,
                "Metal greedy bootstrap configuration failed: " << error);
            IterationOffset = options.iteration_offset;
        }

        void SetScoreNoise(const CBMScoreNoiseOptions& options) {
            char error[2048] = {};
            CB_ENSURE(cbm_greedy_session_set_score_noise(Session.Value, &options, error, sizeof(error)) == 0,
                "Metal greedy score noise configuration failed: " << error);
        }

        void SetBacktracking(ui32 type) {
            char error[2048] = {};
            CB_ENSURE(cbm_greedy_session_set_backtracking(Session.Value, type, error, sizeof(error)) == 0,
                "Metal greedy backtracking configuration failed: " << error);
        }

        TMetalGreedyBootstrapState GetBootstrapState() const noexcept {
            return {IterationOffset + CompletedIterations, 0.f, 0};
        }

        TMetalGreedyTree Step(ui64 absoluteIteration) {
            TMetalGreedyTree tree;
            tree.Nodes.resize(ui64(Params.max_leaves) * 2 - 1);
            tree.Values.resize(Params.max_leaves);
            tree.Weights.resize(Params.max_leaves);
            Step(absoluteIteration, &tree.Info, tree.Nodes, tree.Values, tree.Weights);
            tree.Nodes.resize(tree.Info.node_count);
            tree.Values.resize(tree.Info.leaf_count);
            tree.Weights.resize(tree.Info.leaf_count);
            return tree;
        }

        void Step(
            ui64 absoluteIteration,
            CBMGreedyStepInfo* info,
            TArrayRef<CBMGreedyNode> nodes,
            TArrayRef<float> values,
            TArrayRef<float> weights)
        {
            CB_ENSURE(absoluteIteration == ui64(IterationOffset) + CompletedIterations,
                "Metal greedy step must use the next absolute iteration");
            CB_ENSURE(info && nodes.size() >= ui64(Params.max_leaves) * 2 - 1 &&
                values.size() >= Params.max_leaves && weights.size() >= Params.max_leaves,
                "Metal greedy output buffers are smaller than the configured tree capacity");
            char error[2048] = {};
            CB_ENSURE(cbm_greedy_session_step(Session.Value, info, nodes.data(), values.data(), weights.data(),
                error, sizeof(error)) == 0,
                "Metal greedy iteration " << absoluteIteration << " failed: " << error);
            ++CompletedIterations;
            CB_ENSURE(info->completed_iterations == CompletedIterations &&
                info->leaf_count && info->leaf_count <= Params.max_leaves &&
                info->node_count == ui64(info->leaf_count) * 2 - 1,
                "Metal greedy returned inconsistent iteration or tree dimensions");
        }

        void CopyPredictions(TArrayRef<float> predictions) const {
            CB_ENSURE(predictions.size() == Params.rows,
                "Metal greedy prediction output must match the object count");
            char error[2048] = {};
            CB_ENSURE(cbm_greedy_session_copy_predictions(Session.Value, predictions.data(), error, sizeof(error)) == 0,
                "Metal greedy prediction copy failed: " << error);
        }

        TVector<float> CopyPredictions() const {
            TVector<float> predictions(Params.rows);
            CopyPredictions(predictions);
            return predictions;
        }

    private:
        struct TSessionHandle {
            void* Value = nullptr;
            ~TSessionHandle() {
                cbm_greedy_session_close(Value);
            }
        };

        const CBMGreedyTrainParams Params;
        TSessionHandle Session;
        ui32 IterationOffset = 0;
        ui32 CompletedIterations = 0;
    };
}
