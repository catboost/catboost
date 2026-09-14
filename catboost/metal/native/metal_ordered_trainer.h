#pragma once
#include "metal_trainer.h"
#include "metal_combination.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t rows, features, candidates, iterations;
    uint32_t depth, objective, score_function, leaf_method;
    uint32_t leaf_iterations, permutations, min_fold_size, normalize;
    float learning_rate, l2, bias, fold_growth;
    float objective_param;
    uint32_t reserved0, reserved1, reserved2;
} CBMOrderedParams;

// Numeric symmetric Ordered session. Objectives use the shared scalar IDs
// 0..11. Score 0=Cosine,1=NewtonCosine. Leaf method 0=Newton,1=Gradient,
// 2=Exact (Quantile/MAE/MAPE only; ignores step count and regularization).
// bins uint8[features,rows]; permutation_maps uint32[permutations,rows].
// Maps are explicit position->original-row permutations. Optional initial
// predictions/weights use original row order; buffers are copied on creation.
int cbm_ordered_session_create(const CBMOrderedParams* params, const uint8_t* bins,
    const float* targets, const float* weights, const float* initial_predictions,
    const uint32_t* candidate_features, const uint32_t* candidate_bins,
    const uint32_t* permutation_maps, void** session, char* error, size_t capacity);

// Numeric/one-hot variant. candidate_types[candidates] uses 0 for greater-than
// and 1 for equality. A feature cannot mix types; one-hot bin 255 is valid.
// The original creation entry point remains equivalent to all-zero types.
int cbm_ordered_session_create_typed(const CBMOrderedParams* params, const uint8_t* bins,
    const float* targets, const float* weights, const float* initial_predictions,
    const uint32_t* candidate_features, const uint32_t* candidate_bins, const uint8_t* candidate_types,
    const uint32_t* permutation_maps, void** session, char* error, size_t capacity);

// Group-aware scalar variant. offsets[group_count+1] strictly partitions the
// original rows from zero to params->rows. At least four groups are required;
// every permutation must retain each whole group in its original internal order.
// Each learning permutation receives its own variable-length prefix list.
int cbm_ordered_session_create_grouped(const CBMOrderedParams* params, const uint8_t* bins,
    const float* targets, const float* weights, const float* initial_predictions,
    const uint32_t* candidate_features, const uint32_t* candidate_bins, const uint8_t* candidate_types,
    const uint32_t* permutation_maps, uint32_t group_count, const uint32_t* offsets, double fold_growth,
    void** session, char* error, size_t capacity);

// Permutation-dependent feature variant. bins[bank_count,features,rows] is
// counted by bin_cells; bank_count must be one or params->permutations.
// Every bank shares candidate columns/grids. Search uses its selected bank;
// every prefix/full task routes leaves through its own bank. Optional grouping
// follows the grouped entry point. Original creation ABIs remain unchanged.
int cbm_ordered_session_create_banked(const CBMOrderedParams* params, uint32_t bank_count, uint64_t bin_cells,
    const uint8_t* bins, const float* targets, const float* weights, const float* initial_predictions,
    const uint32_t* candidate_features, const uint32_t* candidate_bins, const uint8_t* candidate_types,
    const uint32_t* permutation_maps, uint32_t group_count, const uint32_t* offsets, double fold_growth,
    void** session, char* error, size_t capacity);

// Whole-query Ordered variants. Offsets and all input columns use original
// source row order; every permutation must preserve entire queries. At least
// four queries are required. IDs12/13=QueryRMSE/QuerySoftMax,14=PairLogit,17=YetiRank.
int cbm_ordered_session_create_query_banked(const CBMOrderedParams* params, uint32_t bank_count, uint64_t bin_cells,
    const uint8_t* bins, const float* targets, const float* weights, const float* initial_predictions,
    const uint32_t* candidate_features, const uint32_t* candidate_bins, const uint8_t* candidate_types,
    const uint32_t* permutation_maps, const CBMQueryOptions* query_options, const uint32_t* group_offsets, double fold_growth,
    void** session, char* error, size_t capacity);
int cbm_ordered_session_create_pair_banked(const CBMOrderedParams* params, uint32_t bank_count, uint64_t bin_cells,
    const uint8_t* bins, const float* initial_predictions, const uint32_t* candidate_features, const uint32_t* candidate_bins,
    const uint8_t* candidate_types, const uint32_t* permutation_maps, const CBMPairOptions* pair_options,
    const uint32_t* pair_winners, const uint32_t* pair_losers, const float* pair_weights,
    const uint32_t* group_offsets, double fold_growth, void** session, char* error, size_t capacity);
int cbm_ordered_session_create_yeti_banked(const CBMOrderedParams* params, uint32_t bank_count, uint64_t bin_cells,
    const uint8_t* bins, const float* targets, const float* weights, const float* initial_predictions,
    const uint32_t* candidate_features, const uint32_t* candidate_bins, const uint8_t* candidate_types,
    const uint32_t* permutation_maps, const CBMYetiRankOptions* yeti_options, const uint32_t* group_offsets, double fold_growth,
    void** session, char* error, size_t capacity);
// Weak seed packets have one Learn then Quality draw for each selected fold,
// including empty quality slices. Leaf packets are evaluation-major/task-major
// over defined prefix tasks and the full task, with I+1 evaluations when I>1.
// Combination with Yeti components requires the callback below for backtracking
// with I>1, whose rejected trials consume a variable number of evaluations.
int cbm_ordered_session_yeti_seed_shape(void* session, uint32_t search_permutation,
    uint32_t* weak_seed_count, uint32_t* leaf_seed_count, char* error, size_t capacity);
int cbm_ordered_session_set_yeti_oracle_seeds(void* session, uint32_t count,
    const uint64_t* seeds, char* error, size_t capacity);
int cbm_ordered_session_set_yeti_leaf_seeds(void* session, uint32_t count,
    const uint64_t* seeds, char* error, size_t capacity);

// Custom ID20 supplies a compiled MSL objective body; pipelines are immutable
// and private to the session. Shared numeric/grouped banks keep the same ABI.
int cbm_ordered_session_create_custom_banked(const CBMOrderedParams* params, uint32_t bank_count, uint64_t bin_cells,
    const uint8_t* bins, const float* targets, const float* weights, const float* initial_predictions,
    const uint32_t* candidate_features, const uint32_t* candidate_bins, const uint8_t* candidate_types,
    const uint32_t* permutation_maps, const char* source, uint32_t group_count, const uint32_t* group_offsets,
    double fold_growth, void** session, char* error, size_t capacity);

// Combination ID19 uses the same independent prefix/full tasks. Grouping
// for Ordered folds is explicit even when every component is pointwise.
int cbm_ordered_session_create_combination_banked(const CBMOrderedParams* params, uint32_t bank_count, uint64_t bin_cells,
    const uint8_t* bins, const float* targets, const float* weights, const float* initial_predictions,
    const uint32_t* candidate_features, const uint32_t* candidate_bins, const uint8_t* candidate_types,
    const uint32_t* permutation_maps, const CBMCombinationOptions* combination_options, const CBMCombinationComponent* components,
    uint32_t group_count, const uint32_t* group_offsets, const uint32_t* pair_winners, const uint32_t* pair_losers,
    const float* pair_weights, double fold_growth, void** session, char* error, size_t capacity);
int cbm_ordered_session_set_combination_yeti_seed_callback(void* session,
    CBMCombinationYetiSeedCallback callback, void* context, char* error, size_t capacity);

// Structure is chosen using the selected learning permutation. Every learning
// fold and the independent full-model estimation task is updated each step.
// Outputs use the existing scalar padded ABI; only the full-model task's tree
// values/weights are exported. Caller supplies depth and 1<<depth strides.
int cbm_ordered_session_step(void* session, uint32_t search_permutation,
    CBMStepInfo* info, uint32_t* depth, uint32_t* split_features, uint32_t* split_bins,
    uint8_t* split_types, float* leaf_values, float* leaf_weights, char* error, size_t capacity);
// Incremental Ordered structure search freezes the selected permutation,
// gradients, bootstrap sample and fold cursors until finish. Finish may end early.
int cbm_ordered_session_begin_tree(void* session, uint32_t search_permutation, char* error, size_t capacity);
int cbm_ordered_session_grow_tree(void* session, CBMStructureInfo* info, char* error, size_t capacity);
int cbm_ordered_session_finish_tree(void* session, CBMStepInfo* info, uint32_t* depth,
    uint32_t* split_features, uint32_t* split_bins, uint8_t* split_types,
    float* leaf_values, float* leaf_weights, char* error, size_t capacity);
// Append-only columns in original row order, with local candidate feature IDs.
// The bank count is one or every permutation; shared banks may be promoted.
// A zero-column append enables dynamic metadata without replacing feature data.
// The scalar flag/weight semantics apply; only dynamic CTR winners become used.
int cbm_ordered_session_append_features(void* session, const CBMAppendFeatureOptions* options,
    const uint8_t* const* permutation_bins, const uint32_t* candidate_features,
    const uint32_t* candidate_bins, const uint8_t* candidate_types,
    const uint32_t* ctr_unique_values, const float* feature_weights,
    const uint8_t* feature_flags, const uint8_t* used_features,
    uint32_t* first_global_feature, char* error, size_t capacity);
int cbm_ordered_session_set_feature_activity(void* session, uint32_t feature_count,
    const uint8_t* active_features, char* error, size_t capacity);
// Metadata snapshots are available only between completed trees. Restore the
// constructed banks' flags only before the first tree; counts/grids are caller-checked.
int cbm_ordered_session_copy_feature_metadata(void* session, uint32_t feature_capacity,
    uint32_t* ctr_unique_values, float* feature_weights, uint8_t* feature_flags,
    uint8_t* used_features, uint8_t* active_features, char* error, size_t capacity);
int cbm_ordered_session_restore_feature_metadata(void* session, uint32_t feature_count,
    const uint8_t* feature_flags, const uint8_t* used_features, const uint8_t* active_features,
    char* error, size_t capacity);
// Static simple CTR penalties follow CUDA update_feature_weights.cpp. Configure
// once before stepping. Simple FeatureParallel CTRs keep their penalty after selection.
int cbm_ordered_session_set_feature_penalties(void* session, const CBMFeaturePenaltyOptions* options,
    const uint32_t* counts, const float* feature_weights, char* error, size_t capacity);
int cbm_ordered_session_info(void* session, CBMStepInfo* info, char* error, size_t capacity);
int cbm_ordered_session_copy_predictions(void* session, float* predictions, char* error, size_t capacity);
int cbm_ordered_session_set_bootstrap(void* session, const CBMBootstrapOptions* options,
    uint32_t test_only, char* error, size_t capacity);
int cbm_ordered_session_set_score_noise(void* session, const CBMScoreNoiseOptions* options,
    char* error, size_t capacity);
int cbm_ordered_session_get_bootstrap_state(void* session, uint32_t* iteration_offset,
    float* mvs_lambda, uint32_t* mvs_lambda_is_set, char* error, size_t capacity);
// 0=No, 1=AnyImprovement, 2=Armijo. Configure before the first step.
int cbm_ordered_session_set_backtracking(void* session, uint32_t type, char* error, size_t capacity);

// First query task_count and cursor_count, then copy arrays with exactly those
// capacities. Descriptor uint32[task_count,4] = estimateEnd,qualityEnd,offset,
// permutationId. Final descriptor is independent full-model estimation.
int cbm_ordered_session_state_shape(void* session, uint32_t* task_count,
    uint32_t* cursor_count, char* error, size_t capacity);
int cbm_ordered_session_copy_state(void* session, uint32_t task_count, uint32_t cursor_count,
    uint32_t* descriptors, float* cursors, char* error, size_t capacity);
// Restore only before the first step. Dataset/options/permutation fingerprints
// must be checked by the caller. Cursor length and finiteness are checked here.
int cbm_ordered_session_restore_cursors(void* session, uint32_t cursor_count,
    const float* cursors, char* error, size_t capacity);
void cbm_ordered_session_close(void* session);

#ifdef __cplusplus
}
#endif
