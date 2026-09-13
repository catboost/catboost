#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Experimental numeric-only RMSE / Plain / symmetric-tree training ABI.
// Quantized data is feature-major, bins[feature * rows + row]. Each candidate
// identifies a feature and a zero-based border; its right child has bin > border.
typedef struct {
    uint32_t rows;
    uint32_t features;
    uint32_t candidates;
    uint32_t bins_per_feature;
    uint32_t iterations;
    uint32_t depth;
    uint32_t score_function; // 0=L2,1=Cosine,2=NewtonL2,3=NewtonCosine,4=SolarL2,5=LOOL2,6=SatL2.
    float learning_rate;
    float l2_leaf_reg;
    float bias;
} CBMTrainParams;

typedef struct {
    uint64_t kernel_dispatches;
    double gpu_seconds;
    char device_name[256];
} CBMTrainStats;

// Returns zero on success, nonzero on failure, with a NUL-terminated error.
int cbm_device_info(char* name, size_t name_capacity, char* error, size_t error_capacity);

// Output arrays are caller-owned. tree_depths has iterations entries;
// split_features/split_bins have iterations * depth entries; leaf_values and
// leaf_weights have iterations * (1 << depth) entries; predictions has rows;
// rmse has iterations + 1 entries, including the initial bias-only loss.
// Leaf values already include learning_rate. The first split is leaf bit 0.
// Arrays use fixed tree strides even when a tree terminates at smaller depth.
int cbm_train(
    const CBMTrainParams* params,
    const uint8_t* bins,
    const float* targets,
    const uint32_t* candidate_features,
    const uint32_t* candidate_bins,
    uint32_t* tree_depths,
    uint32_t* split_features,
    uint32_t* split_bins,
    float* leaf_values,
    float* leaf_weights,
    float* predictions,
    float* rmse,
    CBMTrainStats* stats,
    char* error,
    size_t error_capacity);

// Additive persistent API. Existing CBMTrainParams/cbm_train remain ABI-stable.
typedef struct {
    CBMTrainParams train;
    uint32_t objective; // See CBMObjectiveOptions; all cursors contain raw predictions.
    uint32_t leaf_estimation_iterations;
    uint32_t leaf_estimation_backtracking; // 0=No, 1=AnyImprovement, 2=Armijo.
    uint32_t reserved; // Must be zero.
} CBMSessionParams;

typedef struct {
    uint32_t completed_iterations;
    uint32_t finished;
    float loss; // Weighted objective metric, including the RMSE square root.
    uint32_t reserved;
    CBMTrainStats stats;
} CBMStepInfo;

// Inputs are copied. Null sample_weights means unit weights; null initial_predictions
// means a bias-only cursor. iterations is this session's new-tree capacity, allowing
// callers to resume using prior raw predictions and concatenate prior model trees.
// candidate_types is optional uint8[candidates]: 0=numeric >bin, 1=one-hot ==bin.
// All candidates for a feature must have the same type.
int cbm_session_create(
    const CBMSessionParams* params, const uint8_t* bins, const float* targets,
    const float* sample_weights, const float* initial_predictions,
    const uint32_t* candidate_features, const uint32_t* candidate_bins,
    const uint8_t* candidate_types, void** session, char* error, size_t error_capacity);

// Executes exactly one tree. Caller allocates max-depth/max-leaf output arrays;
// trailing entries are zeroed. depth and info describe the completed tree.
int cbm_session_step(
    void* session, CBMStepInfo* info, uint32_t* depth,
    uint32_t* split_features, uint32_t* split_bins, uint8_t* split_types,
    float* leaf_values, float* leaf_weights, char* error, size_t error_capacity);

// Copies all completed trees, using fixed train.depth and (1<<train.depth) strides.
// capacity_iterations must cover completed trees, and loss has completed+1 values.
// At zero completed trees, tree output pointers may be null; predictions/loss may not.
int cbm_session_result(
    void* session, uint32_t capacity_iterations, uint32_t* completed_iterations,
    uint32_t* tree_depths, uint32_t* split_features, uint32_t* split_bins,
    uint8_t* split_types, float* leaf_values, float* leaf_weights,
    float* predictions, float* loss, CBMTrainStats* stats,
    char* error, size_t error_capacity);

// Idempotent, including null and already-closed handles.
void cbm_session_close(void* session);

typedef struct {
    uint32_t depth, finished, has_split;
    uint32_t feature, bin, type;
    float score, gain;
} CBMStructureInfo;
// Incremental structure search retains the tree's gradients, bootstrap draw,
// partitions and chosen search permutation. Finish may explicitly end early.
int cbm_session_begin_tree(void* session, char* error, size_t error_capacity);
int cbm_session_grow_tree(void* session, CBMStructureInfo* info, char* error, size_t error_capacity);
int cbm_session_finish_tree(void* session, CBMStepInfo* info, uint32_t* depth,
    uint32_t* split_features, uint32_t* split_bins, uint8_t* split_types,
    float* leaf_values, float* leaf_weights, char* error, size_t error_capacity);

typedef struct {
    uint32_t permutation_count, features, candidates, bins_per_feature;
    uint32_t reserved[4];
} CBMAppendFeatureOptions;
// Append-only global feature/candidate identities. Each matrix contains only
// the new feature-major columns, in original row order; candidate feature IDs
// are LOCAL to these columns. bins_per_feature is the new global capacity.
// feature_flags: bit0=dynamic CTR, bit1=globally registered CTR; default3.
// Dynamic CTRs retain their size penalty after use. Initial static features
// are registered; a selected dynamic CTR becomes registered and used.
// Inputs are copied and old banks remain valid until the replacement GPU blit
// finishes. Validation includes the peak footprint of both old and new banks.
int cbm_session_append_features(void* session, const CBMAppendFeatureOptions* options,
    const uint8_t* const* permutation_bins, const uint32_t* candidate_features,
    const uint32_t* candidate_bins, const uint8_t* candidate_types,
    const uint32_t* ctr_unique_values, const float* feature_weights,
    const uint8_t* feature_flags, const uint8_t* used_features,
    uint32_t* first_global_feature, char* error, size_t error_capacity);
// May be changed at a depth boundary. Zero disables all candidates of a feature;
// its data remain available for existing splits and all permutation cursors.
int cbm_session_set_feature_activity(void* session, uint32_t feature_count,
    const uint8_t* active_features, char* error, size_t error_capacity);
// Copies metadata only between completed trees, for exact snapshot restoration.
int cbm_session_copy_feature_metadata(void* session, uint32_t feature_capacity,
    uint32_t* ctr_unique_values, float* feature_weights, uint8_t* feature_flags,
    uint8_t* used_features, uint8_t* active_features, char* error, size_t error_capacity);
// Restores FeatureParallel metadata before the first tree. Configure CTR counts
// and weights and append every restored bank first. Enables dynamic scoring;
// all vectors must cover the complete current feature bank. Used CTRs must be
// registered, and only CTR columns may carry used state.
int cbm_session_restore_feature_metadata(void* session, uint32_t feature_count,
    const uint8_t* feature_flags, const uint8_t* used_features,
    const uint8_t* active_features, char* error, size_t error_capacity);

// Optional configuration setters are additive and allowed before the first step.
typedef struct {
    // 0=RMSE,1=Logloss,2=CrossEntropy,3=Poisson,4=Huber,5=Expectile,
    // 6=Lq,7=Tweedie,8=LogLinQuantile,9=Quantile,10=MAE,11=MAPE,
    // 12=QueryRMSE,13=QuerySoftMax (query constructor),14=PairLogit (pair constructor),
    // 15=PairLogitPairwise,16=QueryCrossEntropy (full-matrix constructors),
    // 17=classic YetiRank (Yeti constructor).
    uint32_t objective;
    uint32_t leaf_estimation_method; // 0=Newton, 1=Gradient, 2=Exact (Quantile/MAE/MAPE), 3=Simple (full matrix).
    float objective_param; // Huber delta, Lq q, Tweedie power, or the loss's alpha.
    uint32_t reserved;
} CBMObjectiveOptions;
// Supplies loss parameters before any initial objective computation. Prefer
// this for parameterized objectives; the legacy constructor remains stable.
int cbm_session_create_configured(
    const CBMSessionParams* params, const CBMObjectiveOptions* objective_options,
    const uint8_t* bins, const float* targets, const float* sample_weights,
    const float* initial_predictions, const uint32_t* candidate_features,
    const uint32_t* candidate_bins, const uint8_t* candidate_types,
    void** session, char* error, size_t error_capacity);

// Group boundaries are in original row order, strictly increasing from zero
// through train.rows. QueryRMSE=12 and QuerySoftMax=13 require this constructor
// so the initial objective uses the supplied groups and parameters.
typedef struct {
    uint32_t group_count;
    float beta;
    float lambda;
    uint32_t reserved;
} CBMQueryOptions;

// YetiRank's target has no scalar objective value (CUDA supplies zero). The
// step/result loss slots are zero; the controller evaluates PFound or another
// selected metric. Every tree consumes one weak-target seed and I leaf seeds,
// plus the CUDA walker's unused final evaluation seed when I>1. Score-noise
// statistics reuse the weak target. Supply all seeds together or just the weak
// seed before begin_tree, then the leaf seeds after structure search.
typedef struct {
    uint32_t group_count, permutations;
    float decay;
    uint32_t legacy_prefix_centering; // 0 centers complete queries; 1 reproduces the pinned CUDA caller defect.
} CBMYetiRankOptions;
typedef struct {
    uint32_t group_count, permutations;
    float decay;
    uint32_t sampling_unit; // Object=0, Group=1; weak Bernoulli sampling only.
} CBMYetiRankPairwiseOptions;
// YetiRankPairwise=18 owns sampled PFound weak edges and a separate fixed
// Bayesian leaf target. Its oracle value is zero; the metric tracker uses PFound.
int cbm_session_create_yeti_pairwise(
    const CBMSessionParams* params, const CBMObjectiveOptions* objective_options,
    const CBMYetiRankPairwiseOptions* yeti_options, float non_diagonal_regularization,
    const uint32_t* group_offsets, const uint8_t* bins, const float* targets,
    const float* sample_weights, const float* initial_predictions,
    const uint32_t* candidate_features, const uint32_t* candidate_bins, const uint8_t* candidate_types,
    void** session, char* error, size_t error_capacity);
int cbm_session_create_yeti(
    const CBMSessionParams* params, const CBMObjectiveOptions* objective_options,
    const CBMYetiRankOptions* yeti_options, const uint32_t* group_offsets,
    const uint8_t* bins, const float* targets, const float* sample_weights,
    const float* initial_predictions, const uint32_t* candidate_features,
    const uint32_t* candidate_bins, const uint8_t* candidate_types,
    void** session, char* error, size_t error_capacity);
int cbm_session_set_yeti_oracle_seeds(void* session, uint32_t count,
    const uint64_t* seeds, char* error, size_t error_capacity);
int cbm_session_set_yeti_leaf_seeds(void* session, uint32_t count,
    const uint64_t* seeds, char* error, size_t error_capacity);
int cbm_session_create_query(
    const CBMSessionParams* params, const CBMObjectiveOptions* objective_options,
    const CBMQueryOptions* query_options, const uint32_t* group_offsets,
    const uint8_t* bins, const float* targets, const float* sample_weights,
    const float* initial_predictions, const uint32_t* candidate_features,
    const uint32_t* candidate_bins, const uint8_t* candidate_types,
    void** session, char* error, size_t error_capacity);

typedef struct {
    uint32_t pair_count;
    uint32_t group_count;
    uint32_t reserved0, reserved1;
} CBMPairOptions;
// PairLogit=14 uses supplied edge weights literally. Its sample/leaf weights
// are incident pair mass, not object weights. Optional group offsets validate
// that every edge remains inside one contiguous query. Labels are unnecessary.
int cbm_session_create_pair(
    const CBMSessionParams* params, const CBMObjectiveOptions* objective_options,
    const CBMPairOptions* pair_options, const uint32_t* pair_winners,
    const uint32_t* pair_losers, const float* pair_weights, const uint32_t* group_offsets,
    const uint8_t* bins, const float* initial_predictions,
    const uint32_t* candidate_features, const uint32_t* candidate_bins,
    const uint8_t* candidate_types, void** session, char* error, size_t error_capacity);

// PairLogitPairwise=15: edge-sampled structure search with full coupled Hessians;
// original document weights are retained for Newton/Gradient leaves. Numeric/one-hot,
// Plain P1, depth 0..8, all three backtracking modes. Simple uses one selected
// split solution and its raw matrix diagonal at depth 1..8. The pinned
// CUDA pairwise search uses edge mass for L2 and curvature for other scores,
// ignores score noise and retains repeated winning splits through max depth.
int cbm_session_create_pair_matrix(
    const CBMSessionParams* params, const CBMObjectiveOptions* objective_options,
    const CBMPairOptions* pair_options, float non_diagonal_regularization,
    const uint32_t* pair_winners, const uint32_t* pair_losers, const float* pair_weights,
    const uint32_t* group_offsets, const uint8_t* bins, const float* sample_weights,
    const float* initial_predictions, const uint32_t* candidate_features,
    const uint32_t* candidate_bins, const uint8_t* candidate_types,
    void** session, char* error, size_t error_capacity);

// Full QueryCrossEntropy point + query matrix, CUDA's 256-row query limit.
// Scales are selected per query from the loss's raw_values_scale table.
typedef struct { uint32_t group_count, reserved0, reserved1, reserved2; } CBMQueryCrossEntropyOptions;
int cbm_session_create_query_cross_entropy(
    const CBMSessionParams* params, const CBMObjectiveOptions* objective_options,
    const CBMQueryCrossEntropyOptions* query_options, float non_diagonal_regularization,
    const uint32_t* group_offsets, const float* query_scales,
    const uint8_t* bins, const float* targets, const float* weights, const float* initial_predictions,
    const uint32_t* candidate_features, const uint32_t* candidate_bins, const uint8_t* candidate_types,
    void** session, char* error, size_t error_capacity);

// Evaluates the CUDA QueryCrossEntropy target metric on Metal, including its
// selected query scales. No fitting or tree construction occurs in this call.
int cbm_query_cross_entropy_metric(uint32_t rows, uint32_t groups,
    const float* targets, const float* weights, const float* predictions,
    const uint32_t* group_offsets, const float* query_scales, float alpha,
    double* result, char* error, size_t error_capacity);

// Persistent metric-only workspace. Metadata is copied once; each evaluation
// supplies new predictions and alpha. Handles serialize evaluations and may
// be destroyed while an already-started evaluation retains its workspace.
int cbm_query_cross_entropy_metric_create(uint32_t rows, uint32_t groups,
    const float* targets, const float* weights, const uint32_t* group_offsets,
    const float* query_scales, uint64_t budget, void** metric, uint64_t* allocated_bytes,
    char* error, size_t error_capacity);
int cbm_query_cross_entropy_metric_evaluate(void* metric, uint32_t rows,
    const float* predictions, float alpha, double* result, uint64_t* evaluations,
    char* error, size_t error_capacity);
void cbm_query_cross_entropy_metric_destroy(void* metric);

int cbm_session_set_objective(void* session, const CBMObjectiveOptions* options,
                              char* error, size_t error_capacity);

typedef struct {
    uint32_t bootstrap_type; // 0=No,1=Bayesian,2=Bernoulli,3=Poisson,4=MVS.
    uint32_t random_seed_low;
    uint32_t random_seed_high;
    uint32_t iteration_offset;
    float bagging_temperature;
    float subsample;
    float mvs_reg;
    uint32_t mvs_reg_is_set;
    float initial_mvs_lambda;
    uint32_t initial_mvs_lambda_is_set;
    uint32_t reserved0;
    uint32_t reserved1;
} CBMBootstrapOptions;
int cbm_session_set_bootstrap(void* session, const CBMBootstrapOptions* options,
                             char* error, size_t error_capacity);
int cbm_session_get_bootstrap_state(void* session, uint32_t* absolute_iterations,
                                    float* mvs_lambda, uint32_t* mvs_lambda_is_set,
                                    char* error, size_t error_capacity);
// Uses the bootstrap seed and absolute iteration configuration. Cosine adds
// per-feature noise; L2 intentionally ignores random_strength, as CUDA does.
typedef struct {
    float random_strength;
    uint32_t reserved0, reserved1, reserved2;
} CBMScoreNoiseOptions;
int cbm_session_set_score_noise(void* session, const CBMScoreNoiseOptions* options,
                                char* error, size_t error_capacity);

// Configure once before training. Every matrix uses the original feature/row
// dimensions and candidate grid. Null cursors clone the create-time cursor.
// Structure search uses the selected permutation; every cursor receives its
// own leaf estimates for that shared structure. Export uses permutation count-1.
int cbm_session_set_permutations(void* session, uint32_t count,
    const uint8_t* const* bins, const float* const* initial_predictions,
    const float* mvs_lambdas, const uint8_t* mvs_valid, char* error, size_t error_capacity);
int cbm_session_select_permutation(void* session, uint32_t search_index,
    char* error, size_t error_capacity);
int cbm_session_copy_permutation_state(void* session, uint32_t capacity,
    float* predictions, float* mvs_lambdas, uint8_t* mvs_valid,
    char* error, size_t error_capacity);

typedef struct {
    float model_size_reg;
    uint32_t reserved0, reserved1, reserved2;
} CBMFeaturePenaltyOptions;
// Zero unique count denotes a non-CTR feature. Used CTR flags persist across
// trees; supplied initial flags restore an existing model or snapshot.
int cbm_session_set_feature_penalties(void* session, const CBMFeaturePenaltyOptions* options,
    const uint32_t* ctr_unique_values, const float* feature_weights,
    const uint8_t* used_features, char* error, size_t error_capacity);
int cbm_session_copy_feature_penalty_state(void* session, uint8_t* used_features,
    char* error, size_t error_capacity);

int cbm_session_get_workspace_info(void* session, uint32_t* histogram_tiles,
    uint64_t* histogram_bytes, uint64_t* estimated_peak_gpu_bytes, char* error, size_t error_capacity);
int cbm_session_copy_predictions(void* session, float* predictions,
                                  char* error, size_t error_capacity);

#ifdef __cplusplus
}
#endif
