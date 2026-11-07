#pragma once

#include "metal_trainer.h"
#include "metal_langevin.h"

#ifdef __cplusplus
extern "C" {
#endif

// Experimental Plain scalar greedy training. Policy:
// 0=Depthwise, 1=Lossguide, 2=Region; objective: shared scalar IDs 0..11,
// QueryRMSE=12, QuerySoftMax=13, PairLogit=14, classic YetiRank=17;
// score: 0=L2, 1=Cosine, 2=NewtonL2, 3=NewtonCosine, 4=SolarL2, 5=LOOL2, 6=SatL2;
// leaf_method: 0=Newton, 1=Gradient, 2=Exact (Quantile/MAE/MAPE), 3=Simple.
// Simple requires one iteration, exports the searched sampled weak model to
// every permutation, and is unavailable for YetiRank (Newton only).
// Backtracking is configured separately; no random score noise by default.
// max_leaves is in [1,65536]. Lossguide accepts any uint32 depth, with
// effective path depth bounded by max_leaves-1. Region caps depth at65535 and
// uses at most depth+1 leaves; Depthwise caps depth at16.
typedef struct {
    uint32_t rows, features, candidates, bins_per_feature;
    uint32_t iterations, depth, max_leaves, min_data_in_leaf;
    uint32_t policy, objective, score_function, leaf_method;
    uint32_t leaf_iterations, reserved0, reserved1, reserved2;
    float learning_rate, l2_leaf_reg, bias;
    uint32_t reserved3;
} CBMGreedyTrainParams;

// Flat binary tree rooted at node 0. Internal nodes have leaf=UINT32_MAX;
// their left/right are node indices. Terminal nodes identify the leaf array
// entry. Type 0 routes bin>border to the right; type 1 routes bin==value right.
// This node order is independent of the retained-parent/appended-right leaf IDs.
typedef struct {
    uint32_t feature, bin, type, left, right, leaf;
} CBMGreedyNode;

typedef struct {
    uint32_t completed_iterations, finished, node_count, leaf_count;
    float loss;
    uint32_t reserved;
    CBMTrainStats stats;
} CBMGreedyStepInfo;

// All inputs are copied. bins is uint8[features,rows] in feature-major order.
// Null weights means unit weights; null initial predictions means constant bias.
// Candidate types may be null (numeric); all candidates of a feature share type.
int cbm_greedy_session_create(const CBMGreedyTrainParams* params,
    const uint8_t* bins, const float* targets, const float* sample_weights,
    const float* initial_predictions, const uint32_t* candidate_features,
    const uint32_t* candidate_bins, const uint8_t* candidate_types,
    void** session, char* error, size_t error_capacity);

// Additive configured constructor; objective/method IDs must agree with params.
// Supplies scalar loss parameters before initial derivatives/loss computation.
// The legacy constructor uses parameter .5 for IDs 8..10, otherwise 1.
int cbm_greedy_session_create_configured(const CBMGreedyTrainParams* params,
    const CBMObjectiveOptions* objective_options,
    const uint8_t* bins, const float* targets, const float* sample_weights,
    const float* initial_predictions, const uint32_t* candidate_features,
    const uint32_t* candidate_bins, const uint8_t* candidate_types,
    void** session, char* error, size_t error_capacity);

// Query-aware additive constructor. Offsets contain group_count+1 strictly
// increasing values spanning [0,rows]; the explicit count is checked before
// reading offsets. All derivatives use original weights and whole queries.
int cbm_greedy_session_create_query(const CBMGreedyTrainParams* params,
    const CBMObjectiveOptions* objective_options, const CBMQueryOptions* query_options,
    const uint32_t* group_offsets, uint64_t group_offsets_count,
    const uint8_t* bins, const float* targets, const float* sample_weights,
    const float* initial_predictions, const uint32_t* candidate_features,
    const uint32_t* candidate_bins, const uint8_t* candidate_types,
    void** session, char* error, size_t error_capacity);

// Literal supplied-edge PairLogit. Training weights are incident edge mass;
// targets are retained only for common input geometry and are not differentiated.
// Counted pair arrays and optional group offsets are checked before reads.
int cbm_greedy_session_create_pair(const CBMGreedyTrainParams* params,
    const CBMObjectiveOptions* objective_options, const CBMPairOptions* pair_options,
    const uint32_t* winners, const uint32_t* losers, const float* pair_weights, uint64_t pair_count,
    const uint32_t* group_offsets, uint64_t group_offsets_count,
    const uint8_t* bins, const float* targets, const float* initial_predictions,
    const uint32_t* candidate_features, const uint32_t* candidate_bins, const uint8_t* candidate_types,
    void** session, char* error, size_t error_capacity);

// Classic stochastic YetiRank uses Newton leaves and no backtracking. Query
// offsets are counted before reads. Loss is zero; the controller computes the
// ranking metric separately. The centering policy matches symmetric YetiRank.
int cbm_greedy_session_create_yeti(const CBMGreedyTrainParams* params,
    const CBMObjectiveOptions* objective_options, const CBMYetiRankOptions* yeti_options,
    const uint32_t* group_offsets, uint64_t group_offsets_count,
    const uint8_t* bins, const float* targets, const float* sample_weights,
    const float* initial_predictions, const uint32_t* candidate_features,
    const uint32_t* candidate_bins, const uint8_t* candidate_types,
    void** session, char* error, size_t error_capacity);
// Configure counted, copied dense numeric feature IDs before the first tree.
// Each feature must have exactly one bin-zero numeric candidate. Repeats are
// permitted. Host adapters resolve CUDA feature-manager IDs into this grid.
int cbm_greedy_session_set_fixed_splits(void* session, uint32_t count,
    const uint32_t* features, char* error, size_t error_capacity);

// Supply one weak seed or the complete weak+leaf packet before search. Leaf
// seeds are in dataset order: P*(I+(I>1)), including unused final evaluations.
int cbm_greedy_session_set_yeti_oracle_seeds(void* session, uint32_t count,
    const uint64_t* seeds, char* error, size_t error_capacity);
// Searches and retains the full topology, bootstrap draw and row partitions.
// The returned scorer attempt count lets the host consume CUDA's exact random
// schedule before drawing leaf seeds. Snapshot/configuration operations are
// unavailable until the existing step function completes the prepared tree.
int cbm_greedy_session_prepare_yeti_tree(void* session, uint32_t* search_attempts,
    char* error, size_t error_capacity);
int cbm_greedy_session_set_yeti_leaf_seeds(void* session, uint32_t count,
    const uint64_t* seeds, char* error, size_t error_capacity);

// One completed tree; caller allocates 2*max_leaves-1 nodes and max_leaves
// values/weights. Only info.node_count/info.leaf_count entries are meaningful.
// Values already include learning_rate. A root-only constant tree is valid.
int cbm_greedy_session_step(void* session, CBMGreedyStepInfo* info,
    CBMGreedyNode* nodes, float* leaf_values, float* leaf_weights,
    char* error, size_t error_capacity);

int cbm_greedy_session_copy_predictions(void* session, float* predictions,
    char* error, size_t error_capacity);
int cbm_greedy_session_info(void* session, CBMGreedyStepInfo* info,
    char* error, size_t error_capacity);
// Configure before the first tree. 0=No, 1=AnyImprovement, 2=Armijo.
// CUDA applies backtracking only when leaf_iterations > 1; Exact ignores it.
int cbm_greedy_session_set_backtracking(void* session, uint32_t type,
    char* error, size_t error_capacity);
// Add lambda/2 * sum(raw_leaf_value^2) to the optimized scalar objective.
// Configure before the first tree; Simple and Exact skip this estimator.
int cbm_greedy_session_set_add_ridge(void* session, uint32_t enabled,
    char* error, size_t error_capacity);
// Greedy Langevin noises leaf derivatives only. The runtime requests source
// target/search seeds and leaf noise synchronously; Simple and Exact have no
// leaf callback events. Configure before the first tree and before Yeti packets.
int cbm_greedy_session_set_langevin(void* session, float temperature,
    CBMLangevinNoiseCallback noise, CBMLangevinSeedCallback seed, void* context,
    char* error, size_t error_capacity);
// Shared option layouts, configured before the first tree. Greedy CUDA does
// not support MVS; supported bootstrap IDs are No/Bayesian/Bernoulli/Poisson.
int cbm_greedy_session_set_bootstrap(void* session, const CBMBootstrapOptions* options,
    char* error, size_t error_capacity);
int cbm_greedy_session_set_score_noise(void* session, const CBMScoreNoiseOptions* options,
    char* error, size_t error_capacity);
// Dense feature multipliers, configured before the first tree. Validation is
// atomic: rejected count/nonfinite/negative input leaves previous weights intact.
int cbm_greedy_session_set_feature_weights(void* session, uint32_t count, const float* weights,
    char* error, size_t error_capacity);
// Plain DocParallel datasets share feature grids and original object order.
// Configure once before training; each dataset owns its raw cursor. Search
// chooses one dataset, then the fixed topology is estimated on every dataset.
// The last dataset supplies exported leaves, loss and predictions. MVS arrays
// use the common permutation ABI but must be absent together or all zero.
int cbm_greedy_session_set_permutations(void* session, uint32_t count,
    const uint8_t* const* bins, const float* const* predictions,
    const float* mvs_lambdas, const uint8_t* mvs_valid,
    char* error, size_t error_capacity);
int cbm_greedy_session_select_permutation(void* session, uint32_t search_index,
    char* error, size_t error_capacity);
int cbm_greedy_session_copy_permutation_state(void* session, uint32_t capacity,
    float* predictions, float* mvs_lambdas, uint8_t* mvs_valid,
    char* error, size_t error_capacity);
// Most recent successfully finished tree, indexed by dataset/history, with
// the shared topology returned by step. Values already include learning_rate
// and any objective centering; Simple copies the searched leaves to every
// history. count and max_leaves must equal the configured history count and
// effective session leaf capacity. Output is float[count][max_leaves], with
// zero padding after the completed tree's leaf_count. Requires a healthy,
// idle session with at least one completed tree; performs no GPU work.
int cbm_greedy_session_copy_last_permutation_leaves(void* session, uint32_t count,
    uint32_t max_leaves, float* output, char* error, size_t error_capacity);
void cbm_greedy_session_close(void* session);

// Resident scalar prediction cursor for variable-node trees. Counts are checked
// before reading each buffer. Bins upload once; each addition uploads only the
// active nodes and leaves and uses the same sequential float32 cursor as fit.
// max_depth is in [0,65535]; full binary topology and all leaf references are
// validated iteratively before dispatch. Zero rows are valid for inference.
typedef struct {
    CBMTrainStats compute;
    uint64_t dataset_uploads, bins_upload_bytes, tree_upload_bytes, resident_bytes;
} CBMGreedyEvaluationStats;
int cbm_greedy_evaluation_create(uint32_t rows, uint32_t features, uint32_t max_depth,
    const uint8_t* bins, uint64_t bins_count, float bias,
    const float* initial_predictions, uint64_t initial_count,
    void** evaluation, char* error, size_t error_capacity);
// Additive vector cursor. dimensions is [1,64], initial_predictions is
// row-major [rows,dimensions] (null means zero); value_count counts floats in
// leaf-major [leaf,dimensions]. Existing predictions/stats/close accept this
// handle; scalar add_tree rejects it when dimensions>1. All counts are checked
// before reading buffers or mutating the cursor. Zero rows are valid.
int cbm_greedy_evaluation_create_vector(uint32_t rows, uint32_t features, uint32_t max_depth,
    uint32_t dimensions, const uint8_t* bins, uint64_t bins_count,
    const float* initial_predictions, uint64_t initial_count,
    void** evaluation, char* error, size_t error_capacity);
int cbm_greedy_evaluation_add_vector_tree(void* evaluation,
    const CBMGreedyNode* nodes, uint64_t node_count,
    const float* leaf_values, uint64_t value_count,
    float* predictions, uint64_t prediction_count, char* error, size_t error_capacity);
int cbm_greedy_evaluation_add_tree(void* evaluation,
    const CBMGreedyNode* nodes, uint64_t node_count,
    const float* leaf_values, uint64_t leaf_count,
    float* predictions, uint64_t prediction_count,
    char* error, size_t error_capacity);
int cbm_greedy_evaluation_predictions(void* evaluation,
    float* predictions, uint64_t prediction_count, char* error, size_t error_capacity);
int cbm_greedy_evaluation_stats(void* evaluation, CBMGreedyEvaluationStats* stats,
    char* error, size_t error_capacity);
void cbm_greedy_evaluation_close(void* evaluation);

#ifdef __cplusplus
}
#endif
