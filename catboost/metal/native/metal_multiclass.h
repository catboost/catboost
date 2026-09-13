#pragma once
#include <stddef.h>
#include <stdint.h>
#include "metal_trainer.h"
#include "metal_greedy_trainer.h"

#ifdef __cplusplus
extern "C" {
#endif

// Additive multiclass session. Numeric and one-hot candidates share each
// symmetric tree across all classes. Internal MultiClass cursors have C-1
// dimensions, but public predictions and leaf values always have C dimensions.
typedef struct {
    uint32_t rows, features, candidates, bins_per_feature;
    uint32_t classes, objective, iterations, depth;
    uint32_t score_function, leaf_method, leaf_iterations, reserved;
    float learning_rate, l2;
    uint32_t reserved1, reserved2;
} CBMMulticlassParams;

// Variable-tree constructor for CUDA-registered vector greedy objectives:
// MultiClass(0), MultiClassOneVsAll(1), RMSEWithUncertainty(3).
// Policies and depth/capacity rules match CBMGreedyTrainParams. Set reserved=0.
// The existing vector setters, cursor recovery functions and close accept this
// handle. Only step_greedy may advance it; leaf values are [leaf,classes].
typedef struct {
    uint32_t policy, max_leaves, min_data_in_leaf, reserved;
} CBMVectorGreedyOptions;
int cbm_multiclass_session_create_greedy(
    const CBMMulticlassParams* params, const CBMVectorGreedyOptions* greedy,
    const uint8_t* bins, const uint32_t* labels, const float* targets,
    const float* weights, const float* initial_predictions,
    const uint32_t* candidate_features, const uint32_t* candidate_bins,
    const uint8_t* candidate_types, void** session, char* error, size_t error_capacity);
// Allocate 2*max_leaves-1 nodes, max_leaves*classes values, max_leaves weights.
// Padding is zeroed. Active counts are in info; values include learning_rate.
int cbm_multiclass_session_step_greedy(void* session, CBMGreedyStepInfo* info,
    CBMGreedyNode* nodes, float* leaf_values, float* leaf_weights,
    char* error, size_t error_capacity);

// Objective: 0=MultiClass, 1=MultiClassOneVsAll.
// Score: 0=L2,1=Cosine,4=SolarL2,5=LOOL2,6=SatL2. Newton structure scores
// remain unsupported, matching CUDA's vector weak-objective restrictions.
// Leaf method: 0=Newton, 1=Gradient. Reserved fields must be zero.
// bins is uint8[features,rows]; labels uint32[rows] in [0,classes).
// Optional initial_predictions is float32[rows,classes], row-major.
// Optional weights defaults to one. Inputs are copied before returning.
int cbm_multiclass_session_create(
    const CBMMulticlassParams* params, const uint8_t* bins, const uint32_t* labels,
    const float* weights, const float* initial_predictions,
    const uint32_t* candidate_features, const uint32_t* candidate_bins,
    const uint8_t* candidate_types, void** session, char* error, size_t error_capacity);

// Additive float-target constructor using the same vector session/setters.
// objective: 2=MultiRMSE, 3=RMSEWithUncertainty, 4=MultiLogloss,
// 5=MultiCrossEntropy. classes denotes output dimensions, without a gauge.
// targets is float32[dimensions,rows], except uncertainty uses [rows] and
// exactly two outputs (mean, log standard deviation). All existing step,
// prediction, sampling, permutation and snapshot functions accept this handle.
int cbm_multioutput_session_create(
    const CBMMulticlassParams* params, const uint8_t* bins, const float* targets,
    const float* weights, const float* initial_predictions,
    const uint32_t* candidate_features, const uint32_t* candidate_bins,
    const uint8_t* candidate_types, void** session, char* error, size_t error_capacity);

// One completed tree, with zero-filled padding. Splits have params.depth entries,
// leaf_values has (1<<depth)*classes entries (leaf-major), and leaf_weights has
// (1<<depth) entries. MultiClass exports class C-1 as zero, matching its cursor
// parameterization. Values include learning_rate. No host tree fitting occurs.
int cbm_multiclass_session_step(
    void* session, CBMStepInfo* info, uint32_t* depth,
    uint32_t* split_features, uint32_t* split_bins, uint8_t* split_types,
    float* leaf_values, float* leaf_weights, char* error, size_t error_capacity);
int cbm_multiclass_session_copy_predictions(
    void* session, float* predictions, char* error, size_t error_capacity);
int cbm_multiclass_session_info(
    void* session, CBMStepInfo* info, char* error, size_t error_capacity);
// Configure before the first step. Bootstrap/noise share the scalar additive
// option structs and RNG state. Backtracking: 0=No,1=AnyImprovement,2=Armijo.
int cbm_multiclass_session_set_bootstrap(void* session, const CBMBootstrapOptions* options,
                                        char* error, size_t error_capacity);
int cbm_multiclass_session_get_bootstrap_state(void* session, uint32_t* absolute_iterations,
    float* mvs_lambda, uint32_t* mvs_lambda_is_set, char* error, size_t error_capacity);
int cbm_multiclass_session_set_score_noise(void* session, const CBMScoreNoiseOptions* options,
                                          char* error, size_t error_capacity);
int cbm_multiclass_session_set_backtracking(void* session, uint32_t type, char* error, size_t error_capacity);
// Independent permutation datasets/cursors with one shared selected structure.
// Prediction matrices are row-major[rows,classes]; the final permutation is
// exported. MVS placeholders must be zero because CUDA rejects multiclass MVS.
int cbm_multiclass_session_set_permutations(void* session, uint32_t count,
    const uint8_t* const* bins, const float* const* initial_predictions,
    const float* mvs_lambdas, const uint8_t* mvs_valid, char* error, size_t error_capacity);
int cbm_multiclass_session_select_permutation(void* session, uint32_t index, char* error, size_t error_capacity);
int cbm_multiclass_session_copy_permutation_state(void* session, uint32_t capacity,
    float* predictions, float* mvs_lambdas, uint8_t* mvs_valid, char* error, size_t error_capacity);
// Exact resume requires the optimizer cursor separately from published raw
// predictions: float32 gauge subtraction is not reversible. Class-major
// [permutations,D,rows], D=C-1 for MultiClass and C for OneVsAll.
int cbm_multiclass_session_copy_optimization_state(void* session, uint32_t capacity,
    float* predictions, char* error, size_t error_capacity);
int cbm_multiclass_session_restore_optimization_state(void* session, uint32_t count,
    const float* predictions, char* error, size_t error_capacity);
int cbm_multiclass_session_set_feature_penalties(void* session, const CBMFeaturePenaltyOptions* options,
    const uint32_t* ctr_unique_values, const float* feature_weights,
    const uint8_t* used_features, char* error, size_t error_capacity);
int cbm_multiclass_session_copy_feature_penalty_state(void* session, uint8_t* used_features,
    char* error, size_t error_capacity);
void cbm_multiclass_session_close(void* session);

// Diagnostic math entrypoint used to validate the actual GPU objective and
// coupled leaf solver. logits/gradients are CLASS-MAJOR[D,rows], D=C-1 for
// MultiClass and C for OVA. probabilities is [C,rows], losses is [rows],
// directions is LEAF-MAJOR[leaves,D]. leaf_ids contains [0,leaves).
// All pointers are required, including outputs. Positive weighted losses are
// averaged over classes for OVA; its derivatives/Hessian are not divided by C.
int cbm_multiclass_math(
    uint32_t rows, uint32_t classes, uint32_t objective, uint32_t leaves,
    uint32_t leaf_method, float l2, const uint32_t* labels, const float* weights,
    const float* logits, const uint32_t* leaf_ids, float* gradients,
    float* probabilities, float* losses, float* directions,
    CBMTrainStats* stats, char* error, size_t error_capacity);

#ifdef __cplusplus
}
#endif
