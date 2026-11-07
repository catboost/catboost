#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CBMEvaluationSession CBMEvaluationSession;

typedef struct {
    uint32_t rows, features, max_depth;
    float bias;
} CBMEvaluationParams;

// Additive matrix ABI: existing scalar parameter/statistics layouts and entry
// points retain their original shapes. Matrix arrays are row-major [rows,C]
// predictions and [leaves,C] leaf values; C is in [1,64].
typedef struct {
    uint32_t rows, features, max_depth, classes;
} CBMEvaluationMatrixParams;

typedef struct {
    uint64_t kernel_dispatches;
    double gpu_seconds;
    uint64_t dataset_uploads, bins_upload_bytes, resident_bytes, tree_upload_bytes;
    char device_name[256];
} CBMEvaluationStats;

// The session owns a copy of feature-major bins and a float32 prediction cursor.
// Optional initial_predictions replace bias and must have exactly rows entries.
// Every array count is checked before the array is read. No caller-owned input
// pointers are retained. Sessions serialize operations and destroy is idempotent.
int cbm_evaluation_create(
    const CBMEvaluationParams* params,
    const uint8_t* bins, uint64_t bins_count,
    const float* initial_predictions, uint64_t initial_predictions_count,
    CBMEvaluationSession** session, char* error, size_t error_capacity);

// Supply only active splits (exactly depth) and leaves (exactly 2**depth).
// Split type 0 compares bin > border; type 1 compares bin == category.
// First split contributes leaf bit zero. An accepted tree performs one Metal
// dispatch for nonempty data, adding leaves with float32 training arithmetic.
int cbm_evaluation_add_tree(
    CBMEvaluationSession* session, uint32_t depth,
    const uint32_t* split_features, uint64_t split_features_count,
    const uint32_t* split_bins, uint64_t split_bins_count,
    const uint8_t* split_types, uint64_t split_types_count,
    const float* leaf_values, uint64_t leaf_values_count,
    float* predictions, uint64_t predictions_count,
    char* error, size_t error_capacity);

int cbm_evaluation_predictions(
    CBMEvaluationSession* session, float* predictions, uint64_t predictions_count,
    char* error, size_t error_capacity);

// Bias contains exactly classes entries, even when an initial cursor replaces
// it. The single feature-major bins matrix is shared by every output column.
int cbm_evaluation_create_matrix(
    const CBMEvaluationMatrixParams* params,
    const uint8_t* bins, uint64_t bins_count,
    const float* bias, uint64_t bias_count,
    const float* initial_predictions, uint64_t initial_predictions_count,
    CBMEvaluationSession** session, char* error, size_t error_capacity);

int cbm_evaluation_add_tree_matrix(
    CBMEvaluationSession* session, uint32_t depth,
    const uint32_t* split_features, uint64_t split_features_count,
    const uint32_t* split_bins, uint64_t split_bins_count,
    const uint8_t* split_types, uint64_t split_types_count,
    const float* leaf_values, uint64_t leaf_values_count,
    float* predictions, uint64_t predictions_count,
    char* error, size_t error_capacity);

int cbm_evaluation_predictions_matrix(
    CBMEvaluationSession* session, float* predictions, uint64_t predictions_count,
    char* error, size_t error_capacity);

int cbm_evaluation_stats(
    CBMEvaluationSession* session, CBMEvaluationStats* stats,
    char* error, size_t error_capacity);

void cbm_evaluation_destroy(CBMEvaluationSession* session);

#ifdef __cplusplus
}
#endif
