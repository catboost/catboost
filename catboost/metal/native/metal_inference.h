#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t rows, features, trees, split_stride, leaf_stride;
    uint32_t tree_start, tree_end, batch_rows;
    double scale, bias;
} CBMInferenceParams;

typedef struct {
    uint64_t kernel_dispatches;
    double gpu_seconds;
    char device_name[256];
} CBMInferenceStats;

// Variable binary tree; internal nodes use leaf=UINT32_MAX and absolute
// left/right node indices. A terminal leaf addresses a row of vector values.
typedef struct {
    uint32_t feature, bin, type, left, right, leaf;
} CBMInferenceNode;

// Compact non-symmetric forest, without exponential padding. roots has one
// absolute node index per tree; leaf_values is [leaf_count,dimensions].
// split_stride and leaf_stride are zero for this additive interface.
int cbm_predict_non_symmetric_bins_multidim(
    const CBMInferenceParams* params, uint32_t dimensions,
    const double* biases, uint64_t biases_count,
    const uint8_t* bins, uint64_t bins_count,
    const uint32_t* roots, uint64_t roots_count,
    const CBMInferenceNode* nodes, uint64_t nodes_count,
    const double* leaf_values, uint64_t leaf_values_count,
    double* predictions, uint64_t predictions_count,
    CBMInferenceStats* stats, char* error, size_t error_capacity);

// Feature-major bins; first split contributes leaf-index bit zero. Every buffer
// has an explicit element count, validated before native code reads its data.
// A tree range starting after zero excludes bias, as in ApplyScaleAndBias.
int cbm_predict_bins(
    const CBMInferenceParams* params,
    const uint8_t* bins, uint64_t bins_count,
    const uint32_t* depths, uint64_t depths_count,
    const uint32_t* split_features, uint64_t split_features_count,
    const uint32_t* split_bins, uint64_t split_bins_count,
    const uint8_t* split_types, uint64_t split_types_count,
    const double* leaf_values, uint64_t leaf_values_count,
    double* predictions, uint64_t predictions_count,
    CBMInferenceStats* stats, char* error, size_t error_capacity);

// Additive multidimensional interface; the scalar ABI above is unchanged.
// dimensions is in [1,64]. Leaf values are [trees,leaf_stride,dimensions] and
// predictions are [rows,dimensions], both row-major. biases has dimensions
// elements. params->scale applies to every dimension; params->bias must be zero
// because this interface uses the explicit bias vector. Tree ranges beginning
// after zero exclude every bias component. Quantized inputs and split buffers
// have exactly the same shapes and predicates as cbm_predict_bins.
int cbm_predict_bins_multidim(
    const CBMInferenceParams* params, uint32_t dimensions,
    const double* biases, uint64_t biases_count,
    const uint8_t* bins, uint64_t bins_count,
    const uint32_t* depths, uint64_t depths_count,
    const uint32_t* split_features, uint64_t split_features_count,
    const uint32_t* split_bins, uint64_t split_bins_count,
    const uint8_t* split_types, uint64_t split_types_count,
    const double* leaf_values, uint64_t leaf_values_count,
    double* predictions, uint64_t predictions_count,
    CBMInferenceStats* stats, char* error, size_t error_capacity);

#ifdef __cplusplus
}
#endif
