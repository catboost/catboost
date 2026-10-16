#pragma once
#include <stddef.h>
#include <stdint.h>
#include "metal_sort.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t rows;
    uint32_t cat_features;
    uint32_t bin_features;
    uint32_t components;
} CBMProjectionParams;

// Exact model CalcHash projection followed by stable unsigned 64-bit grouping.
// Both feature matrices are feature-major. Component types are 0: original
// categorical hash (signed-int32 extension), 1: bin > threshold, 2: bin ==
// threshold. Components must be ordered categorical, numeric, then one-hot,
// matching TStaticCtrProvider; order inside each type is retained exactly.
// Permutation gives source rows in history order; null means original order.
// Equal 64-bit hashes preserve that history order, irrespective of row IDs.
// row_hashes is original row order; sorted_hashes and sorted_rows are grouped.
int cbm_projection_group(const CBMProjectionParams* params,
    const uint32_t* cat_hashes, const uint8_t* bins,
    const uint8_t* component_types, const uint32_t* component_features,
    const uint32_t* component_thresholds, const uint32_t* permutation,
    uint64_t* row_hashes, uint64_t* sorted_hashes, uint32_t* sorted_rows,
    CBMSortStats* stats, char* error, size_t error_capacity);

#ifdef __cplusplus
}
#endif
