#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t rows, categories, ctr_type, target_border;
    float prior_numerator, prior_denominator;
} CBMCtrParams;

typedef struct {
    uint64_t kernel_dispatches;
    double gpu_seconds;
    char device_name[256];
} CBMCtrStats;

// Input indices group categories in ascending order and preserve the caller's
// permutation within each category. Output values are scattered to row order.
// ctr_type: 0 Borders, 1 Buckets, 2 FloatTargetMeanValue, 3 FeatureFreq.
int cbm_compute_ctrs(const CBMCtrParams* params, const uint32_t* categories,
                     const uint32_t* indices, const float* targets,
                     float* values, float* sums, uint32_t* counts,
                     CBMCtrStats* stats, char* error, size_t error_capacity);

// Group-aware target histories read the category cursor before the entire
// current group. group_ids contains arbitrary uint32 labels in original row
// order. indices must come from stable category sorting of a permutation in
// which all rows of each group are contiguous. Each category's group runs are
// validated here; the caller guarantees a consistent global group order.
// A null group_ids pointer has exactly cbm_compute_ctrs semantics. Frequency
// CTRs and final learn statistics are independent of the history unit.
int cbm_compute_ctrs_grouped(const CBMCtrParams* params, const uint32_t* categories,
                             const uint32_t* indices, const float* targets,
                             const uint32_t* group_ids,
                             float* values, float* sums, uint32_t* counts,
                             CBMCtrStats* stats, char* error, size_t error_capacity);

#ifdef __cplusplus
}
#endif
