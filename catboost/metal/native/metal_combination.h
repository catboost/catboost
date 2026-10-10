#pragma once

#include "metal_trainer.h"

#ifdef __cplusplus
extern "C" {
#endif

// Additive Combination objective ABI (objective 19). Components use the scalar
// objective IDs, 12/13 for queries, 14 for PairLogit, or 17 for classic YetiRank.
// Coefficients supplied here are POSITIVE; the runtime applies CUDA's negative
// YetiRank coefficient exactly once. Zero-weight components are omitted by the
// native option parser. Query components execute before pointwise components.
typedef struct {
    uint32_t objective;
    float weight, param, border, beta, lambda;
    uint32_t permutations, reserved;
    float decay;
    uint32_t reserved1[3];
} CBMCombinationComponent;

typedef struct {
    uint32_t component_count, group_count, pair_count, reserved;
} CBMCombinationOptions;

int cbm_session_create_combination(
    const CBMSessionParams* params, const CBMObjectiveOptions* objective_options,
    const CBMCombinationOptions* combination_options, const CBMCombinationComponent* components,
    const uint32_t* group_offsets, const uint32_t* pair_winners,
    const uint32_t* pair_losers, const float* pair_weights,
    const uint8_t* bins, const float* targets, const float* sample_weights,
    const float* initial_predictions, const uint32_t* candidate_features,
    const uint32_t* candidate_bins, const uint8_t* candidate_types,
    void** session, char* error, size_t error_capacity);

// Seed sequence is ordered by derivative call, then active Yeti component.
int cbm_session_set_combination_yeti_seeds(void* session, uint32_t count,
    const uint64_t* seeds, char* error, size_t error_capacity);

// Native controllers can provide a seed source for stochastic backtracking.
// It is invoked synchronously once per evaluated Yeti component; unlike a
// speculative maximum-size packet it advances RNG only for actual trials.
typedef int (*CBMCombinationYetiSeedCallback)(void* context, uint64_t* seed);
int cbm_session_set_combination_yeti_seed_callback(void* session,
    CBMCombinationYetiSeedCallback callback, void* context, char* error, size_t error_capacity);

#ifdef __cplusplus
}
#endif
