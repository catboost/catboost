#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Additive Langevin callback ABI. The native session owns neither callback
// context nor output storage. Callbacks run synchronously and return zero on
// success; a nonzero return aborts the current native operation.
typedef enum {
    CBM_LANGEVIN_WEAK_SEED_CACHE = 0,
    CBM_LANGEVIN_INITIAL_GRADIENT = 1,
    CBM_LANGEVIN_INITIAL_HESSIAN = 2,
    CBM_LANGEVIN_TRIAL_GRADIENT = 3,
    CBM_LANGEVIN_ACCEPTED_GRADIENT = 4,
    CBM_LANGEVIN_YETI_WEAK = 5,
    CBM_LANGEVIN_YETI_LEAF = 6,
    CBM_LANGEVIN_SEARCH = 7,
    CBM_LANGEVIN_BASE_ITERATION = 8,
    CBM_LANGEVIN_PERMUTATION = 9
} CBMLangevinEvent;

// Fill count additive leaf-noise values in source vector order. Each call
// consumes one shared host seed, including when diffusion temperature is zero.
// Only INITIAL_GRADIENT, INITIAL_HESSIAN, TRIAL_GRADIENT, and
// ACCEPTED_GRADIENT are valid here. An accepted CUDA trial noises its gradient
// a second time after computing its clean Hessian; it does not noise that
// Hessian. A FeatureParallel call covers all concatenated estimation tasks.
typedef int (*CBMLangevinNoiseCallback)(void* context, uint32_t event,
    uint32_t count, double* noise);

// WEAK_SEED_CACHE initializes the shared GPU-seed cache once, consuming its
// allocation draw plus 65,536 fill draws, and returns seed=0. Later cache
// events consume no host draws. YETI_WEAK, YETI_LEAF, and SEARCH consume one
// host seed each. BASE_ITERATION and PERMUTATION describe host-only events;
// native code must not repeat them through this callback.
// Device bootstrap/noise state remains the established Metal adaptation;
// matching host consumption does not promise NVIDIA-bitwise device samples.
typedef int (*CBMLangevinSeedCallback)(void* context, uint32_t event,
    uint64_t* seed);

#ifdef __cplusplus
}
#endif
