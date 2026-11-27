#pragma once

#include <stdint.h>

// Budget for live training buffers on the selected unified-memory device.
// Preparation on the host can retain permutation banks alongside GPU copies.
// Reserve half of Metal's recommended working set for those copies and other
// resources. A missing recommendation retains the previous 1 GiB fallback.
static inline uint64_t CBMTrainingWorkingSetLimit(uint64_t recommendedBytes) {
    return recommendedBytes ? recommendedBytes / 2 : (UINT64_C(1) << 30);
}

#ifdef __cplusplus
extern "C" {
#endif
uint64_t cbm_metal_working_set_limit(void);
#ifdef __cplusplus
}
#endif
