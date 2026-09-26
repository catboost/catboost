#pragma once
#include "metal_trainer.h"
#ifdef __cplusplus
extern "C" {
#endif
// CUDA vector objective math. Objective 0=MultiRMSE, 1=RMSEWithUncertainty,
// 2=MultiLogloss, 3=MultiCrossEntropy. The diagnostic IDs are separate from
// the shared training session's objective IDs (2..5).
// Targets are dimension-major[D,N] for MultiRMSE and [N] for uncertainty.
// Predictions/gradients/Hessian diagonals are dimension-major[D,N]; uncertainty
// uses D=2, (mean, log standard deviation). Directions are leaf-major[L,D].
// Weighted losses are SSE summed over D (MultiRMSE), or Gaussian NLL.
int cbm_multioutput_math(uint32_t rows, uint32_t dimensions, uint32_t objective,
    uint32_t leaves, uint32_t leaf_method, float l2, const float* targets,
    const float* weights, const float* predictions, const uint32_t* leaf_ids,
    float* gradients, float* hessian_diagonal, float* losses, float* directions,
    CBMTrainStats* stats, char* error, size_t error_capacity);
#ifdef __cplusplus
}
#endif
