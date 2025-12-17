#pragma once

#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>

namespace NKernel {

    // Fill weights for joined learn+test targets: weights[i]=learnValue for i<learnSize, else testValue.
    void FillLearnTestWeights(
        float* weights,
        ui32 learnSize,
        ui32 totalSize,
        float learnValue,
        float testValue,
        TCudaStream stream
    );

    // Binarize float values into ui8 bins using sorted borders. Output is in [0, borderCount].
    void BinarizeToUi8(
        const float* values,
        ui32 size,
        const float* borders,
        ui32 borderCount,
        ui8* dst,
        TCudaStream stream
    );

    // Binarize float values into ui32 bins using sorted borders. Output is in [0, borderCount].
    void BinarizeToUi32(
        const float* values,
        ui32 size,
        const float* borders,
        ui32 borderCount,
        ui32* dst,
        TCudaStream stream
    );

}
