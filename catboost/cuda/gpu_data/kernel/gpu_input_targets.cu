#include "gpu_input_targets.cuh"

namespace NKernel {
    namespace {

        __global__ void FillLearnTestWeightsImpl(
            float* __restrict weights,
            ui32 learnSize,
            ui32 totalSize,
            float learnValue,
            float testValue
        ) {
            const ui32 i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < totalSize) {
                weights[i] = (i < learnSize) ? learnValue : testValue;
            }
        }

        __device__ __forceinline__ ui32 UpperBoundBorders(const float* borders, ui32 borderCount, float value) {
            ui32 left = 0;
            ui32 right = borderCount;
            while (left < right) {
                const ui32 mid = (left + right) >> 1;
                const float b = borders[mid];
                if (value > b) {
                    left = mid + 1;
                } else {
                    right = mid;
                }
            }
            return left;
        }

        __global__ void BinarizeToUi8Impl(
            const float* __restrict values,
            ui32 size,
            const float* __restrict borders,
            ui32 borderCount,
            ui8* __restrict dst
        ) {
            const ui32 i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < size) {
                const float v = values[i];
                const ui32 bin = (borderCount == 0) ? 0u : UpperBoundBorders(borders, borderCount, v);
                dst[i] = static_cast<ui8>(bin);
            }
        }

        __global__ void BinarizeToUi32Impl(
            const float* __restrict values,
            ui32 size,
            const float* __restrict borders,
            ui32 borderCount,
            ui32* __restrict dst
        ) {
            const ui32 i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < size) {
                const float v = values[i];
                const ui32 bin = (borderCount == 0) ? 0u : UpperBoundBorders(borders, borderCount, v);
                dst[i] = bin;
            }
        }

    }

    void FillLearnTestWeights(
        float* weights,
        ui32 learnSize,
        ui32 totalSize,
        float learnValue,
        float testValue,
        TCudaStream stream
    ) {
        if (totalSize == 0) {
            return;
        }
        const ui32 blockSize = 256;
        const ui32 numBlocks = (totalSize + blockSize - 1) / blockSize;
        FillLearnTestWeightsImpl<<<numBlocks, blockSize, 0, stream>>>(
            weights,
            learnSize,
            totalSize,
            learnValue,
            testValue
        );
    }

    void BinarizeToUi8(
        const float* values,
        ui32 size,
        const float* borders,
        ui32 borderCount,
        ui8* dst,
        TCudaStream stream
    ) {
        if (size == 0) {
            return;
        }
        const ui32 blockSize = 256;
        const ui32 numBlocks = (size + blockSize - 1) / blockSize;
        BinarizeToUi8Impl<<<numBlocks, blockSize, 0, stream>>>(
            values,
            size,
            borders,
            borderCount,
            dst
        );
    }

    void BinarizeToUi32(
        const float* values,
        ui32 size,
        const float* borders,
        ui32 borderCount,
        ui32* dst,
        TCudaStream stream
    ) {
        if (size == 0) {
            return;
        }
        const ui32 blockSize = 256;
        const ui32 numBlocks = (size + blockSize - 1) / blockSize;
        BinarizeToUi32Impl<<<numBlocks, blockSize, 0, stream>>>(
            values,
            size,
            borders,
            borderCount,
            dst
        );
    }

}
