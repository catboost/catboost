#include "gpu_final_ctr.cuh"

#include <catboost/cuda/cuda_lib/cuda_base.h>

namespace NKernel {
    namespace {
        __device__ __forceinline__ ui64 CalcHashDevice(ui64 a, ui64 b) {
            constexpr ui64 MagicMult = 0x4906ba494954cb65ull;
            return MagicMult * (a + MagicMult * b);
        }

        __global__ void UpdateHashesFromCatFeatureImpl(
            const ui32* __restrict bins,
            ui32 size,
            const ui32* __restrict binToHash,
            ui64* __restrict hashes
        ) {
            const ui32 i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < size) {
                const ui32 bin = bins[i];
                const ui32 hash32 = binToHash[bin];
                const ui64 hash64 = (ui64)(int)hash32;
                hashes[i] = CalcHashDevice(hashes[i], hash64);
            }
        }

        __global__ void UpdateHashesFromFloatSplitImpl(
            const float* __restrict values,
            ui32 size,
            float borderValue,
            ui64* __restrict hashes
        ) {
            const ui32 i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < size) {
                const bool isTrue = values[i] > borderValue;
                hashes[i] = CalcHashDevice(hashes[i], static_cast<ui64>(isTrue));
            }
        }

        __global__ void UpdateHashesFromOneHotFeatureImpl(
            const ui32* __restrict bins,
            ui32 size,
            ui32 value,
            ui64* __restrict hashes
        ) {
            const ui32 i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < size) {
                const bool isTrue = (bins[i] == value);
                hashes[i] = CalcHashDevice(hashes[i], static_cast<ui64>(isTrue));
            }
        }

        __global__ void UpdateHashesFromBinarizedSplitImpl(
            const ui8* __restrict binarizedBins,
            ui32 size,
            ui8 threshold,
            ui64* __restrict hashes
        ) {
            const ui32 i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < size) {
                const bool isTrue = (binarizedBins[i] >= threshold);
                hashes[i] = CalcHashDevice(hashes[i], static_cast<ui64>(isTrue));
            }
        }

        __global__ void RemapIndicesImpl(
            const ui32* __restrict src,
            ui32 size,
            const ui32* __restrict remap,
            ui32* __restrict dst
        ) {
            const ui32 i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < size) {
                const ui32 v = src[i];
                dst[i] = remap[v];
            }
        }

        __global__ void AccumulateFloatByIndexImpl(
            const ui32* __restrict indices,
            const float* __restrict values,
            ui32 size,
            float* __restrict sums
        ) {
            const ui32 i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < size) {
                const ui32 idx = indices[i];
                atomicAdd(sums + idx, values[i]);
            }
        }

        __global__ void AccumulateBinarizedTargetSumByIndexImpl(
            const ui32* __restrict indices,
            const ui32* __restrict targetClass,
            ui32 size,
            float invTargetBorderCount,
            float* __restrict sums
        ) {
            const ui32 i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < size) {
                const ui32 idx = indices[i];
                atomicAdd(sums + idx, static_cast<float>(targetClass[i]) * invTargetBorderCount);
            }
        }

        __global__ void AccumulateClassCountsByIndexImpl(
            const ui32* __restrict indices,
            const ui32* __restrict targetClass,
            ui32 size,
            ui32 classCount,
            ui32* __restrict counts
        ) {
            const ui32 i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < size) {
                const ui32 idx = indices[i];
                const ui32 cls = targetClass[i];
                if (cls < classCount) {
                    atomicAdd(counts + (static_cast<ui64>(idx) * classCount + cls), 1u);
                }
            }
        }
    }

    void UpdateHashesFromCatFeature(
        const ui32* bins,
        ui32 size,
        const ui32* binToHash,
        ui64* hashes,
        TCudaStream stream
    ) {
        if (size == 0) {
            return;
        }
        const ui32 blockSize = 256;
        const ui32 numBlocks = (size + blockSize - 1) / blockSize;
        UpdateHashesFromCatFeatureImpl<<<numBlocks, blockSize, 0, stream>>>(bins, size, binToHash, hashes);
    }

    void UpdateHashesFromFloatSplit(
        const float* values,
        ui32 size,
        float borderValue,
        ui64* hashes,
        TCudaStream stream
    ) {
        if (size == 0) {
            return;
        }
        const ui32 blockSize = 256;
        const ui32 numBlocks = (size + blockSize - 1) / blockSize;
        UpdateHashesFromFloatSplitImpl<<<numBlocks, blockSize, 0, stream>>>(values, size, borderValue, hashes);
    }

    void UpdateHashesFromOneHotFeature(
        const ui32* bins,
        ui32 size,
        ui32 value,
        ui64* hashes,
        TCudaStream stream
    ) {
        if (size == 0) {
            return;
        }
        const ui32 blockSize = 256;
        const ui32 numBlocks = (size + blockSize - 1) / blockSize;
        UpdateHashesFromOneHotFeatureImpl<<<numBlocks, blockSize, 0, stream>>>(bins, size, value, hashes);
    }

    void UpdateHashesFromBinarizedSplit(
        const ui8* binarizedBins,
        ui32 size,
        ui8 threshold,
        ui64* hashes,
        TCudaStream stream
    ) {
        if (size == 0) {
            return;
        }
        const ui32 blockSize = 256;
        const ui32 numBlocks = (size + blockSize - 1) / blockSize;
        UpdateHashesFromBinarizedSplitImpl<<<numBlocks, blockSize, 0, stream>>>(
            binarizedBins,
            size,
            threshold,
            hashes
        );
    }

    void RemapIndices(
        const ui32* src,
        ui32 size,
        const ui32* remap,
        ui32* dst,
        TCudaStream stream
    ) {
        if (size == 0) {
            return;
        }
        const ui32 blockSize = 256;
        const ui32 numBlocks = (size + blockSize - 1) / blockSize;
        RemapIndicesImpl<<<numBlocks, blockSize, 0, stream>>>(src, size, remap, dst);
    }

    void AccumulateFloatByIndex(
        const ui32* indices,
        const float* values,
        ui32 size,
        float* sums,
        TCudaStream stream
    ) {
        if (size == 0) {
            return;
        }
        const ui32 blockSize = 256;
        const ui32 numBlocks = (size + blockSize - 1) / blockSize;
        AccumulateFloatByIndexImpl<<<numBlocks, blockSize, 0, stream>>>(indices, values, size, sums);
    }

    void AccumulateBinarizedTargetSumByIndex(
        const ui32* indices,
        const ui32* targetClass,
        ui32 size,
        float invTargetBorderCount,
        float* sums,
        TCudaStream stream
    ) {
        if (size == 0) {
            return;
        }
        const ui32 blockSize = 256;
        const ui32 numBlocks = (size + blockSize - 1) / blockSize;
        AccumulateBinarizedTargetSumByIndexImpl<<<numBlocks, blockSize, 0, stream>>>(
            indices,
            targetClass,
            size,
            invTargetBorderCount,
            sums
        );
    }

    void AccumulateClassCountsByIndex(
        const ui32* indices,
        const ui32* targetClass,
        ui32 size,
        ui32 classCount,
        ui32* counts,
        TCudaStream stream
    ) {
        if (size == 0) {
            return;
        }
        const ui32 blockSize = 256;
        const ui32 numBlocks = (size + blockSize - 1) / blockSize;
        AccumulateClassCountsByIndexImpl<<<numBlocks, blockSize, 0, stream>>>(indices, targetClass, size, classCount, counts);
    }

}
