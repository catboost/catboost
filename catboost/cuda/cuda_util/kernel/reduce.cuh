#pragma once
#include "cub_storage_context.cuh"
#include <catboost/cuda/cuda_util/operator.h>
#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>

namespace NKernel {

    template <typename T>
    cudaError_t Reduce(const T* input, T* output, ui32 size,
                       EOperatorType type,
                       TCubKernelContext& context, TCudaStream stream);

    template <typename T>
    cudaError_t SegmentedReduce(const T* input, ui32 size, const ui32* offsets, ui32 numSegments, T* output,
                                EOperatorType type,
                                TCubKernelContext& context,
                                TCudaStream stream);


    template <typename T, typename K>
    cudaError_t ReduceByKey(const T* input, const K* keys, ui32 size,
                            T* output, K* outKeys, ui32* outputSize,
                            EOperatorType type,
                            TCubKernelContext& context,
                            TCudaStream stream);

}
