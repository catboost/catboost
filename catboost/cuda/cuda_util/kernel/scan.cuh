#pragma once

#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>

namespace NKernel {

    inline constexpr ui32 GetScanBlockSize() {
        return 768;
    }

    inline constexpr ui32 GetScanMaxBlocks() {
        return GetScanBlockSize();
    }

    template <class T, class TOut>
    struct TScanKernelContext : public IKernelContext {
        ui64 NumParts = 0;
        TDevicePointer<char> PartResults;

    };

    template <class T, class TOut>
    ui64 ScanVectorTempSize(ui32 size, bool inclusive);


    template <typename T, typename TOut>
    cudaError_t ScanVector(const T* input, TOut* output, ui32 size, bool inclusive, TScanKernelContext<T, TOut>& context, TCudaStream stream);

    template <typename T, typename TOut>
    cudaError_t SegmentedScanNonNegativeVector(const T* input, TOut* output, ui32 size, bool inclusive, TScanKernelContext<T, TOut>& context, TCudaStream stream);


    template <typename T>
    cudaError_t SegmentedScanAndScatterNonNegativeVector(const T* input, const ui32* indices, T* output,
                                                         ui32 size, bool inclusive,
                                                         TScanKernelContext<T, T>& context,
                                                         TCudaStream stream);



}
