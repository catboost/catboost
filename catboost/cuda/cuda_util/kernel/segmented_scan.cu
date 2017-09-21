#include "scan.cuh"
#include "segmented_scan.cuh"
#include "segmented_scan_helpers.cuh"
#include <contrib/libs/cub/cub/device/device_scan.cuh>

namespace NKernel
{

    template<class T>
    __global__ void ZeroSegmentStartsImpl(const ui32* flags, ui32 flagMask, ui32 size, T* output) {
        const ui32 tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid < size) {
            bool segmentStart = flags[tid] & flagMask;
            if (segmentStart) {
                output[tid] = 0;
            }
        }
    }

    template<typename T>
    cudaError_t SegmentedScanCub(const T* input, const ui32* flags, ui32 flagMask,
                                 T* output,
                                 ui32 size, bool inclusive,
                                 TScanKernelContext<T>& context,
                                 TCudaStream stream) {
        if (inclusive) {
            using TInput = TSegmentedScanInputIterator<T>;
            using TOutput = TSegmentedScanOutputIterator<T, true>;
            TInput inputIter(input, flags, flagMask);
            TOutput outputIter(output, output + size);
            return cub::DeviceScan::InclusiveScan<TInput, TOutput>(context.PartResults, context.NumParts, inputIter, outputIter, TSegmentedSum(), size, stream);
        } else {
            using TInput = TSegmentedScanInputIterator<T>;
            using TOutput = TSegmentedScanOutputIterator<T, false>;
            TInput inputIter(input, flags, flagMask);
            TOutput outputIter(output, output + size);
            cudaError_t errorCode = cub::DeviceScan::InclusiveScan<TInput, TOutput>(context.PartResults, context.NumParts, inputIter, outputIter, TSegmentedSum(), size, stream);
            {
                ui32 blockSize = 256;
                ui32 numBlocks = CeilDivide<ui32>(size, blockSize);
                ZeroSegmentStartsImpl<<<numBlocks, blockSize, 0, stream>>>(flags, flagMask, size, output);
            }
            return errorCode;
        }
    }


    template <class T>
    ui64 SegmentedScanVectorTempSize(ui32 size, bool inclusive) {
        (void)inclusive;

        using TInput = TSegmentedScanInputIterator<T>;
        using TOutput = TSegmentedScanOutputIterator<T, true>;
        ui64 sizeInBytes = 0;
        TInput fakeInput((T*)nullptr, (ui32*)nullptr, 0u);
        TOutput fakeOutput((T*)nullptr, (T*)nullptr);
        cub::DeviceScan::InclusiveScan<TInput, TOutput, TSegmentedSum>(nullptr, sizeInBytes, fakeInput, fakeOutput, TSegmentedSum(), size);
        return sizeInBytes;
    }

    template  cudaError_t SegmentedScanCub<float>(const float* input, const ui32* flags, ui32 mask, float* output, ui32 size, bool inclusive,
                                                  TScanKernelContext<float>& context, TCudaStream stream);

    template  cudaError_t SegmentedScanCub<int>(const int* input, const ui32* flags, ui32 mask, int* output, ui32 size, bool inclusive,
                                                TScanKernelContext<int>& context, TCudaStream stream);

    template  cudaError_t SegmentedScanCub<ui32>(const ui32* input, const ui32* flags, ui32 mask, ui32* output, ui32 size, bool inclusive,
                                                 TScanKernelContext<ui32>& context, TCudaStream stream);

    template ui64 SegmentedScanVectorTempSize<int>(ui32, bool);
    template ui64 SegmentedScanVectorTempSize<ui32>(ui32, bool);
    template ui64 SegmentedScanVectorTempSize<float>(ui32, bool);
}
