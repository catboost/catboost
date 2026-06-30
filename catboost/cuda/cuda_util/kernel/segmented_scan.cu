#include "scan.cuh"
#include "segmented_scan.cuh"
#include "segmented_scan_helpers.cuh"

#include <cub/device/device_scan.cuh>

namespace NKernel
{

    template <class T>
    __global__ void ZeroSegmentStartsImpl(const ui32* flags, ui32 flagMask, ui32 size, T* output) {
        const ui32 tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid < size) {
            bool segmentStart = flags[tid] & flagMask;
            if (segmentStart) {
                output[tid] = 0;
            }
        }
    }

    template <typename T>
    cudaError_t SegmentedScanCub(const T* input, const ui32* flags, ui32 flagMask,
                                 T* output,
                                 ui32 size, bool inclusive,
                                 TScanKernelContext<T, T>& context,
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

    #define SEGMENTED_SCAN_CUB(Type)\
    template  cudaError_t SegmentedScanCub<Type>(const Type* input, const ui32* flags, ui32 mask, Type* output, ui32 size, bool inclusive,\
                                                  TScanKernelContext<Type, Type>& context, TCudaStream stream);

    SEGMENTED_SCAN_CUB(float)
    SEGMENTED_SCAN_CUB(double)
    SEGMENTED_SCAN_CUB(int)
    SEGMENTED_SCAN_CUB(ui32)

    template ui64 SegmentedScanVectorTempSize<int>(ui32, bool);
    template ui64 SegmentedScanVectorTempSize<ui32>(ui32, bool);
    template ui64 SegmentedScanVectorTempSize<float>(ui32, bool);
    template ui64 SegmentedScanVectorTempSize<double>(ui32, bool);
}
