#include "scan.cuh"
#include "segmented_scan_helpers.cuh"
#include "fill.cuh"
#include <contrib/libs/cub/cub/device/device_scan.cuh>

namespace NKernel {

    template<typename T>
    cudaError_t ScanVector(const T* input, T* output, uint size, bool inclusive, TScanKernelContext<T>& context, TCudaStream stream) {
        using TKernelContext = TScanKernelContext<T>;

        if (inclusive) {
            return cub::DeviceScan::InclusiveSum(context.PartResults, context.NumParts, input, output, size, stream);
        } else {
            return cub::DeviceScan::ExclusiveSum(context.PartResults, context.NumParts, input, output, size, stream);
        }
    }

    template <class T>
    struct TToSignedConversion {
        using TSignedType = T;
    };


    template <>
    struct TToSignedConversion<ui32> {
        using TSignedType = int;
    };

    template<typename T_>
    cudaError_t SegmentedScanNonNegativeVector(const T_* input, T_* output, ui32 size, bool inclusive, TScanKernelContext<T_>& context, TCudaStream stream) {
        using TKernelContext = TScanKernelContext<T_>;
        using T = typename TToSignedConversion<T_>::TSignedType;
        T zeroValue = 0.0f;
        if (inclusive) {
            return cub::DeviceScan::InclusiveScan((T*)context.PartResults, context.NumParts, (const T*)input, (T*)output, TNonNegativeSegmentedSum(), size, stream);
        } else {
            return cub::DeviceScan::ExclusiveScan((T*)context.PartResults, context.NumParts, (const T*) input, (T*)output, TNonNegativeSegmentedSum(), zeroValue, size, stream);
        }
    }


    template<typename T_>
    cudaError_t SegmentedScanAndScatterNonNegativeVector(const T_* input, const ui32* indices, T_* output,
                                                         ui32 size, bool inclusive,
                                                         TScanKernelContext<T_>& context,
                                                         TCudaStream stream) {
        using TKernelContext = TScanKernelContext<T_>;
        using T = typename TToSignedConversion<T_>::TSignedType;

        if (inclusive) {
            TNonNegativeSegmentedScanOutputIterator<cub::STORE_CS, T,  ptrdiff_t, true>  outputIterator((T*)output, indices, indices + size);
            return cub::DeviceScan::InclusiveScan((T*)context.PartResults, context.NumParts, (const T*)input, outputIterator, TNonNegativeSegmentedSum(), size, stream);
        } else {
            TNonNegativeSegmentedScanOutputIterator<cub::STORE_CS, T,  ptrdiff_t, false>  outputIterator((T*)output, indices, indices + size);
            FillBuffer<T>((T*)output, 0, size, stream);
            return cub::DeviceScan::InclusiveScan((T*)context.PartResults, context.NumParts, (const T*) input, outputIterator, TNonNegativeSegmentedSum(), size, stream);
        }
    }

    template <class T>
    ui64 ScanVectorTempSize(ui32 size, bool inclusive) {
        ui64 sizeInBytes = 0;
        if (inclusive) {
            cub::DeviceScan::InclusiveSum<const T*, T*>(nullptr, sizeInBytes, nullptr, nullptr, size);
        } else {
            cub::DeviceScan::ExclusiveSum<const T*, T*>(nullptr, sizeInBytes, nullptr, nullptr, size);
        }
        return sizeInBytes;
    }



    template ui64 ScanVectorTempSize<int>(ui32, bool);
    template ui64 ScanVectorTempSize<ui32>(ui32, bool);
    template ui64 ScanVectorTempSize<float>(ui32, bool);
    template ui64 ScanVectorTempSize<double>(ui32, bool);


    template  cudaError_t ScanVector<int>(const int *input, int *output, uint size, bool inclusive, TScanKernelContext<int>& context, TCudaStream stream);

    template  cudaError_t ScanVector<uint>(const uint *input, uint *output, uint size, bool inclusive, TScanKernelContext<uint>& context, TCudaStream stream);

    template  cudaError_t ScanVector<float>(const float *input, float *output, uint size, bool inclusive, TScanKernelContext<float>& context, TCudaStream stream);

    template  cudaError_t SegmentedScanNonNegativeVector<float>(const float *input, float *output, ui32 size, bool inclusive, TScanKernelContext<float>& context, TCudaStream stream);
    template  cudaError_t SegmentedScanNonNegativeVector<int>(const int *input, int *output, ui32 size, bool inclusive, TScanKernelContext<int>& context, TCudaStream stream);
    template  cudaError_t SegmentedScanNonNegativeVector<ui32>(const ui32 *input, ui32 *output, ui32 size, bool inclusive, TScanKernelContext<ui32>& context, TCudaStream stream);

    template  cudaError_t ScanVector<double>(const double *input, double *output, uint size, bool inclusive, TScanKernelContext<double>& context, TCudaStream stream);


    template  cudaError_t SegmentedScanAndScatterNonNegativeVector<float>(const float *input, const ui32* indices, float *output, ui32 size, bool inclusive, TScanKernelContext<float>& context, TCudaStream stream);
    template  cudaError_t SegmentedScanAndScatterNonNegativeVector<int>(const int *input, const ui32* indices, int *output, ui32 size, bool inclusive, TScanKernelContext<int>& context, TCudaStream stream);
    template  cudaError_t SegmentedScanAndScatterNonNegativeVector<ui32>(const ui32 *input, const ui32* indices, ui32 *output, ui32 size, bool inclusive, TScanKernelContext<ui32>& context, TCudaStream stream);


}