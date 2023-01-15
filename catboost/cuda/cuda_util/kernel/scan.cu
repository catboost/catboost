#include "scan.cuh"
#include "segmented_scan_helpers.cuh"
#include "fill.cuh"
#include <contrib/libs/cub/cub/device/device_scan.cuh>

namespace NKernel {

    template <typename T>
    cudaError_t ScanVector(const T* input, T* output, ui32 size, bool inclusive, TScanKernelContext<T>& context, TCudaStream stream) {
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

    template <typename T_>
    cudaError_t SegmentedScanNonNegativeVector(const T_* input, T_* output, ui32 size, bool inclusive, TScanKernelContext<T_>& context, TCudaStream stream) {
        using TKernelContext = TScanKernelContext<T_>;
        using T = typename TToSignedConversion<T_>::TSignedType;
        T zeroValue = 0.0f;
        if (inclusive) {
            return cub::DeviceScan::InclusiveScan((T*)context.PartResults.Get(), context.NumParts, (const T*)input, (T*)output, TNonNegativeSegmentedSum(), size, stream);
        } else {
            return cub::DeviceScan::ExclusiveScan((T*)context.PartResults.Get(), context.NumParts, (const T*) input, (T*)output, TNonNegativeSegmentedSum(), zeroValue, size, stream);
        }
    }


    template <typename T_>
    cudaError_t SegmentedScanAndScatterNonNegativeVector(const T_* input, const ui32* indices, T_* output,
                                                         ui32 size, bool inclusive,
                                                         TScanKernelContext<T_>& context,
                                                         TCudaStream stream) {
        using TKernelContext = TScanKernelContext<T_>;
        using T = typename TToSignedConversion<T_>::TSignedType;

        if (inclusive) {
            TNonNegativeSegmentedScanOutputIterator<cub::STORE_CS, T,  ptrdiff_t, true>  outputIterator((T*)output, indices, indices + size);
            return cub::DeviceScan::InclusiveScan((T*)context.PartResults.Get(), context.NumParts, (const T*)input, outputIterator, TNonNegativeSegmentedSum(), size, stream);
        } else {
            TNonNegativeSegmentedScanOutputIterator<cub::STORE_CS, T,  ptrdiff_t, false>  outputIterator((T*)output, indices, indices + size);
            FillBuffer<T>((T*)output, 0, size, stream);
            return cub::DeviceScan::InclusiveScan((T*)context.PartResults.Get(), context.NumParts, (const T*) input, outputIterator, TNonNegativeSegmentedSum(), size, stream);
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



    #define SCAN_VECTOR(Type) \
    template  cudaError_t ScanVector<Type>(const Type *input, Type *output, ui32 size, bool inclusive, TScanKernelContext<Type>& context, TCudaStream stream); \
    template  cudaError_t SegmentedScanNonNegativeVector<Type>(const Type *input, Type *output, ui32 size, bool inclusive, TScanKernelContext<Type>& context, TCudaStream stream); \
    template  cudaError_t SegmentedScanAndScatterNonNegativeVector<Type>(const Type *input, const ui32* indices, Type *output, ui32 size, bool inclusive, TScanKernelContext<Type>& context, TCudaStream stream); \
    template ui64 ScanVectorTempSize<Type>(ui32, bool);


    SCAN_VECTOR(int)
    SCAN_VECTOR(ui32)
    SCAN_VECTOR(float)
    SCAN_VECTOR(double)


}
