#include "filter.cuh"

#include <util/generic/cast.h>

namespace NKernel {

    struct TZeroWeightFilter {

        __device__ ui32 operator()(float w) {
            return abs(w) > 1e-20f;
        }
    };

    template <class Filter = TZeroWeightFilter, typename TResult>
    __global__ void FilterImpl(const float* weights,
                               ui64 size,
                               TResult* result) {
        const ui64 i = (ui64)blockIdx.x * blockDim.x + threadIdx.x;
        Filter filter;
        if (i < size) {
            result[i] = filter(__ldg(weights + i));
        }
    }


    template <typename TResult>
    void Filter(const float* weights, const ui64 size, TResult* result, TCudaStream stream) {
        if (size > 0) {
            const ui32 blockSize = 512;
            const ui32 numBlocks = SafeIntegerCast<ui32>((size + blockSize - 1) / (blockSize));
            FilterImpl << <numBlocks, blockSize, 0, stream>>>(weights, size, result);
        }
    }

    template
    void Filter<ui32>(const float* weights, const ui64 size, ui32* result, TCudaStream stream);

    template
    void Filter<ui64>(const float* weights, const ui64 size, ui64* result, TCudaStream stream);
}
