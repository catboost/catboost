#include "filter.cuh"

namespace NKernel {

    struct TZeroWeightFilter {

        __device__ ui32 operator()(float w) {
            return abs(w) > 1e-20f;
        }
    };

    template <class Filter = TZeroWeightFilter>
    __global__ void FilterImpl(const float* weights,
                               int size,
                               ui32* result) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        Filter filter;
        if (i < size) {
            result[i] = filter(__ldg(weights + i));
        }
    }


    void Filter(const float* weights, const ui32 size, ui32* result, TCudaStream stream) {
        if (size > 0) {
            const ui32 blockSize = 512;
            const ui32 numBlocks = (size + blockSize - 1) / (blockSize);
            FilterImpl << <numBlocks, blockSize, 0, stream>>>(weights, size, result);
        }
    }
}
