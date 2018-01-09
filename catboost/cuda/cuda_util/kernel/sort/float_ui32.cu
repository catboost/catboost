#include <catboost/cuda/cuda_util/kernel/sort_templ.cuh>

namespace NKernel {
    template cudaError_t RadixSort(float* keys, ui32* values, ui32 size, TRadixSortContext& context,  TCudaStream stream);
}
