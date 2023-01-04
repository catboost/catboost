#include <catboost/cuda/cuda_util/kernel/sort_templ.cuh>

namespace NKernel {
    template cudaError_t RadixSort(ui64* keys, ui32* values, ui32 size, TRadixSortContext& context,  TCudaStream stream);
    template cudaError_t RadixSort(ui64* keys, ui64* values, ui32 size, TRadixSortContext& context,  TCudaStream stream);
}
