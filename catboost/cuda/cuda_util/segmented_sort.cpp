#include <catboost/cuda/cuda_util/segmented_sort.h>

using namespace NKernelHost;

namespace NCudaLib {
    REGISTER_KERNEL_TEMPLATE_2(0xAB0001, TSegmentedRadixSortKernel, ui32, ui32);
}
