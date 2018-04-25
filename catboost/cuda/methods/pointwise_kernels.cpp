#include "pointwise_kernels.h"
#include <catboost/cuda/gpu_data/grid_policy.h>

using namespace NKernelHost;

namespace NCudaLib {
    REGISTER_KERNEL(0x420000, TComputeHist2Kernel);
    REGISTER_KERNEL(0x420001, TComputeHist1Kernel);

    REGISTER_KERNEL(0x420003, TUpdateFoldBinsKernel);
    REGISTER_KERNEL(0x420004, TUpdatePartitionPropsKernel);
    REGISTER_KERNEL(0x420005, TGatherHistogramByLeavesKernel);
    REGISTER_KERNEL(0x420006, TFindOptimalSplitKernel);

}
