#include "pointwise_kernels.h"
#include <catboost/cuda/gpu_data/grid_policy.h>

using namespace NKernelHost;

namespace NCudaLib {
    REGISTER_KERNEL_TEMPLATE(0x420000, TComputeHistKernel, NCatboostCuda::EFeaturesGroupingPolicy::BinaryFeatures);
    REGISTER_KERNEL_TEMPLATE(0x420001, TComputeHistKernel, NCatboostCuda::EFeaturesGroupingPolicy::HalfByteFeatures);
    REGISTER_KERNEL_TEMPLATE(0x420002, TComputeHistKernel, NCatboostCuda::EFeaturesGroupingPolicy::OneByteFeatures);

    REGISTER_KERNEL(0x420003, TUpdateFoldBinsKernel);
    REGISTER_KERNEL(0x420004, TUpdatePartitionPropsKernel);
    REGISTER_KERNEL(0x420005, TGatherHistogramByLeavesKernel);
    REGISTER_KERNEL(0x420006, TFindOptimalSplitKernel);

}
