#include "pointwise_kernels.h"

using namespace NKernelHost;

namespace NCudaLib {

    REGISTER_KERNEL_TEMPLATE(0x420000, TComputeHistKernel, NCatboostCuda::TBinaryFeatureGridPolicy);
    REGISTER_KERNEL_TEMPLATE(0x420001, TComputeHistKernel, NCatboostCuda::THalfByteFeatureGridPolicy);
    REGISTER_KERNEL_TEMPLATE(0x420002, TComputeHistKernel, NCatboostCuda::TByteFeatureGridPolicy);

    REGISTER_KERNEL(0x420003, TUpdateFoldBinsKernel);
    REGISTER_KERNEL(0x420004, TUpdatePartitionPropsKernel);
    REGISTER_KERNEL(0x420005, TGatherHistogramByLeavesKernel);
    REGISTER_KERNEL(0x420006, TFindOptimalSplitKernel);

}
