#include "exact_estimation.h"

using namespace NKernelHost;

namespace NCudaLib {
    REGISTER_KERNEL(0xFF1A00, TComputeNeedWeightsKernel);
    REGISTER_KERNEL(0xFF1A01, TComputeWeightsWithTargetsKernel);
    REGISTER_KERNEL(0xFF1A02, TComputeWeightedQuantileWithBinarySearchKernel);
    REGISTER_KERNEL(0xFF1A03, TMakeEndOfBinsFlagsKernel);
}
