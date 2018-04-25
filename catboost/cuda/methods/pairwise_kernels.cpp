#include "pairwise_kernels.h"

using namespace NKernelHost;

namespace NCudaLib {
    REGISTER_KERNEL(0x421201, TMakeLinearSystemKernel);
    REGISTER_KERNEL(0x421202, TUpdateBinsPairsKernel);
    REGISTER_KERNEL(0x421203, TExtractMatricesAndTargetsKernel);
    REGISTER_KERNEL(0x421204, TRegularizeKernel);
    REGISTER_KERNEL(0x421205, TCholeskySolverKernel);
    REGISTER_KERNEL(0x421206, TCalcScoresKernel);
    REGISTER_KERNEL(0x421207, TCopyReducedTempResultKernel);
    REGISTER_KERNEL(0x421208, TSelectBestSplitKernel);
    REGISTER_KERNEL(0x421209, TComputePairwiseHistogramKernel);
    REGISTER_KERNEL(0x421210, TZeroMeanKernel);


}
