#include "pairwise_kernels.h"


using namespace NKernelHost;

namespace NCudaLib {

    REGISTER_KERNEL(0x421201, TMakeLinearSystemKernel);
    REGISTER_KERNEL(0x421202, TExtractMatricesAndTargetsKernel);
    REGISTER_KERNEL(0x421203, TZeroMeanKernel);
    REGISTER_KERNEL(0x421204, TCholeskySolverKernel);
    REGISTER_KERNEL(0x421205, TCalcScoresKernel);
    REGISTER_KERNEL(0x421206, TCopyReducedTempResultKernel);
    REGISTER_KERNEL(0x421207, TUpdateBinsPairsKernel);
    REGISTER_KERNEL(0x421208, TSelectBestSplitKernel);
    REGISTER_KERNEL(0x421209, TComputePairwiseHistogramKernel);

}
