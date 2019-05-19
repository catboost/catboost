#include "ctr_kernels.h"

using namespace NKernelHost;

namespace NCudaLib {
    REGISTER_KERNEL(0xAEAA01, TUpdateBordersMaskKernel);
    REGISTER_KERNEL(0xAEAA02, TMergeBitsKernel);
    REGISTER_KERNEL(0xAEAA03, TExtractBorderMasksKernel);
    REGISTER_KERNEL(0xAEAA04, TFillBinarizedTargetsStatsKernel);
    REGISTER_KERNEL(0xAEAA05, TMakeMeanKernel);
    REGISTER_KERNEL(0xAEAA06, TMakeMeanAndScatterKernel);
    REGISTER_KERNEL(0xAEAA07, TComputeWeightedBinFreqCtrKernel);
    REGISTER_KERNEL(0xAEAA08, TComputeNonWeightedBinFreqCtrKernel);
    REGISTER_KERNEL(0xAEAA09, TGatherTrivialWeightsKernel);
    REGISTER_KERNEL(0xAEAA10, TWriteMaskKernel);

    REGISTER_KERNEL(0xAEAA11, TFixGroupwiseCtrKernel);
    REGISTER_KERNEL(0xAEAA12, TMakeGroupStartFlagsKernel);
    REGISTER_KERNEL(0xAEAA13, TFillBinIndicesKernel);
    REGISTER_KERNEL(0xAEAA14, TCreateFixedIndicesKernel);
}
