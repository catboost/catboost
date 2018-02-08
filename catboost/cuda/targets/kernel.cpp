#include "kernel.h"

using namespace NKernelHost;

namespace NCudaLib {
    REGISTER_KERNEL(0xA112200, TCrossEntropyTargetKernel);
    REGISTER_KERNEL(0xA112201, TMseTargetKernel);
    REGISTER_KERNEL(0xA112202, TPointwiseTargetImplKernel);
    REGISTER_KERNEL(0xA112203, TQueryRmseKernel);
    REGISTER_KERNEL(0xA112204, TYetiRankKernel);
    REGISTER_KERNEL(0xA112205, TQuerySoftMaxKernel);
}
