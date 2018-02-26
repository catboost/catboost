#include "add_bin_values.h"

using namespace NKernelHost;

namespace NCudaLib {
    REGISTER_KERNEL(0x101300, TAddBinModelValueKernel);
    REGISTER_KERNEL(0x101301, TAddObliviousTreeKernel);
    REGISTER_KERNEL(0x101302, TComputeObliviousTreeLeaveIndicesKernel);
}
