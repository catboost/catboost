#include "helpers.h"

using namespace NKernelHost;

namespace NCudaLib {
    REGISTER_KERNEL_TEMPLATE(0xFFFF00, TDumpPtrs, float);
    REGISTER_KERNEL_TEMPLATE(0xFFFF01, TDumpPtrs, ui32);
    REGISTER_KERNEL_TEMPLATE(0xFFFF02, TDumpPtrs, int);
}
