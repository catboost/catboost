#include "helpers.h"

using namespace NKernelHost;

namespace NCudaLib {
    REGISTER_KERNEL_TEMPLATE(0xFFFF00, TDumpPtrs, float);
    REGISTER_KERNEL_TEMPLATE(0xFFFF01, TDumpPtrs, ui32);
    REGISTER_KERNEL_TEMPLATE(0xFFFF02, TDumpPtrs, int);
    REGISTER_KERNEL_TEMPLATE_2(0xFFFF03, TTailKernel, ui32, NCudaLib::EPtrType::CudaHost);
    REGISTER_KERNEL_TEMPLATE_2(0xFFFF04, TTailKernel, ui32, NCudaLib::EPtrType::CudaDevice);
}
