#include "reduce.h"
using namespace NKernelHost;

namespace NCudaLib {
    REGISTER_KERNEL_TEMPLATE(0xAE0001, TReduceKernel, float)
    REGISTER_KERNEL_TEMPLATE(0xAE0005, TReduceKernel, ui32)
    REGISTER_KERNEL_TEMPLATE(0xAE0006, TReduceKernel, int)
    REGISTER_KERNEL_TEMPLATE_2(0xAE0002, TSegmentedReduceKernel, float, EPtrType::CudaDevice)
    REGISTER_KERNEL_TEMPLATE_2(0xAE0003, TSegmentedReduceKernel, float, EPtrType::CudaHost)
    REGISTER_KERNEL_TEMPLATE_2(0xAE0004, TReduceByKeyKernel, float, ui32)
}
