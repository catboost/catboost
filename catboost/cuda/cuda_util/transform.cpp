#include <catboost/cuda/cuda_util/transform.h>

using namespace NKernelHost;

namespace NCudaLib {
    REGISTER_KERNEL_TEMPLATE(0x110001, TBinOpKernel, float);
    REGISTER_KERNEL_TEMPLATE(0x110002, TBinOpKernel, int);
    REGISTER_KERNEL_TEMPLATE(0x110003, TBinOpKernel, ui32);
    REGISTER_KERNEL_TEMPLATE(0x110004, TBinOpKernel, double);
    REGISTER_KERNEL_TEMPLATE(0x110005, TBinOpKernel, ui8);
    REGISTER_KERNEL_TEMPLATE(0x110006, TBinOpKernel, uint2);
    REGISTER_KERNEL_TEMPLATE(0x110007, TBinOpKernel, ui16);

    REGISTER_KERNEL_TEMPLATE(0x110008, TApplyFuncKernel, float);

    REGISTER_KERNEL_TEMPLATE_2(0x110009, TMapCopyKernel, float, ui32);
    REGISTER_KERNEL_TEMPLATE_2(0x110010, TMapCopyKernel, int, ui32);
    REGISTER_KERNEL_TEMPLATE_2(0x110011, TMapCopyKernel, ui8, ui32);
    REGISTER_KERNEL_TEMPLATE_2(0x110012, TMapCopyKernel, ui32, ui32);
}
