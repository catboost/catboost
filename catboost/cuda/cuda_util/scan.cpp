#include "scan.h"

using namespace NKernelHost;

namespace NCudaLib {
    REGISTER_KERNEL_TEMPLATE_2(0xAD0001, TScanVectorKernel, float, float);
    REGISTER_KERNEL_TEMPLATE_2(0xAD0002, TScanVectorKernel, double, double);
    REGISTER_KERNEL_TEMPLATE_2(0xAD0003, TScanVectorKernel, ui32, ui32);
    REGISTER_KERNEL_TEMPLATE_2(0xAD0004, TScanVectorKernel, int, int);

    REGISTER_KERNEL_TEMPLATE_2(0xAD0006, TScanVectorKernel, ui32, ui64);

    REGISTER_KERNEL_TEMPLATE(0xAD0005, TNonNegativeSegmentedScanAndScatterVectorKernel, float);

}
