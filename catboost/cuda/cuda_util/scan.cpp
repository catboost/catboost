#include "scan.h"

using namespace NKernelHost;

namespace NCudaLib {
    REGISTER_KERNEL_TEMPLATE(0xAD0001, TScanVectorKernel, float);
    REGISTER_KERNEL_TEMPLATE(0xAD0002, TScanVectorKernel, double);
    REGISTER_KERNEL_TEMPLATE(0xAD0003, TScanVectorKernel, ui32);
    REGISTER_KERNEL_TEMPLATE(0xAD0004, TScanVectorKernel, int);

    REGISTER_KERNEL_TEMPLATE(0xAD0005, TNonNegativeSegmentedScanAndScatterVectorKernel, float);

}
