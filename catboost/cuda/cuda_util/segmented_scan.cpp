#include "segmented_scan.h"

using namespace NKernelHost;

namespace NCudaLib {
    REGISTER_KERNEL_TEMPLATE(0xAC0001, TSegmentedScanKernel, float);
    REGISTER_KERNEL_TEMPLATE(0xAC0002, TSegmentedScanKernel, int);
    REGISTER_KERNEL_TEMPLATE(0xAC0003, TSegmentedScanKernel, ui32);
}
