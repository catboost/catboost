#include "fill.h"

using namespace NKernelHost;

namespace NCudaLib {
    REGISTER_KERNEL_TEMPLATE(0x001000, TFillBufferKernel, float);
    REGISTER_KERNEL_TEMPLATE(0x001001, TFillBufferKernel, char);
    REGISTER_KERNEL_TEMPLATE(0x001002, TFillBufferKernel, int);
    REGISTER_KERNEL_TEMPLATE(0x001003, TFillBufferKernel, ui32);
    REGISTER_KERNEL_TEMPLATE(0x001005, TFillBufferKernel, ui64);

    REGISTER_KERNEL_TEMPLATE(0x001006, TMakeSequenceKernel, int);
    REGISTER_KERNEL_TEMPLATE(0x001007, TMakeSequenceKernel, ui32);
    REGISTER_KERNEL_TEMPLATE(0x001010, TMakeSequenceKernel, ui64);

    REGISTER_KERNEL_TEMPLATE(0x001008, TInversePermutationKernel, ui32);
    REGISTER_KERNEL_TEMPLATE(0x001009, TInversePermutationKernel, int);

}
