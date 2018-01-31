#include "dot_product.h"

using namespace NKernelHost;

namespace NCudaLib {
    REGISTER_KERNEL_TEMPLATE(0xDD0001, TDotProductKernel, float)
    REGISTER_KERNEL_TEMPLATE(0xDD0002, TDotProductKernel, double)
}
