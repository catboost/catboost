#include "bootstrap.h"

using namespace NKernelHost;

namespace NCudaLib {
    REGISTER_KERNEL(0x0AAA00, TPoissonBootstrapKernel)
    REGISTER_KERNEL(0x0AAA01, TUniformBootstrapKernel)
    REGISTER_KERNEL(0x0AAA02, TBayesianBootstrapKernel)

}
