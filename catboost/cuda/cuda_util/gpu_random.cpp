#include "gpu_random.h"

using namespace NKernelHost;

namespace NCudaLib {
    REGISTER_KERNEL(0xADD000, TPoissonKernel)
    REGISTER_KERNEL(0xADD001, TUniformRandKernel)
    REGISTER_KERNEL(0xADD002, TGaussianRandKernel)
    REGISTER_KERNEL(0xADD003, TGenerateSeeds)
}
