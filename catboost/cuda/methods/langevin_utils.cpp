#include "langevin_utils.h"

using namespace NKernelHost;

namespace NCudaLib {
    REGISTER_KERNEL(0xFF1B00, TAddLangevinNoiseKernel);
}
