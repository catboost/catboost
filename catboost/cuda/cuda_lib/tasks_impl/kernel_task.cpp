#include "kernel_task.h"

namespace NCudaLib {
#if defined(USE_MPI)
    REGISTER_KERNEL(0x000009, TSyncStreamKernel);
#endif
}
