#include "enable_peers.h"
#include "kernel_task.h"

namespace NCudaLib {
#if defined(USE_MPI)
    REGISTER_KERNEL(0x000004, NKernelHost::TEnablePeersKernel);
    REGISTER_KERNEL(0x000005, NKernelHost::TDisablePeersKernel);
#endif
}
