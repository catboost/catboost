#include "host_tasks.h"
#include "cpu_func.h"
#include <catboost/cuda/cuda_lib/serialization/task_factory.h>

namespace NCudaLib {
#if defined(USE_MPI)
    REGISTER_TASK(0x000000, TWaitSubmitCommand);
    REGISTER_CPU_FUNC(0x000001, TBlockingSyncDevice);
    REGISTER_CPU_FUNC(0x000002, TRequestHandlesTask);
    REGISTER_CPU_FUNC(0x000003, TFreeHandlesTask);
#endif
}
