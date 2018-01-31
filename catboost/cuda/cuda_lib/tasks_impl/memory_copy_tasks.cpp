#include "memory_copy_tasks.h"

namespace NCudaLib {
#if defined(USE_MPI)
    REGISTER_TASK(0x000100, TMasterInterHostMemcpy);
    REGISTER_STREAM_SECTION_TASK(0x000101, TMemoryCopyTasks);
#endif
}
