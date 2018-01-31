#include "memory_state_func.h"

namespace NCudaLib {
#if defined(USE_MPI)
    REGISTER_CPU_FUNC(0x000006, TMemoryStateFunc);
#endif
}
