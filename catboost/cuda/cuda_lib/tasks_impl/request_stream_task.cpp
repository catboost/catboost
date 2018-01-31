#include "request_stream_task.h"
#include <catboost/cuda/cuda_lib/future/mpi_promise_future.h>

namespace NCudaLib {
#if defined(USE_MPI)
    using TStreamPromise = TMpiPromise<ui32>;

    REGISTER_TASK_TEMPLATE(0x000024, TRequestStreamCommand, TStreamPromise);
    REGISTER_TASK(0x000025, TFreeStreamCommand);
#endif
}
