#include "memory_allocation.h"
#include <catboost/cuda/cuda_lib/serialization/task_factory.h>

namespace NCudaLib {
#if defined(USE_MPI)
    REGISTER_TASK_TEMPLATE(0x000011, TCudaMallocTask, EPtrType::CudaHost);
    REGISTER_TASK_TEMPLATE(0x000012, TCudaMallocTask, EPtrType::CudaDevice);
    REGISTER_TASK_TEMPLATE(0x000013, TCudaMallocTask, EPtrType::Host);

    using TCudaHostRawPtr = typename TMemoryProviderImplTrait<EPtrType::CudaHost>::TRawFreeMemory;
    using TCudaDeviceRawPtr = typename TMemoryProviderImplTrait<EPtrType::CudaDevice>::TRawFreeMemory;
    using THostRawPtr = typename TMemoryProviderImplTrait<EPtrType::Host>::TRawFreeMemory;

    REGISTER_TASK_TEMPLATE_2(0x000014, TResetPointerCommand, TCudaHostRawPtr, false);
    REGISTER_TASK_TEMPLATE_2(0x000015, TResetPointerCommand, TCudaDeviceRawPtr, false);
    REGISTER_TASK_TEMPLATE_2(0x000016, TResetPointerCommand, THostRawPtr, false);

    REGISTER_TASK_TEMPLATE_2(0x000017, TResetPointerCommand, TCudaHostRawPtr, true);
    REGISTER_TASK_TEMPLATE_2(0x000018, TResetPointerCommand, TCudaDeviceRawPtr, true);
    REGISTER_TASK_TEMPLATE_2(0x000019, TResetPointerCommand, THostRawPtr, true);

#endif
}
