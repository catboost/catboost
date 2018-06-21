#include "stack_like_memory_pool.h"

namespace NCudaLib {
    template class TStackLikeMemoryPool<EPtrType::CudaDevice>;
    template class TStackLikeMemoryPool<EPtrType::CudaHost>;
}
