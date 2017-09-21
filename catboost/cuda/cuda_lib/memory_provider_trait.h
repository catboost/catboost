#pragma once

#include "cuda_base.h"
#include "gpu_memory_pool.h"

namespace NCudaLib {
    template <EPtrType Type>
    class TMemoryProviderImplTrait {
    public:
//TODO: maybe make it template abd cmd-line arg
#if defined(CB_DEBUG_MODE_WITH_CUDA_MALLOC)
        using TMemoryProvider = TCudaMallocWrapper<Type>;
#else
        using TMemoryProvider = TStackLikeMemoryPool<Type>;
#endif

        using TRawFreeMemory = typename TMemoryProvider::template TMemoryBlock<char>;
    };

    template <>
    class TMemoryProviderImplTrait<Host> {
    public:
        using TMemoryProvider = void;

        using TRawFreeMemory = char*;
    };
}
