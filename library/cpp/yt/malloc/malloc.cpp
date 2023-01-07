#include "malloc.h"

#include <util/system/platform.h>

#include <stdlib.h>

////////////////////////////////////////////////////////////////////////////////

void* aligned_malloc(size_t size, size_t alignment)
{
#if defined(_win_)
    return _aligned_malloc(size, alignment);
#elif defined(_darwin_) || defined(_linux_)
    void* ptr = nullptr;
    ::posix_memalign(&ptr, alignment, size);
    return ptr;
#else
#   error Unsupported platform
#endif
}

////////////////////////////////////////////////////////////////////////////////
