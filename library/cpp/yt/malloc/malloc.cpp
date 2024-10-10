#include "malloc.h"

#include <util/system/compiler.h>
#include <util/system/platform.h>

#include <stdlib.h>

////////////////////////////////////////////////////////////////////////////////

extern "C" Y_WEAK size_t nallocx(size_t size, int /*flags*/) noexcept
{
    return size;
}

#if defined(__ANDROID__)
extern "C" Y_WEAK size_t malloc_usable_size(const void* /*ptr*/)
{
    return 0;
}
#elif !defined(_win_)
extern "C" Y_WEAK size_t malloc_usable_size(void* /*ptr*/) noexcept
{
    return 0;
}
#endif

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
