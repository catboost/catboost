#include <cstddef>


////////////////////////////////////////////////////////////////////////////////

#if defined(__ANDROID__)
extern "C" size_t malloc_usable_size(const void* ptr);
#else
extern "C" size_t malloc_usable_size(void* ptr) noexcept;
#endif

extern "C" size_t nallocx(size_t size, int flags) noexcept;

void* aligned_malloc(size_t size, size_t alignment);

////////////////////////////////////////////////////////////////////////////////
