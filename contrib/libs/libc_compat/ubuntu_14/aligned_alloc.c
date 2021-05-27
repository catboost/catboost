#include <malloc.h>
#include <stdlib.h>

__attribute__((weak)) void* aligned_alloc(size_t alignment, size_t size) {
    return memalign(alignment, size);
}
