#include <malloc.h>
#include <stdlib.h>

void* aligned_alloc(size_t alignment, size_t size) {
    return memalign(alignment, size);
}
