#include <library/cpp/malloc/api/malloc.h>
#include <contrib/libs/tcmalloc/tcmalloc/internal_malloc_extension.h>

using namespace NMalloc;

TMallocInfo NMalloc::MallocInfo() {
    TMallocInfo r;
    r.Name = "tcmalloc";
    return r;
}

void NMalloc::ClearCaches() {
    // not available on darwin, see internal_malloc_extension.h for details
#ifndef _darwin_
    MallocExtension_Internal_ReleaseMemoryToSystem(std::numeric_limits<size_t>::max());
#endif
}
