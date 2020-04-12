#include <library/cpp/malloc/api/malloc.h>

#if defined NALF_DEFINE_GLOBALS && !defined NALF_DONOT_DEFINE_GLOBALS && !defined NALF_FORCE_MALLOC_FREE
NMalloc::TMallocInfo NMalloc::MallocInfo() {
    TMallocInfo r;
    r.Name = "nalf";
    return r;
}
#endif
