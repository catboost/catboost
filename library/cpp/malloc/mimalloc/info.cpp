#include <library/cpp/malloc/api/malloc.h>
#include <contrib/libs/mimalloc/include/mimalloc.h>

using namespace NMalloc;

TMallocInfo NMalloc::MallocInfo() {
    TMallocInfo r;
    r.Name = "mimalloc";
    return r;
}

void NMalloc::ClearCaches() {
    mi_collect(true);
}
