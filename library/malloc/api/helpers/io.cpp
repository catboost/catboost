#include <library/malloc/api/malloc.h>

#include <util/stream/output.h>

using namespace NMalloc;

template <>
void Out<TMallocInfo>(TOutputStream& out, const TMallocInfo& info) {
    out << "malloc (name = " << info.Name << ")";
}
