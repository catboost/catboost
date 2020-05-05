#include <library/cpp/malloc/api/malloc.h>

#include <util/stream/output.h>

using namespace NMalloc;

template <>
void Out<TMallocInfo>(IOutputStream& out, const TMallocInfo& info) {
    out << "malloc (name = " << info.Name << ")";
}
