#include "output.h"

#include <util/stream/output.h>

using namespace NColorizer;

template <>
void Out<TColorHandle>(IOutputStream& o, const TColorHandle& h) {
    o << (*(h.C).*h.F)();
}
