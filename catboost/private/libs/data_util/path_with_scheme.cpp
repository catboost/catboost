#include "path_with_scheme.h"

#include <util/stream/output.h>
#include <util/string/cast.h>

using namespace NCB;


template <>
TPathWithScheme FromStringImpl<TPathWithScheme>(const char* data, size_t len) {
    return TPathWithScheme(TStringBuf(data, len));
}


template <>
void Out<TPathWithScheme>(IOutputStream& o, const TPathWithScheme& pathWithScheme) {
    if (!pathWithScheme.Scheme.empty()) {
        o << pathWithScheme.Scheme << "://";
    }
    o << pathWithScheme.Path;
}
