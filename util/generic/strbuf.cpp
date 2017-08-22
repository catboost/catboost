#include "strbuf.h"

#include <util/stream/output.h>

template <>
void Out<TStringBuf>(IOutputStream& os, const TStringBuf& obj) {
    os.Write(obj.data(), obj.length());
}

template <>
void Out<TWtringBuf>(IOutputStream& os, const TWtringBuf& obj) {
    os << static_cast<const TFixedString<wchar16>&>(obj);
}
