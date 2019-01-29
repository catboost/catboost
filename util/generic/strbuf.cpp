#include "strbuf.h"

#include <util/stream/output.h>
#include <ostream>

template <>
void Out<TStringBuf>(IOutputStream& os, const TStringBuf& obj) {
    os.Write(obj.data(), obj.length());
}

template <>
void Out<TWtringBuf>(IOutputStream& os, const TWtringBuf& obj) {
    os << static_cast<const TFixedString<wchar16>&>(obj);
}

template <>
void Out<TUtf32StringBuf>(IOutputStream& os, const TUtf32StringBuf& obj) {
    os << static_cast<const TFixedString<wchar32>&>(obj);
}

std::ostream& operator<< (std::ostream& os, TStringBuf buf) {
    os.write(buf.data(), buf.size());
    return os;
}
