#include "cow_string.h"

#include <util/charset/wide.h>
#include <util/stream/input.h>
#include <util/string/cast.h>

constexpr size_t MAX_UTF8_BYTES = 4; // UTF-8-encoded code point takes between 1 and 4 bytes

template <typename TCharType>
static void WriteString(IOutputStream& o, const TCharType* w, size_t n) {
    const size_t buflen = (n * MAX_UTF8_BYTES); // * 4 because the conversion functions can convert unicode character into maximum 4 bytes of UTF8
    TTempBuf buffer(buflen + 1);
    size_t written = 0;
    WideToUTF8(w, n, buffer.Data(), written);
    o.Write(buffer.Data(), written);
}

template <>
void Out<TCowString>(IOutputStream& o, const TCowString& p) {
    o.Write(p.data(), p.size());
}

template <>
void Out<TUtf16CowString>(IOutputStream& o, const TUtf16CowString& w) {
    WriteString(o, w.c_str(), w.size());
}

template <>
void Out<TUtf32CowString>(IOutputStream& o, const TUtf32CowString& w) {
    WriteString(o, w.c_str(), w.size());
}

template <>
void Out<TBasicCharRef<TCowString>>(IOutputStream& o, const TBasicCharRef<TCowString>& c) {
    o << static_cast<char>(c);
}

template <>
void Out<TBasicCharRef<TUtf16CowString>>(IOutputStream& o, const TBasicCharRef<TUtf16CowString>& c) {
    o << static_cast<wchar16>(c);
}

template <>
void Out<TBasicCharRef<TUtf32CowString>>(IOutputStream& o, const TBasicCharRef<TUtf32CowString>& c) {
    o << static_cast<wchar32>(c);
}
