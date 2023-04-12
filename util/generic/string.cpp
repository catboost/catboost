#include "string.h"

#include <util/string/ascii.h>
#include <util/system/sanitizers.h>
#include <util/system/sys_alloc.h>
#include <util/charset/wide.h>

#include <iostream>

alignas(32) const char NULL_STRING_REPR[128] = {0};

std::ostream& operator<<(std::ostream& os, const TString& s) {
    return os.write(s.data(), s.size());
}

std::istream& operator>>(std::istream& is, TString& s) {
    return is >> s.MutRef();
}

template <>
bool TBasicString<char, std::char_traits<char>>::to_lower(size_t pos, size_t n) {
    return Transform([](size_t, char c) { return AsciiToLower(c); }, pos, n);
}

template <>
bool TBasicString<char, std::char_traits<char>>::to_upper(size_t pos, size_t n) {
    return Transform([](size_t, char c) { return AsciiToUpper(c); }, pos, n);
}

template <>
bool TBasicString<char, std::char_traits<char>>::to_title(size_t pos, size_t n) {
    if (n == 0) {
        return false;
    }
    bool changed = to_upper(pos, 1);
    return to_lower(pos + 1, n - 1) || changed;
}

template <>
TUtf16String&
TBasicString<wchar16, std::char_traits<wchar16>>::AppendAscii(const ::TStringBuf& s) {
    ReserveAndResize(size() + s.size());

    auto dst = begin() + size() - s.size();

    for (const char* src = s.data(); dst != end(); ++dst, ++src) {
        *dst = static_cast<wchar16>(*src);
    }

    return *this;
}

template <>
TUtf16String&
TBasicString<wchar16, std::char_traits<wchar16>>::AppendUtf8(const ::TStringBuf& s) {
    size_t oldSize = size();
    ReserveAndResize(size() + s.size() * 4);
    size_t written = 0;
    size_t pos = UTF8ToWideImpl(s.data(), s.size(), begin() + oldSize, written);
    if (pos != s.size()) {
        ythrow yexception() << "failed to decode UTF-8 string at pos " << pos << ::NDetail::InStringMsg(s.data(), s.size());
    }
    resize(oldSize + written);

    return *this;
}

template <>
bool TBasicString<wchar16, std::char_traits<wchar16>>::to_lower(size_t pos, size_t n) {
    return ToLower(*this, pos, n);
}

template <>
bool TBasicString<wchar16, std::char_traits<wchar16>>::to_upper(size_t pos, size_t n) {
    return ToUpper(*this, pos, n);
}

template <>
bool TBasicString<wchar16, std::char_traits<wchar16>>::to_title(size_t pos, size_t n) {
    return ToTitle(*this, pos, n);
}

template <>
TUtf32String&
TBasicString<wchar32, std::char_traits<wchar32>>::AppendAscii(const ::TStringBuf& s) {
    ReserveAndResize(size() + s.size());

    auto dst = begin() + size() - s.size();

    for (const char* src = s.data(); dst != end(); ++dst, ++src) {
        *dst = static_cast<wchar32>(*src);
    }

    return *this;
}

template <>
TBasicString<char, std::char_traits<char>>&
TBasicString<char, std::char_traits<char>>::AppendUtf16(const ::TWtringBuf& s) {
    const size_t oldSize = size();
    ReserveAndResize(size() + WideToUTF8BufferSize(s.size()));

    size_t written = 0;
    WideToUTF8(s.data(), s.size(), begin() + oldSize, written);

    resize(oldSize + written);

    return *this;
}

template <>
TUtf32String&
TBasicString<wchar32, std::char_traits<wchar32>>::AppendUtf8(const ::TStringBuf& s) {
    size_t oldSize = size();
    ReserveAndResize(size() + s.size() * 4);
    size_t written = 0;
    size_t pos = UTF8ToWideImpl(s.data(), s.size(), begin() + oldSize, written);
    if (pos != s.size()) {
        ythrow yexception() << "failed to decode UTF-8 string at pos " << pos << ::NDetail::InStringMsg(s.data(), s.size());
    }
    resize(oldSize + written);

    return *this;
}

template <>
TUtf32String&
TBasicString<wchar32, std::char_traits<wchar32>>::AppendUtf16(const ::TWtringBuf& s) {
    size_t oldSize = size();
    ReserveAndResize(size() + s.size() * 2);

    wchar32* oldEnd = begin() + oldSize;
    wchar32* end = oldEnd;
    NDetail::UTF16ToUTF32ImplScalar(s.data(), s.data() + s.size(), end);
    size_t written = end - oldEnd;

    resize(oldSize + written);

    return *this;
}

template <>
bool TBasicString<wchar32, std::char_traits<wchar32>>::to_lower(size_t pos, size_t n) {
    return ToLower(*this, pos, n);
}

template <>
bool TBasicString<wchar32, std::char_traits<wchar32>>::to_upper(size_t pos, size_t n) {
    return ToUpper(*this, pos, n);
}

template <>
bool TBasicString<wchar32, std::char_traits<wchar32>>::to_title(size_t pos, size_t n) {
    return ToTitle(*this, pos, n);
}
