#pragma once

#include <util/str_stl.h>

template <>
struct hash<TCowString>: ::NHashPrivate::TStringHash<char> {
};

template <>
struct hash<TUtf16CowString>: ::NHashPrivate::TStringHash<wchar16> {
};

template <>
struct hash<TUtf32CowString>: ::NHashPrivate::TStringHash<wchar32> {
};

template <>
struct TEqualTo<TCowString>: public TEqualTo<TStringBuf> {
    using is_transparent = void;
};

template <>
struct TEqualTo<TUtf16CowString>: public TEqualTo<TWtringBuf> {
    using is_transparent = void;
};

template <>
struct TEqualTo<TUtf32CowString>: public TEqualTo<TUtf32StringBuf> {
    using is_transparent = void;
};

template <>
struct TCIEqualTo<TCowString> {
    inline bool operator()(const TCowString& a, const TCowString& b) const {
        return a.size() == b.size() && strnicmp(a.data(), b.data(), a.size()) == 0;
    }
};

template <>
struct TLess<TCowString>: public TLess<TStringBuf> {
    using is_transparent = void;
};

template <>
struct TLess<TUtf16CowString>: public TLess<TWtringBuf> {
    using is_transparent = void;
};

template <>
struct TLess<TUtf32CowString>: public TLess<TUtf32StringBuf> {
    using is_transparent = void;
};

template <>
struct TGreater<TCowString>: public TGreater<TStringBuf> {
    using is_transparent = void;
};

template <>
struct TGreater<TUtf16CowString>: public TGreater<TWtringBuf> {
    using is_transparent = void;
};

template <>
struct TGreater<TUtf32CowString>: public TGreater<TUtf32StringBuf> {
    using is_transparent = void;
};
