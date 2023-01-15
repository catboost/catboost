#pragma once

#include <util/generic/vector.h>
#include <util/generic/string.h>
#include <util/generic/strbuf.h>

template <class T>
struct TCompactTrieKeySelector {
    typedef TVector<T> TKey;
    typedef TVector<T> TKeyBuf;
};

template <class TChar>
struct TCompactTrieCharKeySelector {
    typedef TBasicString<TChar> TKey;
    typedef TBasicStringBuf<TChar> TKeyBuf;
};

template <>
struct TCompactTrieKeySelector<char>: public TCompactTrieCharKeySelector<char> {
};

template <>
struct TCompactTrieKeySelector<wchar16>: public TCompactTrieCharKeySelector<wchar16> {
};

template <>
struct TCompactTrieKeySelector<wchar32>: public TCompactTrieCharKeySelector<wchar32> {
};
