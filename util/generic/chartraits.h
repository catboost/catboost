#pragma once

#include "mem_copy.h"
#include "ptr.h"
#include "utility.h"

#include <contrib/libs/libc_compat/string.h>

#include <util/charset/unidata.h>
#include <util/system/yassert.h>
#include <util/system/platform.h>

#include <cctype>
#include <cstring>
#include <string>

template <class TCharType>
class TCharTraits: public std::char_traits<TCharType> {
public:
    static size_t GetLength(const TCharType* s, size_t maxlen) {
        Y_ASSERT(s);
        const TCharType zero(0);
        size_t i = 0;
        while (i < maxlen && s[i] != zero)
            ++i;
        return i;
    }
};

template <>
class TCharTraits<char>: public std::char_traits<char> {
public:
    static size_t GetLength(const char* s, size_t maxlen) {
        Y_ASSERT(s);
        return strnlen(s, maxlen);
    }
};
