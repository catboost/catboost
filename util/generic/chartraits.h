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

Y_PURE_FUNCTION
const char* FastFindFirstOf(const char* s, size_t len, const char* set, size_t setlen);

Y_PURE_FUNCTION
const char* FastFindFirstNotOf(const char* s, size_t len, const char* set, size_t setlen);

template <class TCharType>
class TCharTraits: public std::char_traits<TCharType> {
    using TThis = TCharTraits<TCharType>;
public:
    static size_t GetLength(const TCharType* s, size_t maxlen) {
        Y_ASSERT(s);
        const TCharType zero(0);
        size_t i = 0;
        while (i < maxlen && s[i] != zero)
            ++i;
        return i;
    }

    static int Compare(TCharType a, TCharType b) {
        return a == b ? 0 : (a < b ? -1 : +1);
    }

    static int Compare(const TCharType* s1, const TCharType* s2) {
        int res;
        while (true) {
            res = Compare(*s1, *s2++);
            if (res != 0 || !*s1++)
                break;
        }
        return res;
    }
    static int Compare(const TCharType* s1, const TCharType* s2, size_t n) {
        int res = 0;
        for (size_t i = 0; i < n; ++i) {
            res = Compare(s1[i], s2[i]);
            if (res != 0)
                break;
        }
        return res;
    }

    static int Compare(const TCharType* s1, size_t n1, const TCharType* s2, size_t n2) {
        const size_t n = n1 < n2 ? n1 : n2;
        const int result = Compare(s1, s2, n);
        return result ? result : (n1 < n2 ? -1 : (n1 > n2 ? 1 : 0));
    }

    static const TCharType* Find(const TCharType* s, TCharType c) {
        for (;;) {
            if (TThis::eq(*s, c))
                return s;
            if (!*(s++))
                break;
        }
        return nullptr;
    }
    static const TCharType* Find(const TCharType* s, TCharType c, size_t n) {
        for (; n > 0; ++s, --n) {
            if (TThis::eq(*s, c))
                return s;
        }
        return nullptr;
    }
    static const TCharType* Find(const TCharType* s1, const TCharType* s2) {
        size_t n2 = TCharTraits::length(s2);
        if (!n2)
            return s1;
        size_t n1 = TCharTraits::length(s1);
        return Find(s1, n1, s2, n2);
    }
    static const TCharType* Find(const TCharType* s1, size_t l1, const TCharType* s2, size_t l2) {
        if (!l2)
            return s1;
        while (l1 >= l2) {
            --l1;
            if (!Compare(s1, s2, l2))
                return s1;
            ++s1;
        }
        return nullptr;
    }
    static const TCharType* RFind(const TCharType* s, TCharType c) {
        return RFind(s, c, TCharTraits::length(s));
    }
    static const TCharType* RFind(const TCharType* s, TCharType c, size_t n) {
        if (!n)
            return nullptr;
        for (const TCharType* p = s + n - 1; p >= s; --p) {
            if (TThis::eq(*p, c))
                return p;
        }
        return nullptr;
    }
    static const TCharType* RFind(const TCharType* str1, size_t len1, const TCharType* str2, size_t len2, size_t pos) {
        if (len2 > len1)
            return nullptr;
        for (pos = Min(pos, len1 - len2); pos != (size_t)-1; --pos) {
            if (Compare(str1 + pos, str2, len2) == 0)
                return str1 + pos;
        }
        return nullptr;
    }
    static size_t FindFirstOf(const TCharType* s, const TCharType* set) {
        size_t n = 0;
        for (; *s; ++s, ++n) {
            if (Find(set, *s))
                break;
        }
        return n;
    }
    static const TCharType* FindFirstOf(const TCharType* s, size_t l1, const TCharType* set, size_t l2) {
        return FindBySet<true>(s, l1, set, l2);
    }
    static size_t FindFirstNotOf(const TCharType* s, const TCharType* set) {
        size_t n = 0;
        for (; *s != 0; ++s, ++n) {
            if (!Find(set, *s))
                break;
        }
        return n;
    }
    static const TCharType* FindFirstNotOf(const TCharType* s, size_t l1, const TCharType* set, size_t l2) {
        return FindBySet<false>(s, l1, set, l2);
    }

private:
    template <bool shouldBeInSet>
    static const TCharType* FindBySet(const TCharType* s, size_t l1, const TCharType* set, size_t l2) {
        for (; l1 > 0; ++s, --l1) {
            const bool found = Find(set, *s, l2) != nullptr;
            if (found == shouldBeInSet) {
                return s;
            }
        }
        return nullptr;
    }

public:
    static TCharType ToLower(TCharType c) {
        return ::ToLower((wchar32)c);
    }
};

template <>
class TCharTraits<char>: public std::char_traits<char> {
    using TCharType = char;
    using TThis = TCharTraits<char>;

public:
    static int Compare(TCharType a, TCharType b) {
        return a == b ? 0 : (a < b ? -1 : +1);
    }

    static int Compare(const TCharType* s1, size_t n1, const TCharType* s2, size_t n2) {
        const size_t n = n1 < n2 ? n1 : n2;
        const int result = Compare(s1, s2, n);
        return result ? result : (n1 < n2 ? -1 : (n1 > n2 ? 1 : 0));
    }

    static const TCharType* RFind(const TCharType* s, TCharType c, size_t n) {
        if (!n)
            return nullptr;
        for (const TCharType* p = s + n - 1; p >= s; --p) {
            if (TThis::eq(*p, c))
                return p;
        }
        return nullptr;
    }
    static const TCharType* RFind(const TCharType* str1, size_t len1, const TCharType* str2, size_t len2, size_t pos) {
        if (len2 > len1)
            return nullptr;
        for (pos = Min(pos, len1 - len2); pos != (size_t)-1; --pos) {
            if (Compare(str1 + pos, str2, len2) == 0)
                return str1 + pos;
        }
        return nullptr;
    }

private:
    template <bool shouldBeInSet>
    static const TCharType* FindBySet(const TCharType* s, size_t l1, const TCharType* set, size_t l2) {
        for (; l1 > 0; ++s, --l1) {
            const bool found = Find(set, *s, l2) != nullptr;
            if (found == shouldBeInSet) {
                return s;
            }
        }
        return nullptr;
    }

public:
    // Overriden methods

    static size_t GetLength(const char* s, size_t maxlen) {
        Y_ASSERT(s);
        return strnlen(s, maxlen);
    }
    static int Compare(const char* s1, const char* s2) {
        return strcmp(s1, s2);
    }
    static int Compare(const char* s1, const char* s2, size_t n) {
        return n ? memcmp(s1, s2, n) : 0;
    }

    static const char* Find(const char* s, char c) {
        return strchr(s, c);
    }
    static const char* Find(const char* s, char c, size_t n) {
        return (const char*)memchr(s, c, n);
    }
    static const char* Find(const char* s1, const char* s2) {
        return strstr(s1, s2);
    }
    static const char* Find(const char* s1, size_t l1, const char* s2, size_t l2) {
#if defined(_win_)
        if (!l2)
            return s1;
        while (l1 >= l2) {
            --l1;
            if (!Compare(s1, s2, l2))
                return s1;
            ++s1;
        }
        return nullptr;
#else
#if defined(_darwin_)
        if (l2 == 0) {
            return s1;
        }
#endif // _darwin
        return (const char*)memmem(s1, l1, s2, l2);
#endif // !_win_
    }

    static const char* RFind(const char* s, char c) {
        return strrchr(s, c);
    }
    static size_t FindFirstOf(const char* s, const char* set) {
        return strcspn(s, set);
    }
    static const char* FindFirstOf(const char* s, size_t l1, const char* set, size_t l2) {
        return EndToZero(FastFindFirstOf(s, l1, set, l2), s + l1);
    }
    static size_t FindFirstNotOf(const char* s, const char* set) {
        return strspn(s, set);
    }
    static const char* FindFirstNotOf(const char* s, size_t l1, const char* set, size_t l2) {
        return EndToZero(FastFindFirstNotOf(s, l1, set, l2), s + l1);
    }

private:
    static const char* EndToZero(const char* r, const char* e) {
        return r != e ? r : nullptr;
    }

public:
    static char ToLower(char c) {
        return (char)tolower((ui8)c);
    }
};

template <>
class TCharTraits<wchar16>: public std::char_traits<wchar16> {
    using TCharType = wchar16;
    using TThis = TCharTraits<wchar16>;

public:
    static size_t GetLength(const TCharType* s, size_t maxlen) {
        Y_ASSERT(s);
        const TCharType zero(0);
        size_t i = 0;
        while (i < maxlen && s[i] != zero)
            ++i;
        return i;
    }

    static int Compare(TCharType a, TCharType b) {
        return a == b ? 0 : (a < b ? -1 : +1);
    }

    static int Compare(const TCharType* s1, const TCharType* s2) {
        int res;
        while (true) {
            res = Compare(*s1, *s2++);
            if (res != 0 || !*s1++)
                break;
        }
        return res;
    }
    static int Compare(const TCharType* s1, const TCharType* s2, size_t n) {
        int res = 0;
        for (size_t i = 0; i < n; ++i) {
            res = Compare(s1[i], s2[i]);
            if (res != 0)
                break;
        }
        return res;
    }

    static int Compare(const TCharType* s1, size_t n1, const TCharType* s2, size_t n2) {
        const size_t n = n1 < n2 ? n1 : n2;
        const int result = Compare(s1, s2, n);
        return result ? result : (n1 < n2 ? -1 : (n1 > n2 ? 1 : 0));
    }

    static const TCharType* Find(const TCharType* s, TCharType c) {
        for (;;) {
            if (TThis::eq(*s, c))
                return s;
            if (!*(s++))
                break;
        }
        return nullptr;
    }
    static const TCharType* Find(const TCharType* s, TCharType c, size_t n) {
        for (; n > 0; ++s, --n) {
            if (TThis::eq(*s, c))
                return s;
        }
        return nullptr;
    }
    static const TCharType* Find(const TCharType* s1, const TCharType* s2) {
        size_t n2 = TCharTraits::length(s2);
        if (!n2)
            return s1;
        size_t n1 = TCharTraits::length(s1);
        return Find(s1, n1, s2, n2);
    }
    static const TCharType* Find(const TCharType* s1, size_t l1, const TCharType* s2, size_t l2) {
        if (!l2)
            return s1;
        while (l1 >= l2) {
            --l1;
            if (!Compare(s1, s2, l2))
                return s1;
            ++s1;
        }
        return nullptr;
    }
    static const TCharType* RFind(const TCharType* s, TCharType c) {
        return RFind(s, c, TCharTraits::length(s));
    }
    static const TCharType* RFind(const TCharType* s, TCharType c, size_t n) {
        if (!n)
            return nullptr;
        for (const TCharType* p = s + n - 1; p >= s; --p) {
            if (TThis::eq(*p, c))
                return p;
        }
        return nullptr;
    }
    static const TCharType* RFind(const TCharType* str1, size_t len1, const TCharType* str2, size_t len2, size_t pos) {
        if (len2 > len1)
            return nullptr;
        for (pos = Min(pos, len1 - len2); pos != (size_t)-1; --pos) {
            if (Compare(str1 + pos, str2, len2) == 0)
                return str1 + pos;
        }
        return nullptr;
    }
    static size_t FindFirstOf(const TCharType* s, const TCharType* set) {
        size_t n = 0;
        for (; *s; ++s, ++n) {
            if (Find(set, *s))
                break;
        }
        return n;
    }
    static const TCharType* FindFirstOf(const TCharType* s, size_t l1, const TCharType* set, size_t l2) {
        return FindBySet<true>(s, l1, set, l2);
    }
    static size_t FindFirstNotOf(const TCharType* s, const TCharType* set) {
        size_t n = 0;
        for (; *s != 0; ++s, ++n) {
            if (!Find(set, *s))
                break;
        }
        return n;
    }
    static const TCharType* FindFirstNotOf(const TCharType* s, size_t l1, const TCharType* set, size_t l2) {
        return FindBySet<false>(s, l1, set, l2);
    }

private:
    template <bool shouldBeInSet>
    static const TCharType* FindBySet(const TCharType* s, size_t l1, const TCharType* set, size_t l2) {
        for (; l1 > 0; ++s, --l1) {
            const bool found = Find(set, *s, l2) != nullptr;
            if (found == shouldBeInSet) {
                return s;
            }
        }
        return nullptr;
    }

public:
    static TCharType ToLower(TCharType c) {
        return (TCharType)::ToLower((wchar32)c);
    }
};

template <>
class TCharTraits<wchar32>: public std::char_traits<wchar32> {
    using TCharType = wchar32;
    using TThis = TCharTraits<wchar32>;

public:
    static size_t GetLength(const TCharType* s, size_t maxlen) {
        Y_ASSERT(s);
        const TCharType zero(0);
        size_t i = 0;
        while (i < maxlen && s[i] != zero)
            ++i;
        return i;
    }

    static int Compare(TCharType a, TCharType b) {
        return a == b ? 0 : (a < b ? -1 : +1);
    }

    static int Compare(const TCharType* s1, const TCharType* s2) {
        int res;
        while (true) {
            res = Compare(*s1, *s2++);
            if (res != 0 || !*s1++)
                break;
        }
        return res;
    }
    static int Compare(const TCharType* s1, const TCharType* s2, size_t n) {
        int res = 0;
        for (size_t i = 0; i < n; ++i) {
            res = Compare(s1[i], s2[i]);
            if (res != 0)
                break;
        }
        return res;
    }

    static int Compare(const TCharType* s1, size_t n1, const TCharType* s2, size_t n2) {
        const size_t n = n1 < n2 ? n1 : n2;
        const int result = Compare(s1, s2, n);
        return result ? result : (n1 < n2 ? -1 : (n1 > n2 ? 1 : 0));
    }

    static const TCharType* Find(const TCharType* s, TCharType c) {
        for (;;) {
            if (TThis::eq(*s, c))
                return s;
            if (!*(s++))
                break;
        }
        return nullptr;
    }
    static const TCharType* Find(const TCharType* s, TCharType c, size_t n) {
        for (; n > 0; ++s, --n) {
            if (TThis::eq(*s, c))
                return s;
        }
        return nullptr;
    }
    static const TCharType* Find(const TCharType* s1, const TCharType* s2) {
        size_t n2 = TCharTraits::length(s2);
        if (!n2)
            return s1;
        size_t n1 = TCharTraits::length(s1);
        return Find(s1, n1, s2, n2);
    }
    static const TCharType* Find(const TCharType* s1, size_t l1, const TCharType* s2, size_t l2) {
        if (!l2)
            return s1;
        while (l1 >= l2) {
            --l1;
            if (!Compare(s1, s2, l2))
                return s1;
            ++s1;
        }
        return nullptr;
    }
    static const TCharType* RFind(const TCharType* s, TCharType c) {
        return RFind(s, c, TCharTraits::length(s));
    }
    static const TCharType* RFind(const TCharType* s, TCharType c, size_t n) {
        if (!n)
            return nullptr;
        for (const TCharType* p = s + n - 1; p >= s; --p) {
            if (TThis::eq(*p, c))
                return p;
        }
        return nullptr;
    }
    static const TCharType* RFind(const TCharType* str1, size_t len1, const TCharType* str2, size_t len2, size_t pos) {
        if (len2 > len1)
            return nullptr;
        for (pos = Min(pos, len1 - len2); pos != (size_t)-1; --pos) {
            if (Compare(str1 + pos, str2, len2) == 0)
                return str1 + pos;
        }
        return nullptr;
    }
    static size_t FindFirstOf(const TCharType* s, const TCharType* set) {
        size_t n = 0;
        for (; *s; ++s, ++n) {
            if (Find(set, *s))
                break;
        }
        return n;
    }
    static const TCharType* FindFirstOf(const TCharType* s, size_t l1, const TCharType* set, size_t l2) {
        return FindBySet<true>(s, l1, set, l2);
    }
    static size_t FindFirstNotOf(const TCharType* s, const TCharType* set) {
        size_t n = 0;
        for (; *s != 0; ++s, ++n) {
            if (!Find(set, *s))
                break;
        }
        return n;
    }
    static const TCharType* FindFirstNotOf(const TCharType* s, size_t l1, const TCharType* set, size_t l2) {
        return FindBySet<false>(s, l1, set, l2);
    }

private:
    template <bool shouldBeInSet>
    static const TCharType* FindBySet(const TCharType* s, size_t l1, const TCharType* set, size_t l2) {
        for (; l1 > 0; ++s, --l1) {
            const bool found = Find(set, *s, l2) != nullptr;
            if (found == shouldBeInSet) {
                return s;
            }
        }
        return nullptr;
    }

public:
    static TCharType ToLower(TCharType c) {
        return ::ToLower((wchar32)c);
    }
};
