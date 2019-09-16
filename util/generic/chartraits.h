#pragma once

#include "utility.h"
#include "mem_copy.h"
#include "ptr.h"

#include <util/charset/wide_specific.h>
#include <util/system/compat.h>
#include <util/system/yassert.h>
#include <util/system/platform.h>

#include <cstring>

// Building blocks of TCharTraits:
//
// *** GetLength

template <typename TCharType>
struct TLengthCharTraits {
    static size_t GetLength(const TCharType* s) {
        Y_ASSERT(s);
        const TCharType* sc = s;
        for (; *sc != 0; ++sc) {
        }
        return sc - s;
    }
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
struct TLengthCharTraits<char> {
    static size_t GetLength(const char* s) {
        Y_ASSERT(s);
        return strlen(s);
    }
    static size_t GetLength(const char* s, size_t maxlen) {
        Y_ASSERT(s);
        return strnlen(s, maxlen);
    }
};

// *** Compare/Equal

namespace NCharTraitsImpl {
    template <typename TCharType>
    struct TSingleCharBase {
        static int Compare(TCharType a, TCharType b) {
            return a == b ? 0 : (a < b ? -1 : +1);
        }

        static bool Equal(TCharType a, TCharType b) {
            return a == b;
        }
    };

    template <typename TCharType, typename TSingle>
    struct TCompareBase {
        static int Compare(const TCharType* s1, const TCharType* s2) {
            int res;
            while (true) {
                res = TSingle::Compare(*s1, *s2++);
                if (res != 0 || !*s1++)
                    break;
            }
            return res;
        }
        static int Compare(const TCharType* s1, const TCharType* s2, size_t n) {
            int res = 0;
            for (size_t i = 0; i < n; ++i) {
                res = TSingle::Compare(s1[i], s2[i]);
                if (res != 0)
                    break;
            }
            return res;
        }
    };

    // OS accelerated specialization for char
    template <>
    struct TCompareBase<char, TSingleCharBase<char>> {
        static int Compare(const char* s1, const char* s2) {
            return strcmp(s1, s2);
        }
        static int Compare(const char* s1, const char* s2, size_t n) {
            return n ? memcmp(s1, s2, n) : 0;
        }
    };

    template <typename TCharType, typename TCompare>
    struct TEqualBase {
        static bool Equal(const TCharType* s1, const TCharType* s2) {
            return TCompare::Compare(s1, s2) == 0;
        }
        static bool Equal(const TCharType* s1, const TCharType* s2, size_t n) {
            return TCompare::Compare(s1, s2, n) == 0;
        }
    };

    // OS accelerated specialization for wchar16
    using TCmpWchar16 = TCompareBase<wchar16, TSingleCharBase<wchar16>>;

    template <>
    struct TEqualBase<wchar16, TCmpWchar16> {
        static bool Equal(const wchar16* s1, const wchar16* s2) {
            return TCmpWchar16::Compare(s1, s2) == 0;
        }
        static bool Equal(const wchar16* s1, const wchar16* s2, size_t n) {
            return (n == 0) || (memcmp(s1, s2, n * sizeof(wchar16)) == 0);
        }
    };

    // OS accelerated specialization for wchar32
    using TCmpWchar32 = TCompareBase<wchar32, TSingleCharBase<wchar32>>;

    template <>
    struct TEqualBase<wchar32, TCmpWchar32> {
        static bool Equal(const wchar32* s1, const wchar32* s2) {
            return TCmpWchar32::Compare(s1, s2) == 0;
        }
        static bool Equal(const wchar32* s1, const wchar32* s2, size_t n) {
            return (n == 0) || (memcmp(s1, s2, n * sizeof(wchar32)) == 0);
        }
    };
}

template <typename TCharType,
          typename TSingleCharBase = NCharTraitsImpl::TSingleCharBase<TCharType>,
          typename TCompareBase = NCharTraitsImpl::TCompareBase<TCharType, TSingleCharBase>,
          typename TEqualBase = NCharTraitsImpl::TEqualBase<TCharType, TCompareBase>>
struct TCompareCharTraits: public TSingleCharBase, public TCompareBase, public TEqualBase {
    using TCompareBase::Compare;
    using TEqualBase::Equal;
    using TSingleCharBase::Compare;
    using TSingleCharBase::Equal;

    static int Compare(const TCharType* s1, size_t n1, const TCharType* s2, size_t n2) {
        const size_t n = n1 < n2 ? n1 : n2;
        const int result = TCompareBase::Compare(s1, s2, n);
        return result ? result : (n1 < n2 ? -1 : (n1 > n2 ? 1 : 0));
    }

    static bool Equal(const TCharType* s1, size_t n1, const TCharType* s2, size_t n2) {
        return n1 == n2 && TEqualBase::Equal(s1, s2, n1);
    }

    static bool Equal(const TCharType* s1, size_t n1, const TCharType* s2) {
        const TCharType* end = s1 + n1;

        for (; s1 != end; ++s1, ++s2) {
            if (*s2 == 0 || !TSingleCharBase::Equal(*s1, *s2)) {
                return false;
            }
        }

        return *s2 == 0;
    }
};

// *** Find/RFind/etc

namespace NCharTraitsImpl {
    template <typename TCharType, typename TLength, typename TCompare>
    class TFind {
    public:
        static const TCharType* Find(const TCharType* s, TCharType c) {
            for (;;) {
                if (TCompare::Equal(*s, c))
                    return s;
                if (!*(s++))
                    break;
            }
            return nullptr;
        }
        static const TCharType* Find(const TCharType* s, TCharType c, size_t n) {
            for (; n > 0; ++s, --n) {
                if (TCompare::Equal(*s, c))
                    return s;
            }
            return nullptr;
        }
        static const TCharType* Find(const TCharType* s1, const TCharType* s2) {
            size_t n2 = TLength::GetLength(s2);
            if (!n2)
                return s1;
            size_t n1 = TLength::GetLength(s1);
            return Find(s1, n1, s2, n2);
        }
        static const TCharType* Find(const TCharType* s1, size_t l1, const TCharType* s2, size_t l2) {
            if (!l2)
                return s1;
            while (l1 >= l2) {
                --l1;
                if (!TCompare::Compare(s1, s2, l2))
                    return s1;
                ++s1;
            }
            return nullptr;
        }
        static const TCharType* RFind(const TCharType* s, TCharType c) {
            return RFind(s, c, TLength::GetLength(s));
        }
        static const TCharType* RFind(const TCharType* s, TCharType c, size_t n) {
            if (!n)
                return nullptr;
            for (const TCharType* p = s + n - 1; p >= s; --p) {
                if (TCompare::Equal(*p, c))
                    return p;
            }
            return nullptr;
        }
        static const TCharType* RFind(const TCharType* str1, size_t len1, const TCharType* str2, size_t len2, size_t pos) {
            if (len2 > len1)
                return nullptr;
            for (pos = Min(pos, len1 - len2); pos != (size_t)-1; --pos) {
                if (TCompare::Compare(str1 + pos, str2, len2) == 0)
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
    };
}

template <typename TCharType, typename TLength = TLengthCharTraits<TCharType>, typename TCompare = TCompareCharTraits<TCharType>>
class TFindCharTraits: public NCharTraitsImpl::TFind<TCharType, TLength, TCompare> {
};

Y_PURE_FUNCTION
const char* FastFindFirstOf(const char* s, size_t len, const char* set, size_t setlen);

Y_PURE_FUNCTION
const char* FastFindFirstNotOf(const char* s, size_t len, const char* set, size_t setlen);

// OS accelerated specialization
template <>
class TFindCharTraits<char, TLengthCharTraits<char>, TCompareCharTraits<char>>: public NCharTraitsImpl::TFind<char, TLengthCharTraits<char>, TCompareCharTraits<char>> {
public:
    using TBase = NCharTraitsImpl::TFind<char, TLengthCharTraits<char>, TCompareCharTraits<char>>;

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
        return TBase::Find(s1, l1, s2, l2);
#else
#if defined(_darwin_)
        if(l2 == 0) {
            return s1;
        }
#endif // _darwin
        return (const char*)memmem(s1, l1, s2, l2);
#endif // !_win_
    }
    using TBase::RFind;
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
};

// *** Hashing

template <typename TCharType>
struct THashCharTraits {
    static size_t GetHash(const TCharType* s, size_t n) noexcept {
// reduce code bloat and cycled includes, declare functions here
#if defined(_64_) && !defined(NO_CITYHASH)
        extern ui64 CityHash64(const char* buf, size_t len) noexcept;
        return CityHash64((const char*)s, n * sizeof(TCharType));
#else
        extern size_t MurmurHashSizeT(const char* buf, size_t len) noexcept;
        return MurmurHashSizeT((const char*)s, n * sizeof(TCharType));
#endif
    }
};

// *** Case

// defined in unidata.h
wchar32 ToLower(wchar32 ch);

template <typename TCharType>
struct TCaseCharTraits {
    static TCharType ToLower(TCharType c) {
        return ::ToLower((wchar32)c);
    }
};

template <>
struct TCaseCharTraits<char> {
    static char ToLower(char c) {
        return (char)tolower((ui8)c);
    }
};

// *** Mutation

namespace NCharTraitsImpl {
    template <typename TCharType>
    struct TMutable {
        static TCharType* Move(TCharType* s1, const TCharType* s2, size_t n) {
            return ::MemMove(s1, s2, n);
        }
        static TCharType* Copy(TCharType* s1, const TCharType* s2, size_t n) {
            return ::MemCopy(s1, s2, n);
        }
        static TCharType* Assign(TCharType* s, size_t n, TCharType c) {
            for (TCharType* ptr = s; ptr < s + n; ++ptr) // see REVIEW:52714 for details
                *ptr = c;
            return s;
        }
        static void Reverse(TCharType* s, size_t n) {
            TCharType* f = s;
            TCharType* l = s + n - 1;
            while (f < l) {
                DoSwap(*f++, *l--);
            }
        }
    };
}

template <typename TCharType>
struct TMutableCharTraits: public NCharTraitsImpl::TMutable<TCharType> {
};

// OS accelerated specialization
template <>
struct TMutableCharTraits<char>: public NCharTraitsImpl::TMutable<char> {
    static char* Assign(char* s, size_t n, char c) {
        memset(s, c, n);
        return s;
    }
};

template <>
struct TMutableCharTraits<wchar16>: public NCharTraitsImpl::TMutable<wchar16> {
    static void Reverse(wchar16* start, size_t len);
};

template <class TCharType,
          class TLength = TLengthCharTraits<TCharType>,
          class TCompare = TCompareCharTraits<TCharType>,
          class TFind = TFindCharTraits<TCharType, TLength, TCompare>,
          class THash = THashCharTraits<TCharType>,
          class TCase = TCaseCharTraits<TCharType>,
          class TMutable = TMutableCharTraits<TCharType>>
class TCharTraitsImpl: public TLength, public TCompare, public TFind, public THash, public TCase, public TMutable {
};

template <typename TCharType>
class TCharTraits: public TCharTraitsImpl<TCharType> {
};

template <class T>
class TCharTraits<const T>: public TCharTraits<T> {
};
