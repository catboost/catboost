#include "subst.h"

#include <util/generic/strbuf.h>
#include <util/generic/string.h>
#include <util/system/compiler.h>

#include <string>
#include <type_traits>

// a bit of template magic (to be fast and unreadable)
template <class TStringType, class TTo, bool Main>
static Y_FORCE_INLINE void MoveBlock(typename TStringType::value_type* ptr, size_t& srcPos, size_t& dstPos, const size_t off, const TTo to, const size_t toSize) {
    const size_t unchangedSize = off - srcPos;
    if (dstPos < srcPos) {
        for (size_t i = 0; i < unchangedSize; ++i) {
            ptr[dstPos++] = ptr[srcPos++];
        }
    } else {
        dstPos += unchangedSize;
        srcPos += unchangedSize;
    }

    if (Main) {
        for (size_t i = 0; i < toSize; ++i) {
            ptr[dstPos++] = to[i];
        }
    }
}

template <typename T, typename U>
static bool IsIntersect(const T& a, const U& b) noexcept {
    if (b.data() < a.data()) {
        return IsIntersect(b, a);
    }

    return !a.empty() && !b.empty() &&
           ((a.data() <= b.data() && b.data() < a.data() + a.size()) ||
            (a.data() < b.data() + b.size() && b.data() + b.size() <= a.data() + a.size()));
}

/**
 * Replaces all occurences of substring @c from in string @c s to string @c to.
 * Uses two separate implementations (inplace for shrink and append for grow case)
 * See IGNIETFERRO-394
 **/
template <class TStringType, typename TStringViewType = TBasicStringBuf<typename TStringType::value_type>>
static inline size_t SubstGlobalImpl(TStringType& s, const TStringViewType from, const TStringViewType to, size_t fromPos = 0) {
    if (from.empty()) {
        return 0;
    }

    Y_ASSERT(!IsIntersect(s, from));
    Y_ASSERT(!IsIntersect(s, to));

    const size_t fromSize = from.size();
    const size_t toSize = to.size();
    size_t replacementsCount = 0;
    size_t off = fromPos;
    size_t srcPos = 0;

    if (toSize > fromSize) {
        // string will grow: append to another string
        TStringType result;
        for (; (off = TStringViewType(s).find(from, off)) != TStringType::npos; off += fromSize) {
            if (!replacementsCount) {
                // first replacement occured, we can prepare result string
                result.reserve(s.size() + s.size() / 3);
            }
            result.append(s.begin() + srcPos, s.begin() + off);
            result.append(to.data(), to.size());
            srcPos = off + fromSize;
            ++replacementsCount;
        }
        if (replacementsCount) {
            // append tail
            result.append(s.begin() + srcPos, s.end());
            s = std::move(result);
        }
        return replacementsCount;
    }

    // string will not grow: use inplace algo
    size_t dstPos = 0;
    typename TStringType::value_type* ptr = &*s.begin();
    for (; (off = TStringViewType(s).find(from, off)) != TStringType::npos; off += fromSize) {
        Y_ASSERT(dstPos <= srcPos);
        MoveBlock<TStringType, TStringViewType, true>(ptr, srcPos, dstPos, off, to, toSize);
        srcPos = off + fromSize;
        ++replacementsCount;
    }

    if (replacementsCount) {
        // append tail
        MoveBlock<TStringType, TStringViewType, false>(ptr, srcPos, dstPos, s.size(), to, toSize);
        s.resize(dstPos);
    }
    return replacementsCount;
}

/// Replaces all occurences of the 'from' symbol in a string to the 'to' symbol.
template <class TStringType>
inline size_t SubstCharGlobalImpl(TStringType& s, typename TStringType::value_type from, typename TStringType::value_type to, size_t fromPos = 0) {
    if (fromPos >= s.size()) {
        return 0;
    }

    size_t result = 0;
    fromPos = s.find(from, fromPos);

    // s.begin() might cause memory copying, so call it only if needed
    if (fromPos != TStringType::npos) {
        auto* it = &*s.begin() + fromPos;
        *it = to;
        ++result;
        // at this point string is copied and it's safe to use constant s.end() to iterate
        const auto* const sEnd = &*s.end();
        // unrolled loop goes first because it is more likely that `it` will be properly aligned
        for (const auto* const end = sEnd - (sEnd - it) % 4; it < end;) {
            if (*it == from) {
                *it = to;
                ++result;
            }
            ++it;
            if (*it == from) {
                *it = to;
                ++result;
            }
            ++it;
            if (*it == from) {
                *it = to;
                ++result;
            }
            ++it;
            if (*it == from) {
                *it = to;
                ++result;
            }
            ++it;
        }
        for (; it < sEnd; ++it) {
            if (*it == from) {
                *it = to;
                ++result;
            }
        }
    }

    return result;
}

/* Standard says that `char16_t` is a distinct type and has same size, signedness and alignment as
 * `std::uint_least16_t`, so we check if `char16_t` has same signedness and size as `wchar16` to be
 * sure that we can make safe casts between values of these types and pointers.
 */
static_assert(sizeof(wchar16) == sizeof(char16_t), "");
static_assert(sizeof(wchar32) == sizeof(char32_t), "");
static_assert(std::is_unsigned<wchar16>::value == std::is_unsigned<char16_t>::value, "");
static_assert(std::is_unsigned<wchar32>::value == std::is_unsigned<char32_t>::value, "");

size_t SubstGlobal(TString& text, const TStringBuf what, const TStringBuf with, size_t from) {
    return SubstGlobalImpl(text, what, with, from);
}

size_t SubstGlobal(std::string& text, const TStringBuf what, const TStringBuf with, size_t from) {
    return SubstGlobalImpl(text, what, with, from);
}

size_t SubstGlobal(TUtf16String& text, const TWtringBuf what, const TWtringBuf with, size_t from) {
    return SubstGlobalImpl(text, what, with, from);
}

size_t SubstGlobal(TUtf32String& text, const TUtf32StringBuf what, const TUtf32StringBuf with, size_t from) {
    return SubstGlobalImpl(text, what, with, from);
}

size_t SubstGlobal(std::u16string& text, const TWtringBuf what, const TWtringBuf with, size_t from) {
    return SubstGlobalImpl(text,
                           std::u16string_view(reinterpret_cast<const char16_t*>(what.data()), what.size()),
                           std::u16string_view(reinterpret_cast<const char16_t*>(with.data()), with.size()),
                           from);
}

size_t SubstGlobal(TString& text, char what, char with, size_t from) {
    return SubstCharGlobalImpl(text, what, with, from);
}

size_t SubstGlobal(std::string& text, char what, char with, size_t from) {
    return SubstCharGlobalImpl(text, what, with, from);
}

size_t SubstGlobal(TUtf16String& text, wchar16 what, wchar16 with, size_t from) {
    return SubstCharGlobalImpl(text, (char16_t)what, (char16_t)with, from);
}

size_t SubstGlobal(std::u16string& text, wchar16 what, wchar16 with, size_t from) {
    return SubstCharGlobalImpl(text, (char16_t)what, (char16_t)with, from);
}

size_t SubstGlobal(TUtf32String& text, wchar32 what, wchar32 with, size_t from) {
    return SubstCharGlobalImpl(text, (char32_t)what, (char32_t)with, from);
}
