#pragma once

// Some of these includes are just a legacy from previous implementation.
// We don't need them here, but removing them is tricky because it breaks all
// kinds of builds downstream
#include "mem_copy.h"
#include "ptr.h"
#include "utility.h"

#include <util/charset/unidata.h>
#include <util/system/platform.h>
#include <util/system/yassert.h>

#include <contrib/libs/libc_compat/string.h>

#include <cctype>
#include <cstring>
#include <string>
#include <string_view>

namespace NStringPrivate {
    template <class TCharType>
    size_t GetStringLengthWithLimit(const TCharType* s, size_t maxlen) {
        Y_ASSERT(s);
        size_t i = 0;
        for (; i != maxlen && s[i]; ++i)
            ;
        return i;
    }

    inline size_t GetStringLengthWithLimit(const char* s, size_t maxlen) {
        Y_ASSERT(s);
        return strnlen(s, maxlen);
    }
}

template <typename TDerived, typename TCharType, typename TTraitsType = std::char_traits<TCharType>>
class TStringBase {
    using TStringView = std::basic_string_view<TCharType>;
    using TStringViewWithTraits = std::basic_string_view<TCharType, TTraitsType>;

public:
    using TChar = TCharType;
    using TTraits = TTraitsType;
    using TSelf = TStringBase<TDerived, TChar, TTraits>;

    using size_type = size_t;
    using difference_type = ptrdiff_t;
    static constexpr size_t npos = size_t(-1);

    using const_iterator = const TCharType*;
    using const_reference = const TCharType&;

    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    static constexpr size_t StrLen(const TCharType* s) noexcept {
        if (Y_LIKELY(s)) {
            return TTraits::length(s);
        }
        return 0;
    }

    template <class TCharTraits>
    inline constexpr operator std::basic_string_view<TCharType, TCharTraits>() const {
        return std::basic_string_view<TCharType, TCharTraits>(data(), size());
    }

    template <class TCharTraits, class Allocator>
    inline explicit operator std::basic_string<TCharType, TCharTraits, Allocator>() const {
        return std::basic_string<TCharType, TCharTraits, Allocator>(Ptr(), Len());
    }

    /**
     * @param                           Pointer to character inside the string, or nullptr.
     * @return                          Offset from string beginning (in chars), or npos on nullptr.
     */
    inline size_t off(const TCharType* ret) const noexcept {
        return ret ? (size_t)(ret - Ptr()) : npos;
    }

    inline size_t IterOff(const_iterator it) const noexcept {
        return begin() <= it && end() > it ? size_t(it - begin()) : npos;
    }

    constexpr const_iterator begin() const noexcept {
        return Ptr();
    }

    constexpr const_iterator end() const noexcept {
        return Ptr() + size();
    }

    constexpr const_iterator cbegin() const noexcept {
        return begin();
    }

    constexpr const_iterator cend() const noexcept {
        return end();
    }

    constexpr const_reverse_iterator rbegin() const noexcept {
        return const_reverse_iterator(Ptr() + size());
    }

    constexpr const_reverse_iterator rend() const noexcept {
        return const_reverse_iterator(Ptr());
    }

    constexpr const_reverse_iterator crbegin() const noexcept {
        return rbegin();
    }

    constexpr const_reverse_iterator crend() const noexcept {
        return rend();
    }

    inline TCharType back() const noexcept {
        Y_ASSERT(!this->empty());
        return Ptr()[Len() - 1];
    }

    inline TCharType front() const noexcept {
        Y_ASSERT(!empty());
        return Ptr()[0];
    }

    constexpr const TCharType* data() const noexcept {
        return Ptr();
    }

    constexpr inline size_t size() const noexcept {
        return Len();
    }

    constexpr inline bool is_null() const noexcept {
        return *Ptr() == 0;
    }

    Y_PURE_FUNCTION constexpr inline bool empty() const noexcept {
        return Len() == 0;
    }

    constexpr inline explicit operator bool() const noexcept {
        return !empty();
    }

public: // style-guide compliant methods
    constexpr const TCharType* Data() const noexcept {
        return Ptr();
    }

    constexpr size_t Size() const noexcept {
        return Len();
    }

    Y_PURE_FUNCTION constexpr bool Empty() const noexcept {
        return 0 == Len();
    }

private:
    static constexpr TStringView LegacySubString(const TStringView view, size_t p, size_t n) noexcept {
        p = Min(p, view.length());
        return view.substr(p, n);
    }

public:
    // ~~~ Comparison ~~~ : FAMILY0(int, compare)
    static constexpr int compare(const TSelf& s1, const TSelf& s2) noexcept {
        return s1.AsStringView().compare(s2.AsStringView());
    }

    static constexpr int compare(const TCharType* p, const TSelf& s2) noexcept {
        TCharType null{0};
        return TStringViewWithTraits(p ? p : &null).compare(s2.AsStringView());
    }

    static constexpr int compare(const TSelf& s1, const TCharType* p) noexcept {
        TCharType null{0};
        return s1.AsStringView().compare(p ? p : &null);
    }

    static constexpr int compare(const TStringView s1, const TStringView s2) noexcept {
        return TStringViewWithTraits(s1.data(), s1.size()).compare(TStringViewWithTraits(s2.data(), s2.size()));
    }

    template <class T>
    constexpr int compare(const T& t) const noexcept {
        return compare(*this, t);
    }

    constexpr int compare(size_t p, size_t n, const TStringView t) const noexcept {
        return compare(LegacySubString(*this, p, n), t);
    }

    constexpr int compare(size_t p, size_t n, const TStringView t, size_t p1, size_t n1) const noexcept {
        return compare(LegacySubString(*this, p, n), LegacySubString(t, p1, n1));
    }

    constexpr int compare(size_t p, size_t n, const TStringView t, size_t n1) const noexcept {
        return compare(LegacySubString(*this, p, n), LegacySubString(t, 0, n1));
    }

    constexpr int compare(const TCharType* p, size_t len) const noexcept {
        return compare(*this, TStringView(p, len));
    }

    static constexpr bool equal(const TSelf& s1, const TSelf& s2) noexcept {
        return s1.AsStringView() == s2.AsStringView();
    }

    static constexpr bool equal(const TSelf& s1, const TCharType* p) noexcept {
        if (p == nullptr) {
            return s1.Len() == 0;
        }

        return s1.AsStringView() == p;
    }

    static constexpr bool equal(const TCharType* p, const TSelf& s2) noexcept {
        return equal(s2, p);
    }

    static constexpr bool equal(const TStringView s1, const TStringView s2) noexcept {
        return TStringViewWithTraits{s1.data(), s1.size()} == TStringViewWithTraits{s2.data(), s2.size()};
    }

    template <class T>
    constexpr bool equal(const T& t) const noexcept {
        return equal(*this, t);
    }

    constexpr bool equal(size_t p, size_t n, const TStringView t) const noexcept {
        return equal(LegacySubString(*this, p, n), t);
    }

    constexpr bool equal(size_t p, size_t n, const TStringView t, size_t p1, size_t n1) const noexcept {
        return equal(LegacySubString(*this, p, n), LegacySubString(t, p1, n1));
    }

    constexpr bool equal(size_t p, size_t n, const TStringView t, size_t n1) const noexcept {
        return equal(LegacySubString(*this, p, n), LegacySubString(t, 0, n1));
    }

    static constexpr bool StartsWith(const TCharType* what, size_t whatLen, const TCharType* with, size_t withLen) noexcept {
        return withLen <= whatLen && TStringViewWithTraits(what, withLen) == TStringViewWithTraits(with, withLen);
    }

    static constexpr bool EndsWith(const TCharType* what, size_t whatLen, const TCharType* with, size_t withLen) noexcept {
        return withLen <= whatLen && TStringViewWithTraits(what + whatLen - withLen, withLen) == TStringViewWithTraits(with, withLen);
    }

    constexpr bool StartsWith(const TCharType* s, size_t n) const noexcept {
        return StartsWith(Ptr(), Len(), s, n);
    }

    constexpr bool StartsWith(const TStringView s) const noexcept {
        return StartsWith(s.data(), s.length());
    }

    constexpr bool StartsWith(TCharType ch) const noexcept {
        return !empty() && TTraits::eq(*Ptr(), ch);
    }

    constexpr bool EndsWith(const TCharType* s, size_t n) const noexcept {
        return EndsWith(Ptr(), Len(), s, n);
    }

    constexpr bool EndsWith(const TStringView s) const noexcept {
        return EndsWith(s.data(), s.length());
    }

    constexpr bool EndsWith(TCharType ch) const noexcept {
        return !empty() && TTraits::eq(Ptr()[Len() - 1], ch);
    }

    template <typename TDerived2, typename TTraits2>
    constexpr bool operator==(const TStringBase<TDerived2, TChar, TTraits2>& s2) const noexcept {
        return equal(*this, s2);
    }

    constexpr bool operator==(TStringView s2) const noexcept {
        return equal(*this, s2);
    }

    constexpr bool operator==(const TCharType* pc) const noexcept {
        return equal(*this, pc);
    }

#ifndef __cpp_impl_three_way_comparison
    friend constexpr bool operator==(const TCharType* pc, const TSelf& s) noexcept {
        return equal(pc, s);
    }

    template <typename TDerived2, typename TTraits2>
    friend constexpr bool operator!=(const TSelf& s1, const TStringBase<TDerived2, TChar, TTraits2>& s2) noexcept {
        return !(s1 == s2);
    }

    friend constexpr bool operator!=(const TSelf& s1, TStringView s2) noexcept {
        return !(s1 == s2);
    }

    friend constexpr bool operator!=(const TSelf& s, const TCharType* pc) noexcept {
        return !(s == pc);
    }

    friend constexpr bool operator!=(const TCharType* pc, const TSelf& s) noexcept {
        return !(pc == s);
    }
#endif

    template <typename TDerived2, typename TTraits2>
    friend constexpr bool operator<(const TSelf& s1, const TStringBase<TDerived2, TChar, TTraits2>& s2) noexcept {
        return compare(s1, s2) < 0;
    }

    friend constexpr bool operator<(const TSelf& s1, TStringView s2) noexcept {
        return compare(s1, s2) < 0;
    }

    friend constexpr bool operator<(const TSelf& s, const TCharType* pc) noexcept {
        return compare(s, pc) < 0;
    }

    friend constexpr bool operator<(const TCharType* pc, const TSelf& s) noexcept {
        return compare(pc, s) < 0;
    }

    template <typename TDerived2, typename TTraits2>
    friend constexpr bool operator<=(const TSelf& s1, const TStringBase<TDerived2, TChar, TTraits2>& s2) noexcept {
        return compare(s1, s2) <= 0;
    }

    friend constexpr bool operator<=(const TSelf& s1, TStringView s2) noexcept {
        return compare(s1, s2) <= 0;
    }

    friend constexpr bool operator<=(const TSelf& s, const TCharType* pc) noexcept {
        return compare(s, pc) <= 0;
    }

    friend constexpr bool operator<=(const TCharType* pc, const TSelf& s) noexcept {
        return compare(pc, s) <= 0;
    }

    template <typename TDerived2, typename TTraits2>
    friend constexpr bool operator>(const TSelf& s1, const TStringBase<TDerived2, TChar, TTraits2>& s2) noexcept {
        return compare(s1, s2) > 0;
    }

    friend constexpr bool operator>(const TSelf& s1, TStringView s2) noexcept {
        return compare(s1, s2) > 0;
    }

    friend constexpr bool operator>(const TSelf& s, const TCharType* pc) noexcept {
        return compare(s, pc) > 0;
    }

    friend constexpr bool operator>(const TCharType* pc, const TSelf& s) noexcept {
        return compare(pc, s) > 0;
    }

    template <typename TDerived2, typename TTraits2>
    friend constexpr bool operator>=(const TSelf& s1, const TStringBase<TDerived2, TChar, TTraits2>& s2) noexcept {
        return compare(s1, s2) >= 0;
    }

    friend constexpr bool operator>=(const TSelf& s1, TStringView s2) noexcept {
        return compare(s1, s2) >= 0;
    }

    friend constexpr bool operator>=(const TSelf& s, const TCharType* pc) noexcept {
        return compare(s, pc) >= 0;
    }

    friend constexpr bool operator>=(const TCharType* pc, const TSelf& s) noexcept {
        return compare(pc, s) >= 0;
    }

    // ~~ Read access ~~
    inline TCharType at(size_t pos) const noexcept {
        if (Y_LIKELY(pos < Len())) {
            return (Ptr())[pos];
        }
        return 0;
    }

    inline TCharType operator[](size_t pos) const noexcept {
        Y_ASSERT(pos < this->size());

        return Ptr()[pos];
    }

    //~~~~Search~~~~
    /**
     * @return                          Position of the substring inside this string, or `npos` if not found.
     */
    inline size_t find(const TStringView s, size_t pos = 0) const noexcept {
        return find(s.data(), pos, s.size());
    }

    inline size_t find(const TCharType* s, size_t pos, size_t count) const noexcept {
        return AsStringView().find(s, pos, count);
    }

    inline size_t find(TCharType c, size_t pos = 0) const noexcept {
        return AsStringView().find(c, pos);
    }

    inline size_t rfind(TCharType c) const noexcept {
        return AsStringView().rfind(c);
    }

    inline size_t rfind(TCharType c, size_t pos) const noexcept {
        if (pos == 0) {
            return npos;
        }
        return AsStringView().rfind(c, pos - 1);
    }

    inline size_t rfind(const TStringView str, size_t pos = npos) const {
        return AsStringView().rfind(str.data(), pos, str.size());
    }

    //~~~~Contains~~~~
    /**
     * @returns                         Whether this string contains the provided substring.
     */
    inline bool Contains(const TStringView s, size_t pos = 0) const noexcept {
        return !s.length() || find(s, pos) != npos;
    }

    inline bool Contains(TChar c, size_t pos = 0) const noexcept {
        return find(c, pos) != npos;
    }

    inline void Contains(std::enable_if<std::is_unsigned<TCharType>::value, char> c, size_t pos = 0) const noexcept {
        return find(ui8(c), pos) != npos;
    }

    //~~~~Character Set Search~~~
    inline size_t find_first_of(TCharType c) const noexcept {
        return find_first_of(c, 0);
    }

    inline size_t find_first_of(TCharType c, size_t pos) const noexcept {
        return find(c, pos);
    }

    inline size_t find_first_of(const TStringView set) const noexcept {
        return find_first_of(set, 0);
    }

    inline size_t find_first_of(const TStringView set, size_t pos) const noexcept {
        return AsStringView().find_first_of(set.data(), pos, set.size());
    }

    inline size_t find_first_not_of(TCharType c) const noexcept {
        return find_first_not_of(c, 0);
    }

    inline size_t find_first_not_of(TCharType c, size_t pos) const noexcept {
        return find_first_not_of(TStringView(&c, 1), pos);
    }

    inline size_t find_first_not_of(const TStringView set) const noexcept {
        return find_first_not_of(set, 0);
    }

    inline size_t find_first_not_of(const TStringView set, size_t pos) const noexcept {
        return AsStringView().find_first_not_of(set.data(), pos, set.size());
    }

    inline size_t find_last_of(TCharType c, size_t pos = npos) const noexcept {
        return find_last_of(&c, pos, 1);
    }

    inline size_t find_last_of(const TStringView set, size_t pos = npos) const noexcept {
        return find_last_of(set.data(), pos, set.length());
    }

    inline size_t find_last_of(const TCharType* set, size_t pos, size_t n) const noexcept {
        return AsStringView().find_last_of(set, pos, n);
    }

    inline size_t find_last_not_of(TCharType c, size_t pos = npos) const noexcept {
        return AsStringView().find_last_not_of(c, pos);
    }

    inline size_t find_last_not_of(const TStringView set, size_t pos = npos) const noexcept {
        return find_last_not_of(set.data(), pos, set.length());
    }

    inline size_t find_last_not_of(const TCharType* set, size_t pos, size_t n) const noexcept {
        return AsStringView().find_last_not_of(set, pos, n);
    }

    inline size_t copy(TCharType* pc, size_t n, size_t pos) const {
        if (pos > Len()) {
            throw std::out_of_range("TStringBase::copy");
        }

        return CopyImpl(pc, n, pos);
    }

    inline size_t copy(TCharType* pc, size_t n) const noexcept {
        return CopyImpl(pc, n, 0);
    }

    inline size_t strcpy(TCharType* pc, size_t n) const noexcept {
        if (n) {
            n = copy(pc, n - 1);
            pc[n] = 0;
        }

        return n;
    }

    inline TDerived copy() const Y_WARN_UNUSED_RESULT {
        return TDerived(Ptr(), Len());
    }

    // ~~~ Partial copy ~~~~
    TDerived substr(size_t pos, size_t n = npos) const Y_WARN_UNUSED_RESULT {
        return TDerived(*This(), pos, n);
    }

private:
    using GenericFinder = const TCharType* (*)(const TCharType*, size_t, const TCharType*, size_t);

    constexpr TStringViewWithTraits AsStringView() const {
        return static_cast<TStringViewWithTraits>(*this);
    }

    constexpr inline const TCharType* Ptr() const noexcept {
        return This()->data();
    }

    constexpr inline size_t Len() const noexcept {
        return This()->length();
    }

    constexpr inline const TDerived* This() const noexcept {
        return static_cast<const TDerived*>(this);
    }

    inline size_t CopyImpl(TCharType* pc, size_t n, size_t pos) const noexcept {
        const size_t toCopy = Min(Len() - pos, n);

        TTraits::copy(pc, Ptr() + pos, toCopy);

        return toCopy;
    }
};

/**
 * @def Y_STRING_LIFETIME_BOUND
 *
 * The attribute on a string-like function parameter can be used to tell the compiler
 * that function return value may refer that parameter.
 * this macro differs from the Y_LIFETIME_BOUND  in that it does not check
 * the lifetime of copy-on-write strings if that implementation is used.
 */
#if defined(TSTRING_IS_STD_STRING)
    #define Y_STRING_LIFETIME_BOUND Y_LIFETIME_BOUND
#else
    // It is difficult to determine the lifetime of a copy-on-write
    // string using static analysis, as some copies of the string may
    // extend the buffer's lifetime.
    // Therefore, checking the lifetime of such strings has not yet been implemented.
    #define Y_STRING_LIFETIME_BOUND
#endif
