#pragma once

#include <string_view>
#include <util/system/yassert.h>

#include "chartraits.h"
#include "utility.h"

template <typename TDerived, typename TCharType, typename TTraits>
class TStringBase;

// temporarily here IGNIETFERRO-1198
template <typename TCharType, typename TTraits = TCharTraits<TCharType>>
struct TFixedString {
    constexpr TFixedString()
        : Start(nullptr)
        , Length(0)
    {
    }

    template <typename T>
    TFixedString(const TStringBase<T, TCharType, TTraits>& s)
        : Start(s.data())
        , Length(s.size())
    {
    }

    template <typename T, typename A>
    TFixedString(const std::basic_string<TCharType, T, A>& s)
        : Start(s.data())
        , Length(s.size())
    {
    }

    TFixedString(const TCharType* s)
        : Start(s)
        , Length(s ? TTraits::GetLength(s) : 0)
    {
    }

    constexpr TFixedString(const TCharType* s, size_t n)
        : Start(s)
        , Length(n)
    {
    }

    constexpr TFixedString(const TCharType* begin, const TCharType* end)
        : Start(begin)
        , Length(end - begin)
    {
    }

    Y_PURE_FUNCTION
    inline TFixedString SubString(size_t pos, size_t n) const noexcept {
        pos = Min(pos, Length);
        n = Min(n, Length - pos);
        return TFixedString(Start + pos, n);
    }

    const TCharType* Start;
    size_t Length;
};

template <typename TDerived, typename TCharType, typename TTraitsType = TCharTraits<TCharType>>
class TStringBase {
public:
    using TChar = TCharType;
    using TTraits = TTraitsType;
    using TSelf = TStringBase<TDerived, TChar, TTraits>;
    using TFixedString = ::TFixedString<TChar, TTraits>;

    using size_type = size_t;
    static constexpr size_t npos = size_t(-1);

    static size_t hashVal(const TCharType* s, size_t n) noexcept {
        return TTraits::GetHash(s, n);
    }

    using const_iterator = const TCharType*;

    template <typename TBase>
    struct TReverseIteratorBase {
        constexpr TReverseIteratorBase() noexcept = default;
        explicit constexpr TReverseIteratorBase(TBase p)
            : P_(p)
        {
        }

        TReverseIteratorBase operator++() noexcept {
            --P_;
            return *this;
        }

        TReverseIteratorBase operator++(int) noexcept {
            TReverseIteratorBase old(*this);
            --P_;
            return old;
        }

        TReverseIteratorBase& operator--() noexcept {
            ++P_;
            return *this;
        }

        TReverseIteratorBase operator--(int) noexcept {
            TReverseIteratorBase old(*this);
            ++P_;
            return old;
        }

        constexpr auto operator*() const noexcept -> std::remove_pointer_t<TBase>& {
            return *TBase(*this);
        }

        explicit constexpr operator TBase() const noexcept {
            return TBase(P_ - 1);
        }

        constexpr auto operator-(const TReverseIteratorBase o) const noexcept {
            return o.P_ - P_;
        }

        constexpr bool operator==(const TReverseIteratorBase o) const noexcept {
            return P_ == o.P_;
        }

        constexpr bool operator!=(const TReverseIteratorBase o) const noexcept {
            return !(*this == o);
        }

    private:
        TBase P_ = nullptr;
    };
    using const_reverse_iterator = TReverseIteratorBase<const_iterator>;

    static inline size_t StrLen(const TCharType* s) noexcept {
        return s ? TTraits::GetLength(s) : 0;
    }

    template <class TCharTraits>
    inline constexpr operator std::basic_string_view<TCharType, TCharTraits>() const {
        return std::basic_string_view<TCharType>(data(), size());
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

    inline const_iterator begin() const noexcept {
        return Ptr();
    }

    inline const_iterator end() const noexcept {
        return Ptr() + size();
    }

    inline const_iterator cbegin() const noexcept {
        return begin();
    }

    inline const_iterator cend() const noexcept {
        return end();
    }

    inline const_reverse_iterator rbegin() const noexcept {
        return const_reverse_iterator(Ptr() + size());
    }

    inline const_reverse_iterator rend() const noexcept {
        return const_reverse_iterator(Ptr());
    }

    inline const_reverse_iterator crbegin() const noexcept {
        return rbegin();
    }

    inline const_reverse_iterator crend() const noexcept {
        return rend();
    }

    inline TCharType back() const noexcept {
        Y_ASSERT(!this->empty());
        return Ptr()[Len() - 1];
    }

    inline const TCharType front() const noexcept {
        Y_ASSERT(!empty());
        return Ptr()[0];
    }

    constexpr const TCharType* data() const noexcept {
        return Ptr();
    }

    constexpr inline size_t size() const noexcept {
        return Len();
    }

    inline size_t hash() const noexcept {
        return hashVal(Ptr(), size());
    }

    constexpr inline bool is_null() const noexcept {
        return *Ptr() == 0;
    }

    Y_PURE_FUNCTION
    constexpr inline bool empty() const noexcept {
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

    Y_PURE_FUNCTION
    constexpr bool Empty() const noexcept {
        return 0 == Len();
    }

public:
    // ~~~ Comparison ~~~ : FAMILY0(int, compare)
    static int compare(const TSelf& s1, const TSelf& s2) noexcept {
        return TTraits::Compare(s1.Ptr(), s1.Len(), s2.Ptr(), s2.Len());
    }

    static int compare(const TCharType* p, const TSelf& s2) noexcept {
        return TTraits::Compare(p, StrLen(p), s2.Ptr(), s2.Len());
    }

    static int compare(const TSelf& s1, const TCharType* p) noexcept {
        return TTraits::Compare(s1.Ptr(), s1.Len(), p, StrLen(p));
    }

    static int compare(const TFixedString s1, const TFixedString s2) noexcept {
        return TTraits::Compare(s1.Start, s1.Length, s2.Start, s2.Length);
    }

    template <class T>
    inline int compare(const T& t) const noexcept {
        return compare(*this, t);
    }

    inline int compare(size_t p, size_t n, const TFixedString t) const noexcept {
        return compare(TFixedString(*this).SubString(p, n), t);
    }

    inline int compare(size_t p, size_t n, const TFixedString t, size_t p1, size_t n1) const noexcept {
        return compare(TFixedString(*this).SubString(p, n), t.SubString(p1, n1));
    }

    inline int compare(size_t p, size_t n, const TFixedString t, size_t n1) const noexcept {
        return compare(TFixedString(*this).SubString(p, n), t.SubString(0, n1));
    }

    inline int compare(const TCharType* p, size_t len) const noexcept {
        return compare(*this, TFixedString(p, len));
    }

    static bool equal(const TSelf& s1, const TSelf& s2) noexcept {
        return TTraits::Equal(s1.Ptr(), s1.Len(), s2.Ptr(), s2.Len());
    }

    static bool equal(const TSelf& s1, const TCharType* p) noexcept {
        if (p == nullptr) {
            return s1.Len() == 0;
        }

        return TTraits::Equal(s1.Ptr(), s1.Len(), p);
    }

    static bool equal(const TCharType* p, const TSelf& s2) noexcept {
        return equal(s2, p);
    }

    static bool equal(const TFixedString s1, const TFixedString s2) noexcept {
        return TTraits::Equal(s1.Start, s1.Length, s2.Start, s2.Length);
    }

    template <class T>
    inline bool equal(const T& t) const noexcept {
        return equal(*this, t);
    }

    inline bool equal(size_t p, size_t n, const TFixedString t) const noexcept {
        return equal(TFixedString(*this).SubString(p, n), t);
    }

    inline bool equal(size_t p, size_t n, const TFixedString t, size_t p1, size_t n1) const noexcept {
        return equal(TFixedString(*this).SubString(p, n), t.SubString(p1, n1));
    }

    inline bool equal(size_t p, size_t n, const TFixedString t, size_t n1) const noexcept {
        return equal(TFixedString(*this).SubString(p, n), t.SubString(0, n1));
    }

    static inline bool StartsWith(const TCharType* what, size_t whatLen, const TCharType* with, size_t withLen) noexcept {
        return withLen <= whatLen && TTraits::Equal(what, withLen, with, withLen);
    }

    static inline bool EndsWith(const TCharType* what, size_t whatLen, const TCharType* with, size_t withLen) noexcept {
        return withLen <= whatLen && TTraits::Equal(what + whatLen - withLen, withLen, with, withLen);
    }

    inline bool StartsWith(const TCharType* s, size_t n) const noexcept {
        return StartsWith(Ptr(), Len(), s, n);
    }

    inline bool StartsWith(const TFixedString s) const noexcept {
        return StartsWith(s.Start, s.Length);
    }

    inline bool StartsWith(TCharType ch) const noexcept {
        return !empty() && TTraits::Equal(*Ptr(), ch);
    }

    inline bool EndsWith(const TCharType* s, size_t n) const noexcept {
        return EndsWith(Ptr(), Len(), s, n);
    }

    inline bool EndsWith(const TFixedString s) const noexcept {
        return EndsWith(s.Start, s.Length);
    }

    inline bool EndsWith(TCharType ch) const noexcept {
        return !empty() && TTraits::Equal(Ptr()[Len() - 1], ch);
    }

    template <typename TDerived2, typename TTraits2>
    friend bool operator==(const TSelf& s1, const TStringBase<TDerived2, TChar, TTraits2>& s2) noexcept {
        return equal(s1, s2);
    }

    friend bool operator==(const TSelf& s, const TCharType* pc) noexcept {
        return equal(s, pc);
    }

    friend bool operator==(const TCharType* pc, const TSelf& s) noexcept {
        return equal(pc, s);
    }

    template <typename TDerived2, typename TTraits2>
    friend bool operator!=(const TSelf& s1, const TStringBase<TDerived2, TChar, TTraits2>& s2) noexcept {
        return !(s1 == s2);
    }

    friend bool operator!=(const TSelf& s, const TCharType* pc) noexcept {
        return !(s == pc);
    }

    friend bool operator!=(const TCharType* pc, const TSelf& s) noexcept {
        return !(pc == s);
    }

    template <typename TDerived2, typename TTraits2>
    friend bool operator<(const TSelf& s1, const TStringBase<TDerived2, TChar, TTraits2>& s2) noexcept {
        return compare(s1, s2) < 0;
    }

    friend bool operator<(const TSelf& s, const TCharType* pc) noexcept {
        return compare(s, pc) < 0;
    }

    friend bool operator<(const TCharType* pc, const TSelf& s) noexcept {
        return compare(pc, s) < 0;
    }

    template <typename TDerived2, typename TTraits2>
    friend bool operator<=(const TSelf& s1, const TStringBase<TDerived2, TChar, TTraits2>& s2) noexcept {
        return compare(s1, s2) <= 0;
    }

    friend bool operator<=(const TSelf& s, const TCharType* pc) noexcept {
        return compare(s, pc) <= 0;
    }

    friend bool operator<=(const TCharType* pc, const TSelf& s) noexcept {
        return compare(pc, s) <= 0;
    }

    template <typename TDerived2, typename TTraits2>
    friend bool operator>(const TSelf& s1, const TStringBase<TDerived2, TChar, TTraits2>& s2) noexcept {
        return compare(s1, s2) > 0;
    }

    friend bool operator>(const TSelf& s, const TCharType* pc) noexcept {
        return compare(s, pc) > 0;
    }

    friend bool operator>(const TCharType* pc, const TSelf& s) noexcept {
        return compare(pc, s) > 0;
    }

    template <typename TDerived2, typename TTraits2>
    friend bool operator>=(const TSelf& s1, const TStringBase<TDerived2, TChar, TTraits2>& s2) noexcept {
        return compare(s1, s2) >= 0;
    }

    friend bool operator>=(const TSelf& s, const TCharType* pc) noexcept {
        return compare(s, pc) >= 0;
    }

    friend bool operator>=(const TCharType* pc, const TSelf& s) noexcept {
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
    inline size_t find(const TFixedString s, size_t pos = 0) const noexcept {
        if (Y_UNLIKELY(!s.Length)) {
            return pos <= Len() ? pos : npos;
        }
        return GenericFind<TTraits::Find>(s.Start, s.Length, pos);
    }

    inline size_t find(TCharType c, size_t pos = 0) const noexcept {
        if (pos >= Len()) {
            return npos;
        }
        return off(TTraits::Find(Ptr() + pos, c, Len() - pos));
    }

    inline size_t rfind(TCharType c) const noexcept {
        return off(TTraits::RFind(Ptr(), c, Len()));
    }

    inline size_t rfind(TCharType c, size_t pos) const noexcept {
        if (pos > Len()) {
            pos = Len();
        }

        return off(TTraits::RFind(Ptr(), c, pos));
    }

    inline size_t rfind(const TFixedString str, size_t pos = npos) const {
        return off(TTraits::RFind(Ptr(), Len(), str.Start, str.Length, pos));
    }

    //~~~~Contains~~~~
    /**
     * @returns                         Whether this string contains the provided substring.
     */
    inline bool Contains(const TFixedString s, size_t pos = 0) const noexcept {
        return !s.Length || find(s, pos) != npos;
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

    inline size_t find_first_of(const TFixedString set) const noexcept {
        return find_first_of(set, 0);
    }

    inline size_t find_first_of(const TFixedString set, size_t pos) const noexcept {
        return GenericFind<TTraits::FindFirstOf>(set.Start, set.Length, pos);
    }

    inline size_t find_first_not_of(TCharType c) const noexcept {
        return find_first_not_of(c, 0);
    }

    inline size_t find_first_not_of(TCharType c, size_t pos) const noexcept {
        return find_first_not_of(TFixedString(&c, 1), pos);
    }

    inline size_t find_first_not_of(const TFixedString set) const noexcept {
        return find_first_not_of(set, 0);
    }

    inline size_t find_first_not_of(const TFixedString set, size_t pos) const noexcept {
        return GenericFind<TTraits::FindFirstNotOf>(set.Start, set.Length, pos);
    }

    inline size_t find_last_of(TCharType c, size_t pos = npos) const noexcept {
        return find_last_of(&c, pos, 1);
    }

    inline size_t find_last_of(const TFixedString set, size_t pos = npos) const noexcept {
        return find_last_of(set.Start, pos, set.Length);
    }

    inline size_t find_last_of(const TCharType* set, size_t pos, size_t n) const noexcept {
        ssize_t startpos = pos >= size() ? static_cast<ssize_t>(size()) - 1 : static_cast<ssize_t>(pos);

        for (ssize_t i = startpos; i >= 0; --i) {
            const TCharType c = Ptr()[i];

            for (const TCharType* p = set; p < set + n; ++p) {
                if (TTraits::Equal(c, *p)) {
                    return static_cast<size_t>(i);
                }
            }
        }

        return npos;
    }

    inline size_t find_last_not_of(TCharType c, size_t pos = npos) const noexcept {
        return find_last_not_of(&c, pos, 1);
    }

    inline size_t find_last_not_of(const TFixedString set, size_t pos = npos) const noexcept {
        return find_last_not_of(set.Start, pos, set.Length);
    }

    inline size_t find_last_not_of(const TCharType* set, size_t pos, size_t n) const noexcept {
        ssize_t startpos = pos >= size() ? static_cast<ssize_t>(size()) - 1 : static_cast<ssize_t>(pos);

        for (ssize_t i = startpos; i >= 0; --i) {
            const TCharType c = Ptr()[i];

            bool found = true;
            for (const TCharType* p = set; p < set + n; ++p) {
                if (TTraits::Equal(c, *p)) {
                    found = false;
                    break;
                }
            }
            if (found) {
                return static_cast<size_t>(i);
            }
        }

        return npos;
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

    template <GenericFinder finder>
    inline size_t GenericFind(const TCharType* s, size_t n, size_t pos = npos) const noexcept {
        if (pos >= Len()) {
            return npos;
        }

        return off(finder(Ptr() + pos, Len() - pos, s, n));
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

        TTraits::Copy(pc, Ptr() + pos, toCopy);

        return toCopy;
    }
};
