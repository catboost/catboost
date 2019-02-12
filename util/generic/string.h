#pragma once

#include <cstddef>
#include <cstring>
#include <stlfwd>

#include <util/system/compat.h>
#include <util/system/yassert.h>
#include <util/system/atomic.h>

#include "utility.h"
#include "chartraits.h"
#include "bitops.h"
#include "explicit_type.h"
#include "reserve.h"

#if defined(address_sanitizer_enabled) || defined(thread_sanitizer_enabled)
#include "hide_ptr.h"
#endif

[[noreturn]] void ThrowLengthError(const char* descr);
[[noreturn]] void ThrowRangeError(const char* descr);

namespace NDetail {
    extern void const* STRING_DATA_NULL;

    /** Represents string data shared between instances of string objects. */
    struct TStringData {
        TAtomic Refs;
        size_t BufLen; /**< Maximum number of characters that this data can fit. */
        size_t Length; /**< Actual string data length. */
    };

    template <typename TCharType>
    struct TStringDataTraits {
        using TData = TStringData;

        enum : size_t {
            Overhead = sizeof(TData) + sizeof(TCharType), // + null terminated symbol
            MaxSize = (std::numeric_limits<size_t>::max() - Overhead) / sizeof(TCharType)
        };

        static constexpr size_t CalcAllocationSize(const size_t len) noexcept {
            return len * sizeof(TCharType) + Overhead;
        }

        static TData* GetData(TCharType* p) {
            return ((TData*)(void*)p) - 1;
        }

        static TCharType* GetChars(TData* data) {
            return (TCharType*)(void*)(data + 1);
        }

        static TCharType* GetNull() {
            return (TCharType*)STRING_DATA_NULL;
        }
    };

    /**
     * Allocates new string data that fits at least @c newLen of characters.
     *
     * @throw std::length_error
     */
    template <typename TCharType>
    TCharType* Allocate(size_t oldLen, size_t newLen, TStringData* oldData = nullptr);

    void Deallocate(void* data);

}

template <typename TDerived, typename TCharType, typename TTraits>
class TStringBase;

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

    static constexpr size_t max_size() noexcept {
        return ::NDetail::TStringDataTraits<TCharType>::MaxSize;
    }

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

    template <class T, class A>
    inline operator std::basic_string<TCharType, T, A>() const {
        return std::basic_string<TCharType, T, A>(Ptr(), Len());
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
    constexpr bool Empty() const noexcept  {
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
        Y_ASSERT(pos < this->Size());

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
            ThrowRangeError("TStringBase::copy");
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

template <class TStringType>
class TBasicCharRef {
public:
    using TChar = typename TStringType::TChar;

    TBasicCharRef(TStringType& s, size_t pos)
        : S_(s)
        , Pos_(pos)
    {
    }

    operator TChar() const {
        return S_.at(Pos_);
    }

    TBasicCharRef& operator=(TChar c) {
        Y_ASSERT(Pos_ < S_.size());

        TChar* p = S_.Detach();
        p[Pos_] = c;

        return *this;
    }

    TBasicCharRef& operator=(const TBasicCharRef& other) {
        return this->operator=(static_cast<TChar>(other));
    }

private:
    TStringType& S_;
    size_t Pos_;
};

template <typename TDerived, typename TCharType, typename TTraitsType>
const size_t TStringBase<TDerived, TCharType, TTraitsType>::npos;

template <typename TDerived, typename TCharType, typename TTraits>
class TBasicString: public TStringBase<TDerived, TCharType, TTraits> {
public:
    using TSelf = TBasicString;
    using TBase = TStringBase<TDerived, TCharType, TTraits>;
    using TDataTraits = ::NDetail::TStringDataTraits<TCharType>;
    using TData = typename TDataTraits::TData;
    using TFixedString = typename TBase::TFixedString;

    using TdChar = TCharType;
    using TCharRef = TBasicCharRef<TDerived>;
    using char_type = TCharType;
    using value_type = TCharType;
    using traits_type = TTraits;

    using iterator = TCharType*;
    using reverse_iterator = typename TBase::template TReverseIteratorBase<iterator>;
    using const_iterator = typename TBase::const_iterator;
    using const_reverse_iterator = typename TBase::const_reverse_iterator;

    struct TUninitialized {
        explicit TUninitialized(size_t size)
            : Size(size)
        {
        }

        size_t Size;
    };

private:
    TDerived* This() {
        return static_cast<TDerived*>(this);
    }

    const TDerived* This() const {
        return static_cast<const TDerived*>(this);
    }

protected:
    /**
     * Allocates new string data that fits at least the specified number of characters.
     *
     * @param len                       Number of characters.
     * @throw std::length_error
     */
    static TCharType* Allocate(size_t len, TData* oldData = nullptr) {
        return Allocate(len, len, oldData);
    }

    static TCharType* Allocate(size_t oldLen, size_t newLen, TData* oldData) {
        return ::NDetail::Allocate<TCharType>(oldLen, newLen, oldData);
    }

    // ~~~ Data member ~~~
    TCharType* Data_;

    // ~~~ Core functions ~~~
    inline void Ref() noexcept {
        if (Data_ != TDataTraits::GetNull()) {
            AtomicIncrement(GetData()->Refs);
        }
    }

    inline void UnRef() noexcept {
        if (Data_ != TDataTraits::GetNull()) {
            // IsDetached() check is a common case optimization
            if (IsDetached() || AtomicDecrement(GetData()->Refs) == 0) {
                ::NDetail::Deallocate(GetData());
            }
        }
    }

    inline TData* GetData() const noexcept {
        return TDataTraits::GetData(Data_);
    }

    void Relink(TCharType* tmp) {
        UnRef();
        Data_ = tmp;
    }

    /**
     * Makes a distinct copy of this string. `IsDetached()` is always true after this call.
     *
     * @throw std::length_error
     */
    void Clone() {
        const size_t len = length();

        Relink(TTraits::Copy(Allocate(len), Data_, len));
    }

    void TruncNonShared(size_t n) {
        GetData()->Length = n;
        Data_[n] = 0;
    }

    void ResizeNonShared(size_t n) {
        if (capacity() < n) {
            Data_ = Allocate(n, GetData());
        } else {
            TruncNonShared(n);
        }
    }

public:
    inline TCharType operator[](size_t pos) const noexcept {
        Y_ASSERT(pos <= length());

        return this->data()[pos];
    }

    inline TCharRef operator[](size_t pos) noexcept {
        Y_ASSERT(pos <= length());

        return TCharRef(*This(), pos);
    }

    using TBase::back;

    inline TCharRef back() noexcept {
        Y_ASSERT(!this->empty());

        if (Y_UNLIKELY(this->empty())) {
            return TCharRef(*This(), 0);
        }
        return TCharRef(*This(), length() - 1);
    }

    using TBase::front;

    inline TCharRef front() noexcept {
        Y_ASSERT(!this->empty());
        return TCharRef(*This(), 0);
    }

    inline size_t length() const noexcept {
        return GetData()->Length;
    }

    inline const TCharType* data() const noexcept {
        return Data_;
    }

    inline const TCharType* c_str() const noexcept {
        return Data_;
    }

    // ~~~ STL compatible method to obtain data pointer ~~~
    iterator begin() {
        Detach();

        return Data_;
    }

    iterator vend() {
        Detach();

        return Data_ + length();
    }

    reverse_iterator rbegin() {
        Detach();

        return reverse_iterator(Data_ + length());
    }

    reverse_iterator rend() {
        Detach();

        return reverse_iterator(Data_);
    }

    using TBase::begin;   //!< const_iterator TStringBase::begin() const
    using TBase::cbegin;  //!< const_iterator TStringBase::cbegin() const
    using TBase::cend;    //!< const_iterator TStringBase::cend() const
    using TBase::crbegin; //!< const_reverse_iterator TStringBase::crbegin() const
    using TBase::crend;   //!< const_reverse_iterator TStringBase::crend() const
    using TBase::end;     //!< const_iterator TStringBase::end() const
    using TBase::rbegin;  //!< const_reverse_iterator TStringBase::rbegin() const
    using TBase::rend;    //!< const_reverse_iterator TStringBase::rend() const

    inline size_t reserve() const noexcept {
        return GetData()->BufLen;
    }

    inline size_t capacity() const noexcept {
        return reserve();
    }

    TCharType* Detach() {
        if (IsDetached()) {
            return Data_;
        }

        Clone();
        return Data_;
    }

    bool IsDetached() const {
        return 1 == AtomicGet(GetData()->Refs);
    }

    // ~~~ Size and capacity ~~~
    TDerived& resize(size_t n, TCharType c = ' ') { // remove or append
        const size_t len = length();

        if (n > len) {
            ReserveAndResize(n);
            TTraits::Assign(Data_ + len, n - len, c);

            return *This();
        }

        return remove(n);
    }

    // ~~~ Constructor ~~~ : FAMILY0(,TBasicString)
    inline TBasicString()
        : Data_(TDataTraits::GetNull())
    {
    }

    inline TBasicString(const TDerived& s)
        : Data_(s.Data_)
    {
        Ref();
    }

    template <typename T, typename A>
    explicit inline TBasicString(const std::basic_string<TCharType, T, A>& s)
        : Data_(TDataTraits::GetNull())
    {
        AssignNoAlias(s.data(), s.length());
    }

    TBasicString(const TDerived& s, size_t pos, size_t n) {
        size_t len = s.length();
        pos = Min(pos, len);
        n = Min(n, len - pos);
        Data_ = Allocate(n);
        TTraits::Copy(Data_, s.Data_ + pos, n);
    }

    TBasicString(const TCharType* pc) {
        const size_t len = TBase::StrLen(pc);

        Data_ = Allocate(len);
        TTraits::Copy(Data_, pc, len);
    }

    TBasicString(const TCharType* pc, size_t n) {
        Data_ = Allocate(n);
        TTraits::Copy(Data_, pc, n);
    }

    TBasicString(const TCharType* pc, size_t pos, size_t n) {
        Data_ = Allocate(n);
        TTraits::Copy(Data_, pc + pos, n);
    }

    explicit TBasicString(TExplicitType<TCharType> c) {
        Data_ = Allocate(1);
        Data_[0] = c;
    }

    explicit TBasicString(const TCharRef& c) {
        Data_ = Allocate(1);
        Data_[0] = c;
    }

    TBasicString(size_t n, TCharType c) {
        Data_ = Allocate(n);
        TTraits::Assign(Data_, n, c);
    }

    /**
     * Constructs an uninitialized string of size `uninitialized.Size`. The proper
     * way to use this ctor is via `TBasicString::Uninitialized` factory function.
     *
     * @throw std::length_error
     */
    TBasicString(TUninitialized uninitialized) {
        Data_ = Allocate(uninitialized.Size);
    }

    TBasicString(const TCharType* b, const TCharType* e) {
        Data_ = Allocate(e - b);
        TTraits::Copy(Data_, b, e - b);
    }

    explicit TBasicString(const TFixedString s)
        : Data_(Allocate(s.Length))
    {
        if (0 != s.Length) {
            TTraits::Copy(Data_, s.Start, s.Length);
        }
    }

    static TDerived Uninitialized(size_t n) {
        return TDerived(TUninitialized(n));
    }

private:
    template <typename... R>
    static size_t SumLength(const TFixedString s1, const R&... r) noexcept {
        return s1.Length + SumLength(r...);
    }

    template <typename... R>
    static size_t SumLength(const TCharType /*s1*/, const R&... r) noexcept {
        return 1 + SumLength(r...);
    }

    static constexpr size_t SumLength() noexcept {
        return 0;
    }

    template <typename... R>
    static void CopyAll(TCharType* p, const TFixedString s, const R&... r) {
        TTraits::Copy(p, s.Start, s.Length);
        CopyAll(p + s.Length, r...);
    }

    template <typename... R, class TNextCharType, typename = std::enable_if_t<std::is_same<TCharType, TNextCharType>::value>>
    static void CopyAll(TCharType* p, const TNextCharType s, const R&... r) {
        p[0] = s;
        CopyAll(p + 1, r...);
    }

    static void CopyAll(TCharType*) noexcept {
    }

public:
    // ~~~ Destructor ~~~
    inline ~TBasicString() {
        UnRef();
    }

    inline void clear() noexcept {
        if (IsDetached()) {
            TruncNonShared(0);
            return;
        }

        Relink(TDataTraits::GetNull());
    }

    template <typename... R>
    static inline TDerived Join(const R&... r) {
        TDerived s;

        s.Data_ = Allocate(SumLength(r...));
        CopyAll(s.Data_, r...);

        return s;
    }

    // ~~~ Assignment ~~~ : FAMILY0(TBasicString&, assign);
    TDerived& assign(const TDerived& s) {
        TDerived(s).swap(*This());

        return *This();
    }

    TDerived& assign(const TDerived& s, size_t pos, size_t n) {
        return assign(TDerived(s, pos, n));
    }

    TDerived& assign(const TCharType* pc) {
        return assign(pc, TBase::StrLen(pc));
    }

    TDerived& assign(TCharType ch) {
        return assign(&ch, 1);
    }

    TDerived& assign(const TCharType* pc, size_t len) {
#if defined(address_sanitizer_enabled) || defined(thread_sanitizer_enabled)
        pc = (const TCharType*)HidePointerOrigin((void*)pc);
#endif

        if (Y_LIKELY(IsDetached() && (pc + len <= TBase::begin() || pc >= TBase::end()))) {
            ResizeNonShared(len);
            TTraits::Copy(Data_, pc, len);
        } else if (IsDetached() && pc == data() && capacity() >= len) {
            TruncNonShared(len);
        } else {
            Relink(TTraits::Copy(Allocate(len), pc, len));
        }

        return *This();
    }

    TDerived& assign(const TCharType* first, const TCharType* last) {
        return assign(first, last - first);
    }

    TDerived& assign(const TCharType* pc, size_t pos, size_t n) {
        return assign(pc + pos, n);
    }

    inline TDerived& AssignNoAlias(const TCharType* pc, size_t len) {
        if (IsDetached()) {
            ResizeNonShared(len);
        } else {
            Relink(Allocate(len));
        }

        TTraits::Copy(Data_, pc, len);

        return *This();
    }

    inline TDerived& AssignNoAlias(const TCharType* b, const TCharType* e) {
        return AssignNoAlias(b, e - b);
    }

    TDerived& assign(const TFixedString s) {
        return assign(s.Start, s.Length);
    }

    TDerived& assign(const TFixedString s, size_t spos, size_t sn = TBase::npos) {
        return assign(s.SubString(spos, sn));
    }

    TDerived& AssignNoAlias(const TFixedString s) {
        return AssignNoAlias(s.Start, s.Length);
    }

    TDerived& AssignNoAlias(const TFixedString s, size_t spos, size_t sn = TBase::npos) {
        return AssignNoAlias(s.SubString(spos, sn));
    }

    TDerived& operator=(const TDerived& s) {
        return assign(s);
    }

    TDerived& operator=(const TFixedString s) {
        return assign(s);
    }

    TDerived& operator=(const TCharType* s) {
        return assign(s);
    }

    TDerived& operator=(TCharType ch) {
        return assign(ch);
    }

    inline void reserve(size_t len) {
        if (IsDetached()) {
            if (capacity() < len) {
                Data_ = Allocate(length(), len, GetData());
            }
        } else {
            const size_t sufficientLen = Max(length(), len);
            Relink(TTraits::Copy(Allocate(length(), sufficientLen, nullptr), Data_, length()));
        }
    }

    // ~~~ Appending ~~~ : FAMILY0(TBasicString&, append);
    inline TDerived& append(size_t count, TCharType ch) {
        while (count--) {
            append(ch);
        }

        return *This();
    }

    inline TDerived& append(const TDerived& s) {
        if (&s != This()) {
            return AppendNoAlias(s.data(), s.size());
        }

        return append(s.data(), s.size());
    }

    inline TDerived& append(const TDerived& s, size_t pos, size_t n) {
        return append(s.data(), pos, n, s.size());
    }

    inline TDerived& append(const TCharType* pc) {
        return append(pc, TBase::StrLen(pc));
    }

    inline TDerived& append(TCharType c) {
        const size_t olen = length();

        ReserveAndResize(olen + 1);
        *(Data_ + olen) = c;

        return *This();
    }

    inline TDerived& append(const TCharType* first, const TCharType* last) {
        return append(first, last - first);
    }

    inline TDerived& append(const TCharType* pc, size_t len) {
        if (pc + len <= TBase::begin() || pc >= TBase::end()) {
            return AppendNoAlias(pc, len);
        }

        return append(pc, 0, len, len);
    }

    inline void ReserveAndResize(size_t len) {
        if (IsDetached()) {
            ResizeNonShared(len);
        } else {
            Relink(TTraits::Copy(Allocate(len), Data_, Min(len, length())));
        }
    }

    inline TDerived& AppendNoAlias(const TCharType* pc, size_t len) {
        const size_t olen = length();
        const size_t nlen = olen + len;

        ReserveAndResize(nlen);
        TTraits::Copy(Data_ + olen, pc, len);

        return *This();
    }

    TDerived& AppendNoAlias(const TFixedString s) {
        return AppendNoAlias(s.Start, s.Length);
    }

    TDerived& AppendNoAlias(const TFixedString s, size_t spos, size_t sn = TBase::npos) {
        return AppendNoAlias(s.SubString(spos, sn));
    }

    TDerived& append(const TFixedString s) {
        return append(s.Start, s.Length);
    }

    TDerived& append(const TFixedString s, size_t spos, size_t sn = TBase::npos) {
        return append(s.SubString(spos, sn));
    }

    inline TDerived& append(const TCharType* pc, size_t pos, size_t n, size_t pc_len = TBase::npos) {
        return replace(length(), 0, pc, pos, n, pc_len);
    }

    inline void push_back(TCharType c) {
        append(c);
    }

    template <class T>
    TDerived& operator+=(const T& s) {
        return append(s);
    }

    template <class T>
    friend TDerived operator*(const TDerived& s, T count) {
        TDerived result;

        for (T i = 0; i < count; ++i) {
            result += s;
        }

        return result;
    }

    template <class T>
    TDerived& operator*=(T count) {
        TDerived temp;

        for (T i = 0; i < count; ++i) {
            temp += *This();
        }

        swap(temp);

        return *This();
    }

    /*
     * Following overloads of "operator+" aim to choose the cheapest implementation depending on
     * summand types: lvalues, detached rvalues, shared rvalues.
     *
     * General idea is to use the detached-rvalue argument (left of right) to store the result
     * wherever possible. If a buffer in rvalue is large enough this saves a re-allocation. If
     * both arguments are rvalues we check which one is detached. If both of them are detached then
     * the left argument is obviously preferrable because you won't need to shift the data.
     *
     * If an rvalue is shared then it's basically the same as lvalue because you cannot use its
     * buffer to store the sum. However, we rely on the fact that append() and prepend() are already
     * optimized for the shared case and detach the string into the buffer large enough to store
     * the sum (compared to the detach+reallocation). This way, if we have only one rvalue argument
     * (left or right) then we simply append/prepend into it, without checking if it's detached or
     * not. This will be checked inside ReserveAndResize anyway.
     *
     * If both arguments cannot be used to store the sum (e.g. two lvalues) then we fall back to the
     * Join function that constructs a resulting string in the new buffer with the minimum overhead:
     * malloc + memcpy + memcpy.
     */

    friend TDerived operator+(TDerived&& s1, const TDerived& s2) Y_WARN_UNUSED_RESULT {
        s1 += s2;
        return std::move(s1);
    }

    friend TDerived operator+(const TDerived& s1, TDerived&& s2) Y_WARN_UNUSED_RESULT {
        s2.prepend(s1);
        return std::move(s2);
    }

    friend TDerived operator+(TDerived&& s1, TDerived&& s2) Y_WARN_UNUSED_RESULT {
        if (!s1.IsDetached() && s2.IsDetached()) {
            s2.prepend(s1);
            return std::move(s2);
        }
        s1 += s2;
        return std::move(s1);
    }

    friend TDerived operator+(TDerived&& s1, const TFixedString s2) Y_WARN_UNUSED_RESULT {
        s1 += s2;
        return std::move(s1);
    }

    friend TDerived operator+(TDerived&& s1, const TCharType* s2) Y_WARN_UNUSED_RESULT {
        s1 += s2;
        return std::move(s1);
    }

    friend TDerived operator+(TDerived&& s1, TCharType s2) Y_WARN_UNUSED_RESULT {
        s1 += s2;
        return std::move(s1);
    }

    friend TDerived operator+(const TDerived& s1, const TDerived& s2) Y_WARN_UNUSED_RESULT {
        return Join(s1, s2);
    }

    friend TDerived operator+(const TDerived& s1, const TFixedString s2) Y_WARN_UNUSED_RESULT {
        return Join(s1, s2);
    }

    friend TDerived operator+(const TDerived& s1, const TCharType* s2) Y_WARN_UNUSED_RESULT {
        return Join(s1, s2);
    }

    friend TDerived operator+(const TDerived& s1, TCharType s2) Y_WARN_UNUSED_RESULT {
        return Join(s1, TFixedString(&s2, 1));
    }

    friend TDerived operator+(const TCharType* s1, TDerived&& s2) Y_WARN_UNUSED_RESULT {
        s2.prepend(s1);
        return std::move(s2);
    }

    friend TDerived operator+(const TFixedString s1, TDerived&& s2) Y_WARN_UNUSED_RESULT {
        s2.prepend(s1);
        return std::move(s2);
    }

    friend TDerived operator+(const TFixedString s1, const TDerived& s2) Y_WARN_UNUSED_RESULT {
        return Join(s1, s2);
    }

    friend TDerived operator+(const TCharType* s1, const TDerived& s2) Y_WARN_UNUSED_RESULT {
        return Join(s1, s2);
    }

    // ~~~ Prepending ~~~ : FAMILY0(TDerived&, prepend);
    TDerived& prepend(const TDerived& s) {
        return replace(0, 0, s.Data_, 0, TBase::npos, s.length());
    }

    TDerived& prepend(const TDerived& s, size_t pos, size_t n) {
        return replace(0, 0, s.Data_, pos, n, s.length());
    }

    TDerived& prepend(const TCharType* pc) {
        return replace(0, 0, pc);
    }

    TDerived& prepend(size_t n, TCharType c) {
        return insert(size_t(0), n, c);
    }

    TDerived& prepend(TCharType c) {
        return replace(0, 0, &c, 0, 1, 1);
    }

    TDerived& prepend(const TFixedString s, size_t spos = 0, size_t sn = TBase::npos) {
        return insert(0, s, spos, sn);
    }

    // ~~~ Insertion ~~~ : FAMILY1(TDerived&, insert, size_t pos);
    TDerived& insert(size_t pos, const TDerived& s) {
        return replace(pos, 0, s.Data_, 0, TBase::npos, s.length());
    }

    TDerived& insert(size_t pos, const TDerived& s, size_t pos1, size_t n1) {
        return replace(pos, 0, s.Data_, pos1, n1, s.length());
    }

    TDerived& insert(size_t pos, const TCharType* pc) {
        return replace(pos, 0, pc);
    }

    TDerived& insert(size_t pos, const TCharType* pc, size_t len) {
        return insert(pos, TFixedString(pc, len));
    }

    TDerived& insert(const_iterator pos, const_iterator b, const_iterator e) {
        return insert(this->off(pos), b, e - b);
    }

    TDerived& insert(size_t pos, size_t n, TCharType c) {
        if (n == 1) {
            return replace(pos, 0, &c, 0, 1, 1);
        } else {
            return insert(pos, TDerived(n, c));
        }
    }

    TDerived& insert(const_iterator pos, size_t len, TCharType ch) {
        return this->insert(this->off(pos), len, ch);
    }

    TDerived& insert(const_iterator pos, TCharType ch) {
        return this->insert(pos, 1, ch);
    }

    TDerived& insert(size_t pos, const TFixedString s, size_t spos = 0, size_t sn = TBase::npos) {
        return replace(pos, 0, s, spos, sn);
    }

    // ~~~ Removing ~~~
    TDerived& remove(size_t pos, size_t n) {
        return replace(pos, n, TDataTraits::GetNull(), 0, 0, 0);
    }

    TDerived& remove(size_t pos = 0) {
        if (pos < length()) {
            Detach();
            TruncNonShared(pos);
        }

        return *This();
    }

    TDerived& erase(size_t pos = 0, size_t n = TBase::npos) {
        return remove(pos, n);
    }

    TDerived& erase(const_iterator b, const_iterator e) {
        return erase(this->off(b), e - b);
    }

    TDerived& erase(const_iterator i) {
        return erase(i, i + 1);
    }

    TDerived& pop_back() {
        Y_ASSERT(!this->empty());
        return erase(this->length() - 1, 1);
    }

    // ~~~ replacement ~~~ : FAMILY2(TDerived&, replace, size_t pos, size_t n);
    TDerived& replace(size_t pos, size_t n, const TDerived& s) {
        return replace(pos, n, s.Data_, 0, TBase::npos, s.length());
    }

    TDerived& replace(size_t pos, size_t n, const TDerived& s, size_t pos1, size_t n1) {
        return replace(pos, n, s.Data_, pos1, n1, s.length());
    }

    TDerived& replace(size_t pos, size_t n, const TCharType* pc) {
        return replace(pos, n, TFixedString(pc));
    }

    TDerived& replace(size_t pos, size_t n, const TCharType* s, size_t len) {
        return replace(pos, n, s, 0, len, len);
    }

    TDerived& replace(size_t pos, size_t n, const TCharType* s, size_t spos, size_t sn) {
        return replace(pos, n, s, spos, sn, sn);
    }

    TDerived& replace(size_t pos, size_t n1, size_t n2, TCharType c) {
        if (n2 == 1) {
            return replace(pos, n1, &c, 0, 1, 1);
        } else {
            return replace(pos, n1, TDerived(n2, c));
        }
    }

    TDerived& replace(size_t pos, size_t n, const TFixedString s, size_t spos = 0, size_t sn = TBase::npos) {
        return replace(pos, n, s.Start, spos, sn, s.Length);
    }

    // ~~~ main driver: should be protected (in the future)
    TDerived& replace(size_t pos, size_t del, const TCharType* pc, size_t pos1, size_t ins, size_t len1) {
        size_t len = length();
        // 'pc' can point to a single character that is not null terminated, so in this case TTraits::GetLength must not be called
        len1 = pc ? (len1 == TBase::npos ? (ins == TBase::npos ? TTraits::GetLength(pc) : TTraits::GetLength(pc, ins + pos1)) : len1) : 0;

        pos = Min(pos, len);
        pos1 = Min(pos1, len1);

        del = Min(del, len - pos);
        ins = Min(ins, len1 - pos1);

        if (len - del > this->max_size() - ins) { // len-del+ins -- overflow
            ThrowLengthError("TBasicString::replace");
        }

        size_t total = len - del + ins;

        if (!total) {
            clear();
            return *This();
        }

        size_t rem = len - del - pos;

        if (!IsDetached() || (pc && (pc >= Data_ && pc < Data_ + len))) {
            // malloc
            // 1. alias
            // 2. overlapped
            TCharType* temp = Allocate(total);
            TTraits::Copy(temp, Data_, pos);
            TTraits::Copy(temp + pos, pc + pos1, ins);
            TTraits::Copy(temp + pos + ins, Data_ + pos + del, rem);
            Relink(temp);
        } else if (reserve() < total) {
            // realloc (increasing)
            // 3. not enough room
            Data_ = Allocate(total, GetData());
            TTraits::Move(Data_ + pos + ins, Data_ + pos + del, rem);
            TTraits::Copy(Data_ + pos, pc + pos1, ins);
        } else {
            // 1. not alias
            // 2. not overlapped
            // 3. enough room
            // 4. not too much room
            TTraits::Move(Data_ + pos + ins, Data_ + pos + del, rem);
            TTraits::Copy(Data_ + pos, pc + pos1, ins);
            //GetData()->SetLength(total);
            TruncNonShared(total);
        }

        return *This();
    }

    // ~~~ Reversion ~~~~
    void reverse() {
        Detach();
        TTraits::Reverse(Data_, length());
    }

    void swap(TDerived& s) noexcept {
        DoSwap(Data_, s.Data_);
    }

    /**
     * @returns                         String suitable for debug printing (like Python's `repr()`).
     *                                  Format of the string is unspecified and may be changed over time.
     */
    TDerived Quote() const {
        extern TDerived EscapeC(const TDerived&);

        return TDerived() + '"' + EscapeC(*This()) + '"';
    }
};

class TString: public TBasicString<TString, char, TCharTraits<char>> {
    using TBase = TBasicString<TString, char, TCharTraits<char>>;

public:
    using TFixedString = TBase::TFixedString;

    using TBase::TBase;

    TString() {
    }

    TString(const TString& s)
        : TBase(s)
    {
    }

    TString(::NDetail::TReserveTag rt) {
        this->reserve(rt.Capacity);
    }

    TString(TString&& s) noexcept {
        swap(s);
    }

    TString& operator=(const TString& s) {
        return assign(s);
    }

    TString& operator=(TString&& s) noexcept {
        swap(s);

        return *this;
    }

    TString& operator=(const TFixedString s) {
        return assign(s);
    }

    TString& operator=(const TdChar* s) {
        return assign(s);
    }

public:
    /**
     * Modifies the substring of length `n` starting from `pos`, applying `f` to each position and symbol.
     *
     * @return                          false if no changes have been made.
     */
    template <typename T>
    bool Transform(T&& f, size_t pos = 0, size_t n = TBase::npos) {
        size_t len = length();

        if (pos > len) {
            pos = len;
        }

        if (n > len - pos) {
            n = len - pos;
        }

        bool changed = false;

        for (size_t i = pos; i != pos + n; ++i) {
            char c = f(i, Data_[i]);

            if (c != Data_[i]) {
                if (!changed) {
                    Detach();
                    changed = true;
                }

                Data_[i] = c;
            }
        }

        return changed;
    }

    bool to_lower(size_t pos = 0, size_t n = TBase::npos);
    bool to_upper(size_t pos = 0, size_t n = TBase::npos);
    bool to_title(size_t pos = 0, size_t n = TBase::npos);

    /**
     * @warning doesn't work with non-ASCII letters.
     */
    friend TString to_lower(const TString& s) {
        TString ret(s);
        ret.to_lower();
        return ret;
    }

    /**
     * @warning doesn't work with non-ASCII letters.
     */
    friend TString to_upper(const TString& s) {
        TString ret(s);
        ret.to_upper();
        return ret;
    }

    friend TString to_title(const TString& s) {
        TString ret(s);
        ret.to_title();
        return ret;
    }
};

class TUtf16String: public TBasicString<TUtf16String, wchar16, TCharTraits<wchar16>> {
    using TBase = TBasicString<TUtf16String, wchar16, TCharTraits<wchar16>>;

public:
    using TFixedString = TBase::TFixedString;

    using TBase::TBase;

    TUtf16String() = default;

    TUtf16String(TUtf16String&& s) noexcept {
        swap(s);
    }

    TUtf16String(const TUtf16String& s)
        : TBase(s)
    {
    }

    TUtf16String(::NDetail::TReserveTag rt) {
        this->reserve(rt.Capacity);
    }

    static TUtf16String FromUtf8(const ::TFixedString<char>& s) {
        return TUtf16String().AppendUtf8(s);
    }

    static TUtf16String FromAscii(const ::TFixedString<char>& s) {
        return TUtf16String().AppendAscii(s);
    }

    TUtf16String& AssignUtf8(const ::TFixedString<char>& s) {
        clear();
        return AppendUtf8(s);
    }

    TUtf16String& AssignAscii(const ::TFixedString<char>& s) {
        clear();
        return AppendAscii(s);
    }

    TUtf16String& AppendUtf8(const ::TFixedString<char>& s);
    TUtf16String& AppendAscii(const ::TFixedString<char>& s);

    TUtf16String& operator=(const TUtf16String& s) {
        return assign(s);
    }

    TUtf16String& operator=(TUtf16String&& s) noexcept {
        swap(s);

        return *this;
    }

    TUtf16String& operator=(const TFixedString s) {
        return assign(s);
    }

    TUtf16String& operator=(const TdChar* s) {
        return assign(s);
    }

    TUtf16String& operator=(wchar16 ch) {
        return assign(ch);
    }

    // @{
    /**
     * Modifies the case of the string, depending on the operation.
     * @return false if no changes have been made
     */
    bool to_lower(size_t pos = 0, size_t n = TBase::npos);
    bool to_upper(size_t pos = 0, size_t n = TBase::npos);
    bool to_title();
    // @}

    friend TUtf16String to_lower(const TUtf16String& s) {
        TUtf16String ret(s);
        ret.to_lower();
        return ret;
    }

    friend TUtf16String to_upper(const TUtf16String& s) {
        TUtf16String ret(s);
        ret.to_upper();
        return ret;
    }

    friend TUtf16String to_title(const TUtf16String& s) {
        TUtf16String ret(s);
        ret.to_title();
        return ret;
    }
};

class TUtf32String: public TBasicString<TUtf32String, wchar32, TCharTraits<wchar32>> {
    using TBase = TBasicString<TUtf32String, wchar32, TCharTraits<wchar32>>;

public:
    using TFixedString = TBase::TFixedString;

    using TBase::TBase;

    TUtf32String() = default;

    TUtf32String(TUtf32String&& s) noexcept {
        swap(s);
    }

    TUtf32String(const TUtf32String& s)
        : TBase(s)
    {
    }

    TUtf32String(::NDetail::TReserveTag rt) {
        this->reserve(rt.Capacity);
    }

    static TUtf32String FromUtf8(const ::TFixedString<char>& s) {
        return TUtf32String().AppendUtf8(s);
    }

    static TUtf32String FromUtf16(const ::TFixedString<wchar16>& s) {
        return TUtf32String().AppendUtf16(s);
    }

    static TUtf32String FromAscii(const ::TFixedString<char>& s) {
        return TUtf32String().AppendAscii(s);
    }

    TUtf32String& AssignUtf8(const ::TFixedString<char>& s) {
        clear();
        return AppendUtf8(s);
    }

    TUtf32String& AssignUtf16(const ::TFixedString<wchar16>& s) {
        clear();
        return AppendUtf16(s);
    }

    TUtf32String& AssignAscii(const ::TFixedString<char>& s) {
        clear();
        return AppendAscii(s);
    }

    TUtf32String& AppendUtf8(const ::TFixedString<char>& s);
    TUtf32String& AppendAscii(const ::TFixedString<char>& s);
    TUtf32String& AppendUtf16(const ::TFixedString<wchar16>& s);

    TUtf32String& operator=(const TUtf32String& s) {
        return assign(s);
    }

    TUtf32String& operator=(TUtf32String&& s) noexcept {
        swap(s);

        return *this;
    }

    TUtf32String& operator=(const TFixedString s) {
        return assign(s);
    }

    TUtf32String& operator=(const TdChar* s) {
        return assign(s);
    }

    TUtf32String& operator=(wchar32 ch) {
        return assign(ch);
    }

    // @{
    /**
    * Modifies the case of the string, depending on the operation.
    * @return false if no changes have been made
    */
    bool to_lower(size_t pos = 0, size_t n = TBase::npos);
    bool to_upper(size_t pos = 0, size_t n = TBase::npos);
    bool to_title();
    // @}

    friend TUtf32String to_lower(const TUtf32String& s) {
        TUtf32String ret(s);
        ret.to_lower();
        return ret;
    }

    friend TUtf32String to_upper(const TUtf32String& s) {
        TUtf32String ret(s);
        ret.to_upper();
        return ret;
    }

    friend TUtf32String to_title(const TUtf32String& s) {
        TUtf32String ret(s);
        ret.to_title();
        return ret;
    }
};

std::ostream& operator<<(std::ostream&, const TString&);

namespace NPrivate {
    template <class Char>
    struct TCharToString {
        // TODO: switch to TBaseString derived type when compilation with nvcc on windows will succeed
        using type = TFixedString<Char>;
    };

    template <>
    struct TCharToString<char> {
        using type = TString;
    };

    template <>
    struct TCharToString<wchar16> {
        using type = TUtf16String;
    };

    template <>
    struct TCharToString<wchar32> {
        using type = TUtf32String;
    };
}

template <class Char>
using TGenericString = typename NPrivate::TCharToString<Char>::type;

namespace std {
    template <>
    struct hash<TString> {
        using argument_type = TString;
        using result_type = size_t;
        inline result_type operator()(argument_type const& s) const noexcept {
            return s.hash();
        }
    };
}
