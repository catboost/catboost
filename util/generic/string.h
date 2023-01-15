#pragma once

#include <cstddef>
#include <cstring>
#include <stlfwd>
#include <stdexcept>
#ifdef TSTRING_IS_STD_STRING
#include <string>
#endif
#include <string_view>

#include <util/system/yassert.h>
#include <util/system/atomic.h>

#include "utility.h"
#include "bitops.h"
#include "explicit_type.h"
#include "reserve.h"
#include "strbase.h"
#include "strbuf.h"
#include "string_hash.h"

#if defined(address_sanitizer_enabled) || defined(thread_sanitizer_enabled)
#include "hide_ptr.h"
#endif

#ifndef TSTRING_IS_STD_STRING
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
            MaxSize = (std::numeric_limits<size_t>::max() / 2 + 1 - Overhead) / sizeof(TCharType)
        };

        static constexpr size_t CalcAllocationSizeAndCapacity(size_t& len) noexcept {
            // buffer should be multiple to 2^n to fit allocator's memory block size
            size_t alignedSize = FastClp2(len * sizeof(TCharType) + Overhead);
            // calc capacity
            len = (alignedSize - Overhead) / sizeof(TCharType);
            return alignedSize;
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

    TChar* operator& () {
        return S_.begin() + Pos_;
    }

    const TChar* operator& () const {
        return S_.cbegin() + Pos_;
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
#endif

template <typename TCharType, typename TTraits>
class TBasicString: public TStringBase<TBasicString<TCharType, TTraits>, TCharType, TTraits> {
public:
    // TODO: Move to private section
    using TBase = TStringBase<TBasicString, TCharType, TTraits>;
#ifdef TSTRING_IS_STD_STRING
    using TStorage = std::basic_string<TCharType>;
    using reference = typename TStorage::reference;
#else
    using TDataTraits = ::NDetail::TStringDataTraits<TCharType>;
    using TData = typename TDataTraits::TData;

    using reference = TBasicCharRef<TBasicString>;
#endif
    using char_type = TCharType; // TODO: DROP
    using value_type = TCharType;
    using traits_type = TTraits;

    using iterator = TCharType*;
    using reverse_iterator = typename TBase::template TReverseIteratorBase<iterator>;
    using typename TBase::const_iterator;
    using typename TBase::const_reverse_iterator;
    using typename TBase::const_reference;

    struct TUninitialized {
        explicit TUninitialized(size_t size)
            : Size(size)
        {
        }

        size_t Size;
    };

    static size_t max_size() noexcept {
#ifdef TSTRING_IS_STD_STRING
        static size_t result = TStorage{}.max_size();
        return result;
#else
        return ::NDetail::TStringDataTraits<TCharType>::MaxSize;
#endif
    }

protected:
#ifdef TSTRING_IS_STD_STRING
    TStorage Storage_;
#else
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

        Relink(TTraits::copy(Allocate(len), Data_, len));
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
#endif

public:
    inline const_reference operator[](size_t pos) const noexcept {
        Y_ASSERT(pos <= length());

        return this->data()[pos];
    }

    inline reference operator[](size_t pos) noexcept {
        Y_ASSERT(pos <= length());

#ifdef TSTRING_IS_STD_STRING
        return Storage_[pos];
#else
        return reference(*this, pos);
#endif
    }

    using TBase::back;

    inline reference back() noexcept {
        Y_ASSERT(!this->empty());

#ifdef TSTRING_IS_STD_STRING
        return Storage_.back();
#else
        if (Y_UNLIKELY(this->empty())) {
            return reference(*this, 0);
        }
        return reference(*this, length() - 1);
#endif
    }

    using TBase::front;

    inline reference front() noexcept {
        Y_ASSERT(!this->empty());
#ifdef TSTRING_IS_STD_STRING
        return Storage_.front();
#else
        return reference(*this, 0);
#endif
    }

    inline size_t length() const noexcept {
#ifdef TSTRING_IS_STD_STRING
        return Storage_.length();
#else
        return GetData()->Length;
#endif
    }

    inline const TCharType* data() const noexcept {
#ifdef TSTRING_IS_STD_STRING
        return Storage_.data();
#else
        return Data_;
#endif
    }

    inline const TCharType* c_str() const noexcept {
#ifdef TSTRING_IS_STD_STRING
        return Storage_.c_str();
#else
        return Data_;
#endif
    }

    // ~~~ STL compatible method to obtain data pointer ~~~
    iterator begin() {
#ifdef TSTRING_IS_STD_STRING
        return Storage_.data();
#else
        Detach();
        return Data_;
#endif
    }

    iterator vend() {
#ifdef TSTRING_IS_STD_STRING
        return Storage_.data() + length();
#else
        Detach();
        return Data_ + length();
#endif

    }

    reverse_iterator rbegin() {
#ifdef TSTRING_IS_STD_STRING
        return reverse_iterator(Storage_.data() + length());
#else
        Detach();
        return reverse_iterator(Data_ + length());
#endif
    }

    reverse_iterator rend() {
#ifdef TSTRING_IS_STD_STRING
        return reverse_iterator(Storage_.data());
#else
        Detach();
        return reverse_iterator(Data_);
#endif
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
#ifdef TSTRING_IS_STD_STRING
        return Storage_.capacity();
#else
        return GetData()->BufLen;
#endif
    }

    inline size_t capacity() const noexcept {
#ifdef TSTRING_IS_STD_STRING
        return Storage_.capacity();
#else
        return reserve();
#endif
    }

    TCharType* Detach() {
#ifdef TSTRING_IS_STD_STRING
        return Storage_.data();
#else
        if (IsDetached()) {
            return Data_;
        }

        Clone();
        return Data_;
#endif
    }

    bool IsDetached() const {
#ifdef TSTRING_IS_STD_STRING
        return true;
#else
        return 1 == AtomicGet(GetData()->Refs);
#endif
    }

    // ~~~ Size and capacity ~~~
    TBasicString& resize(size_t n, TCharType c = ' ') { // remove or append
#ifdef TSTRING_IS_STD_STRING
        Storage_.resize(n, c);

        return *this;
#else
        const size_t len = length();

        if (n > len) {
            ReserveAndResize(n);
            TTraits::assign(Data_ + len, n - len, c);

            return *this;
        }

        return remove(n);
#endif
    }

    // ~~~ Constructor ~~~ : FAMILY0(,TBasicString)
    TBasicString()
#ifndef TSTRING_IS_STD_STRING
        : Data_(TDataTraits::GetNull())
#endif
    {
    }

    inline TBasicString(::NDetail::TReserveTag rt)
#ifndef TSTRING_IS_STD_STRING
        : Data_(TDataTraits::GetNull())
#endif
    {
        reserve(rt.Capacity);
    }

    inline TBasicString(const TBasicString& s)
#ifdef TSTRING_IS_STD_STRING
        : Storage_(s.Storage_)
#else
        : Data_(s.Data_)
#endif
    {
#ifndef TSTRING_IS_STD_STRING
        Ref();
#endif
    }

    inline TBasicString(TBasicString&& s) noexcept
#ifdef TSTRING_IS_STD_STRING
        : Storage_(std::move(s.Storage_))
#else
        : Data_(TDataTraits::GetNull())
#endif
    {
#ifndef TSTRING_IS_STD_STRING
        swap(s);
#endif
    }

    template <typename T, typename A>
    explicit inline TBasicString(const std::basic_string<TCharType, T, A>& s)
#ifdef TSTRING_IS_STD_STRING
        : Storage_(s)
#else
        : Data_(TDataTraits::GetNull())
#endif
    {
#ifndef TSTRING_IS_STD_STRING
        AssignNoAlias(s.data(), s.length());
#endif
    }

    TBasicString(const TBasicString& s, size_t pos, size_t n)
#ifdef TSTRING_IS_STD_STRING
        : Storage_(s.Storage_, pos, n)
#endif
    {
#ifndef TSTRING_IS_STD_STRING
        size_t len = s.length();
        pos = Min(pos, len);
        n = Min(n, len - pos);
        Data_ = Allocate(n);
        TTraits::copy(Data_, s.Data_ + pos, n);
#endif
    }

    TBasicString(const TCharType* pc)
#ifdef TSTRING_IS_STD_STRING
        : Storage_(pc, TBase::StrLen(pc))
#endif
    {
#ifndef TSTRING_IS_STD_STRING
        const size_t len = TBase::StrLen(pc);

        Data_ = Allocate(len);
        TTraits::copy(Data_, pc, len);
#endif
    }

    TBasicString(const TCharType* pc, size_t n)
#ifdef TSTRING_IS_STD_STRING
        : Storage_(pc, n)
#endif
    {
#ifndef TSTRING_IS_STD_STRING
        Data_ = Allocate(n);
        TTraits::copy(Data_, pc, n);
#endif
    }

    TBasicString(const TCharType* pc, size_t pos, size_t n)
#ifdef TSTRING_IS_STD_STRING
        : Storage_(pc + pos, n)
#endif
    {
#ifndef TSTRING_IS_STD_STRING
        Data_ = Allocate(n);
        TTraits::copy(Data_, pc + pos, n);
#endif
    }

#ifdef TSTRING_IS_STD_STRING
    explicit TBasicString(TExplicitType<TCharType> c) {
        Storage_.push_back(c);
    }
#else
    explicit TBasicString(TExplicitType<TCharType> c) {
        Data_ = Allocate(1);
        Data_[0] = c;
    }
    explicit TBasicString(const reference& c) {
        Data_ = Allocate(1);
        Data_[0] = c;
    }
#endif

    TBasicString(size_t n, TCharType c)
#ifdef TSTRING_IS_STD_STRING
        : Storage_(n, c)
#endif
    {
#ifndef TSTRING_IS_STD_STRING
        Data_ = Allocate(n);
        TTraits::assign(Data_, n, c);
#endif
    }

    /**
     * Constructs an uninitialized string of size `uninitialized.Size`. The proper
     * way to use this ctor is via `TBasicString::Uninitialized` factory function.
     *
     * @throw std::length_error
     */
    TBasicString(TUninitialized uninitialized) {
#ifdef TSTRING_IS_STD_STRING
        Storage_.resize_uninitialized(uninitialized.Size);
#else
        Data_ = Allocate(uninitialized.Size);
#endif
    }

    TBasicString(const TCharType* b, const TCharType* e)
#ifdef TSTRING_IS_STD_STRING
        : Storage_(b, e - b)
#endif
    {
#ifndef TSTRING_IS_STD_STRING
        Data_ = Allocate(e - b);
        TTraits::copy(Data_, b, e - b);
#endif
    }

    explicit TBasicString(const TBasicStringBuf<TCharType, TTraits> s)
#ifdef TSTRING_IS_STD_STRING
        : Storage_(s.data(), s.size())
#else
        : Data_(Allocate(s.size()))
#endif
    {
#ifndef TSTRING_IS_STD_STRING
        if (0 != s.size()) {
            TTraits::copy(Data_, s.data(), s.size());
        }
#endif
    }

    template <typename Traits>
    explicit inline TBasicString(const std::basic_string_view<TCharType, Traits>& s)
#ifdef TSTRING_IS_STD_STRING
        : Storage_(s)
#endif
    {
#ifndef TSTRING_IS_STD_STRING
        Data_ = Allocate(s.size());
        TTraits::copy(Data_, s.data(), s.size());
#endif
    }

    /**
     * WARN:
     *    Certain invocations of this method will result in link-time error.
     *    You are free to implement corresponding methods in string.cpp if you need them.
     */
    static TBasicString FromAscii(const ::TStringBuf& s) {
        return TBasicString().AppendAscii(s);
    }

    static TBasicString FromUtf8(const ::TStringBuf& s) {
        return TBasicString().AppendUtf8(s);
    }

    static TBasicString FromUtf16(const ::TWtringBuf& s) {
        return TBasicString().AppendUtf16(s);
    }

    static TBasicString Uninitialized(size_t n) {
        return TBasicString(TUninitialized(n));
    }

private:
    template <typename... R>
    static size_t SumLength(const TBasicStringBuf<TCharType, TTraits> s1, const R&... r) noexcept {
        return s1.size() + SumLength(r...);
    }

    template <typename... R>
    static size_t SumLength(const TCharType /*s1*/, const R&... r) noexcept {
        return 1 + SumLength(r...);
    }

    static constexpr size_t SumLength() noexcept {
        return 0;
    }

    template <typename... R>
    static void CopyAll(TCharType* p, const TBasicStringBuf<TCharType, TTraits> s, const R&... r) {
        TTraits::copy(p, s.data(), s.size());
        CopyAll(p + s.size(), r...);
    }

    template <typename... R, class TNextCharType, typename = std::enable_if_t<std::is_same<TCharType, TNextCharType>::value>>
    static void CopyAll(TCharType* p, const TNextCharType s, const R&... r) {
        p[0] = s;
        CopyAll(p + 1, r...);
    }

    static void CopyAll(TCharType*) noexcept {
    }

public:
#ifndef TSTRING_IS_STD_STRING
    // ~~~ Destructor ~~~
    inline ~TBasicString() {
        UnRef();
    }
#endif

    inline void clear() noexcept {
#ifdef TSTRING_IS_STD_STRING
        Storage_.clear();
#else
        if (IsDetached()) {
            TruncNonShared(0);
            return;
        }

        Relink(TDataTraits::GetNull());
#endif
    }

    template <typename... R>
    static inline TBasicString Join(const R&... r) {
#ifdef TSTRING_IS_STD_STRING
        TBasicString s{TUninitialized{SumLength(r...)}};

        TBasicString::CopyAll(s.Storage_.data(), r...);
#else
        TBasicString s;

        s.Data_ = Allocate(SumLength(r...));
        CopyAll(s.Data_, r...);
#endif

        return s;
    }

    // ~~~ Assignment ~~~ : FAMILY0(TBasicString&, assign);
    TBasicString& assign(size_t size, TCharType ch) {
        ReserveAndResize(size);
        std::fill(begin(), vend(), ch);
        return *this;
    }

    TBasicString& assign(const TBasicString& s) {
        TBasicString(s).swap(*this);

        return *this;
    }

    TBasicString& assign(const TBasicString& s, size_t pos, size_t n) {
        return assign(TBasicString(s, pos, n));
    }

    TBasicString& assign(const TCharType* pc) {
        return assign(pc, TBase::StrLen(pc));
    }

    TBasicString& assign(TCharType ch) {
        return assign(&ch, 1);
    }

    TBasicString& assign(const TCharType* pc, size_t len) {
#if defined(address_sanitizer_enabled) || defined(thread_sanitizer_enabled)
        pc = (const TCharType*)HidePointerOrigin((void*)pc);
#endif
#ifdef TSTRING_IS_STD_STRING
        Storage_.assign(pc, len);
#else

        if (Y_LIKELY(IsDetached() && (pc + len <= TBase::begin() || pc >= TBase::end()))) {
            ResizeNonShared(len);
            TTraits::copy(Data_, pc, len);
        } else if (IsDetached() && pc == data() && capacity() >= len) {
            TruncNonShared(len);
        } else {
            Relink(TTraits::copy(Allocate(len), pc, len));
        }
#endif

        return *this;
    }

    TBasicString& assign(const TCharType* first, const TCharType* last) {
        return assign(first, last - first);
    }

    TBasicString& assign(const TCharType* pc, size_t pos, size_t n) {
        return assign(pc + pos, n);
    }

    inline TBasicString& AssignNoAlias(const TCharType* pc, size_t len) {
#ifdef TSTRING_IS_STD_STRING
        assign(pc, len);
#else
        if (IsDetached()) {
            ResizeNonShared(len);
        } else {
            Relink(Allocate(len));
        }

        TTraits::copy(Data_, pc, len);
#endif

        return *this;
    }

    inline TBasicString& AssignNoAlias(const TCharType* b, const TCharType* e) {
#ifdef TSTRING_IS_STD_STRING
        return assign(b, e - b);
#else
        return AssignNoAlias(b, e - b);
#endif
    }

    TBasicString& assign(const TBasicStringBuf<TCharType, TTraits> s) {
        return assign(s.data(), s.size());
    }

    TBasicString& assign(const TBasicStringBuf<TCharType, TTraits> s, size_t spos, size_t sn = TBase::npos) {
        return assign(s.SubString(spos, sn));
    }

    TBasicString& AssignNoAlias(const TBasicStringBuf<TCharType, TTraits> s) {
#ifdef TSTRING_IS_STD_STRING
        return assign(s.data(), s.size());
#else
        return AssignNoAlias(s.data(), s.size());
#endif
    }

    TBasicString& AssignNoAlias(const TBasicStringBuf<TCharType, TTraits> s, size_t spos, size_t sn = TBase::npos) {
#ifdef TSTRING_IS_STD_STRING
        return assign(s.SubString(spos, sn));
#else
        return AssignNoAlias(s.SubString(spos, sn));
#endif
    }

    /**
     * WARN:
     *    Certain invocations of this method will result in link-time error.
     *    You are free to implement corresponding methods in string.cpp if you need them.
     */
    auto AssignAscii(const ::TStringBuf& s) {
        clear();
        return AppendAscii(s);
    }

    auto AssignUtf8(const ::TStringBuf& s) {
        clear();
        return AppendUtf8(s);
    }

    auto AssignUtf16(const ::TWtringBuf& s) {
        clear();
        return AppendUtf16(s);
    }

    TBasicString& operator=(const TBasicString& s) {
        return assign(s);
    }

    TBasicString& operator=(TBasicString&& s) noexcept {
        swap(s);
        return *this;
    }

    TBasicString& operator=(const TBasicStringBuf<TCharType, TTraits> s) {
        return assign(s);
    }

    TBasicString& operator=(std::initializer_list<TCharType> il) {
        return assign(il.begin(), il.end());
    }

    TBasicString& operator=(const TCharType* s) {
        return assign(s);
    }

    TBasicString& operator=(TExplicitType<TCharType> ch) {
        return assign(ch);
    }

    inline void reserve(size_t len) {
#ifdef TSTRING_IS_STD_STRING
        Storage_.reserve(len);
#else
        if (IsDetached()) {
            if (capacity() < len) {
                Data_ = Allocate(length(), len, GetData());
            }
        } else {
            const size_t sufficientLen = Max(length(), len);
            Relink(TTraits::copy(Allocate(length(), sufficientLen, nullptr), Data_, length()));
        }
#endif
    }

    // ~~~ Appending ~~~ : FAMILY0(TBasicString&, append);
    inline TBasicString& append(size_t count, TCharType ch) {
#ifdef TSTRING_IS_STD_STRING
        Storage_.append(count, ch);
#else
        while (count--) {
            append(ch);
        }
#endif

        return *this;
    }

    inline TBasicString& append(const TBasicString& s) {
#ifdef TSTRING_IS_STD_STRING
        Storage_.append(s.Storage_);

        return *this;
#else
        if (&s != this) {
            return AppendNoAlias(s.data(), s.size());
        }

        return append(s.data(), s.size());
#endif
    }

    inline TBasicString& append(const TBasicString& s, size_t pos, size_t n) {
#ifdef TSTRING_IS_STD_STRING
        Storage_.append(s.Storage_, pos, n);

        return *this;
#else
        return append(s.data(), pos, n, s.size());
#endif
    }

    inline TBasicString& append(const TCharType* pc) {
#ifdef TSTRING_IS_STD_STRING
        Storage_.append(pc);

        return *this;
#else
        return append(pc, TBase::StrLen(pc));
#endif
    }

    inline TBasicString& append(TCharType c) {
#ifdef TSTRING_IS_STD_STRING
        Storage_.push_back(c);
#else
        const size_t olen = length();

        ReserveAndResize(olen + 1);
        *(Data_ + olen) = c;
#endif

        return *this;
    }

    inline TBasicString& append(const TCharType* first, const TCharType* last) {
#ifdef TSTRING_IS_STD_STRING
        Storage_.append(first, last);

        return *this;
#else
        return append(first, last - first);
#endif
    }

    inline TBasicString& append(const TCharType* pc, size_t len) {
#ifdef TSTRING_IS_STD_STRING
        Storage_.append(pc, len);

        return *this;
#else
        if (pc + len <= TBase::begin() || pc >= TBase::end()) {
            return AppendNoAlias(pc, len);
        }

        return append(pc, 0, len, len);
#endif
    }

    inline void ReserveAndResize(size_t len) {
#ifdef TSTRING_IS_STD_STRING
        Storage_.resize_uninitialized(len);
#else
        if (IsDetached()) {
            ResizeNonShared(len);
        } else {
            Relink(TTraits::copy(Allocate(len), Data_, Min(len, length())));
        }
#endif
    }

    inline TBasicString& AppendNoAlias(const TCharType* pc, size_t len) {
#ifdef TSTRING_IS_STD_STRING
        return append(pc, len);
#else
        const size_t olen = length();
        const size_t nlen = olen + len;

        ReserveAndResize(nlen);
        TTraits::copy(Data_ + olen, pc, len);
#endif

        return *this;
    }

    TBasicString& AppendNoAlias(const TBasicStringBuf<TCharType, TTraits> s) {
#ifdef TSTRING_IS_STD_STRING
        return append(s.data(), s.size());
#else
        return AppendNoAlias(s.data(), s.size());
#endif
    }

    TBasicString& AppendNoAlias(const TBasicStringBuf<TCharType, TTraits> s, size_t spos, size_t sn = TBase::npos) {
#ifdef TSTRING_IS_STD_STRING
        return append(s.SubString(spos, sn));
#else
        return AppendNoAlias(s.SubString(spos, sn));
#endif
    }

    TBasicString& append(const TBasicStringBuf<TCharType, TTraits> s) {
        return append(s.data(), s.size());
    }

    TBasicString& append(const TBasicStringBuf<TCharType, TTraits> s, size_t spos, size_t sn = TBase::npos) {
        return append(s.SubString(spos, sn));
    }

    inline TBasicString& append(const TCharType* pc, size_t pos, size_t n, size_t pc_len = TBase::npos) {
#ifdef TSTRING_IS_STD_STRING
        return append(pc + pos, Min(n, pc_len - pos));
#else
        return replace(length(), 0, pc, pos, n, pc_len);
#endif
    }

    /**
     * WARN:
     *    Certain invocations of this method will result in link-time error.
     *    You are free to implement corresponding methods in string.cpp if you need them.
     */
    TBasicString& AppendAscii(const ::TStringBuf& s);

    TBasicString& AppendUtf8(const ::TStringBuf& s);

    TBasicString& AppendUtf16(const ::TWtringBuf& s);

    inline void push_back(TCharType c) {
        append(c);
    }

    template <class T>
    TBasicString& operator+=(const T& s) {
        return append(s);
    }

    template <class T>
    friend TBasicString operator*(const TBasicString& s, T count) {
        TBasicString result;

        for (T i = 0; i < count; ++i) {
            result += s;
        }

        return result;
    }

    template <class T>
    TBasicString& operator*=(T count) {
        TBasicString temp;

        for (T i = 0; i < count; ++i) {
            temp += *this;
        }

        swap(temp);

        return *this;
    }

    template <class TCharTraits, class Allocator>
    /* implicit */ operator std::basic_string<TCharType, TCharTraits, Allocator>() const {
        // NB(eeight) MSVC cannot compiler direct reference to TBase::operator std::basic_string<...>
        // so we are using static_cast to force the needed operator call.
        return static_cast<std::basic_string<TCharType, TCharTraits, Allocator>>(
                static_cast<const TBase&>(*this));
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

    friend TBasicString operator+(TBasicString&& s1, const TBasicString& s2) Y_WARN_UNUSED_RESULT {
        s1 += s2;
        return std::move(s1);
    }

    friend TBasicString operator+(const TBasicString& s1, TBasicString&& s2) Y_WARN_UNUSED_RESULT {
        s2.prepend(s1);
        return std::move(s2);
    }

    friend TBasicString operator+(TBasicString&& s1, TBasicString&& s2) Y_WARN_UNUSED_RESULT {
#ifndef TSTRING_IS_STD_STRING
        if (!s1.IsDetached() && s2.IsDetached()) {
            s2.prepend(s1);
            return std::move(s2);
        }
#endif
        s1 += s2;
        return std::move(s1);
    }

    friend TBasicString operator+(TBasicString&& s1, const TBasicStringBuf<TCharType, TTraits> s2) Y_WARN_UNUSED_RESULT {
        s1 += s2;
        return std::move(s1);
    }

    friend TBasicString operator+(TBasicString&& s1, const TCharType* s2) Y_WARN_UNUSED_RESULT {
        s1 += s2;
        return std::move(s1);
    }

    friend TBasicString operator+(TBasicString&& s1, TCharType s2) Y_WARN_UNUSED_RESULT {
        s1 += s2;
        return std::move(s1);
    }

    friend TBasicString operator+ (TExplicitType<TCharType> ch, const TBasicString& s) Y_WARN_UNUSED_RESULT {
        return Join(TCharType(ch), s);
    }

    friend TBasicString operator+(const TBasicString& s1, const TBasicString& s2) Y_WARN_UNUSED_RESULT {
        return Join(s1, s2);
    }

    friend TBasicString operator+(const TBasicString& s1, const TBasicStringBuf<TCharType, TTraits> s2) Y_WARN_UNUSED_RESULT {
        return Join(s1, s2);
    }

    friend TBasicString operator+(const TBasicString& s1, const TCharType* s2) Y_WARN_UNUSED_RESULT {
        return Join(s1, s2);
    }

    friend TBasicString operator+(const TBasicString& s1, TCharType s2) Y_WARN_UNUSED_RESULT {
        return Join(s1, TBasicStringBuf<TCharType, TTraits>(&s2, 1));
    }

    friend TBasicString operator+(const TCharType* s1, TBasicString&& s2) Y_WARN_UNUSED_RESULT {
        s2.prepend(s1);
        return std::move(s2);
    }

    friend TBasicString operator+(const TBasicStringBuf<TCharType, TTraits> s1, TBasicString&& s2) Y_WARN_UNUSED_RESULT {
        s2.prepend(s1);
        return std::move(s2);
    }

    friend TBasicString operator+(const TBasicStringBuf<TCharType, TTraits> s1, const TBasicString& s2) Y_WARN_UNUSED_RESULT {
        return Join(s1, s2);
    }

    friend TBasicString operator+(const TCharType* s1, const TBasicString& s2) Y_WARN_UNUSED_RESULT {
        return Join(s1, s2);
    }

    // ~~~ Prepending ~~~ : FAMILY0(TBasicString&, prepend);
    TBasicString& prepend(const TBasicString& s) {
#ifdef TSTRING_IS_STD_STRING
        Storage_.insert(0, s.Storage_);

        return *this;
#else
        return replace(0, 0, s.Data_, 0, TBase::npos, s.length());
#endif
    }

    TBasicString& prepend(const TBasicString& s, size_t pos, size_t n) {
#ifdef TSTRING_IS_STD_STRING
        Storage_.insert(0, s.Storage_, pos, n);

        return *this;
#else
        return replace(0, 0, s.Data_, pos, n, s.length());
#endif
    }

    TBasicString& prepend(const TCharType* pc) {
#ifdef TSTRING_IS_STD_STRING
        Storage_.insert(0, pc);

        return *this;
#else
        return replace(0, 0, pc);
#endif
    }

    TBasicString& prepend(size_t n, TCharType c) {
#ifdef TSTRING_IS_STD_STRING
        Storage_.insert(size_t(0), n, c);

        return *this;
#else
        return insert(size_t(0), n, c);
#endif
    }

    TBasicString& prepend(TCharType c) {
#ifdef TSTRING_IS_STD_STRING
        Storage_.insert(size_t(0), 1, c);

        return *this;
#else
        return replace(0, 0, &c, 0, 1, 1);
#endif
    }

    TBasicString& prepend(const TBasicStringBuf<TCharType, TTraits> s, size_t spos = 0, size_t sn = TBase::npos) {
#ifdef TSTRING_IS_STD_STRING
        Storage_.insert(0, s, spos, sn);

        return *this;
#else
        return insert(0, s, spos, sn);
#endif
    }

    // ~~~ Insertion ~~~ : FAMILY1(TBasicString&, insert, size_t pos);
    TBasicString& insert(size_t pos, const TBasicString& s) {
#ifdef TSTRING_IS_STD_STRING
        Storage_.insert(pos, s.Storage_);

        return *this;
#else
        return replace(pos, 0, s.Data_, 0, TBase::npos, s.length());
#endif
    }

    TBasicString& insert(size_t pos, const TBasicString& s, size_t pos1, size_t n1) {
#ifdef TSTRING_IS_STD_STRING
        Storage_.insert(pos, s.Storage_, pos1, n1);

        return *this;
#else
        return replace(pos, 0, s.Data_, pos1, n1, s.length());
#endif
    }

    TBasicString& insert(size_t pos, const TCharType* pc) {
#ifdef TSTRING_IS_STD_STRING
        Storage_.insert(pos, pc);

        return *this;
#else
        return replace(pos, 0, pc);
#endif
    }

    TBasicString& insert(size_t pos, const TCharType* pc, size_t len) {
#ifdef TSTRING_IS_STD_STRING
        Storage_.insert(pos, pc, len);

        return *this;
#else
        return insert(pos, TBasicStringBuf<TCharType, TTraits>(pc, len));
#endif
    }

    TBasicString& insert(const_iterator pos, const_iterator b, const_iterator e) {
#ifdef TSTRING_IS_STD_STRING
        Storage_.insert(Storage_.begin() + this->off(pos), b, e);

        return *this;
#else
        return insert(this->off(pos), b, e - b);
#endif
    }

    TBasicString& insert(size_t pos, size_t n, TCharType c) {
#ifdef TSTRING_IS_STD_STRING
        Storage_.insert(pos, n, c);

        return *this;
#else
        if (n == 1) {
            return replace(pos, 0, &c, 0, 1, 1);
        } else {
            return insert(pos, TBasicString(n, c));
        }
#endif
    }

    TBasicString& insert(const_iterator pos, size_t len, TCharType ch) {
#ifdef TSTRING_IS_STD_STRING
        Storage_.insert(this->off(pos), len, ch);

        return *this;
#else
        return this->insert(this->off(pos), len, ch);
#endif
    }

    TBasicString& insert(const_iterator pos, TCharType ch) {
#ifdef TSTRING_IS_STD_STRING
        Storage_.insert(this->off(pos), 1, ch);

        return *this;
#else
        return this->insert(pos, 1, ch);
#endif
    }

    TBasicString& insert(size_t pos, const TBasicStringBuf<TCharType, TTraits> s, size_t spos = 0, size_t sn = TBase::npos) {
#ifdef TSTRING_IS_STD_STRING
        Storage_.insert(pos, s, spos, sn);

        return *this;
#else
        return replace(pos, 0, s, spos, sn);
#endif
    }

    // ~~~ Removing ~~~
    TBasicString& remove(size_t pos, size_t n) {
#ifdef TSTRING_IS_STD_STRING
        Storage_.erase(pos, n);

        return *this;
#else
        return replace(pos, n, TDataTraits::GetNull(), 0, 0, 0);
#endif
    }

    TBasicString& remove(size_t pos = 0) {
        if (pos < length()) {
#ifdef TSTRING_IS_STD_STRING
            Storage_.erase(pos);
#else
            Detach();
            TruncNonShared(pos);
#endif
        }

        return *this;
    }

    TBasicString& erase(size_t pos = 0, size_t n = TBase::npos) {
#ifdef TSTRING_IS_STD_STRING
        Storage_.erase(pos, n);

        return *this;
#else
        return remove(pos, n);
#endif
    }

    TBasicString& erase(const_iterator b, const_iterator e) {
        return erase(this->off(b), e - b);
    }

    TBasicString& erase(const_iterator i) {
        return erase(i, i + 1);
    }

    TBasicString& pop_back() {
        Y_ASSERT(!this->empty());
#ifdef TSTRING_IS_STD_STRING
        Storage_.pop_back();

        return *this;
#else
        return erase(this->length() - 1, 1);
#endif
    }

    // ~~~ replacement ~~~ : FAMILY2(TBasicString&, replace, size_t pos, size_t n);
    TBasicString& replace(size_t pos, size_t n, const TBasicString& s) {
#ifdef TSTRING_IS_STD_STRING
        Storage_.replace(pos, n, s.Storage_);

        return *this;
#else
        return replace(pos, n, s.Data_, 0, TBase::npos, s.length());
#endif
    }

    TBasicString& replace(size_t pos, size_t n, const TBasicString& s, size_t pos1, size_t n1) {
#ifdef TSTRING_IS_STD_STRING
        Storage_.replace(pos, n, s.Storage_, pos1, n1);

        return *this;
#else
        return replace(pos, n, s.Data_, pos1, n1, s.length());
#endif
    }

    TBasicString& replace(size_t pos, size_t n, const TCharType* pc) {
#ifdef TSTRING_IS_STD_STRING
        Storage_.replace(pos, n, pc);

        return *this;
#else
        return replace(pos, n, TBasicStringBuf<TCharType, TTraits>(pc));
#endif
    }

    TBasicString& replace(size_t pos, size_t n, const TCharType* s, size_t len) {
#ifdef TSTRING_IS_STD_STRING
        Storage_.replace(pos, n, s, len);

        return *this;
#else
        return replace(pos, n, s, 0, len, len);
#endif
    }

    TBasicString& replace(size_t pos, size_t n, const TCharType* s, size_t spos, size_t sn) {
#ifdef TSTRING_IS_STD_STRING
        Storage_.replace(pos, n, s + spos, sn - spos);

        return *this;
#else
        return replace(pos, n, s, spos, sn, sn);
#endif
    }

    TBasicString& replace(size_t pos, size_t n1, size_t n2, TCharType c) {
#ifdef TSTRING_IS_STD_STRING
        Storage_.replace(pos, n1, n2, c);

        return *this;
#else
        if (n2 == 1) {
            return replace(pos, n1, &c, 0, 1, 1);
        } else {
            return replace(pos, n1, TBasicString(n2, c));
        }
#endif
    }

    TBasicString& replace(size_t pos, size_t n, const TBasicStringBuf<TCharType, TTraits> s, size_t spos = 0, size_t sn = TBase::npos) {
#ifdef TSTRING_IS_STD_STRING
        Storage_.replace(pos, n, s, spos, sn);

        return *this;
#else
        return replace(pos, n, s.data(), spos, sn, s.size());
#endif
    }

#ifndef TSTRING_IS_STD_STRING
private:
    // ~~~ main driver
    TBasicString& replace(size_t pos, size_t del, const TCharType* pc, size_t pos1, size_t ins, size_t len1) {
        size_t len = length();
        // 'pc' can point to a single character that is not null terminated, so in this case TTraits::length must not be called
        len1 = pc ? (len1 == TBase::npos ? (ins == TBase::npos ? TTraits::length(pc) : NStringPrivate::GetStringLengthWithLimit(pc, ins + pos1)) : len1) : 0;

        pos = Min(pos, len);
        pos1 = Min(pos1, len1);

        del = Min(del, len - pos);
        ins = Min(ins, len1 - pos1);

        if (len - del > this->max_size() - ins) { // len-del+ins -- overflow
            throw std::length_error("TBasicString::replace");
        }

        size_t total = len - del + ins;

        if (!total) {
            clear();
            return *this;
        }

        size_t rem = len - del - pos;

        if (!IsDetached() || (pc && (pc >= Data_ && pc < Data_ + len))) {
            // malloc
            // 1. alias
            // 2. overlapped
            TCharType* temp = Allocate(total);
            TTraits::copy(temp, Data_, pos);
            TTraits::copy(temp + pos, pc + pos1, ins);
            TTraits::copy(temp + pos + ins, Data_ + pos + del, rem);
            Relink(temp);
        } else if (reserve() < total) {
            // realloc (increasing)
            // 3. not enough room
            Data_ = Allocate(total, GetData());
            TTraits::move(Data_ + pos + ins, Data_ + pos + del, rem);
            TTraits::copy(Data_ + pos, pc + pos1, ins);
        } else {
            // 1. not alias
            // 2. not overlapped
            // 3. enough room
            // 4. not too much room
            TTraits::move(Data_ + pos + ins, Data_ + pos + del, rem);
            TTraits::copy(Data_ + pos, pc + pos1, ins);
            //GetData()->SetLength(total);
            TruncNonShared(total);
        }

        return *this;
    }
#endif

public:
    void swap(TBasicString& s) noexcept {
#ifdef TSTRING_IS_STD_STRING
        std::swap(Storage_, s.Storage_);
#else
        DoSwap(Data_, s.Data_);
#endif
    }

    /**
     * @returns                         String suitable for debug printing (like Python's `repr()`).
     *                                  Format of the string is unspecified and may be changed over time.
     */
    TBasicString Quote() const {
        extern TBasicString EscapeC(const TBasicString&);

        return TBasicString() + '"' + EscapeC(*this) + '"';
    }

    /**
     * Modifies the case of the string, depending on the operation.
     * @return false if no changes have been made.
     *
     * @warning when the value_type is char, these methods will not work with non-ASCII letters.
     */
    bool to_lower(size_t pos = 0, size_t n = TBase::npos);
    bool to_upper(size_t pos = 0, size_t n = TBase::npos);
    bool to_title(size_t pos = 0, size_t n = TBase::npos);

public:
    /**
     * Modifies the substring of length `n` starting from `pos`, applying `f` to each position and symbol.
     *
     * @return false if no changes have been made.
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
#ifdef TSTRING_IS_STD_STRING
            auto c = f(i, Storage_[i]);

            if (c != Storage_[i]) {
                changed = true;

                Storage_[i] = c;
            }
#else
            auto c = f(i, Data_[i]);
            if (c != Data_[i]) {
                if (!changed) {
                    Detach();
                    changed = true;
                }

                Data_[i] = c;
            }
#endif
        }

        return changed;
    }
};

std::ostream& operator<<(std::ostream&, const TString&);
std::istream& operator>>(std::istream&, TString&);

template<typename TCharType, typename TTraits>
TBasicString<TCharType> to_lower(const TBasicString<TCharType, TTraits>& s) {
    TBasicString<TCharType> ret(s);
    ret.to_lower();
    return ret;
}

template<typename TCharType, typename TTraits>
TBasicString<TCharType> to_upper(const TBasicString<TCharType, TTraits>& s) {
    TBasicString<TCharType> ret(s);
    ret.to_upper();
    return ret;
}

template<typename TCharType, typename TTraits>
TBasicString<TCharType> to_title(const TBasicString<TCharType, TTraits>& s) {
    TBasicString<TCharType> ret(s);
    ret.to_title();
    return ret;
}

namespace std {
    template <>
    struct hash<TString> {
        using argument_type = TString;
        using result_type = size_t;
        inline result_type operator()(argument_type const& s) const noexcept {
            return NHashPrivate::ComputeStringHash(s.data(), s.size());
        }
    };
}
