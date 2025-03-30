#pragma once

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <stlfwd>
#include <string>
#include <string_view>
#include <type_traits>

#include <util/system/compiler.h>
#include <util/system/yassert.h>

#include "iterator.h"
#include "ptr.h"
#include "utility.h"
#include "explicit_type.h"
#include "reserve.h"
#ifndef _LIBCPP_VERSION
    #include "singleton.h"
#endif
#include "strbase.h"
#include "strbuf.h"
#include "string_hash.h"
#include "ylimits.h"

#if defined(address_sanitizer_enabled) || defined(thread_sanitizer_enabled)
    #include "hide_ptr.h"
#endif

template <class TCharType, class TCharTraits, class TAllocator>
void ResizeUninitialized(std::basic_string<TCharType, TCharTraits, TAllocator>& s, size_t len) {
#if defined(_YNDX_LIBCXX_ENABLE_STRING_RESIZE_UNINITIALIZED)
    s.resize_uninitialized(len);
#else
    s.resize(len);
#endif
}

#define Y_NOEXCEPT

template <class T>
class TStringPtrOps {
public:
    static inline void Ref(T* t) noexcept {
        if (t != T::NullStr()) {
            t->Ref();
        }
    }

    static inline void UnRef(T* t) noexcept {
        if (t != T::NullStr()) {
            t->UnRef();
        }
    }

    static inline long RefCount(const T* t) noexcept {
        if (t == T::NullStr()) {
            return -1;
        }

        return t->RefCount();
    }
};

alignas(32) extern const char NULL_STRING_REPR[128];

struct TRefCountHolder {
    TAtomicCounter C = 1;
};

template <class B>
struct TStdString: public TRefCountHolder, public B {
    template <typename... Args>
    inline TStdString(Args&&... args)
        : B(std::forward<Args>(args)...)
    {
    }

    inline bool IsNull() const noexcept {
        return this == NullStr();
    }

    static TStdString* NullStr() noexcept {
#ifdef _LIBCPP_VERSION
        return (TStdString*)NULL_STRING_REPR;
#else
        return Singleton<TStdString>();
#endif
    }

private:
    friend TStringPtrOps<TStdString>;
    inline void Ref() noexcept {
        C.Inc();
    }

    inline void UnRef() noexcept {
        if (C.Val() == 1 || C.Dec() == 0) {
            delete this;
        }
    }

    inline long RefCount() const noexcept {
        return C.Val();
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

    TChar* operator&() {
        return S_.begin() + Pos_;
    }

    const TChar* operator&() const {
        return S_.cbegin() + Pos_;
    }

    TBasicCharRef& operator=(TChar c) {
        Y_ASSERT(Pos_ < S_.size() || (Pos_ == S_.size() && !c));

        S_.Detach()[Pos_] = c;

        return *this;
    }

    TBasicCharRef& operator=(const TBasicCharRef& other) {
        return this->operator=(static_cast<TChar>(other));
    }

    /*
     * WARN:
     * Though references are copyable types according to the standard,
     * the behavior of this explicit default specification is different from the one
     * implemented by the assignment operator above.
     *
     * An attempt to explicitly delete it will break valid invocations like
     * auto c = flag ? s[i] : s[j];
     */
    TBasicCharRef(const TBasicCharRef&) = default;

private:
    TStringType& S_;
    size_t Pos_;
};

template <typename TCharType, typename TTraits>
class TBasicString: public TStringBase<TBasicString<TCharType, TTraits>, TCharType, TTraits> {
public:
    // TODO: Move to private section
    using TBase = TStringBase<TBasicString, TCharType, TTraits>;
    using TStringType = std::basic_string<TCharType, TTraits>;
#ifdef TSTRING_IS_STD_STRING
    using TStorage = TStringType;
    using reference = typename TStorage::reference;
#else
    using TStdStr = TStdString<TStringType>;
    using TStorage = TIntrusivePtr<TStdStr, TStringPtrOps<TStdStr>>;
    using reference = TBasicCharRef<TBasicString>;
#endif
    using char_type = TCharType; // TODO: DROP
    using value_type = TCharType;
    using traits_type = TTraits;

    using iterator = TCharType*;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using typename TBase::const_iterator;
    using typename TBase::const_reference;
    using typename TBase::const_reverse_iterator;

    struct TUninitialized {
        explicit TUninitialized(size_t size)
            : Size(size)
        {
        }

        size_t Size;
    };

    size_t max_size() noexcept {
        static size_t res = TStringType().max_size();

        return res;
    }

protected:
#ifdef TSTRING_IS_STD_STRING
    TStorage Storage_;
#else
    TStorage S_;

    template <typename... A>
    static TStorage Construct(A&&... a) {
        return {new TStdStr(std::forward<A>(a)...), typename TStorage::TNoIncrement()};
    }

    static TStorage Construct() noexcept {
        return TStdStr::NullStr();
    }

    TStdStr& StdStr() noexcept {
        return *S_;
    }

    const TStdStr& StdStr() const noexcept {
        return *S_;
    }

    /**
     * Makes a distinct copy of this string. `IsDetached()` is always true after this call.
     *
     * @throw std::length_error
     */
    void Clone() {
        Construct(StdStr()).Swap(S_);
    }

    size_t RefCount() const noexcept {
        return S_.RefCount();
    }
#endif

public:
    inline const TStringType& ConstRef() const Y_LIFETIME_BOUND {
#ifdef TSTRING_IS_STD_STRING
        return Storage_;
#else
        return StdStr();
#endif
    }

    inline TStringType& MutRef() Y_LIFETIME_BOUND {
#ifdef TSTRING_IS_STD_STRING
        return Storage_;
#else
        Detach();

        return StdStr();
#endif
    }

    inline const_reference operator[](size_t pos) const noexcept Y_LIFETIME_BOUND {
        Y_ASSERT(pos <= length());

        return this->data()[pos];
    }

    inline reference operator[](size_t pos) noexcept Y_LIFETIME_BOUND {
        Y_ASSERT(pos <= length());

#ifdef TSTRING_IS_STD_STRING
        return Storage_[pos];
#else
        return reference(*this, pos);
#endif
    }

    using TBase::back;

    inline reference back() noexcept Y_LIFETIME_BOUND {
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

    inline reference front() noexcept Y_LIFETIME_BOUND {
        Y_ASSERT(!this->empty());

#ifdef TSTRING_IS_STD_STRING
        return Storage_.front();
#else
        return reference(*this, 0);
#endif
    }

    inline size_t length() const noexcept {
        return ConstRef().length();
    }

    inline const TCharType* data() const noexcept Y_LIFETIME_BOUND {
        return ConstRef().data();
    }

    inline const TCharType* c_str() const noexcept Y_LIFETIME_BOUND {
        return ConstRef().c_str();
    }

    // ~~~ STL compatible method to obtain data pointer ~~~
    iterator begin() Y_LIFETIME_BOUND {
        return &*MutRef().begin();
    }

    iterator vend() Y_LIFETIME_BOUND {
        return &*MutRef().end();
    }

    reverse_iterator rbegin() Y_LIFETIME_BOUND {
        return reverse_iterator(vend());
    }

    reverse_iterator rend() Y_LIFETIME_BOUND {
        return reverse_iterator(begin());
    }

    const_iterator begin() const noexcept Y_LIFETIME_BOUND {
        return TBase::begin();
    }
    const_iterator cbegin() const noexcept Y_LIFETIME_BOUND {
        return TBase::cbegin();
    }

    const_iterator cend() const noexcept Y_LIFETIME_BOUND {
        return TBase::cend();
    }

    const_reverse_iterator crbegin() const noexcept Y_LIFETIME_BOUND {
        return TBase::crbegin();
    }

    const_reverse_iterator crend() const noexcept Y_LIFETIME_BOUND {
        return TBase::crend();
    }

    const_iterator end() const noexcept Y_LIFETIME_BOUND {
        return TBase::end();
    }

    const_reverse_iterator rbegin() const noexcept Y_LIFETIME_BOUND {
        return TBase::rbegin();
    }

    const_reverse_iterator rend() const noexcept Y_LIFETIME_BOUND {
        return TBase::rend();
    }

    inline size_t capacity() const noexcept {
#ifdef TSTRING_IS_STD_STRING
        return Storage_.capacity();
#else
        if (S_->IsNull()) {
            return 0;
        }

        return S_->capacity();
#endif
    }

    TCharType* Detach() Y_LIFETIME_BOUND {
#ifdef TSTRING_IS_STD_STRING
        return Storage_.data();
#else
        if (Y_UNLIKELY(!IsDetached())) {
            Clone();
        }

        return (TCharType*)S_->data();
#endif
    }

    bool IsDetached() const {
#ifdef TSTRING_IS_STD_STRING
        return true;
#else
        return 1 == RefCount();
#endif
    }

    // ~~~ Size and capacity ~~~
    TBasicString& resize(size_t n, TCharType c = ' ') Y_LIFETIME_BOUND { // remove or append
        MutRef().resize(n, c);

        return *this;
    }

    // ~~~ Constructor ~~~ : FAMILY0(,TBasicString)
    TBasicString() noexcept
#ifndef TSTRING_IS_STD_STRING
        : S_(Construct())
#endif
    {
    }

    inline explicit TBasicString(::NDetail::TReserveTag rt)
#ifndef TSTRING_IS_STD_STRING
        : S_(Construct<>())
#endif
    {
        reserve(rt.Capacity);
    }

    inline TBasicString(const TBasicString& s)
#ifdef TSTRING_IS_STD_STRING
        : Storage_(s.Storage_)
#else
        : S_(s.S_)
#endif
    {
    }

    inline TBasicString(TBasicString&& s) noexcept
#ifdef TSTRING_IS_STD_STRING
        : Storage_(std::move(s.Storage_))
#else
        : S_(Construct())
#endif
    {
#ifdef TSTRING_IS_STD_STRING
#else
        s.swap(*this);
#endif
    }

    template <typename T, typename A>
    explicit inline TBasicString(const std::basic_string<TCharType, T, A>& s)
        : TBasicString(s.data(), s.size())
    {
    }

    template <typename T, typename A>
    inline TBasicString(std::basic_string<TCharType, T, A>&& s)
#ifdef TSTRING_IS_STD_STRING
        : Storage_(std::move(s))
#else
        : S_(s.empty() ? Construct() : Construct(std::move(s)))
#endif
    {
    }

    TBasicString(const TBasicString& s, size_t pos, size_t n) Y_NOEXCEPT
#ifdef TSTRING_IS_STD_STRING
        : Storage_(s.Storage_, pos, n)
#else
        : S_(n ? Construct(s, pos, n) : Construct())
#endif
    {
    }

    TBasicString(const TCharType* pc)
        : TBasicString(pc, TBase::StrLen(pc))
    {
    }
    TBasicString(std::nullptr_t) = delete;

    TBasicString(const TCharType* pc, size_t n)
#ifdef TSTRING_IS_STD_STRING
        : Storage_(pc, n)
#else
        : S_(n ? Construct(pc, n) : Construct())
#endif
    {
    }
    TBasicString(std::nullptr_t, size_t) = delete;

    TBasicString(const TCharType* pc, size_t pos, size_t n)
        : TBasicString(pc + pos, n)
    {
    }

#ifdef TSTRING_IS_STD_STRING
    explicit TBasicString(TExplicitType<TCharType> c) {
        Storage_.push_back(c);
    }
#else
    explicit TBasicString(TExplicitType<TCharType> c)
        : TBasicString(&c.Value(), 1)
    {
    }
    explicit TBasicString(const reference& c)
        : TBasicString(&c, 1)
    {
    }
#endif

    TBasicString(size_t n, TCharType c)
#ifdef TSTRING_IS_STD_STRING
        : Storage_(n, c)
#else
        : S_(Construct(n, c))
#endif
    {
    }

    /**
     * Constructs an uninitialized string of size `uninitialized.Size`. The proper
     * way to use this ctor is via `TBasicString::Uninitialized` factory function.
     *
     * @throw std::length_error
     */
    TBasicString(TUninitialized uninitialized)
#if !defined(TSTRING_IS_STD_STRING)
        : S_(Construct<>())
#endif
    {
        ReserveAndResize(uninitialized.Size);
    }

    TBasicString(const TCharType* b, const TCharType* e)
        : TBasicString(b, NonNegativeDistance(b, e))
    {
    }

    explicit TBasicString(const TBasicStringBuf<TCharType, TTraits> s)
        : TBasicString(s.data(), s.size())
    {
    }

    template <typename Traits>
    explicit inline TBasicString(const std::basic_string_view<TCharType, Traits>& s)
        : TBasicString(s.data(), s.size())
    {
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
    template <typename T>
    using TJoinParam = std::conditional_t<std::is_same_v<T, TCharType>, TCharType, TBasicStringBuf<TCharType, TTraits>>;

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

    template <typename... R>
    static inline TBasicString JoinImpl(const R&... r) {
        TBasicString s{TUninitialized{SumLength(r...)}};

        TBasicString::CopyAll((TCharType*)s.data(), r...);

        return s;
    }

public:
    Y_REINITIALIZES_OBJECT inline void clear() noexcept {
#ifdef TSTRING_IS_STD_STRING
        Storage_.clear();
#else
        if (IsDetached()) {
            S_->clear();

            return;
        }

        Construct().Swap(S_);
#endif
    }

    template <typename... R>
    static inline TBasicString Join(const R&... r) {
        return JoinImpl(TJoinParam<R>(r)...);
    }

    // ~~~ Assignment ~~~ : FAMILY0(TBasicString&, assign);
    TBasicString& assign(size_t size, TCharType ch) Y_LIFETIME_BOUND {
        ReserveAndResize(size);
        std::fill(begin(), vend(), ch);
        return *this;
    }

    TBasicString& assign(const TBasicString& s) Y_LIFETIME_BOUND {
        TBasicString(s).swap(*this);

        return *this;
    }

    TBasicString& assign(const TBasicString& s, size_t pos, size_t n) Y_LIFETIME_BOUND {
        return assign(TBasicString(s, pos, n));
    }

    TBasicString& assign(const TCharType* pc) Y_LIFETIME_BOUND {
        return assign(pc, TBase::StrLen(pc));
    }

    TBasicString& assign(TCharType ch) Y_LIFETIME_BOUND {
        return assign(&ch, 1);
    }

    TBasicString& assign(const TCharType* pc, size_t len) Y_LIFETIME_BOUND {
#if defined(address_sanitizer_enabled) || defined(thread_sanitizer_enabled)
        pc = (const TCharType*)HidePointerOrigin((void*)pc);
#endif
        if (IsDetached()) {
            MutRef().assign(pc, len);
        } else {
            TBasicString(pc, len).swap(*this);
        }

        return *this;
    }

    TBasicString& assign(const TCharType* first, const TCharType* last) Y_LIFETIME_BOUND {
        return assign(first, NonNegativeDistance(first, last));
    }

    TBasicString& assign(const TCharType* pc, size_t pos, size_t n) Y_LIFETIME_BOUND {
        return assign(pc + pos, n);
    }

    TBasicString& assign(const TBasicStringBuf<TCharType, TTraits> s) Y_LIFETIME_BOUND {
        return assign(s.data(), s.size());
    }

    TBasicString& assign(const TBasicStringBuf<TCharType, TTraits> s, size_t spos, size_t sn = TBase::npos) Y_LIFETIME_BOUND {
        return assign(s.SubString(spos, sn));
    }

    inline TBasicString& AssignNoAlias(const TCharType* pc, size_t len) Y_LIFETIME_BOUND {
        return assign(pc, len);
    }

    inline TBasicString& AssignNoAlias(const TCharType* b, const TCharType* e) Y_LIFETIME_BOUND {
        return AssignNoAlias(b, e - b);
    }

    TBasicString& AssignNoAlias(const TBasicStringBuf<TCharType, TTraits> s) Y_LIFETIME_BOUND {
        return AssignNoAlias(s.data(), s.size());
    }

    TBasicString& AssignNoAlias(const TBasicStringBuf<TCharType, TTraits> s, size_t spos, size_t sn = TBase::npos) Y_LIFETIME_BOUND {
        return AssignNoAlias(s.SubString(spos, sn));
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

    TBasicString& operator=(const TBasicString& s) Y_LIFETIME_BOUND {
        return assign(s);
    }

    TBasicString& operator=(TBasicString&& s) noexcept Y_LIFETIME_BOUND {
        swap(s);
        return *this;
    }

    template <typename T, typename A>
    TBasicString& operator=(std::basic_string<TCharType, T, A>&& s) noexcept Y_LIFETIME_BOUND {
        TBasicString(std::move(s)).swap(*this);

        return *this;
    }

    TBasicString& operator=(const TBasicStringBuf<TCharType, TTraits> s) Y_LIFETIME_BOUND {
        return assign(s);
    }

    TBasicString& operator=(std::initializer_list<TCharType> il) Y_LIFETIME_BOUND {
        return assign(il.begin(), il.end());
    }

    TBasicString& operator=(const TCharType* s) Y_LIFETIME_BOUND {
        return assign(s);
    }
    TBasicString& operator=(std::nullptr_t) Y_LIFETIME_BOUND = delete;

    TBasicString& operator=(TExplicitType<TCharType> ch) Y_LIFETIME_BOUND {
        return assign(ch);
    }

    inline void reserve(size_t len) {
        MutRef().reserve(len);
    }

    // ~~~ Appending ~~~ : FAMILY0(TBasicString&, append);
    inline TBasicString& append(size_t count, TCharType ch) Y_LIFETIME_BOUND {
        MutRef().append(count, ch);

        return *this;
    }

    inline TBasicString& append(const TBasicString& s) Y_LIFETIME_BOUND {
        MutRef().append(s.ConstRef());

        return *this;
    }

    inline TBasicString& append(const TBasicString& s, size_t pos, size_t n) Y_LIFETIME_BOUND {
        MutRef().append(s.ConstRef(), pos, n);

        return *this;
    }

    inline TBasicString& append(const TCharType* pc) Y_NOEXCEPT Y_LIFETIME_BOUND {
        MutRef().append(pc);

        return *this;
    }

    inline TBasicString& append(TCharType c) Y_LIFETIME_BOUND {
        MutRef().push_back(c);

        return *this;
    }

    inline TBasicString& append(const TCharType* first, const TCharType* last) Y_LIFETIME_BOUND {
        MutRef().append(first, last);

        return *this;
    }

    inline TBasicString& append(const TCharType* pc, size_t len) Y_LIFETIME_BOUND {
        MutRef().append(pc, len);

        return *this;
    }

    inline void ReserveAndResize(size_t len) {
        ::ResizeUninitialized(MutRef(), len);
    }

    TBasicString& AppendNoAlias(const TCharType* pc, size_t len) Y_LIFETIME_BOUND {
        if (len) {
            auto s = this->size();

            ReserveAndResize(s + len);
            memcpy(&*(begin() + s), pc, len * sizeof(*pc));
        }

        return *this;
    }

    TBasicString& AppendNoAlias(const TBasicStringBuf<TCharType, TTraits> s) Y_LIFETIME_BOUND {
        return AppendNoAlias(s.data(), s.size());
    }

    TBasicString& AppendNoAlias(const TBasicStringBuf<TCharType, TTraits> s, size_t spos, size_t sn = TBase::npos) Y_LIFETIME_BOUND {
        return AppendNoAlias(s.SubString(spos, sn));
    }

    TBasicString& append(const TBasicStringBuf<TCharType, TTraits> s) Y_LIFETIME_BOUND {
        return append(s.data(), s.size());
    }

    TBasicString& append(const TBasicStringBuf<TCharType, TTraits> s, size_t spos, size_t sn = TBase::npos) Y_LIFETIME_BOUND {
        return append(s.SubString(spos, sn));
    }

    TBasicString& append(const TCharType* pc, size_t pos, size_t n, size_t pc_len = TBase::npos) Y_LIFETIME_BOUND {
        return append(pc + pos, Min(n, pc_len - pos));
    }

    /**
     * WARN:
     *    Certain invocations of this method will result in link-time error.
     *    You are free to implement corresponding methods in string.cpp if you need them.
     */
    TBasicString& AppendAscii(const ::TStringBuf& s) Y_LIFETIME_BOUND;

    TBasicString& AppendUtf8(const ::TStringBuf& s) Y_LIFETIME_BOUND;

    TBasicString& AppendUtf16(const ::TWtringBuf& s) Y_LIFETIME_BOUND;

    inline void push_back(TCharType c) {
        // TODO
        append(c);
    }

    template <class T>
    TBasicString& operator+=(const T& s) Y_LIFETIME_BOUND {
        return append(s);
    }

    template <class T>
    friend TBasicString operator*(const TBasicString& s, T count) {
        static_assert(std::is_integral<T>::value, "Integral type required.");

        TBasicString result;

        if (count > 0) {
            result.reserve(s.length() * count);
        }

        for (T i = 0; i < count; ++i) {
            result += s;
        }

        return result;
    }

    template <class T>
    TBasicString& operator*=(T count) Y_LIFETIME_BOUND {
        static_assert(std::is_integral<T>::value, "Integral type required.");

        TBasicString temp;

        if (count > 0) {
            temp.reserve(length() * count);
        }

        for (T i = 0; i < count; ++i) {
            temp += *this;
        }

        swap(temp);

        return *this;
    }

    operator const TStringType&() const noexcept Y_LIFETIME_BOUND {
        return this->ConstRef();
    }

    template <typename T, typename = std::enable_if_t<std::is_same_v<T, TStringType>>>
        operator T&() & Y_LIFETIME_BOUND {
        return this->MutRef();
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
#if 0 && !defined(TSTRING_IS_STD_STRING)
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

    friend TBasicString operator+(TExplicitType<TCharType> ch, const TBasicString& s) Y_WARN_UNUSED_RESULT {
        return Join(TCharType(ch), s);
    }

    friend TBasicString operator+(TExplicitType<TCharType> ch, TBasicString&& s) Y_WARN_UNUSED_RESULT {
        s.prepend(ch);
        return std::move(s);
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

    friend TBasicString operator+(std::basic_string<TCharType, TTraits> l, TBasicString r) {
        return std::move(l) + r.ConstRef();
    }

    friend TBasicString operator+(TBasicString l, std::basic_string<TCharType, TTraits> r) {
        return l.ConstRef() + std::move(r);
    }

    // ~~~ Prepending ~~~ : FAMILY0(TBasicString&, prepend);
    TBasicString& prepend(const TBasicString& s) Y_LIFETIME_BOUND {
        MutRef().insert(0, s.ConstRef());

        return *this;
    }

    TBasicString& prepend(const TBasicString& s, size_t pos, size_t n) Y_LIFETIME_BOUND {
        MutRef().insert(0, s.ConstRef(), pos, n);

        return *this;
    }

    TBasicString& prepend(const TCharType* pc) Y_LIFETIME_BOUND {
        MutRef().insert(0, pc);

        return *this;
    }

    TBasicString& prepend(size_t n, TCharType c) Y_LIFETIME_BOUND {
        MutRef().insert(size_t(0), n, c);

        return *this;
    }

    TBasicString& prepend(TCharType c) Y_LIFETIME_BOUND {
        MutRef().insert(size_t(0), 1, c);

        return *this;
    }

    TBasicString& prepend(const TBasicStringBuf<TCharType, TTraits> s, size_t spos = 0, size_t sn = TBase::npos) Y_LIFETIME_BOUND {
        return insert(0, s, spos, sn);
    }

    // ~~~ Insertion ~~~ : FAMILY1(TBasicString&, insert, size_t pos);
    TBasicString& insert(size_t pos, const TBasicString& s) Y_LIFETIME_BOUND {
        MutRef().insert(pos, s.ConstRef());

        return *this;
    }

    TBasicString& insert(size_t pos, const TBasicString& s, size_t pos1, size_t n1) Y_LIFETIME_BOUND {
        MutRef().insert(pos, s.ConstRef(), pos1, n1);

        return *this;
    }

    TBasicString& insert(size_t pos, const TCharType* pc) Y_LIFETIME_BOUND {
        MutRef().insert(pos, pc);

        return *this;
    }

    TBasicString& insert(size_t pos, const TCharType* pc, size_t len) Y_LIFETIME_BOUND {
        MutRef().insert(pos, pc, len);

        return *this;
    }

    TBasicString& insert(const_iterator pos, const_iterator b, const_iterator e) Y_LIFETIME_BOUND {
#ifdef TSTRING_IS_STD_STRING
        Storage_.insert(Storage_.begin() + this->off(pos), b, e);

        return *this;
#else
        return insert(this->off(pos), b, e - b);
#endif
    }

    TBasicString& insert(size_t pos, size_t n, TCharType c) Y_LIFETIME_BOUND {
        MutRef().insert(pos, n, c);

        return *this;
    }

    TBasicString& insert(const_iterator pos, size_t len, TCharType ch) Y_LIFETIME_BOUND {
        return this->insert(this->off(pos), len, ch);
    }

    TBasicString& insert(const_iterator pos, TCharType ch) Y_LIFETIME_BOUND {
        return this->insert(pos, 1, ch);
    }

    TBasicString& insert(size_t pos, const TBasicStringBuf<TCharType, TTraits> s, size_t spos = 0, size_t sn = TBase::npos) Y_LIFETIME_BOUND {
        MutRef().insert(pos, s, spos, sn);

        return *this;
    }

    // ~~~ Removing ~~~
    TBasicString& remove(size_t pos, size_t n) Y_NOEXCEPT Y_LIFETIME_BOUND {
        if (pos < length()) {
            MutRef().erase(pos, n);
        }

        return *this;
    }

    TBasicString& remove(size_t pos = 0) Y_NOEXCEPT Y_LIFETIME_BOUND {
        if (pos < length()) {
            MutRef().erase(pos);
        }

        return *this;
    }

    TBasicString& erase(size_t pos = 0, size_t n = TBase::npos) Y_NOEXCEPT Y_LIFETIME_BOUND {
        MutRef().erase(pos, n);

        return *this;
    }

    TBasicString& erase(const_iterator b, const_iterator e) Y_NOEXCEPT Y_LIFETIME_BOUND {
        return erase(this->off(b), e - b);
    }

    TBasicString& erase(const_iterator i) Y_NOEXCEPT Y_LIFETIME_BOUND {
        return erase(i, i + 1);
    }

    TBasicString& pop_back() Y_NOEXCEPT Y_LIFETIME_BOUND {
        Y_ASSERT(!this->empty());

        MutRef().pop_back();

        return *this;
    }

    // ~~~ replacement ~~~ : FAMILY2(TBasicString&, replace, size_t pos, size_t n);
    TBasicString& replace(size_t pos, size_t n, const TBasicString& s) Y_NOEXCEPT Y_LIFETIME_BOUND {
        MutRef().replace(pos, n, s.ConstRef());

        return *this;
    }

    TBasicString& replace(size_t pos, size_t n, const TBasicString& s, size_t pos1, size_t n1) Y_NOEXCEPT Y_LIFETIME_BOUND {
        MutRef().replace(pos, n, s.ConstRef(), pos1, n1);

        return *this;
    }

    TBasicString& replace(size_t pos, size_t n, const TCharType* pc) Y_NOEXCEPT Y_LIFETIME_BOUND {
        MutRef().replace(pos, n, pc);

        return *this;
    }

    TBasicString& replace(size_t pos, size_t n, const TCharType* s, size_t len) Y_NOEXCEPT Y_LIFETIME_BOUND {
        MutRef().replace(pos, n, s, len);

        return *this;
    }

    TBasicString& replace(size_t pos, size_t n, const TCharType* s, size_t spos, size_t sn) Y_NOEXCEPT Y_LIFETIME_BOUND {
        MutRef().replace(pos, n, s + spos, sn - spos);

        return *this;
    }

    TBasicString& replace(size_t pos, size_t n1, size_t n2, TCharType c) Y_NOEXCEPT Y_LIFETIME_BOUND {
        MutRef().replace(pos, n1, n2, c);

        return *this;
    }

    TBasicString& replace(size_t pos, size_t n, const TBasicStringBuf<TCharType, TTraits> s, size_t spos = 0, size_t sn = TBase::npos) Y_NOEXCEPT Y_LIFETIME_BOUND {
        MutRef().replace(pos, n, s, spos, sn);

        return *this;
    }

    void swap(TBasicString& s) noexcept {
#ifdef TSTRING_IS_STD_STRING
        std::swap(Storage_, s.Storage_);
#else
        S_.Swap(s.S_);
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

    constexpr const TCharType* Data() const noexcept = delete;
    constexpr size_t Size() noexcept = delete;
    Y_PURE_FUNCTION constexpr bool Empty() const noexcept = delete;

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
            auto c = f(i, data()[i]);
            if (c != data()[i]) {
                if (!changed) {
                    Detach();
                    changed = true;
                }

                begin()[i] = c;
            }
#endif
        }

        return changed;
    }
};

std::ostream& operator<<(std::ostream&, const TString&);
std::istream& operator>>(std::istream&, TString&);

template <typename TCharType, typename TTraits>
TBasicString<TCharType> to_lower(const TBasicString<TCharType, TTraits>& s) {
    TBasicString<TCharType> ret(s);
    ret.to_lower();
    return ret;
}

template <typename TCharType, typename TTraits>
TBasicString<TCharType> to_upper(const TBasicString<TCharType, TTraits>& s) {
    TBasicString<TCharType> ret(s);
    ret.to_upper();
    return ret;
}

template <typename TCharType, typename TTraits>
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
} // namespace std

#undef Y_NOEXCEPT

template <class S>
inline S LegacySubstr(const S& s, size_t pos, size_t n = S::npos) {
    size_t len = s.length();

    pos = Min(pos, len);
    n = Min(n, len - pos);

    return S(s, pos, n);
}

template <typename S, typename... Args>
inline S&& LegacyReplace(S&& s, size_t pos, Args&&... args) {
    if (pos <= s.length()) {
        s.replace(pos, std::forward<Args>(args)...);
    }

    return s;
}

template <typename S, typename... Args>
inline S&& LegacyErase(S&& s, size_t pos, Args&&... args) {
    if (pos <= s.length()) {
        s.erase(pos, std::forward<Args>(args)...);
    }

    return s;
}

inline const char* LegacyStr(const char* s) noexcept {
    return s ? s : "";
}

// interop
template <class TCharType, class TTraits>
auto& MutRef(TBasicString<TCharType, TTraits>& s Y_LIFETIME_BOUND) {
    return s.MutRef();
}

template <class TCharType, class TTraits>
const auto& ConstRef(const TBasicString<TCharType, TTraits>& s Y_LIFETIME_BOUND) noexcept {
    return s.ConstRef();
}

template <class TCharType, class TCharTraits, class TAllocator>
auto& MutRef(std::basic_string<TCharType, TCharTraits, TAllocator>& s Y_LIFETIME_BOUND) noexcept {
    return s;
}

template <class TCharType, class TCharTraits, class TAllocator>
const auto& ConstRef(const std::basic_string<TCharType, TCharTraits, TAllocator>& s Y_LIFETIME_BOUND) noexcept {
    return s;
}

template <class TCharType, class TTraits>
void ResizeUninitialized(TBasicString<TCharType, TTraits>& s, size_t len) {
    s.ReserveAndResize(len);
}
