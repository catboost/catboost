#pragma once

#include <util/generic/string.h>

template <typename TCharType, typename TTraits = std::char_traits<TCharType>>
class Y_EMPTY_BASES TBasicCowString: public TStringBase<TBasicCowString<TCharType, TTraits>, TCharType, TTraits>,
                                     public TStdStringCompatibilityBase<TBasicString<TCharType, TTraits>, TCharType, TTraits> {
public:
    // TODO: Move to private section
    using TBase = TStringBase<TBasicCowString, TCharType, TTraits>;
    using TStringType = std::basic_string<TCharType, TTraits>;
    using TStdStr = TStdString<TStringType>;
    using TStorage = TIntrusivePtr<TStdStr, TStringPtrOps<TStdStr>>;
    using reference = TBasicCharRef<TBasicCowString>;
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

public:
    inline const TStringType& ConstRef() const Y_LIFETIME_BOUND {
        return StdStr();
    }

    inline TStringType& MutRef() Y_LIFETIME_BOUND {
        Detach();

        return StdStr();
    }

    inline const_reference operator[](size_t pos) const noexcept Y_LIFETIME_BOUND {
        Y_ASSERT(pos <= length());

        return this->data()[pos];
    }

    inline reference operator[](size_t pos) noexcept Y_LIFETIME_BOUND {
        Y_ASSERT(pos <= length());

        return reference(*this, pos);
    }

    using TBase::back;

    inline reference back() noexcept Y_LIFETIME_BOUND {
        Y_ASSERT(!this->empty());

        if (Y_UNLIKELY(this->empty())) {
            return reference(*this, 0);
        }

        return reference(*this, length() - 1);
    }

    using TBase::front;

    inline reference front() noexcept Y_LIFETIME_BOUND {
        Y_ASSERT(!this->empty());

        return reference(*this, 0);
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

    iterator end() Y_LIFETIME_BOUND {
        return &*MutRef().end();
    }

    reverse_iterator rbegin() Y_LIFETIME_BOUND {
        return reverse_iterator(end());
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
        if (S_->IsNull()) {
            return 0;
        }

        return S_->capacity();
    }

    TCharType* Detach() Y_LIFETIME_BOUND {
        if (Y_UNLIKELY(!IsDetached())) {
            Clone();
        }

        return (TCharType*)S_->data();
    }

    bool IsDetached() const {
        return 1 == RefCount();
    }

    // ~~~ Size and capacity ~~~
    TBasicCowString& resize(size_t n, TCharType c = ' ') Y_LIFETIME_BOUND { // remove or append
        MutRef().resize(n, c);

        return *this;
    }

    // ~~~ Constructor ~~~ : FAMILY0(,TBasicCowString)
    TBasicCowString() noexcept
        : S_(Construct())
    {
    }

    inline explicit TBasicCowString(::NDetail::TReserveTag rt)
        : S_(Construct<>())
    {
        reserve(rt.Capacity);
    }

    inline TBasicCowString(const TBasicCowString& s)
        : S_(s.S_)
    {
    }

    inline TBasicCowString(TBasicCowString&& s) noexcept
        : S_(Construct())
    {
        s.swap(*this);
    }

    template <typename T, typename A>
    explicit inline TBasicCowString(const std::basic_string<TCharType, T, A>& s)
        : TBasicCowString(s.data(), s.size())
    {
    }

    template <typename T, typename A>
    inline TBasicCowString(std::basic_string<TCharType, T, A>&& s)
        : S_(s.empty() ? Construct() : Construct(std::move(s)))
    {
    }

    TBasicCowString(const TBasicCowString& s, size_t pos, size_t n)
        : S_(n ? Construct(s, pos, n) : Construct())
    {
    }

    TBasicCowString(const TCharType* pc)
        : TBasicCowString(pc, TBase::StrLen(pc))
    {
    }
    TBasicCowString(std::nullptr_t) = delete;

    TBasicCowString(const TCharType* pc, size_t n)
        : S_(n ? Construct(pc, n) : Construct())
    {
    }
    TBasicCowString(std::nullptr_t, size_t) = delete;

    TBasicCowString(const TCharType* pc, size_t pos, size_t n)
        : TBasicCowString(pc + pos, n)
    {
    }

    explicit TBasicCowString(TExplicitType<TCharType> c)
        : TBasicCowString(&c.Value(), 1)
    {
    }
    explicit TBasicCowString(const reference& c)
        : TBasicCowString(&c, 1)
    {
    }

    TBasicCowString(size_t n, TCharType c)
        : S_(Construct(n, c))
    {
    }

    /**
     * Constructs an uninitialized string of size `uninitialized.Size`. The proper
     * way to use this ctor is via `TBasicCowString::Uninitialized` factory function.
     *
     * @throw std::length_error
     */
    TBasicCowString(TUninitialized uninitialized)
        : S_(Construct<>())
    {
        ReserveAndResize(uninitialized.Size);
    }

    TBasicCowString(const TCharType* b, const TCharType* e)
        : TBasicCowString(b, NonNegativeDistance(b, e))
    {
    }

    explicit TBasicCowString(const TBasicStringBuf<TCharType, TTraits> s)
        : TBasicCowString(s.data(), s.size())
    {
    }

    template <typename Traits>
    explicit inline TBasicCowString(const std::basic_string_view<TCharType, Traits>& s)
        : TBasicCowString(s.data(), s.size())
    {
    }

    /**
     * WARN:
     *    Certain invocations of this method will result in link-time error.
     *    You are free to implement corresponding methods in string.cpp if you need them.
     */
    static TBasicCowString FromAscii(const ::TStringBuf& s) {
        return TBasicCowString().AppendAscii(s);
    }

    static TBasicCowString FromUtf8(const ::TStringBuf& s) {
        return TBasicCowString().AppendUtf8(s);
    }

    static TBasicCowString FromUtf16(const ::TWtringBuf& s) {
        return TBasicCowString().AppendUtf16(s);
    }

    static TBasicCowString Uninitialized(size_t n) {
        return TBasicCowString(TUninitialized(n));
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
    static inline TBasicCowString JoinImpl(const R&... r) {
        TBasicCowString s{TUninitialized{SumLength(r...)}};

        TBasicCowString::CopyAll((TCharType*)s.data(), r...);

        return s;
    }

public:
    Y_REINITIALIZES_OBJECT inline void clear() noexcept {
        if (IsDetached()) {
            S_->clear();

            return;
        }

        Construct().Swap(S_);
    }

    template <typename... R>
    static inline TBasicCowString Join(const R&... r) {
        return JoinImpl(TJoinParam<R>(r)...);
    }

    // ~~~ Assignment ~~~ : FAMILY0(TBasicCowString&, assign);
    TBasicCowString& assign(size_t size, TCharType ch) Y_LIFETIME_BOUND {
        ReserveAndResize(size);
        std::fill(begin(), end(), ch);
        return *this;
    }

    TBasicCowString& assign(const TBasicCowString& s) Y_LIFETIME_BOUND {
        TBasicCowString(s).swap(*this);

        return *this;
    }

    TBasicCowString& assign(const TBasicCowString& s, size_t pos, size_t n) Y_LIFETIME_BOUND {
        return assign(TBasicCowString(s, pos, n));
    }

    TBasicCowString& assign(const TCharType* pc) Y_LIFETIME_BOUND {
        return assign(pc, TBase::StrLen(pc));
    }

    TBasicCowString& assign(TCharType ch) Y_LIFETIME_BOUND {
        return assign(&ch, 1);
    }

    TBasicCowString& assign(const TCharType* pc, size_t len) Y_LIFETIME_BOUND {
#if defined(address_sanitizer_enabled) || defined(thread_sanitizer_enabled)
        pc = (const TCharType*)HidePointerOrigin((void*)pc);
#endif
        if (IsDetached()) {
            MutRef().assign(pc, len);
        } else {
            TBasicCowString(pc, len).swap(*this);
        }

        return *this;
    }

    TBasicCowString& assign(const TCharType* first, const TCharType* last) Y_LIFETIME_BOUND {
        return assign(first, NonNegativeDistance(first, last));
    }

    TBasicCowString& assign(const TCharType* pc, size_t pos, size_t n) Y_LIFETIME_BOUND {
        return assign(pc + pos, n);
    }

    TBasicCowString& assign(const TBasicStringBuf<TCharType, TTraits> s) Y_LIFETIME_BOUND {
        return assign(s.data(), s.size());
    }

    TBasicCowString& assign(const TBasicStringBuf<TCharType, TTraits> s, size_t spos, size_t sn = TBase::npos) Y_LIFETIME_BOUND {
        return assign(s.SubString(spos, sn));
    }

    inline TBasicCowString& AssignNoAlias(const TCharType* pc, size_t len) Y_LIFETIME_BOUND {
        return assign(pc, len);
    }

    inline TBasicCowString& AssignNoAlias(const TCharType* b, const TCharType* e) Y_LIFETIME_BOUND {
        return AssignNoAlias(b, e - b);
    }

    TBasicCowString& AssignNoAlias(const TBasicStringBuf<TCharType, TTraits> s) Y_LIFETIME_BOUND {
        return AssignNoAlias(s.data(), s.size());
    }

    TBasicCowString& AssignNoAlias(const TBasicStringBuf<TCharType, TTraits> s, size_t spos, size_t sn = TBase::npos) Y_LIFETIME_BOUND {
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

    TBasicCowString& operator=(const TBasicCowString& s) Y_LIFETIME_BOUND {
        return assign(s);
    }

    TBasicCowString& operator=(TBasicCowString&& s) noexcept Y_LIFETIME_BOUND {
        swap(s);
        return *this;
    }

    template <typename T, typename A>
    TBasicCowString& operator=(std::basic_string<TCharType, T, A>&& s) noexcept Y_LIFETIME_BOUND {
        TBasicCowString(std::move(s)).swap(*this);

        return *this;
    }

    TBasicCowString& operator=(const TBasicStringBuf<TCharType, TTraits> s) Y_LIFETIME_BOUND {
        return assign(s);
    }

    TBasicCowString& operator=(std::initializer_list<TCharType> il) Y_LIFETIME_BOUND {
        return assign(il.begin(), il.end());
    }

    TBasicCowString& operator=(const TCharType* s) Y_LIFETIME_BOUND {
        return assign(s);
    }
    TBasicCowString& operator=(std::nullptr_t) Y_LIFETIME_BOUND = delete;

    TBasicCowString& operator=(TExplicitType<TCharType> ch) Y_LIFETIME_BOUND {
        return assign(ch);
    }

    inline void reserve(size_t len) {
        MutRef().reserve(len);
    }

    // ~~~ Appending ~~~ : FAMILY0(TBasicCowString&, append);
    inline TBasicCowString& append(size_t count, TCharType ch) Y_LIFETIME_BOUND {
        MutRef().append(count, ch);

        return *this;
    }

    inline TBasicCowString& append(const TBasicCowString& s) Y_LIFETIME_BOUND {
        MutRef().append(s.ConstRef());

        return *this;
    }

    inline TBasicCowString& append(const TBasicCowString& s, size_t pos, size_t n) Y_LIFETIME_BOUND {
        MutRef().append(s.ConstRef(), pos, n);

        return *this;
    }

    inline TBasicCowString& append(const TCharType* pc) Y_LIFETIME_BOUND {
        MutRef().append(pc);

        return *this;
    }

    inline TBasicCowString& append(TCharType c) Y_LIFETIME_BOUND {
        MutRef().push_back(c);

        return *this;
    }

    inline TBasicCowString& append(const TCharType* first, const TCharType* last) Y_LIFETIME_BOUND {
        MutRef().append(first, last);

        return *this;
    }

    inline TBasicCowString& append(const TCharType* pc, size_t len) Y_LIFETIME_BOUND {
        MutRef().append(pc, len);

        return *this;
    }

    inline void ReserveAndResize(size_t len) {
        ::ResizeUninitialized(MutRef(), len);
    }

    TBasicCowString& AppendNoAlias(const TCharType* pc, size_t len) Y_LIFETIME_BOUND {
        if (len) {
            auto s = this->size();

            ReserveAndResize(s + len);
            memcpy(&*(begin() + s), pc, len * sizeof(*pc));
        }

        return *this;
    }

    TBasicCowString& AppendNoAlias(const TBasicStringBuf<TCharType, TTraits> s) Y_LIFETIME_BOUND {
        return AppendNoAlias(s.data(), s.size());
    }

    TBasicCowString& AppendNoAlias(const TBasicStringBuf<TCharType, TTraits> s, size_t spos, size_t sn = TBase::npos) Y_LIFETIME_BOUND {
        return AppendNoAlias(s.SubString(spos, sn));
    }

    TBasicCowString& append(const TBasicStringBuf<TCharType, TTraits> s) Y_LIFETIME_BOUND {
        return append(s.data(), s.size());
    }

    TBasicCowString& append(const TBasicStringBuf<TCharType, TTraits> s, size_t spos, size_t sn = TBase::npos) Y_LIFETIME_BOUND {
        return append(s.SubString(spos, sn));
    }

    TBasicCowString& append(const TCharType* pc, size_t pos, size_t n, size_t pc_len = TBase::npos) Y_LIFETIME_BOUND {
        return append(pc + pos, Min(n, pc_len - pos));
    }

    /**
     * WARN:
     *    Certain invocations of this method will result in link-time error.
     *    You are free to implement corresponding methods in string.cpp if you need them.
     */
    TBasicCowString& AppendAscii(const ::TStringBuf& s) Y_LIFETIME_BOUND;

    TBasicCowString& AppendUtf8(const ::TStringBuf& s) Y_LIFETIME_BOUND;

    TBasicCowString& AppendUtf16(const ::TWtringBuf& s) Y_LIFETIME_BOUND;

    inline void push_back(TCharType c) {
        // TODO
        append(c);
    }

    template <class T>
    TBasicCowString& operator+=(const T& s) Y_LIFETIME_BOUND {
        return append(s);
    }

    template <class T>
    friend TBasicCowString operator*(const TBasicCowString& s, T count) {
        static_assert(std::is_integral<T>::value, "Integral type required.");

        TBasicCowString result;

        if (count > 0) {
            result.reserve(s.length() * count);
        }

        for (T i = 0; i < count; ++i) {
            result += s;
        }

        return result;
    }

    template <class T>
    TBasicCowString& operator*=(T count) Y_LIFETIME_BOUND {
        static_assert(std::is_integral<T>::value, "Integral type required.");

        TBasicCowString temp;

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

    /* We have operator casting TString to `const std::string&` but we explicitly don't support
    * casting TString to `std::string&` since such casting requires detaching TString and therefore
    * modifies TString object. Sometimes compiler might call `operator std::string&`
    * implicitly and it might lead to problems. Check IGNIETFERRO-2155 for details.
    */
    template <typename T, typename = std::enable_if_t<std::is_same_v<T, TStringType>>>
    operator T&() & Y_LIFETIME_BOUND requires false {
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

    friend TBasicCowString operator+(TBasicCowString&& s1, const TBasicCowString& s2) Y_WARN_UNUSED_RESULT {
        s1 += s2;
        return std::move(s1);
    }

    friend TBasicCowString operator+(const TBasicCowString& s1, TBasicCowString&& s2) Y_WARN_UNUSED_RESULT {
        s2.prepend(s1);
        return std::move(s2);
    }

    friend TBasicCowString operator+(TBasicCowString&& s1, TBasicCowString&& s2) Y_WARN_UNUSED_RESULT {
#if 0
        if (!s1.IsDetached() && s2.IsDetached()) {
            s2.prepend(s1);
            return std::move(s2);
        }
#endif
        s1 += s2;
        return std::move(s1);
    }

    friend TBasicCowString operator+(TBasicCowString&& s1, const TBasicStringBuf<TCharType, TTraits> s2) Y_WARN_UNUSED_RESULT {
        s1 += s2;
        return std::move(s1);
    }

    friend TBasicCowString operator+(TBasicCowString&& s1, const TCharType* s2) Y_WARN_UNUSED_RESULT {
        s1 += s2;
        return std::move(s1);
    }

    friend TBasicCowString operator+(TBasicCowString&& s1, TCharType s2) Y_WARN_UNUSED_RESULT {
        s1 += s2;
        return std::move(s1);
    }

    friend TBasicCowString operator+(TExplicitType<TCharType> ch, const TBasicCowString& s) Y_WARN_UNUSED_RESULT {
        return Join(TCharType(ch), s);
    }

    friend TBasicCowString operator+(TExplicitType<TCharType> ch, TBasicCowString&& s) Y_WARN_UNUSED_RESULT {
        s.prepend(ch);
        return std::move(s);
    }

    friend TBasicCowString operator+(const TBasicCowString& s1, const TBasicCowString& s2) Y_WARN_UNUSED_RESULT {
        return Join(s1, s2);
    }

    friend TBasicCowString operator+(const TBasicCowString& s1, const TBasicStringBuf<TCharType, TTraits> s2) Y_WARN_UNUSED_RESULT {
        return Join(s1, s2);
    }

    friend TBasicCowString operator+(const TBasicCowString& s1, const TCharType* s2) Y_WARN_UNUSED_RESULT {
        return Join(s1, s2);
    }

    friend TBasicCowString operator+(const TBasicCowString& s1, TCharType s2) Y_WARN_UNUSED_RESULT {
        return Join(s1, TBasicStringBuf<TCharType, TTraits>(&s2, 1));
    }

    friend TBasicCowString operator+(const TCharType* s1, TBasicCowString&& s2) Y_WARN_UNUSED_RESULT {
        s2.prepend(s1);
        return std::move(s2);
    }

    friend TBasicCowString operator+(const TBasicStringBuf<TCharType, TTraits> s1, TBasicCowString&& s2) Y_WARN_UNUSED_RESULT {
        s2.prepend(s1);
        return std::move(s2);
    }

    friend TBasicCowString operator+(const TBasicStringBuf<TCharType, TTraits> s1, const TBasicCowString& s2) Y_WARN_UNUSED_RESULT {
        return Join(s1, s2);
    }

    friend TBasicCowString operator+(const TCharType* s1, const TBasicCowString& s2) Y_WARN_UNUSED_RESULT {
        return Join(s1, s2);
    }

    friend TBasicCowString operator+(std::basic_string<TCharType, TTraits> l, TBasicCowString r) {
        return std::move(l) + r.ConstRef();
    }

    friend TBasicCowString operator+(TBasicCowString l, std::basic_string<TCharType, TTraits> r) {
        return l.ConstRef() + std::move(r);
    }

    // ~~~ Prepending ~~~ : FAMILY0(TBasicCowString&, prepend);
    TBasicCowString& prepend(const TBasicCowString& s) Y_LIFETIME_BOUND {
        MutRef().insert(0, s.ConstRef());

        return *this;
    }

    TBasicCowString& prepend(const TBasicCowString& s, size_t pos, size_t n) Y_LIFETIME_BOUND {
        MutRef().insert(0, s.ConstRef(), pos, n);

        return *this;
    }

    TBasicCowString& prepend(const TCharType* pc) Y_LIFETIME_BOUND {
        MutRef().insert(0, pc);

        return *this;
    }

    TBasicCowString& prepend(size_t n, TCharType c) Y_LIFETIME_BOUND {
        MutRef().insert(size_t(0), n, c);

        return *this;
    }

    TBasicCowString& prepend(TCharType c) Y_LIFETIME_BOUND {
        MutRef().insert(size_t(0), 1, c);

        return *this;
    }

    TBasicCowString& prepend(const TBasicStringBuf<TCharType, TTraits> s, size_t spos = 0, size_t sn = TBase::npos) Y_LIFETIME_BOUND {
        return insert(0, s, spos, sn);
    }

    // ~~~ Insertion ~~~ : FAMILY1(TBasicCowString&, insert, size_t pos);
    TBasicCowString& insert(size_t pos, const TBasicCowString& s) Y_LIFETIME_BOUND {
        MutRef().insert(pos, s.ConstRef());

        return *this;
    }

    TBasicCowString& insert(size_t pos, const TBasicCowString& s, size_t pos1, size_t n1) Y_LIFETIME_BOUND {
        MutRef().insert(pos, s.ConstRef(), pos1, n1);

        return *this;
    }

    TBasicCowString& insert(size_t pos, const TCharType* pc) Y_LIFETIME_BOUND {
        MutRef().insert(pos, pc);

        return *this;
    }

    TBasicCowString& insert(size_t pos, const TCharType* pc, size_t len) Y_LIFETIME_BOUND {
        MutRef().insert(pos, pc, len);

        return *this;
    }

    TBasicCowString& insert(const_iterator pos, const_iterator b, const_iterator e) Y_LIFETIME_BOUND {
        return insert(this->off(pos), b, e - b);
    }

    TBasicCowString& insert(size_t pos, size_t n, TCharType c) Y_LIFETIME_BOUND {
        MutRef().insert(pos, n, c);

        return *this;
    }

    TBasicCowString& insert(const_iterator pos, size_t len, TCharType ch) Y_LIFETIME_BOUND {
        return this->insert(this->off(pos), len, ch);
    }

    TBasicCowString& insert(const_iterator pos, TCharType ch) Y_LIFETIME_BOUND {
        return this->insert(pos, 1, ch);
    }

    TBasicCowString& insert(size_t pos, const TBasicStringBuf<TCharType, TTraits> s, size_t spos = 0, size_t sn = TBase::npos) Y_LIFETIME_BOUND {
        MutRef().insert(pos, s, spos, sn);

        return *this;
    }

    // ~~~ Removing ~~~
    TBasicCowString& remove(size_t pos, size_t n) Y_LIFETIME_BOUND {
        if (pos < length()) {
            MutRef().erase(pos, n);
        }

        return *this;
    }

    TBasicCowString& remove(size_t pos = 0) Y_LIFETIME_BOUND {
        if (pos < length()) {
            MutRef().erase(pos);
        }

        return *this;
    }

    TBasicCowString& erase(size_t pos = 0, size_t n = TBase::npos) Y_LIFETIME_BOUND {
        MutRef().erase(pos, n);

        return *this;
    }

    TBasicCowString& erase(const_iterator b, const_iterator e) Y_LIFETIME_BOUND {
        return erase(this->off(b), e - b);
    }

    TBasicCowString& erase(const_iterator i) Y_LIFETIME_BOUND {
        return erase(i, i + 1);
    }

    TBasicCowString& pop_back() Y_LIFETIME_BOUND {
        Y_ASSERT(!this->empty());

        MutRef().pop_back();

        return *this;
    }

    // ~~~ replacement ~~~ : FAMILY2(TBasicCowString&, replace, size_t pos, size_t n);
    TBasicCowString& replace(size_t pos, size_t n, const TBasicCowString& s) Y_LIFETIME_BOUND {
        MutRef().replace(pos, n, s.ConstRef());

        return *this;
    }

    TBasicCowString& replace(size_t pos, size_t n, const TBasicCowString& s, size_t pos1, size_t n1) Y_LIFETIME_BOUND {
        MutRef().replace(pos, n, s.ConstRef(), pos1, n1);

        return *this;
    }

    TBasicCowString& replace(size_t pos, size_t n, const TCharType* pc) Y_LIFETIME_BOUND {
        MutRef().replace(pos, n, pc);

        return *this;
    }

    TBasicCowString& replace(size_t pos, size_t n, const TCharType* s, size_t len) Y_LIFETIME_BOUND {
        MutRef().replace(pos, n, s, len);

        return *this;
    }

    TBasicCowString& replace(size_t pos, size_t n, const TCharType* s, size_t spos, size_t sn) Y_LIFETIME_BOUND {
        MutRef().replace(pos, n, s + spos, sn - spos);

        return *this;
    }

    TBasicCowString& replace(size_t pos, size_t n1, size_t n2, TCharType c) Y_LIFETIME_BOUND {
        MutRef().replace(pos, n1, n2, c);

        return *this;
    }

    TBasicCowString& replace(size_t pos, size_t n, const TBasicStringBuf<TCharType, TTraits> s, size_t spos = 0, size_t sn = TBase::npos) Y_LIFETIME_BOUND {
        MutRef().replace(pos, n, s, spos, sn);

        return *this;
    }

    void swap(TBasicCowString& s) noexcept {
        S_.Swap(s.S_);
    }

    /**
     * @returns                         String suitable for debug printing (like Python's `repr()`).
     *                                  Format of the string is unspecified and may be changed over time.
     */
    TBasicCowString Quote() const {
        extern TBasicCowString EscapeC(const TBasicCowString&);

        return TBasicCowString() + '"' + EscapeC(*this) + '"';
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
            auto c = f(i, data()[i]);
            if (c != data()[i]) {
                if (!changed) {
                    Detach();
                    changed = true;
                }

                begin()[i] = c;
            }
        }

        return changed;
    }
};

using TCowString = TBasicCowString<char>;
using TUtf16CowString = TBasicCowString<wchar16>;
using TUtf32CowString = TBasicCowString<wchar32>;

std::ostream& operator<<(std::ostream&, const TCowString&);
std::istream& operator>>(std::istream&, TCowString&);

template <typename TCharType, typename TTraits>
TBasicCowString<TCharType> to_lower(const TBasicCowString<TCharType, TTraits>& s) {
    TBasicCowString<TCharType> ret(s);
    ret.to_lower();
    return ret;
}

template <typename TCharType, typename TTraits>
TBasicCowString<TCharType> to_upper(const TBasicCowString<TCharType, TTraits>& s) {
    TBasicCowString<TCharType> ret(s);
    ret.to_upper();
    return ret;
}

template <typename TCharType, typename TTraits>
TBasicCowString<TCharType> to_title(const TBasicCowString<TCharType, TTraits>& s) {
    TBasicCowString<TCharType> ret(s);
    ret.to_title();
    return ret;
}

namespace std {
    template <>
    struct hash<TCowString> {
        using argument_type = TCowString;
        using result_type = size_t;
        inline result_type operator()(argument_type const& s) const noexcept {
            return NHashPrivate::ComputeStringHash(s.data(), s.size());
        }
    };
} // namespace std

// interop
template <class TCharType, class TTraits>
auto& MutRef(TBasicCowString<TCharType, TTraits>& s Y_LIFETIME_BOUND) {
    return s.MutRef();
}

template <class TCharType, class TTraits>
const auto& ConstRef(const TBasicCowString<TCharType, TTraits>& s Y_LIFETIME_BOUND) noexcept {
    return s.ConstRef();
}

template <class TCharType, class TTraits>
void ResizeUninitialized(TBasicCowString<TCharType, TTraits>& s, size_t len) {
    s.ReserveAndResize(len);
}
