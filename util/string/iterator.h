#pragma once

#include "cast.h"
#include "split.h"

#include <util/generic/algorithm.h>
#include <util/generic/iterator.h>
#include <util/generic/typetraits.h>

#include <utility>
#include <stlfwd>

//
// Provides convenient way to split strings.
// Check iterator_ut.cpp to get examples of usage.

namespace NPrivate {
    Y_HAS_MEMBER(push_back, PushBack);
    Y_HAS_MEMBER(insert, Insert);

    /**
     * This one is needed here so that `std::string_view -> std::string_view`
     * conversion works.
     */
    template<class Src, class Dst>
    inline void DoFromString(const Src& src, Dst* dst) {
        *dst = ::FromString<Dst>(src);
    }

    template<class T>
    inline void DoFromString(const T& src, T* dst) noexcept {
        *dst = src;
    }

    template<class Src, class Dst>
    inline Y_WARN_UNUSED_RESULT bool TryDoFromString(const Src& src, Dst* dst) noexcept {
        return ::TryFromString(src, *dst);
    }

    template<class T>
    inline Y_WARN_UNUSED_RESULT bool TryDoFromString(const T& src, T* dst) noexcept {
        *dst = src;
        return true;
    }

    /**
     * Consumer that places provided elements into a container. Not using
     * `emplace(iterator)` for efficiency.
     */
    template <class Container>
    struct TContainerConsumer {
        using value_type = typename Container::value_type;

        TContainerConsumer(Container* c)
            : C_(c)
        {
        }

        template<class StringBuf>
        void operator()(StringBuf e) const {
            this->operator()(C_, e);
        }

    private:
        template<class OtherContainer, class StringBuf>
        auto operator()(OtherContainer* c, StringBuf e) const -> decltype(c->emplace_back()) {
            return c->emplace_back(value_type(e));
        }

        template<class OtherContainer, class StringBuf>
        auto operator()(OtherContainer* c, StringBuf e) const -> decltype(c->emplace()) {
            return c->emplace(value_type(e));
        }

        Container* C_;
    };

    /**
     * Consumer that converts provided elements via `FromString` and places them
     * into a container.
     */
    template <class Container>
    struct TContainerConvertingConsumer {
        using value_type = typename Container::value_type;

        TContainerConvertingConsumer(Container* c)
            : C_(c)
        {
        }

        template <class StringBuf>
        void operator()(StringBuf e) const {
            this->operator()(C_, e);
        }

    private:
        template<class OtherContainer, class StringBuf>
        auto operator()(OtherContainer* c, StringBuf e) const -> decltype(c->emplace_back()) {
            value_type v;
            ::NPrivate::DoFromString(e, &v);
            return c->emplace_back(std::move(v));
        }

        template<class OtherContainer, class StringBuf>
        auto operator()(OtherContainer* c, StringBuf e) const -> decltype(c->emplace()) {
            value_type v;
            ::NPrivate::DoFromString(e, &v);
            return c->emplace(std::move(v));
        }

        Container* C_;
    };

    template <class String>
    struct TStringBufOfImpl {
        using type = TBasicStringBuf<typename String::value_type>;
    };

    template <class Char, class Traits, class Allocator>
    struct TStringBufOfImpl<std::basic_string<Char, Traits, Allocator>> {
        using type = std::basic_string_view<Char, Traits>;
    };

    template <class Char, class Traits>
    struct TStringBufOfImpl<std::basic_string_view<Char, Traits>> {
        using type = std::basic_string_view<Char, Traits>;
    };

    /**
     * Metafunction that returns a string buffer for the given type. This is to
     * make sure that splitting `std::string` returns `std::string_view`.
     */
    template<class String>
    using TStringBufOf = typename TStringBufOfImpl<String>::type;

    template<class Iterator, class StrBuf, class Storage>
    struct TIteratorState {
        inline TIteratorState(const Storage& storage) noexcept
            : Storage_(storage)
            , B(Storage_.Begin())
            , E(Storage_.End())
            , TokS()
            , TokD()
        {
        }

        operator StrBuf() const noexcept {
            return Token();
        }

        inline Iterator TokenStart() const noexcept {
            return TokS;
        }

        inline Iterator TokenDelim() const noexcept {
            return TokD;
        }

        inline Iterator TokenEnd() const noexcept {
            return B;
        }

        Y_PURE_FUNCTION
        inline bool Empty() const noexcept {
            return TokenStart() == TokenDelim();
        }

        inline StrBuf Token() const noexcept {
            return StrBuf(TokenStart(), TokenDelim() - TokenStart());
        }

        inline StrBuf Delim() const noexcept {
            return StrBuf(TokenDelim(), TokenEnd() - TokenDelim());
        }

        Storage Storage_; // TODO: doesn't belong here, terribly broken =(

        Iterator B;
        Iterator E;

        Iterator TokS;
        Iterator TokD;
    };

}

template <class Base>
struct TSplitRange: public Base, public TInputRangeAdaptor<TSplitRange<Base>> {
    using TStrBuf = decltype(std::declval<Base>().Next()->Token());

    template <typename... Args>
    inline TSplitRange(Args&&... args)
        : Base(std::forward<Args>(args)...)
    {
    }

    template <class F>
    inline void Consume(F&& f) {
        for (auto&& it : *this) {
            f(it.Token());
        }
    }

    template<class Container, class = std::enable_if_t<::NPrivate::THasInsert<Container>::value || ::NPrivate::THasPushBack<Container>::value>>
    operator Container() {
        Container result;
        AddTo(&result);
        return result;
    }

    template <class S>
    inline TVector<S> ToList() {
        TVector<S> result;
        for (auto&& it : *this) {
            result.push_back(S(it.Token()));
        }
        return result;
    }

    template <class Container>
    inline void Collect(Container* c) {
        Y_ASSERT(c);
        c->clear();
        AddTo(c);
    }

    template <class Container>
    inline void AddTo(Container* c) {
        Y_ASSERT(c);
        ::NPrivate::TContainerConsumer<Container> consumer(c);
        Consume(consumer);
    }

    template <class Container>
    inline void ParseInto(Container* c) {
        Y_ASSERT(c);
        ::NPrivate::TContainerConvertingConsumer<Container> consumer(c);
        Consume(consumer);
    }

    // TODO: this is actually TryParseInto
    /**
     * Same as `CollectInto`, just doesn't throw.
     *
     * \param[out] args                 Output arguments.
     * \returns                         Whether parsing was successful.
     */
    template <typename... Args>
    inline bool TryCollectInto(Args*... args) noexcept {
        size_t successfullyFilled = 0;
        auto it = this->begin();

        //FIXME: actually, some kind of TryApplyToMany is needed in order to stop iteration upon first failure
        ApplyToMany([&](auto&& arg) {
            if (it != this->end()) {
                if (::NPrivate::TryDoFromString(it->Token(), arg)) {
                    ++successfullyFilled;
                }
                ++it;
            }
        }, args...);

        return successfullyFilled == sizeof...(args) && it == this->end();
    }

    // TODO: this is actually ParseInto
    /**
     * Splits and parses everything that's in this splitter into `args`.
     *
     * Example usage:
     * \code
     * int l, r;
     * StringSplitter("100*200").Split('*').CollectInto(&l, &r);
     * \endcode
     *
     * \param[out] args                 Output arguments.
     * \throws                          If not all items were parsed, or
     *                                  if there were too many items in the split.
     */
    template <typename... Args>
    inline void CollectInto(Args*... args) {
        Y_ENSURE(TryCollectInto<Args...>(args...));
    }

    inline size_t Count() const {
        size_t cnt = 0;
        for (auto&& it : *this) {
            Y_UNUSED(it);
            ++cnt;
        }
        return cnt;
    }
};

// TODO: NPrivate
template <class TIt>
class TExternalOwnerPolicy {
public:
    TExternalOwnerPolicy(TIt b, TIt e)
        : B_(b)
        , E_(e)
    {
    }

    template <class T>
    TExternalOwnerPolicy(T&& t)
        : B_(std::cbegin(t))
        , E_(std::cend(t))
    {
    }

    TIt Begin() const {
        return B_;
    }

    TIt End() const {
        return E_;
    }

private:
    TIt B_;
    TIt E_;
};

// TODO: NPrivate
template <class TStr>
class TCopyOwnerPolicy {
    using TIt = decltype(std::cbegin(TStr()));

public:
    TCopyOwnerPolicy(TIt b, TIt e)
        : B_(b)
        , E_(e)
    {
    }

    TCopyOwnerPolicy(TStr s)
        : S_(std::move(s))
        , B_(std::cbegin(S_))
        , E_(std::cend(S_))
    {
    }

    TCopyOwnerPolicy(const TCopyOwnerPolicy& o)
        : S_(o.S_)
        , B_(std::cbegin(S_))
        , E_(std::cend(S_))
    {
    }

    TCopyOwnerPolicy(TCopyOwnerPolicy&& o)
        : S_(std::move(o.S_))
        , B_(std::cbegin(S_))
        , E_(std::cend(S_))
    {
    }

    TIt Begin() const {
        return B_;
    }

    TIt End() const {
        return E_;
    }

private:
    TStr S_;
    TIt B_;
    TIt E_;
};

template <class It, class StringBuf = void /* =default */, class TOwnerPolicy = TExternalOwnerPolicy<It>>
class TStringSplitter {
    using TCVChar = std::remove_reference_t<decltype(*std::declval<It>())>;
    using TChar = std::remove_cv_t<TCVChar>;
    using TStrBuf = std::conditional_t<std::is_same<StringBuf, void>::value, TBasicStringBuf<TChar>, StringBuf>;

    static_assert(std::is_same<typename TStrBuf::value_type, TChar>::value, "Character type mismatch.");

    using TIteratorState = ::NPrivate::TIteratorState<It, TStrBuf, TOwnerPolicy>;

    /**
     * Base class for all split ranges that actually does the splitting.
     */
    template <class DelimStorage>
    struct TSplitRangeBase: public TIteratorState, public DelimStorage {
        template <typename... Args>
        inline TSplitRangeBase(const TOwnerPolicy& s, Args&&... args)
            : TIteratorState(s)
            , DelimStorage(std::forward<Args>(args)...)
        {
        }

        inline TIteratorState* Next() {
            if (this->TokD == this->B) {
                return nullptr;
            }

            this->TokS = this->B;
            this->TokD = this->Ptr()->Find(this->B, this->E);

            return this;
        }
    };

    template <class Base, class Filter>
    struct TFilterRange: public Base {
        template <typename... Args>
        inline TFilterRange(const Base& base, Args&&... args)
            : Base(base)
            , Filter_(std::forward<Args>(args)...)
        {
        }

        inline TIteratorState* Next() {
            TIteratorState* ret;

            do {
                ret = Base::Next();
            } while (ret && !Filter_.Accept(ret));

            return ret;
        }

        Filter Filter_;
    };

    struct TNonEmptyFilter {
        template <class TToken>
        inline bool Accept(const TToken* token) noexcept {
            return !token->Empty();
        }
    };

    template <class TIter>
    struct TStopIteration;

    template <class TIter>
    struct TFilters: public TIter {
        template <class TFilter>
        using TIt = TSplitRange<TStopIteration<TFilters<TFilterRange<TIter, TFilter>>>>;

        template <typename... Args>
        inline TFilters(Args&&... args)
            : TIter(std::forward<Args>(args)...)
        {
        }

        inline TIt<TNonEmptyFilter> SkipEmpty() const {
            return {*this};
        }
    };

    template <class Base, class Stopper>
    struct TStopRange: public Base {
        template <typename... Args>
        inline TStopRange(const Base& base, Args&&... args)
            : Base(base)
            , Stopper_(std::forward<Args>(args)...)
        {
        }

        inline TIteratorState* Next() {
            TIteratorState* ret = Base::Next();
            if (!ret || Stopper_.Stop(ret)) {
                return nullptr;
            }
            return ret;
        }

        Stopper Stopper_;
    };

    struct TTake {
        TTake() = default;

        TTake(size_t count)
            : Count(count)
        {
        }

        template <class TToken>
        inline bool Stop(const TToken*) noexcept {
            if (Count > 0) {
                --Count;
                return false;
            } else {
                return true;
            }
        }

        size_t Count = 0;
    };

    template <class TIter>
    struct TStopIteration: public TIter {
        template <class TStopper>
        using TIt = TSplitRange<TStopIteration<TFilters<TStopRange<TIter, TStopper>>>>;

        template <typename... Args>
        inline TStopIteration(Args&&... args)
            : TIter(std::forward<Args>(args)...)
        {
        }

        inline TIt<TTake> Take(size_t count) {
            return {*this, count};
        }
    };

    template <class TPolicy>
    using TIt = TSplitRange<TStopIteration<TFilters<TSplitRangeBase<TPolicy>>>>;

public:
    inline TStringSplitter(It b, It e)
        : Base_(b, e)
    {
    }

    explicit inline TStringSplitter(TOwnerPolicy o)
        : Base_(std::move(o))
    {
    }

    //does not own TDelim
    template <class TDelim>
    inline TIt<TPtrPolicy<const TDelim>> Split(const TDelim& d) const noexcept {
        return {Base_, &d};
    }

    inline TIt<TEmbedPolicy<TCharDelimiter<TCVChar>>> Split(TChar ch) const noexcept {
        return {Base_, ch};
    }

    inline TIt<TSimpleRefPolicy<TSetDelimiter<TCVChar>>> SplitBySet(const TChar* set) const noexcept {
        return {Base_, set};
    }

    inline TIt<TEmbedPolicy<TStringDelimiter<TCVChar>>> SplitByString(const TStrBuf& str) const noexcept {
        return {Base_, str.data(), str.size()};
    }

    template <class TFunc>
    inline TIt<TEmbedPolicy<TFuncDelimiter<It, TFunc>>> SplitByFunc(TFunc f) const noexcept {
        return {Base_, f};
    }

    template <class TDelim>
    inline TIt<TEmbedPolicy<TLimitedDelimiter<It, TDelim>>> SplitLimited(const TDelim& d, size_t limit) const noexcept {
        return {Base_, limit, d};
    }

    inline TIt<TEmbedPolicy<TLimitedDelimiter<It, TCharDelimiter<TCVChar>>>> SplitLimited(TChar ch, size_t limit) const noexcept {
        return {Base_, limit, ch};
    }

    inline TIt<TEmbedPolicy<TLimitedDelimiter<It, TSetDelimiter<TCVChar>>>> SplitBySetLimited(const TChar* set, size_t limit) const noexcept {
        return {Base_, limit, set};
    }

    inline TIt<TEmbedPolicy<TLimitedDelimiter<It, TStringDelimiter<TCVChar>>>> SplitByStringLimited(const TStrBuf& str, size_t limit) const noexcept {
        return {Base_, limit, str.data(), str.size()};
    }

    template <class TFunc>
    inline TIt<TEmbedPolicy<TLimitedDelimiter<It, TFuncDelimiter<It, TFunc>>>> SplitByFuncLimited(TFunc f, size_t limit) const noexcept {
        return {Base_, limit, f};
    }

private:
    TOwnerPolicy Base_;
};

template <class It>
TStringSplitter<It> StringSplitter(It begin, It end) {
    return {begin, end};
}

template <class It>
TStringSplitter<It> StringSplitter(It begin, size_t len) {
    return {begin, begin + len};
}

template <class Str, class NakedStr = std::remove_cv_t<std::remove_reference_t<Str>>, std::enable_if_t<!std::is_pointer<NakedStr>::value, int> = 0>
auto StringSplitter(Str& s) {
    return TStringSplitter<decltype(std::cbegin(s)), ::NPrivate::TStringBufOf<NakedStr>>(std::cbegin(s), std::cend(s));
}

template <class Str, class NakedStr = std::remove_cv_t<std::remove_reference_t<Str>>, std::enable_if_t<!std::is_pointer<NakedStr>::value, int> = 0>
auto StringSplitter(Str&& s) {
    return TStringSplitter<decltype(std::cbegin(s)), ::NPrivate::TStringBufOf<NakedStr>, TCopyOwnerPolicy<NakedStr>>(std::move(s));
}

template <class TChar>
auto StringSplitter(const TChar* str) {
    return StringSplitter(TBasicStringBuf<TChar>(str));
}
