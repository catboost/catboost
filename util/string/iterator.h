#pragma once

#include "cast.h"
#include "split.h"

#include <util/generic/algorithm.h>
#include <util/generic/iterator.h>
#include <util/generic/typetraits.h>
#include <util/generic/store_policy.h>

#include <utility>

//
// Provides convenient way to split strings.
// Check iterator_ut.cpp to get examples of usage.

namespace NPrivate {
    Y_HAS_MEMBER(push_back, PushBack);
    Y_HAS_MEMBER(insert, Insert);

    template <class Container>
    struct TContainerConsumer {
        using value_type = typename Container::value_type;

        TContainerConsumer(Container* c)
            : C_(c)
        {
        }

        template<class Char>
        void operator()(TGenericStringBuf<Char> e) const {
            this->operator()(C_, e);
        }

    private:
        template<class OtherContainer, class Char>
        auto operator()(OtherContainer* c, TGenericStringBuf<Char> e) const -> decltype(c->push_back(value_type(e))) {
            return c->push_back(value_type(e));
        }

        template<class OtherContainer, class Char>
        auto operator()(OtherContainer* c, TGenericStringBuf<Char> e) const -> decltype(c->insert(value_type(e))) {
            return c->insert(value_type(e));
        }

        Container* C_;
    };

    template <class Container>
    struct TContainerConvertingConsumer {
        using value_type = typename Container::value_type;

        TContainerConvertingConsumer(Container* c)
            : C_(c)
        {
        }

        template <class Char>
        void operator()(TGenericStringBuf<Char> e) const {
            this->operator()(C_, e);
        }

    private:
        template<class OtherContainer, class Char>
        auto operator()(OtherContainer* c, TGenericStringBuf<Char> e) const -> decltype(c->push_back(FromString<value_type>(e))) {
            return c->push_back(FromString<value_type>(e));
        }

        template<class OtherContainer, class Char>
        auto operator()(OtherContainer* c, TGenericStringBuf<Char> e) const -> decltype(c->insert(FromString<value_type>(e))) {
            return c->insert(FromString<value_type>(e));
        }

        Container* C_;
    };
}

template <class It>
struct TStlIteratorFace: public It, public TInputRangeAdaptor<TStlIteratorFace<It>> {
    using TStrBuf = decltype(std::declval<It>().Next()->Token());

    template <typename... Args>
    inline TStlIteratorFace(Args&&... args)
        : It(std::forward<Args>(args)...)
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

    /**
        * Collects all splitted arguments into args
        * @param args: Output arguments
        * @return bool: true, if all items collected successfully, else - false
    */
    template <typename... Args>
    inline bool TryCollectInto(Args*... args) {
        size_t filled = 0;
        auto it = this->begin();

        ApplyToMany([&](auto&& arg) {
            if (it != this->end()) {
                ++filled;
                *arg = ::FromString<std::remove_reference_t<decltype(*arg)>>(it->Token());
                ++it;
            }
        }, args...);

        return filled == sizeof...(args) && it == this->end();
    }

    /**
        * Collects all splitted arguments into args
        * Throws exception, if not all items collected successfully
        * @param args: Output arguments
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

template <class It, class TOwnerPolicy = TExternalOwnerPolicy<It>>
class TStringSplitter {
    using TCVChar = std::remove_reference_t<decltype(*std::declval<It>())>;
    using TChar = std::remove_cv_t<TCVChar>;
    using TStrBuf = TGenericStringBuf<TChar>;

    struct TIteratorState {
        inline TIteratorState(const TStringSplitter* s) noexcept
            : Base(s->Base)
            , B(Base.Begin())
            , E(Base.End())
            , TokS()
            , TokD()
        {
        }

        operator TStrBuf() const noexcept {
            return Token();
        }

        inline It TokenStart() const noexcept {
            return TokS;
        }

        inline It TokenDelim() const noexcept {
            return TokD;
        }

        inline It TokenEnd() const noexcept {
            return B;
        }

        Y_PURE_FUNCTION
        inline bool Empty() const noexcept {
            return TokenStart() == TokenDelim();
        }

        inline TStrBuf Token() const noexcept {
            return {TokenStart(), TokenDelim()};
        }

        inline TStrBuf Delim() const noexcept {
            return {TokenDelim(), TokenEnd()};
        }

        TOwnerPolicy Base;

        It B;
        It E;

        It TokS;
        It TokD;
    };

    template <class TDelim>
    struct TIterator: public TIteratorState, public TDelim {
        template <typename... Args>
        inline TIterator(const TStringSplitter* s, Args&&... args)
            : TIteratorState(s)
            , TDelim(std::forward<Args>(args)...)
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

    template <class TIter, class TFilter>
    struct TApplyFilter: public TIter {
        template <typename... Args>
        inline TApplyFilter(const TIter& iter, Args&&... args)
            : TIter(iter)
            , Filter(std::forward<Args>(args)...)
        {
        }

        inline TIteratorState* Next() {
            TIteratorState* ret;

            do {
                ret = TIter::Next();
            } while (ret && !Filter.Accept(ret));

            return ret;
        }

        TFilter Filter;
    };

    struct TSkipEmpty {
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
        using TIt = TStlIteratorFace<TStopIteration<TFilters<TApplyFilter<TIter, TFilter>>>>;

        template <typename... Args>
        inline TFilters(Args&&... args)
            : TIter(std::forward<Args>(args)...)
        {
        }

        inline TIt<TSkipEmpty> SkipEmpty() const {
            return {*this};
        }
    };

    template <class TIter, class TStopper>
    struct TApplyStopper: public TIter {
        template <typename... Args>
        inline TApplyStopper(const TIter& iter, Args&&... args)
            : TIter(iter)
            , Stopper(std::forward<Args>(args)...)
        {
        }

        inline TIteratorState* Next() {
            TIteratorState* ret = TIter::Next();
            if (!ret || Stopper.Stop(ret)) {
                return nullptr;
            }
            return ret;
        }

        TStopper Stopper;
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
        using TIt = TStlIteratorFace<TStopIteration<TFilters<TApplyStopper<TIter, TStopper>>>>;

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
    using TIt = TStlIteratorFace<TStopIteration<TFilters<TIterator<TPolicy>>>>;

public:
    inline TStringSplitter(It b, It e)
        : Base(b, e)
    {
    }

    explicit inline TStringSplitter(TOwnerPolicy o)
        : Base(std::move(o))
    {
    }

    //does not own TDelim
    template <class TDelim>
    inline TIt<TPtrPolicy<const TDelim>> Split(const TDelim& d) const noexcept {
        return {this, &d};
    }

    inline TIt<TEmbedPolicy<TCharDelimiter<TCVChar>>> Split(TChar ch) const noexcept {
        return {this, ch};
    }

    inline TIt<TSimpleRefPolicy<TSetDelimiter<TCVChar>>> SplitBySet(const TChar* set) const noexcept {
        return {this, set};
    }

    inline TIt<TEmbedPolicy<TStringDelimiter<TCVChar>>> SplitByString(const TStrBuf& str) const noexcept {
        return {this, str.data(), str.size()};
    }

    template <class TFunc>
    inline TIt<TEmbedPolicy<TFuncDelimiter<It, TFunc>>> SplitByFunc(TFunc f) const noexcept {
        return {this, f};
    }

    template <class TDelim>
    inline TIt<TEmbedPolicy<TLimitedDelimiter<It, TDelim>>> SplitLimited(const TDelim& d, size_t limit) const noexcept {
        return {this, limit, d};
    }

    inline TIt<TEmbedPolicy<TLimitedDelimiter<It, TCharDelimiter<TCVChar>>>> SplitLimited(TChar ch, size_t limit) const noexcept {
        return {this, limit, ch};
    }

    inline TIt<TEmbedPolicy<TLimitedDelimiter<It, TSetDelimiter<TCVChar>>>> SplitBySetLimited(const TChar* set, size_t limit) const noexcept {
        return {this, limit, set};
    }

    inline TIt<TEmbedPolicy<TLimitedDelimiter<It, TStringDelimiter<TCVChar>>>> SplitByStringLimited(const TStrBuf& str, size_t limit) const noexcept {
        return {this, limit, str.data(), str.size()};
    }

    template <class TFunc>
    inline TIt<TEmbedPolicy<TLimitedDelimiter<It, TFuncDelimiter<It, TFunc>>>> SplitByFuncLimited(TFunc f, size_t limit) const noexcept {
        return {this, limit, f};
    }

private:
    TOwnerPolicy Base;
};

template <class It>
static inline TStringSplitter<It> StringSplitter(It begin, It end) {
    return {begin, end};
}

template <class It>
static inline TStringSplitter<It> StringSplitter(It begin, size_t len) {
    return {begin, begin + len};
}

// enable_if for solving ambiguous overload for raw pointers. char* a = something; StringSplitter(a)...
template <class Str, std::enable_if_t<!std::is_pointer<std::remove_cv_t<std::remove_reference_t<Str>>>::value, int> = 0>
static inline auto StringSplitter(Str& s) {
    return TStringSplitter<decltype(std::cbegin(s))>(std::cbegin(s), std::cend(s));
}

// enable_if for solving ambiguous overload for raw pointers. char* a = something; StringSplitter(a)...
template <class Str, std::enable_if_t<!std::is_pointer<std::remove_cv_t<std::remove_reference_t<Str>>>::value, int> = 0>
static inline auto StringSplitter(Str&& s) {
    using TRes = TStringSplitter<decltype(std::cbegin(s)), TCopyOwnerPolicy<std::remove_cv_t<std::remove_reference_t<Str>>>>;
    return TRes(std::move(s));
}

template <class TChar>
static inline auto StringSplitter(const TChar* str) {
    return StringSplitter(TGenericStringBuf<TChar>(str));
}
