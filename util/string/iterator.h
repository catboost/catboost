#pragma once

#include "split.h"

#include <util/generic/iterator.h>
#include <utility>
#include <util/generic/typetraits.h>
#include <util/generic/store_policy.h>

//
// Provides convenient way to split strings.
// Check iterator_ut.cpp to get examples of usage.

namespace NStringSplitterContainerConsumer {
    Y_HAS_MEMBER(push_back, PushBack);

    template <class Container, class StrBuf>
    struct TContainerPushBackConsumer {
        using T = typename Container::value_type;

        TContainerPushBackConsumer(Container* cont)
            : Cont(cont)
        {
        }

        inline void operator()(const StrBuf& token) {
            Cont->push_back(T(token));
        }

        Container* Cont;
    };

    template <class Container, class StrBuf>
    struct TContainerInsertConsumer {
        using T = typename Container::value_type;

        TContainerInsertConsumer(Container* cont)
            : Cont(cont)
        {
        }

        inline void operator()(const StrBuf& token) {
            Cont->insert(T(token));
        }

        Container* Cont;
    };
} // namespace NStringSplitterContainerConsumer

template <class It>
struct TStlIteratorFace: public It, public TStlIterator<TStlIteratorFace<It>> {
    using TRetVal = decltype(std::declval<It>().Next());
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

    template <class S>
    inline yvector<S> ToList() {
        yvector<S> result;
        for (auto&& it : *this) {
            result.push_back(S(it.Token()));
        }
        return result;
    }

    template <class Container>
    inline void Collect(Container* c) {
        Y_ASSERT(c);
        using namespace NStringSplitterContainerConsumer;
        using TConsumer = std::conditional_t<THasPushBack<Container>::Result, TContainerPushBackConsumer<Container, TStrBuf>, TContainerInsertConsumer<Container, TStrBuf>>;
        TConsumer consumer(c);
        this->Consume(consumer);
    }

    template <typename... TArgs>
    inline void CollectInto(TArgs*... args) {
        size_t filled = 0;
        auto it = this->begin();
        auto dummy = {
            (
                it != this->end() ? (
                                        ++filled,
                                        *args = ::FromString<TArgs>(it->Token()),
                                        ++it,
                                        0)
                                  : 0)...};
        Y_UNUSED(dummy);
        Y_ENSURE(filled == sizeof...(args) && it == this->end());
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

template <class It>
class TStringSplitter {
    using TCVChar = std::remove_reference_t<decltype(*std::declval<It>())>;
    using TChar = std::remove_cv_t<TCVChar>;
    using TStrBuf = TGenericStringBuf<TChar>;

    struct TIteratorState {
        inline TIteratorState(const TStringSplitter* s) noexcept
            : B(s->B_)
            , E(s->E_)
            , TokS()
            , TokD()
        {
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

        inline bool Empty() const noexcept {
            return TokenStart() == TokenDelim();
        }

        inline TStrBuf Token() const noexcept {
            return {TokenStart(), TokenDelim()};
        }

        inline TStrBuf Delim() const noexcept {
            return {TokenDelim(), TokenEnd()};
        }

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
        TTake() {
        }

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
        : B_(b)
        , E_(e)
    {
    }

    //does not own TDelim
    template <class TDelim>
    inline TIt<TPtrPolicy<TDelim>> Split(const TDelim& d) const noexcept {
        return {this, &d};
    }

    inline TIt<TEmbedPolicy<TCharDelimiter<TCVChar>>> Split(TChar ch) const noexcept {
        return {this, ch};
    }

    inline TIt<TSimpleRefPolicy<TSetDelimiter<TCVChar>>> SplitBySet(const TChar* set) const noexcept {
        return {this, set};
    }

    inline TIt<TEmbedPolicy<TStringDelimiter<TCVChar>>> SplitByString(const TStrBuf& str) const noexcept {
        return {this, ~str, +str};
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
        return {this, limit, ~str, +str};
    }

    template <class TFunc>
    inline TIt<TEmbedPolicy<TLimitedDelimiter<It, TFuncDelimiter<It, TFunc>>>> SplitByFuncLimited(TFunc f, size_t limit) const noexcept {
        return {this, limit, f};
    }

private:
    It B_;
    It E_;
};

template <class It>
static inline TStringSplitter<It> StringSplitter(It begin, It end) {
    return {begin, end};
}

template <class It>
static inline TStringSplitter<It> StringSplitter(It begin, size_t len) {
    return {begin, begin + len};
}

/*
 * should not be used with temporaries and range-based for loop, see http://pg.at.yandex-team.ru/3580
 * consider StringSplitter(SomeFunc()).Split(...).Consume([...](...) {...});
 */
template <class Str>
static inline auto StringSplitter(const Str& s) {
    return StringSplitter(s.begin(), s.end());
}

template <class TChar>
static inline auto StringSplitter(const TChar* str) {
    return StringSplitter(TGenericStringBuf<TChar>(str));
}
