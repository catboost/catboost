#pragma once

#include "cast.h"
#include "split.h"

#include <util/generic/algorithm.h>
#include <util/generic/iterator.h>
#include <util/generic/typetraits.h>
#include <util/generic/store_policy.h>
#include <util/generic/iterator_range.h>

#include <utility>
#include <stlfwd>

/**
 * \fn auto StringSplitter(...)
 *
 * Creates a string splitter object. The only use for it is to call one of its
 * `Split*` methods, and then do something with the resulting proxy range.
 *
 * Some examples:
 * \code
 * TVector<TStringBuf> values = StringSplitter("1\t2\t3").Split('\t');
 *
 * for(TStringBuf part: StringSplitter("1::2::::3").SplitByString("::").SkipEmpty()) {
 *     Cerr << part;
 * }
 *
 * TVector<TString> firstTwoValues = StringSplitter("1\t2\t3").Split('\t').Take(2);
 * \endcode
 *
 * Use `Collect` or `AddTo` to store split results into an existing container:
 * \code
 * TVector<TStringBuf> values = {"0"};
 * StringSplitter("1\t2\t3").Split('\t').AddTo(&values);
 * \endcode
 * Note that `Collect` clears target container, while `AddTo` just inserts values.
 * You can use these methods with any container that has `emplace` / `emplace_back`.
 *
 * Use `ParseInto` to also perform string conversions before inserting values
 * into target container:
 * \code
 * TSet<int> values;
 * StringSplitter("1\t2\t3").Split('\t').ParseInto(&values);
 * \endcode
 */

namespace NPrivate {
    Y_HAS_MEMBER(push_back, PushBack);
    Y_HAS_MEMBER(insert, Insert);
    Y_HAS_MEMBER(data, Data);

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

        // TODO: return bool (continue)
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
            DoFromString(e, &v);
            return c->emplace_back(std::move(v));
        }

        template<class OtherContainer, class StringBuf>
        auto operator()(OtherContainer* c, StringBuf e) const -> decltype(c->emplace()) {
            value_type v;
            DoFromString(e, &v);
            return c->emplace(std::move(v));
        }

        Container* C_;
    };

    template <class String>
    struct TStringBufOfImpl {
        using type = std::conditional_t<
            THasData<String>::value,
            TBasicStringBuf<typename String::value_type>,
            TIteratorRange<typename String::const_iterator>
        >;
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

    template<class StringBuf, class Iterator>
    StringBuf DoMakeStringBuf(Iterator b, Iterator e, StringBuf*) {
        return StringBuf(b, e);
    }

    template<class Char, class Traits, class Iterator>
    std::basic_string_view<Char, Traits> DoMakeStringBuf(Iterator b, Iterator e, std::basic_string_view<Char, Traits>*) {
        return std::basic_string_view<Char, Traits>(b, e - b);
    }

    template<class StringBuf, class Iterator>
    StringBuf MakeStringBuf(Iterator b, Iterator e) {
        return DoMakeStringBuf(b, e, static_cast<StringBuf*>(nullptr));
    }

    template<class String>
    struct TIteratorOfImpl {
        using type = std::conditional_t<
            THasData<String>::value,
            const typename String::value_type*,
            typename String::const_iterator
        >;
    };

    template<class String>
    using TIteratorOf = typename TIteratorOfImpl<String>::type;

    template<class String>
    struct TIteratorState {
        using TStringBufType = TStringBufOf<String>;
        using TIterator = TIteratorOf<String>;

        TIteratorState(const String& string) noexcept
            : TokS()
            , TokD()
        {
            Init(string, THasData<String>());
        }

        operator TStringBufType() const noexcept {
            return Token();
        }

        explicit operator bool() const {
            return !Empty();
        }

        TIterator TokenStart() const noexcept {
            return TokS;
        }

        TIterator TokenDelim() const noexcept {
            return TokD;
        }

        TIterator TokenEnd() const noexcept {
            return B;
        }

        Y_PURE_FUNCTION
        bool Empty() const noexcept {
            return TokenStart() == TokenDelim();
        }

        TStringBufType Token() const noexcept {
            return MakeStringBuf<TStringBufType>(TokenStart(), TokenDelim());
        }

        TStringBufType Delim() const noexcept {
            return MakeStringBuf<TStringBufType>(TokenDelim(), TokenEnd());
        }

        TIterator B;
        TIterator E;

        TIterator TokS;
        TIterator TokD;

    private:
        void Init(const String& string, std::true_type) {
            B = string.data();
            E = string.data() + string.size();
        }

        void Init(const String& string, std::false_type) {
            B = string.begin();
            E = string.end();
        }
    };

    template <class Base>
    class TSplitRange : public Base, public TInputRangeAdaptor<TSplitRange<Base>> {
        using TStringBufType = decltype(std::declval<Base>().Next()->Token());

    public:
        template <typename... Args>
        inline TSplitRange(Args&&... args)
            : Base(std::forward<Args>(args)...)
        {
        }

        template <class Consumer, std::enable_if_t<std::is_same<decltype(std::declval<Consumer>()(std::declval<TStringBufType>())), void>::value, int>* = nullptr>
        inline void Consume(Consumer&& f) {
            for (auto&& it : *this) {
                f(it.Token());
            }
        }

        template <class Consumer, std::enable_if_t<std::is_same<decltype(std::declval<Consumer>()(std::declval<TStringBufType>())), bool>::value, int>* = nullptr>
        inline bool Consume(Consumer&& f) {
            for (auto&& it : *this) {
                if (!f(it.Token())) {
                    return false;
                }
            }
            return true;
        }

        template<class Container, class = std::enable_if_t<THasInsert<Container>::value || THasPushBack<Container>::value>>
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
            TContainerConsumer<Container> consumer(c);
            Consume(consumer);
        }

        template <class Container>
        inline void ParseInto(Container* c) {
            Y_ASSERT(c);
            TContainerConvertingConsumer<Container> consumer(c);
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
                    if (TryDoFromString(it->Token(), arg)) {
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

    template <class String>
    class TStringSplitter {
        using TStringType = String;
        using TStringBufType = TStringBufOf<TStringType>;
        using TChar = typename TStringType::value_type;
        using TIterator = TIteratorOf<TStringType>;
        using TIteratorState = TIteratorState<TStringType>;

        /**
         * Base class for all split ranges that actually does the splitting.
         */
        template <class DelimStorage>
        struct TSplitRangeBase {
            template <class OtherString, class... Args>
            inline TSplitRangeBase(OtherString&& s, Args&&... args)
                : String_(std::forward<OtherString>(s))
                , State_(String_)
                , Delim_(std::forward<Args>(args)...)
            {
            }

            inline TIteratorState* Next() {
                if (State_.TokD == State_.B) {
                    return nullptr;
                }

                State_.TokS = State_.B;
                State_.TokD = Delim_.Ptr()->Find(State_.B, State_.E);

                return &State_;
            }

        private:
            TStringType String_;
            TIteratorState State_;
            DelimStorage Delim_;
        };

        template <class Base, class Filter>
        struct TFilterRange : public Base {
            template <class... Args>
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

        template <class Base>
        struct TFilters : public Base {
            template <class TFilter>
            using TIt = TSplitRange<TStopIteration<TFilters<TFilterRange<Base, TFilter>>>>;

            template <typename... Args>
            inline TFilters(Args&&... args)
                : Base(std::forward<Args>(args)...)
            {
            }

            inline TIt<TNonEmptyFilter> SkipEmpty() const {
                return { *this };
            }
        };

        template <class Base, class Stopper>
        struct TStopRange : public Base {
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

        template <class Base>
        struct TStopIteration : public Base {
            template <class TStopper>
            using TIt = TSplitRange<TStopIteration<TFilters<TStopRange<Base, TStopper>>>>;

            template <typename... Args>
            inline TStopIteration(Args&&... args)
                : Base(std::forward<Args>(args)...)
            {
            }

            inline TIt<TTake> Take(size_t count) {
                return { *this, count };
            }
        };

        template <class TPolicy>
        using TIt = TSplitRange<TStopIteration<TFilters<TSplitRangeBase<TPolicy>>>>;

    public:
        template<class OtherString>
        explicit TStringSplitter(OtherString&& s)
            : String_(std::forward<OtherString>(s))
        {
        }

        //does not own TDelim
        template <class TDelim>
        inline TIt<TPtrPolicy<const TDelim>> Split(const TDelim& d) const noexcept {
            return { String_, &d };
        }

        inline TIt<TEmbedPolicy<TCharDelimiter<const TChar>>> Split(TChar ch) const noexcept {
            return { String_, ch };
        }

        inline TIt<TSimpleRefPolicy<TSetDelimiter<const TChar>>> SplitBySet(const TChar* set) const noexcept {
            return { String_, set };
        }

        inline TIt<TEmbedPolicy<TStringDelimiter<const TChar>>> SplitByString(const TStringBufType& str) const noexcept {
            return { String_, str.data(), str.size() };
        }

        template <class TFunc>
        inline TIt<TEmbedPolicy<TFuncDelimiter<TIterator, TFunc>>> SplitByFunc(TFunc f) const noexcept {
            return { String_, f };
        }

        template <class TDelim>
        inline TIt<TEmbedPolicy<TLimitedDelimiter<const TChar*, TDelim>>> SplitLimited(const TDelim& d, size_t limit) const noexcept {
            return { String_, limit, d };
        }

        inline TIt<TEmbedPolicy<TLimitedDelimiter<const TChar*, TCharDelimiter<const TChar>>>> SplitLimited(TChar ch, size_t limit) const noexcept {
            return { String_, limit, ch };
        }

        inline TIt<TEmbedPolicy<TLimitedDelimiter<const TChar*, TSetDelimiter<const TChar>>>> SplitBySetLimited(const TChar* set, size_t limit) const noexcept {
            return { String_, limit, set };
        }

        inline TIt<TEmbedPolicy<TLimitedDelimiter<const TChar*, TStringDelimiter<const TChar>>>> SplitByStringLimited(const TStringBufType& str, size_t limit) const noexcept {
            return { String_, limit, str.data(), str.size() };
        }

        template <class TFunc>
        inline TIt<TEmbedPolicy<TLimitedDelimiter<TIterator, TFuncDelimiter<TIterator, TFunc>>>> SplitByFuncLimited(TFunc f, size_t limit) const noexcept {
            return { String_, limit, f };
        }

    private:
        TStringType String_;
    };

    template<class String>
    auto MakeStringSplitter(String&& s) {
        return TStringSplitter<std::remove_reference_t<String>>(std::forward<String>(s));
    }
}

template <class Iterator>
auto StringSplitter(Iterator begin, Iterator end) {
    return ::NPrivate::MakeStringSplitter(TIteratorRange<Iterator>(begin, end));
}

template <class Char>
auto StringSplitter(const Char* begin, const Char* end) {
    return ::NPrivate::MakeStringSplitter(TBasicStringBuf<Char>(begin, end));
}

template <class Char>
auto StringSplitter(const Char* begin, size_t len) {
    return ::NPrivate::MakeStringSplitter(TBasicStringBuf<Char>(begin, len));
}

template <class Char>
auto StringSplitter(const Char* str) {
    return ::NPrivate::MakeStringSplitter(TBasicStringBuf<Char>(str));
}

template <class String, std::enable_if_t<!std::is_pointer<std::remove_reference_t<String>>::value, int> = 0>
auto StringSplitter(String& s) {
    return ::NPrivate::MakeStringSplitter(::NPrivate::TStringBufOf<String>(s.data(), s.size()));
}

template <class String, std::enable_if_t<!std::is_pointer<std::remove_reference_t<String>>::value, int> = 0>
auto StringSplitter(String&& s) {
    return ::NPrivate::MakeStringSplitter(std::move(s));
}

