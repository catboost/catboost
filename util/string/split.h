#pragma once

#include "strspn.h"
#include "cast.h"

#include <util/generic/algorithm.h>
#include <util/generic/fwd.h>
#include <util/generic/iterator.h>
#include <util/generic/iterator_range.h>
#include <util/generic/store_policy.h>
#include <util/generic/strbuf.h>
#include <util/generic/string.h>
#include <util/generic/typetraits.h>
#include <util/generic/vector.h>
#include <util/generic/ylimits.h>
#include <util/system/compat.h>
#include <util/system/defaults.h>

#include <utility>
#include <stlfwd>

// NOTE: Check StringSplitter below to get more convenient split string interface.

namespace NStringSplitPrivate {

    template <class T, class I, class = void>
    struct TIsConsumer: std::false_type {};

    template <class T, class I>
    struct TIsConsumer<
        T, I,
        TVoidT<decltype(std::declval<T>().Consume(
            std::declval<I>(), std::declval<I>(), std::declval<I>()))>>
        : std::true_type {};

    template <class T, class I>
    constexpr bool TIsConsumerV = TIsConsumer<T, I>::value;

    template <class T>
    T* Find(T* str, std::common_type_t<T> ch) {
        for (; *str; ++str) {
            if (*str == ch) {
                return str;
            }
        }

        return nullptr;
    }

}

template <class I, class TDelim, class TConsumer>
std::enable_if_t<::NStringSplitPrivate::TIsConsumerV<TConsumer, I>>
SplitString(I b, I e, const TDelim& d, TConsumer&& c) {
    I l, i;

    do {
        l = b;
        i = d.Find(b, e);
    } while (c.Consume(l, i, b) && (b != i));
}

template <class I, class TDelim, class TConsumer>
std::enable_if_t<::NStringSplitPrivate::TIsConsumerV<TConsumer, I>>
SplitString(I b, const TDelim& d, TConsumer&& c) {
    I l, i;

    do {
        l = b;
        i = d.Find(b);
    } while (c.Consume(l, i, b) && (b != i));
}

template <class I1, class I2>
static inline I1* FastStrChr(I1* str, I2 f) noexcept {
    I1* ret = NStringSplitPrivate::Find(str, f);

    if (!ret) {
        ret = str + std::char_traits<I1>::length(str);
    }

    return ret;
}

template <class I>
static inline I* FastStrStr(I* str, I* f, size_t l) noexcept {
    std::basic_string_view<I> strView(str);
    const auto ret = strView.find(*f);

    if (ret != std::string::npos) {
        std::basic_string_view<I> fView(f, l);
        strView = strView.substr(ret);
        for (; strView.size() >= l; strView = strView.substr(1)) {
            if (strView.substr(0, l) == fView) {
                break;
            }
        }

        return strView.size() >= l ? strView.data() : strView.data() + strView.size();
    } else {
        return strView.data() + strView.size();
    }
}

template <class Char>
struct TStringDelimiter {
    inline TStringDelimiter(Char* delim) noexcept
        : Delim(delim)
        , Len(std::char_traits<Char>::length(delim))
    {
    }

    inline TStringDelimiter(Char* delim, size_t len) noexcept
        : Delim(delim)
        , Len(len)
    {
    }

    inline Char* Find(Char*& b, Char* e) const noexcept {
        const auto ret = std::basic_string_view<Char>(b, e - b).find(Delim, 0, Len);

        if (ret != std::string::npos) {
            const auto result = b + ret;
            b = result + Len;
            return result;
        }

        return (b = e);
    }

    inline Char* Find(Char*& b) const noexcept {
        Char* ret = FastStrStr(b, Delim, Len);

        b = *ret ? ret + Len : ret;

        return ret;
    }

    Char* Delim;
    const size_t Len;
};

template <class Char>
struct TCharDelimiter {
    inline TCharDelimiter(Char ch) noexcept
        : Ch(ch)
    {
    }

    inline Char* Find(Char*& b, Char* e) const noexcept {
        const auto ret = std::basic_string_view<Char>(b, e - b).find(Ch);

        if (ret != std::string::npos) {
            const auto result = b + ret;
            b = result + 1;
            return result;
        }

        return (b = e);
    }

    inline Char* Find(Char*& b) const noexcept {
        Char* ret = FastStrChr(b, Ch);

        if (*ret) {
            b = ret + 1;
        } else {
            b = ret;
        }

        return ret;
    }

    Char Ch;
};

template <class Iterator, class Condition>
struct TFuncDelimiter {
public:
    template <class... Args>
    TFuncDelimiter(Args&&... args)
        : Fn(std::forward<Args>(args)...)
    {
    }

    inline Iterator Find(Iterator& b, Iterator e) const noexcept {
        if ((b = std::find_if(b, e, Fn)) != e) {
            return b++;
        }

        return b;
    }

private:
    Condition Fn;
};

template <class Char>
struct TFindFirstOf {
    inline TFindFirstOf(Char* set)
        : Set(set)
    {
    }

    inline Char* FindFirstOf(Char* b, Char* e) const noexcept {
        Char* ret = b;
        for (; ret != e; ++ret) {
            if (NStringSplitPrivate::Find(Set, *ret))
                break;
        }
        return ret;
    }

    inline Char* FindFirstOf(Char* b) const noexcept {
        const std::basic_string_view<Char> bView(b);
        const auto ret = bView.find_first_of(Set);
        return ret != std::string::npos ? b + ret : b + bView.size();
    }

    Char* Set;
};

template <>
struct TFindFirstOf<const char>: public TCompactStrSpn {
    inline TFindFirstOf(const char* set, const char* e)
        : TCompactStrSpn(set, e)
    {
    }

    inline TFindFirstOf(const char* set)
        : TCompactStrSpn(set)
    {
    }
};

template <class Char>
struct TSetDelimiter: private TFindFirstOf<const Char> {
    using TFindFirstOf<const Char>::TFindFirstOf;

    inline Char* Find(Char*& b, Char* e) const noexcept {
        Char* ret = const_cast<Char*>(this->FindFirstOf(b, e));

        if (ret != e) {
            b = ret + 1;
            return ret;
        }

        return (b = e);
    }

    inline Char* Find(Char*& b) const noexcept {
        Char* ret = const_cast<Char*>(this->FindFirstOf(b));

        if (*ret) {
            b = ret + 1;
            return ret;
        }

        return (b = ret);
    }
};

namespace NSplitTargetHasPushBack {
    Y_HAS_MEMBER(push_back, PushBack);
}

template <class T, class = void>
struct TConsumerBackInserter;

template <class T>
struct TConsumerBackInserter<T, std::enable_if_t<NSplitTargetHasPushBack::TClassHasPushBack<T>::value>> {
    static void DoInsert(T* C, const typename T::value_type& i) {
        C->push_back(i);
    }
};

template <class T>
struct TConsumerBackInserter<T, std::enable_if_t<!NSplitTargetHasPushBack::TClassHasPushBack<T>::value>> {
    static void DoInsert(T* C, const typename T::value_type& i) {
        C->insert(C->end(), i);
    }
};

template <class T>
struct TContainerConsumer {
    inline TContainerConsumer(T* c) noexcept
        : C(c)
    {
    }

    template <class I>
    inline bool Consume(I* b, I* d, I* /*e*/) {
        TConsumerBackInserter<T>::DoInsert(C, typename T::value_type(b, d));

        return true;
    }

    T* C;
};

template <class T>
struct TContainerConvertingConsumer {
    inline TContainerConvertingConsumer(T* c) noexcept
        : C(c)
    {
    }

    template <class I>
    inline bool Consume(I* b, I* d, I* /*e*/) {
        TConsumerBackInserter<T>::DoInsert(C, FromString<typename T::value_type>(TStringBuf(b, d)));

        return true;
    }

    T* C;
};

template <class S, class I>
struct TLimitingConsumer {
    inline TLimitingConsumer(size_t cnt, S* slave) noexcept
        : Cnt(cnt ? cnt - 1 : Max<size_t>())
        , Slave(slave)
        , Last(nullptr)
    {
    }

    inline bool Consume(I* b, I* d, I* e) {
        if (!Cnt) {
            Last = b;

            return false;
        }

        --Cnt;

        return Slave->Consume(b, d, e);
    }

    size_t Cnt;
    S* Slave;
    I* Last;
};

template <class S>
struct TSkipEmptyTokens {
    inline TSkipEmptyTokens(S* slave) noexcept
        : Slave(slave)
    {
    }

    template <class I>
    inline bool Consume(I* b, I* d, I* e) {
        if (b != d) {
            return Slave->Consume(b, d, e);
        }

        return true;
    }

    S* Slave;
};

template <class S>
struct TKeepDelimiters {
    inline TKeepDelimiters(S* slave) noexcept
        : Slave(slave)
    {
    }

    template <class I>
    inline bool Consume(I* b, I* d, I* e) {
        if (Slave->Consume(b, d, d)) {
            if (d != e) {
                return Slave->Consume(d, e, e);
            }

            return true;
        }

        return false;
    }

    S* Slave;
};

template <class T>
struct TSimplePusher {
    inline bool Consume(char* b, char* d, char*) {
        *d = 0;
        C->push_back(b);

        return true;
    }

    T* C;
};

template <class T>
static inline void Split(char* buf, char ch, T* res) {
    res->resize(0);
    if (*buf == 0)
        return;

    TCharDelimiter<char> delim(ch);
    TSimplePusher<T> pusher = {res};

    SplitString(buf, delim, pusher);
}

/// Split string into res vector. Res vector is cleared before split.
/// Old good slow split function.
/// Field delimter is any number of symbols specified in delim (no empty strings in res vector)
/// @return number of elements created
size_t Split(const char* in, const char* delim, TVector<TString>& res);
size_t Split(const TString& in, const TString& delim, TVector<TString>& res);

/// Old split reimplemented for TStringBuf using the new code
/// Note that delim can be constructed from char* automatically (it is not cheap though)
inline size_t Split(const TStringBuf s, const TSetDelimiter<const char>& delim, TVector<TStringBuf>& res) {
    res.clear();
    TContainerConsumer<TVector<TStringBuf>> res1(&res);
    TSkipEmptyTokens<TContainerConsumer<TVector<TStringBuf>>> consumer(&res1);
    SplitString(s.data(), s.data() + s.size(), delim, consumer);
    return res.size();
}

template <class P, class D>
void GetNext(TStringBuf& s, D delim, P& param) {
    TStringBuf next = s.NextTok(delim);
    Y_ENSURE(next.IsInited(), TStringBuf("Split: number of fields less than number of Split output arguments"));
    param = FromString<P>(next);
}

template <class P, class D>
void GetNext(TStringBuf& s, D delim, TMaybe<P>& param) {
    TStringBuf next = s.NextTok(delim);
    if (next.IsInited()) {
        param = FromString<P>(next);
    } else {
        param.Clear();
    }
}

// example:
// Split(TStringBuf("Sherlock,2014,36.6"), ',', name, year, temperature);
template <class D, class P1, class P2>
void Split(TStringBuf s, D delim, P1& p1, P2& p2) {
    GetNext(s, delim, p1);
    GetNext(s, delim, p2);
    Y_ENSURE(!s.IsInited(), TStringBuf("Split: number of fields more than number of Split output arguments"));
}

template <class D, class P1, class P2, class... Other>
void Split(TStringBuf s, D delim, P1& p1, P2& p2, Other&... other) {
    GetNext(s, delim, p1);
    Split(s, delim, p2, other...);
}

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

namespace NStringSplitPrivate {
    Y_HAS_MEMBER(push_back, PushBack);
    Y_HAS_MEMBER(insert, Insert);
    Y_HAS_MEMBER(data, Data);

    /**
     * This one is needed here so that `std::string_view -> std::string_view`
     * conversion works.
     */
    template <class Src, class Dst>
    inline void DoFromString(const Src& src, Dst* dst) {
        *dst = ::FromString<Dst>(src);
    }

    template <class T>
    inline void DoFromString(const T& src, T* dst) noexcept {
        *dst = src;
    }

    template <class T>
    inline void DoFromString(const T& src, decltype(std::ignore)* dst) noexcept {
        *dst = src;
    }

    template <class Src, class Dst>
    inline Y_WARN_UNUSED_RESULT bool TryDoFromString(const Src& src, Dst* dst) noexcept {
        return ::TryFromString(src, *dst);
    }

    template <class T>
    inline Y_WARN_UNUSED_RESULT bool TryDoFromString(const T& src, T* dst) noexcept {
        *dst = src;
        return true;
    }

    template <class T>
    inline Y_WARN_UNUSED_RESULT bool TryDoFromString(const T& src, decltype(std::ignore)* dst) noexcept {
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
        template <class StringBuf>
        void operator()(StringBuf e) const {
            this->operator()(C_, e);
        }

    private:
        template <class OtherContainer, class StringBuf>
        auto operator()(OtherContainer* c, StringBuf e) const -> decltype(c->emplace_back()) {
            return c->emplace_back(value_type(e));
        }

        template <class OtherContainer, class StringBuf>
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
        template <class OtherContainer, class StringBuf>
        auto operator()(OtherContainer* c, StringBuf e) const -> decltype(c->emplace_back()) {
            value_type v;
            DoFromString(e, &v);
            return c->emplace_back(std::move(v));
        }

        template <class OtherContainer, class StringBuf>
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
            TIteratorRange<typename String::const_iterator>>;
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
    template <class String>
    using TStringBufOf = typename TStringBufOfImpl<String>::type;

    template <class StringBuf, class Iterator>
    StringBuf DoMakeStringBuf(Iterator b, Iterator e, StringBuf*) {
        return StringBuf(b, e);
    }

    template <class Char, class Traits, class Iterator>
    std::basic_string_view<Char, Traits> DoMakeStringBuf(Iterator b, Iterator e, std::basic_string_view<Char, Traits>*) {
        return std::basic_string_view<Char, Traits>(b, e - b);
    }

    template <class StringBuf, class Iterator>
    StringBuf MakeStringBuf(Iterator b, Iterator e) {
        return DoMakeStringBuf(b, e, static_cast<StringBuf*>(nullptr));
    }

    template <class String>
    struct TIteratorOfImpl {
        using type = std::conditional_t<
            THasData<String>::value,
            const typename String::value_type*,
            typename String::const_iterator>;
    };

    template <class String>
    using TIteratorOf = typename TIteratorOfImpl<String>::type;

    template <class String>
    class TStringSplitter;

    template <class String>
    struct TIterState: public TStringBufOf<String> {
    public:
        using TStringBufType = TStringBufOf<String>;
        using TIterator = TIteratorOf<String>;
        friend class TStringSplitter<String>;

        TIterState(const String& string) noexcept
            : TStringBufType()
            , DelimiterEnd_(std::begin(string))
            , OriginEnd_(std::end(string))
        {
        }

        template <
            typename Other,
            typename = std::enable_if_t<
                std::is_convertible<Other, TStringBufType>::value>>
        bool operator==(const Other& toCompare) const {
            return TStringBufType(*this) == TStringBufType(toCompare);
        }

        TIterator TokenStart() const noexcept {
            return this->begin();
        }

        TIterator TokenDelim() const noexcept {
            return this->end();
        }

        TStringBufType Token() const noexcept {
            return *this;
        }

        TStringBufType Delim() const noexcept {
            return MakeStringBuf<TStringBufType>(TokenDelim(), DelimiterEnd_);
        }

    private:
        void UpdateParentBuf(TIterator tokenStart, TIterator tokenDelim) noexcept {
            *static_cast<TStringBufType*>(this) = MakeStringBuf<TStringBufType>(tokenStart, tokenDelim);
        }

        bool DelimiterIsEmpty() const noexcept {
            return TokenDelim() == DelimiterEnd_;
        }

    private:
        TIterator DelimiterEnd_;
        const TIterator OriginEnd_;
    };

    template <class Base>
    class TSplitRange: public Base, public TInputRangeAdaptor<TSplitRange<Base>> {
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

        template <class Container, class = std::enable_if_t<THasInsert<Container>::value || THasPushBack<Container>::value>>
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

            // FIXME: actually, some kind of TryApplyToMany is needed in order to stop iteration upon first failure
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

        inline size_t Count() {
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
        using TChar = typename TStringType::value_type;
        using TIteratorState = TIterState<TStringType>;
        using TStringBufType = typename TIteratorState::TStringBufType;
        using TIterator = typename TIteratorState::TIterator;

        /**
         * Base class for all split ranges that actually does the splitting.
         */
        template <class DelimStorage>
        struct TSplitRangeBase {
            template <class OtherString, class... Args>
            inline TSplitRangeBase(OtherString&& s, Args&&... args)
                : String_(std::forward<OtherString>(s))
                , State_(String_)
                , Delimiter_(std::forward<Args>(args)...)
            {
            }

            inline TIteratorState* Next() {
                if (State_.DelimiterIsEmpty()) {
                    return nullptr;
                }

                const auto tokenBegin = State_.DelimiterEnd_;
                const auto tokenEnd = Delimiter_.Ptr()->Find(State_.DelimiterEnd_, State_.OriginEnd_);
                State_.UpdateParentBuf(tokenBegin, tokenEnd);

                return &State_;
            }

        private:
            TStringType String_;
            TIteratorState State_;
            DelimStorage Delimiter_;
        };

        template <class Base, class Filter>
        struct TFilterRange: public Base {
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
                return !token->empty();
            }
        };

        template <class TIter>
        struct TStopIteration;

        template <class Base>
        struct TFilters: public Base {
            template <class TFilter>
            using TIt = TSplitRange<TStopIteration<TFilters<TFilterRange<Base, TFilter>>>>;

            template <typename... Args>
            inline TFilters(Args&&... args)
                : Base(std::forward<Args>(args)...)
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
            inline bool Stop(TToken*) noexcept {
                if (Count > 0) {
                    --Count;
                    return false;
                } else {
                    return true;
                }
            }

            size_t Count = 0;
        };

        struct TLimit {
            TLimit() = default;

            TLimit(size_t count)
                : Count(count)
            {
                Y_ASSERT(Count > 0);
            }

            template <class TToken>
            inline bool Stop(TToken* token) noexcept {
                if (Count > 1) {
                    --Count;
                    return false;
                } else if (Count == 1) {
                    token->DelimiterEnd_ = token->OriginEnd_;
                    token->UpdateParentBuf(token->TokenStart(), token->DelimiterEnd_);
                    return false;
                }
                return true;
            }

            size_t Count = 0;
        };

        template <class Base>
        struct TStopIteration: public Base {
            template <class TStopper>
            using TIt = TSplitRange<TStopIteration<TFilters<TStopRange<Base, TStopper>>>>;

            template <typename... Args>
            inline TStopIteration(Args&&... args)
                : Base(std::forward<Args>(args)...)
            {
            }

            inline TIt<TTake> Take(size_t count) {
                return {*this, count};
            }

            inline TIt<TLimit> Limit(size_t count) {
                return {*this, count};
            }
        };

        template <class TPolicy>
        using TIt = TSplitRange<TStopIteration<TFilters<TSplitRangeBase<TPolicy>>>>;

    public:
        template <class OtherString>
        explicit TStringSplitter(OtherString&& s)
            : String_(std::forward<OtherString>(s))
        {
        }

        // does not own TDelim
        template <class TDelim>
        inline TIt<TPtrPolicy<const TDelim>> Split(const TDelim& d) const noexcept {
            return {String_, &d};
        }

        inline TIt<TEmbedPolicy<TCharDelimiter<const TChar>>> Split(TChar ch) const noexcept {
            return {String_, ch};
        }

        inline TIt<TSimpleRefPolicy<TSetDelimiter<const TChar>>> SplitBySet(const TChar* set) const noexcept {
            return {String_, set};
        }

        inline TIt<TEmbedPolicy<TStringDelimiter<const TChar>>> SplitByString(const TStringBufType& str) const noexcept {
            return {String_, str.data(), str.size()};
        }

        template <class TFunc>
        inline TIt<TEmbedPolicy<TFuncDelimiter<TIterator, TFunc>>> SplitByFunc(TFunc f) const noexcept {
            return {String_, f};
        }

    private:
        TStringType String_;
    };

    template <class String>
    auto MakeStringSplitter(String&& s) {
        return TStringSplitter<std::remove_reference_t<String>>(std::forward<String>(s));
    }
}

template <class Iterator>
auto StringSplitter(Iterator begin, Iterator end) {
    return ::NStringSplitPrivate::MakeStringSplitter(TIteratorRange<Iterator>(begin, end));
}

template <class Char>
auto StringSplitter(const Char* begin, const Char* end) {
    return ::NStringSplitPrivate::MakeStringSplitter(TBasicStringBuf<Char>(begin, end));
}

template <class Char>
auto StringSplitter(const Char* begin, size_t len) {
    return ::NStringSplitPrivate::MakeStringSplitter(TBasicStringBuf<Char>(begin, len));
}

template <class Char>
auto StringSplitter(const Char* str) {
    return ::NStringSplitPrivate::MakeStringSplitter(TBasicStringBuf<Char>(str));
}

template <class String, std::enable_if_t<!std::is_pointer<std::remove_reference_t<String>>::value, int> = 0>
auto StringSplitter(String& s) {
    return ::NStringSplitPrivate::MakeStringSplitter(::NStringSplitPrivate::TStringBufOf<String>(s.data(), s.size()));
}

template <class String, std::enable_if_t<!std::is_pointer<std::remove_reference_t<String>>::value, int> = 0>
auto StringSplitter(String&& s) {
    return ::NStringSplitPrivate::MakeStringSplitter(std::move(s));
}
