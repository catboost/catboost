#pragma once

#include "strspn.h"
#include "cast.h"

#include <util/system/compat.h>
#include <util/system/defaults.h>
#include <util/generic/fwd.h>
#include <util/generic/string.h>
#include <util/generic/strbuf.h>
#include <util/generic/typetraits.h>
#include <util/generic/ylimits.h>
#include <util/generic/vector.h>

//
// NOTE: Check util/string/iterator.h to get more convenient split string interface.

namespace NPrivate {

    template <class T, class I, class = void>
    struct TIsConsumer : std::false_type {};

    template <class T, class I>
    struct TIsConsumer<
        T, I,
        TVoidT<decltype(std::declval<T>().Consume(
            std::declval<I>(), std::declval<I>(), std::declval<I>()))>>
        : std::true_type {};

    template <class T, class I>
    constexpr bool TIsConsumerV = TIsConsumer<T, I>::value;

}

template <class I, class TDelim, class TConsumer>
std::enable_if_t<::NPrivate::TIsConsumerV<TConsumer, I>>
SplitString(I b, I e, const TDelim& d, TConsumer&& c) {
    I l, i;

    do {
        l = b;
        i = d.Find(b, e);
    } while (c.Consume(l, i, b) && (b != i));
}

template <class I, class TDelim, class TConsumer>
std::enable_if_t<::NPrivate::TIsConsumerV<TConsumer, I>>
SplitString(I b, const TDelim& d, TConsumer&& c) {
    I l, i;

    do {
        l = b;
        i = d.Find(b);
    } while (c.Consume(l, i, b) && (b != i));
}

template <class I1, class I2>
static inline I1* FastStrChr(I1* str, I2 f) noexcept {
    I1* ret = (I1*)TCharTraits<I1>::Find(str, f);

    if (!ret) {
        ret = str + TCharTraits<I1>::GetLength(str);
    }

    return ret;
}

template <class I>
static inline I* FastStrStr(I* str, I* f, size_t l) noexcept {
    (void)l;
    I* ret = (I*)TCharTraits<I>::Find(str, *f);

    if (ret) {
        ret = (I*)TCharTraits<I>::Find(ret, f);
    }

    if (!ret) {
        ret = str + TCharTraits<I>::GetLength(str);
    }

    return ret;
}

template <class I>
struct TStringDelimiter {
    inline TStringDelimiter(I* delim) noexcept
        : Delim(delim)
        , Len(TCharTraits<I>::GetLength(delim))
    {
    }

    inline TStringDelimiter(I* delim, size_t len) noexcept
        : Delim(delim)
        , Len(len)
    {
    }

    inline I* Find(I*& b, I* e) const noexcept {
        I* ret = (I*)TCharTraits<I>::Find(b, e - b, Delim, Len);

        if (ret) {
            b = ret + Len;
            return ret;
        }

        return (b = e);
    }

    inline I* Find(I*& b) const noexcept {
        I* ret = FastStrStr(b, Delim, Len);

        if (*ret) {
            b = ret + Len;
        } else {
            b = ret;
        }

        return ret;
    }

    I* Delim;
    const size_t Len;
};

template <class I>
struct TCharDelimiter {
    inline TCharDelimiter(I ch) noexcept
        : Ch(ch)
    {
    }

    inline I* Find(I*& b, I* e) const noexcept {
        I* ret = (I*)TCharTraits<I>::Find(b, Ch, e - b);

        if (ret) {
            b = ret + 1;
            return ret;
        }

        return (b = e);
    }

    inline I* Find(I*& b) const noexcept {
        I* ret = FastStrChr(b, Ch);

        if (*ret) {
            b = ret + 1;
        } else {
            b = ret;
        }

        return ret;
    }

    I Ch;
};

template <class I, class TFunc>
struct TFuncDelimiter {
public:
    TFuncDelimiter(const TFunc& f)
        : Fn(f)
    {
    }

    inline I Find(I& b, I e) const noexcept {
        if ((b = std::find_if(b, e, Fn)) != e) {
            return b++;
        }

        return b;
    }

private:
    TFunc Fn;
};

template <class I, class TDelim>
class TLimitedDelimiter {
public:
    template <class... Args>
    TLimitedDelimiter(size_t limit, Args&&... args)
        : Delim(std::forward<Args>(args)...)
        , Limit(limit)
    {
        Y_ASSERT(limit > 0);
    }

    inline I Find(I& b, I e) noexcept {
        if (Limit > 1) {
            --Limit;
            return Delim.Find(b, e);
        } else {
            return (b = e);
        }
    }

private:
    TDelim Delim;
    size_t Limit = Max<size_t>();
};

template <class I>
struct TFindFirstOf {
    inline TFindFirstOf(I* set)
        : Set(set)
    {
    }

    //TODO
    inline I* FindFirstOf(I* b, I* e) const noexcept {
        I* ret = b;
        for (; ret != e; ++ret) {
            if (TCharTraits<I>::Find(Set, *ret))
                break;
        }
        return ret;
    }

    inline I* FindFirstOf(I* b) const noexcept {
        return b + TCharTraits<I>::FindFirstOf(b, Set);
    }

    I* Set;
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

template <class I>
struct TSetDelimiter: private TFindFirstOf<const I> {
    using TFindFirstOf<const I>::TFindFirstOf;

    inline I* Find(I*& b, I* e) const noexcept {
        I* ret = const_cast<I*>(this->FindFirstOf(b, e));

        if (ret != e) {
            b = ret + 1;
            return ret;
        }

        return (b = e);
    }

    inline I* Find(I*& b) const noexcept {
        I* ret = const_cast<I*>(this->FindFirstOf(b));

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
    Y_ENSURE(next.IsInited(), AsStringBuf("Split: number of fields less than number of Split output arguments"));
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
// Split(AsStringBuf("Sherlock,2014,36.6"), ',', name, year, temperature);
template <class D, class P1, class P2>
void Split(TStringBuf s, D delim, P1& p1, P2& p2) {
    GetNext(s, delim, p1);
    GetNext(s, delim, p2);
    Y_ENSURE(!s.IsInited(), AsStringBuf("Split: number of fields more than number of Split output arguments"));
}

template <class D, class P1, class P2, class... Other>
void Split(TStringBuf s, D delim, P1& p1, P2& p2, Other&... other) {
    GetNext(s, delim, p1);
    Split(s, delim, p2, other...);
}
