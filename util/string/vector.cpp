#include "util.h"
#include "split.h"
#include "vector.h"

#include <util/system/defaults.h>

template <class TConsumer, class TDelim, typename TChr>
static inline void DoSplit2(TConsumer& c, TDelim& d, const TBasicStringBuf<TChr> str, int) {
    SplitString(str.data(), str.data() + str.size(), d, c);
}

template <class TConsumer, class TDelim, typename TChr>
static inline void DoSplit1(TConsumer& cc, TDelim& d, const TBasicStringBuf<TChr> str, int opts) {
    if (opts & KEEP_EMPTY_TOKENS) {
        DoSplit2(cc, d, str, opts);
    } else {
        TSkipEmptyTokens<TConsumer> sc(&cc);

        DoSplit2(sc, d, str, opts);
    }
}

template <class C, class TDelim, typename TChr>
static inline void DoSplit0(C* res, const TBasicStringBuf<TChr> str, TDelim& d, size_t maxFields, int options) {
    using TStringType = std::conditional_t<std::is_same<TChr, wchar16>::value, TUtf16String, TString>;
    res->clear();

    if (!str.data()) {
        return;
    }

    using TConsumer = TContainerConsumer<C>;
    TConsumer cc(res);

    if (maxFields) {
        TLimitingConsumer<TConsumer, const TChr> lc(maxFields, &cc);

        DoSplit1(lc, d, str, options);

        if (lc.Last) {
            res->push_back(TStringType(lc.Last, str.data() + str.size() - lc.Last));
        }
    } else {
        DoSplit1(cc, d, str, options);
    }
}

template <typename TChr>
static void SplitStringImplT(TVector<std::conditional_t<std::is_same<TChr, wchar16>::value, TUtf16String, TString>>* res,
                             const TBasicStringBuf<TChr> str, const TChr* delim, size_t maxFields, int options) {
    if (!*delim) {
        return;
    }

    if (*(delim + 1)) {
        TStringDelimiter<const TChr> d(delim, std::char_traits<TChr>::length(delim));

        DoSplit0(res, str, d, maxFields, options);
    } else {
        TCharDelimiter<const TChr> d(*delim);

        DoSplit0(res, str, d, maxFields, options);
    }
}

void ::NPrivate::SplitStringImpl(TVector<TString>* res, const char* ptr, const char* delim, size_t maxFields, int options) {
    return SplitStringImplT<char>(res, TStringBuf(ptr), delim, maxFields, options);
}

void ::NPrivate::SplitStringImpl(TVector<TString>* res, const char* ptr, size_t len, const char* delim, size_t maxFields, int options) {
    return SplitStringImplT<char>(res, TStringBuf(ptr, len), delim, maxFields, options);
}

void ::NPrivate::SplitStringImpl(TVector<TUtf16String>* res, const wchar16* ptr, const wchar16* delimiter, size_t maxFields, int options) {
    return SplitStringImplT<wchar16>(res, TWtringBuf(ptr), delimiter, maxFields, options);
}

void ::NPrivate::SplitStringImpl(TVector<TUtf16String>* res, const wchar16* ptr, size_t len, const wchar16* delimiter, size_t maxFields, int options) {
    return SplitStringImplT<wchar16>(res, TWtringBuf(ptr, len), delimiter, maxFields, options);
}

TUtf16String JoinStrings(const TVector<TUtf16String>& v, const TWtringBuf delim) {
    return JoinStrings(v.begin(), v.end(), delim);
}

TUtf16String JoinStrings(const TVector<TUtf16String>& v, size_t index, size_t count, const TWtringBuf delim) {
    const size_t f = Min(index, v.size());
    const size_t l = f + Min(count, v.size() - f);

    return JoinStrings(v.begin() + f, v.begin() + l, delim);
}
