#include "util.h"
#include "split.h"
#include "vector.h"

#include <util/system/defaults.h>

template <class TConsumer, class TDelim, typename TChr>
static inline void DoSplit3(TConsumer& c, TDelim& d, const TFixedString<TChr> str, int) {
    SplitString(str.Start, str.Start + str.Length, d, c);
}

template <class TConsumer, class TDelim, typename TChr>
static inline void DoSplit1(TConsumer& cc, TDelim& d, const TFixedString<TChr> str, int opts) {
    if (opts & KEEP_DELIMITERS) {
        TKeepDelimiters<TConsumer> kc(&cc);

        DoSplit2(kc, d, str, opts);
    } else {
        DoSplit2(cc, d, str, opts);
    }
}

template <class TConsumer, class TDelim, typename TChr>
static inline void DoSplit2(TConsumer& cc, TDelim& d, const TFixedString<TChr> str, int opts) {
    if (opts & KEEP_EMPTY_TOKENS) {
        DoSplit3(cc, d, str, opts);
    } else {
        TSkipEmptyTokens<TConsumer> sc(&cc);

        DoSplit3(sc, d, str, opts);
    }
}

template <class C, class TDelim, typename TChr>
static inline void DoSplit0(C* res, const TFixedString<TChr> str, TDelim& d, size_t maxFields, int options) {
    using TStringType = std::conditional_t<std::is_same<TChr, wchar16>::value, TUtf16String, TString>;
    res->clear();

    if (!str.Start) {
        return;
    }

    using TConsumer = TContainerConsumer<C>;
    TConsumer cc(res);

    if (maxFields) {
        TLimitingConsumer<TConsumer, const TChr> lc(maxFields, &cc);

        DoSplit1(lc, d, str, options);

        if (lc.Last) {
            res->push_back(TStringType(lc.Last, str.Start + str.Length - lc.Last));
        }
    } else {
        DoSplit1(cc, d, str, options);
    }
}

template <typename TChr>
static void SplitStroku(yvector<std::conditional_t<std::is_same<TChr, wchar16>::value, TUtf16String, TString>>* res,
                        const TFixedString<TChr> str, const TChr* delim, size_t maxFields, int options) {
    if (!*delim) {
        return;
    }

    if (*(delim + 1)) {
        TStringDelimiter<const TChr> d(delim, TCharTraits<TChr>::GetLength(delim));

        DoSplit0(res, str, d, maxFields, options);
    } else {
        TCharDelimiter<const TChr> d(*delim);

        DoSplit0(res, str, d, maxFields, options);
    }
}

void SplitStroku(yvector<TString>* res, const char* ptr, const char* delim, size_t maxFields, int options) {
    return SplitStroku<char>(res, TFixedString<char>(ptr), delim, maxFields, options);
}

void SplitStroku(yvector<TString>* res, const char* ptr, size_t len, const char* delim, size_t maxFields, int options) {
    return SplitStroku<char>(res, TFixedString<char>(ptr, len), delim, maxFields, options);
}

void SplitStroku(yvector<TUtf16String>* res, const wchar16* ptr, const wchar16* delimiter, size_t maxFields, int options) {
    return SplitStroku<wchar16>(res, TFixedString<wchar16>(ptr), delimiter, maxFields, options);
}

void SplitStroku(yvector<TUtf16String>* res, const wchar16* ptr, size_t len, const wchar16* delimiter, size_t maxFields, int options) {
    return SplitStroku<wchar16>(res, TFixedString<wchar16>(ptr, len), delimiter, maxFields, options);
}

template <class T>
void SplitStrokuBySetImpl(yvector<T>* res, const typename T::char_type* ptr, const typename T::char_type* delimiters, size_t maxFields, int options) {
    TSetDelimiter<const typename T::char_type> d(delimiters);
    DoSplit0(res, TFixedString<typename T::char_type>(ptr), d, maxFields, options);
}

template <class T>
void SplitStrokuBySetImpl(yvector<T>* res, const typename T::char_type* ptr, size_t len, const typename T::char_type* delimiters, size_t maxFields, int options) {
    TSetDelimiter<const typename T::char_type> d(delimiters);
    DoSplit0(res, TFixedString<typename T::char_type>(ptr, len), d, maxFields, options);
}

void SplitStrokuBySet(yvector<TString>* res, const char* ptr, const char* delimiters, size_t maxFields, int options) {
    SplitStrokuBySetImpl<TString>(res, ptr, delimiters, maxFields, options);
}

void SplitStrokuBySet(yvector<TString>* res, const char* ptr, size_t len, const char* delimiters, size_t maxFields, int options) {
    SplitStrokuBySetImpl<TString>(res, ptr, len, delimiters, maxFields, options);
}

void SplitStrokuBySet(yvector<TUtf16String>* res, const wchar16* ptr, const wchar16* delimiters, size_t maxFields, int options) {
    SplitStrokuBySetImpl<TUtf16String>(res, ptr, delimiters, maxFields, options);
}

void SplitStrokuBySet(yvector<TUtf16String>* res, const wchar16* ptr, size_t len, const wchar16* delimiters, size_t maxFields, int options) {
    SplitStrokuBySetImpl<TUtf16String>(res, ptr, len, delimiters, maxFields, options);
}

TUtf16String JoinStrings(const yvector<TUtf16String>& v, const TWtringBuf delim) {
    return JoinStrings(v.begin(), v.end(), delim);
}

TUtf16String JoinStrings(const yvector<TUtf16String>& v, size_t index, size_t count, const TWtringBuf delim) {
    const size_t f = Min(index, +v);
    const size_t l = f + Min(count, +v - f);

    return JoinStrings(v.begin() + f, v.begin() + l, delim);
}

size_t SplitStroku(char* str, char delim, char* tokens[], size_t maxCount) {
    if (!str)
        return 0;

    size_t i = 0;
    while (i < maxCount) {
        tokens[i++] = str;
        str = strchr(str, delim);
        if (!str)
            break;
        *str++ = 0;
    }

    return i;
}
