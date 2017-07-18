#pragma once

#include "cast.h"

#include <util/generic/map.h>
#include <util/generic/strbuf.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/system/yassert.h>

#define KEEP_EMPTY_TOKENS 0x01
#define KEEP_DELIMITERS 0x02

//
// NOTE: Check util/string/iterator.h to get more convenient split string interface.

void SplitStroku(yvector<TString>* res, const char* ptr, const char* delimiter, size_t maxFields = 0, int options = 0);
void SplitStroku(yvector<TString>* res, const char* ptr, size_t len, const char* delimiter, size_t maxFields = 0, int options = 0);

void SplitStrokuBySet(yvector<TString>* res, const char* ptr, const char* delimiters, size_t maxFields = 0, int options = 0);
void SplitStrokuBySet(yvector<TString>* res, const char* ptr, size_t len, const char* delimiters, size_t maxFields = 0, int options = 0);

void SplitStrokuBySet(yvector<TUtf16String>* res, const wchar16* ptr, const wchar16* delimiters, size_t maxFields = 0, int options = 0);
void SplitStrokuBySet(yvector<TUtf16String>* res, const wchar16* ptr, size_t len, const wchar16* delimiters, size_t maxFields = 0, int options = 0);

inline void SplitStroku(yvector<TString>* res, const TString& str, const char* delimiter, size_t maxFields = 0, int options = 0) {
    SplitStroku(res, ~str, +str, delimiter, maxFields, options);
}

void SplitStroku(yvector<TUtf16String>* res, const wchar16* ptr, const wchar16* delimiter, size_t maxFields = 0, int options = 0);
void SplitStroku(yvector<TUtf16String>* res, const wchar16* ptr, size_t len, const wchar16* delimiter, size_t maxFields = 0, int options = 0);

inline void SplitStroku(yvector<TUtf16String>* res, const TUtf16String& str, const wchar16* delimiter, size_t maxFields = 0, int options = 0) {
    SplitStroku(res, ~str, +str, delimiter, maxFields, options);
}

inline yvector<TString> splitStroku(const char* ptr, const char* delimiter, size_t maxFields = 0, int options = 0) {
    yvector<TString> res;
    SplitStroku(&res, ptr, delimiter, maxFields, options);
    return res;
}

inline yvector<TString> splitStroku(const char* ptr, size_t len, const char* delimiter, size_t maxFields = 0, int options = 0) {
    yvector<TString> res;
    SplitStroku(&res, ptr, len, delimiter, maxFields, options);
    return res;
}

inline yvector<TString> splitStroku(const TString& str, const char* delimiter, size_t maxFields = 0, int options = 0) {
    return splitStroku(~str, +str, delimiter, maxFields, options);
}

inline yvector<TString> splitStrokuBySet(const char* ptr, const char* delimiters, size_t maxFields = 0, int options = 0) {
    yvector<TString> res;
    SplitStrokuBySet(&res, ptr, delimiters, maxFields, options);
    return res;
}

inline yvector<TString> splitStrokuBySet(const char* ptr, size_t len, const char* delimiters, size_t maxFields = 0, int options = 0) {
    yvector<TString> res;
    SplitStrokuBySet(&res, ptr, len, delimiters, maxFields, options);
    return res;
}

/// Splits input string by given delimiter character.
/*! @param[in, out] str input string
        (will be modified: delimiter will be replaced by NULL character)
    @param[in] delim delimiter character
    @param[out] arr output array of substrings
    @param[in] maxCount max number of substrings to return
    @return count of substrings
*/
size_t SplitStroku(char* str, char delim, char* arr[], size_t maxCount);

template <class TIter>
inline TString JoinStrings(TIter begin, TIter end, const TStringBuf delim) {
    if (begin == end)
        return TString();

    TString result = ToString(*begin);

    for (++begin; begin != end; ++begin) {
        result.append(delim);
        result.append(ToString(*begin));
    }

    return result;
}

template <class TIter>
inline TUtf16String JoinStrings(TIter begin, TIter end, const TWtringBuf delim) {
    if (begin == end)
        return TUtf16String();

    TUtf16String result = ToWtring(*begin);

    for (++begin; begin != end; ++begin) {
        result.append(delim);
        result.append(ToWtring(*begin));
    }

    return result;
}

/// Concatenates elements of given yvector<TString>.
inline TString JoinStrings(const yvector<TString>& v, const TStringBuf delim) {
    return JoinStrings(v.begin(), v.end(), delim);
}

inline TString JoinStrings(const yvector<TString>& v, size_t index, size_t count, const TStringBuf delim) {
    Y_ASSERT(index + count <= v.size() && "JoinStrings(): index or count out of range");
    return JoinStrings(v.begin() + index, v.begin() + index + count, delim);
}

template <typename T>
inline TString JoinVectorIntoStroku(const yvector<T>& v, const TStringBuf delim) {
    return JoinStrings(v.begin(), v.end(), delim);
}

template <typename T>
inline TString JoinVectorIntoStroku(const yvector<T>& v, size_t index, size_t count, const TStringBuf delim) {
    Y_ASSERT(index + count <= v.size() && "JoinVectorIntoStroku(): index or count out of range");
    return JoinStrings(v.begin() + index, v.begin() + index + count, delim);
}

TUtf16String JoinStrings(const yvector<TUtf16String>& v, const TWtringBuf delim);
TUtf16String JoinStrings(const yvector<TUtf16String>& v, size_t index, size_t count, const TWtringBuf delim);

//! Converts vector of strings to vector of type T variables
template <typename T, typename TStringType>
yvector<T> Scan(const yvector<TStringType>& input) {
    yvector<T> output;
    output.reserve(input.size());
    for (int i = 0; i < input.ysize(); ++i) {
        output.push_back(FromString<T>(input[i]));
    }
    return output;
}
