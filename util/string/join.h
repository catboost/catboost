#pragma once

#include <util/generic/string.h>
#include <util/generic/typetraits.h>
#include "cast.h"

template <typename T>
inline void AppendToString(TString& dst, const T& t) {
    dst.AppendNoAlias(ToString(t));

    // Currently we have only ToString() as a base conversion routine,
    // which allocates and returns temporary string on each call.
    // It would be more efficient to define AppendToString() as the base instead,
    // and then implement ToString(), Out(), Join(), etc. via AppendToString().
}

inline void AppendToString(TString& dst, const char* t) {
    dst.append(t);
}

inline void AppendToString(TString& dst, const TStringBuf t) {
    dst.append(t);
}

inline void AppendJoin(TString&, const TStringBuf) {
}

template <typename TFirst, typename... TRest>
inline void AppendJoin(TString& dst, const TStringBuf delim, const TFirst& f, const TRest&... r) {
    AppendToString(dst, delim);
    AppendToString(dst, f);
    AppendJoin(dst, delim, r...);
}

template <typename TFirst, typename... TRest>
inline TString Join(const TStringBuf delim, const TFirst& f, const TRest&... r) {
    TString ret = ToString(f);
    AppendJoin(ret, delim, r...);
    return ret;
}

// Note that char delimeter @cdelim will be printed as single char string,
// but any char value @v will be printed as corresponding numeric code.
// For example, Join('a', 'a', 'a') will print "97a97" (see unit-test).
template <typename... TValues>
inline TString Join(char cdelim, const TValues&... v) {
    return Join(TStringBuf(&cdelim, 1), v...);
}

template <typename TIter>
inline TString JoinRange(const TStringBuf delim, TIter beg, TIter end) {
    TString out;
    if (beg != end) {
        AppendToString(out, *beg);
        for (++beg; beg != end; ++beg) {
            AppendJoin(out, delim, *beg);
        }
    }

    return out;
}

template <typename TContainer>
TString JoinSeq(const TStringBuf delim, const TContainer& data) {
    return JoinRange(delim, data.begin(), data.end());
}

template <typename T>
TString JoinSeq(const TStringBuf delim, const std::initializer_list<T>& data) {
    return JoinRange(delim, data.begin(), data.end());
}
