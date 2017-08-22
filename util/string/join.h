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

namespace NPrivate {
    template <typename T>
    inline size_t GetLength(const T&) {
        // By default don't pre-allocate space when joining and appending non-string types.
        // This code can be extended by estimating stringified length for specific types (e.g. 10 for ui32).
        return 0;
    }

    template <>
    inline size_t GetLength(const TString& s) {
        return s.length();
    }

    template <>
    inline size_t GetLength(const TStringBuf& s) {
        return s.length();
    }

    template <>
    inline size_t GetLength(const char* const& s) {
        return (s ? TCharTraits<char>::GetLength(s) : 0);
    }

    inline size_t GetAppendLength(const TStringBuf /*delim*/) {
        return 0;
    }

    template <typename TFirst, typename... TRest>
    size_t GetAppendLength(const TStringBuf delim, const TFirst& f, const TRest&... r) {
        return delim.length() + ::NPrivate::GetLength(f) + ::NPrivate::GetAppendLength(delim, r...);
    }
}

inline void AppendJoinNoReserve(TString&, const TStringBuf) {
}

template <typename TFirst, typename... TRest>
inline void AppendJoinNoReserve(TString& dst, const TStringBuf delim, const TFirst& f, const TRest&... r) {
    AppendToString(dst, delim);
    AppendToString(dst, f);
    AppendJoinNoReserve(dst, delim, r...);
}

template <typename... TValues>
inline void AppendJoin(TString& dst, const TStringBuf delim, const TValues&... values) {
    const size_t appendLength = ::NPrivate::GetAppendLength(delim, values...);
    if (appendLength > 0) {
        dst.reserve(dst.length() + appendLength);
    }
    AppendJoinNoReserve(dst, delim, values...);
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
inline TString JoinRange(const TStringBuf delim, const TIter beg, const TIter end) {
    TString out;
    if (beg != end) {
        size_t total = ::NPrivate::GetLength(*beg);
        for (TIter pos = beg; ++pos != end; ) {
            total += delim.length() + ::NPrivate::GetLength(*pos);
        }
        if (total > 0) {
            out.reserve(total);
        }

        AppendToString(out, *beg);
        for (TIter pos = beg; ++pos != end; ) {
            AppendJoinNoReserve(out, delim, *pos);
        }
    }

    return out;
}

template <typename TContainer>
TString JoinSeq(const TStringBuf delim, const TContainer& data) {
    return JoinRange(delim, data.begin(), data.end());
}

/* We force (std::initializer_list<TStringBuf>) input type for (TString) and (const char*) types because:
 * # When (std::initializer_list<TString>) is used, TString objects are copied into the initializer_list object.
 *   Storing TStringBufs instead is faster, even with COW-enabled strings.
 * # For (const char*) we calculate length only once and store it in TStringBuf. Otherwise strlen scan would be executed
 *   in both GetAppendLength and AppendToString. For string literals constant lengths get propagated in compile-time.
 *
 * This way JoinSeq(",", { s1, s2 }) always does the right thing whatever types s1 and s2 have.
 *
 * If someone needs to join std::initializer_list<TString> -- it still works because of the TContainer template above.
*/

template <typename T>
inline
std::enable_if_t<
       !std::is_same<std::decay_t<T>, TString>::value
    && !std::is_same<std::decay_t<T>, const char*>::value,
TString>
JoinSeq(const TStringBuf delim, const std::initializer_list<T>& data) {
    return JoinRange(delim, data.begin(), data.end());
}

inline TString JoinSeq(const TStringBuf delim, const std::initializer_list<TStringBuf>& data) {
    return JoinRange(delim, data.begin(), data.end());
}
