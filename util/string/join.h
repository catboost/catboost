#pragma once

#include <util/generic/string.h>
#include <util/generic/typetraits.h>
#include <util/string/cast.h>
#include "cast.h"

/*
 * Default implementation of AppendToString uses a temporary TString object which is inefficient. You can overload it
 * for your type to speed up string joins. If you already have an Out() or operator<<() implementation you can simply
 * do the following:
 *
 *      inline void AppendToString(TString& dst, const TMyType& t) {
 *          TStringOutput o(dst);
 *          o << t;
 *      }
 *
 * Unfortunately we can't do this by default because for some types ToString() is defined while Out() is not.
 * For standard types (strings of all kinds and arithmetic types) we don't use a temporary TString in AppendToString().
 */

template <typename T>
inline std::enable_if_t<!std::is_arithmetic<std::remove_cv_t<T>>::value, void>
AppendToString(TString& dst, const T& t) {
    dst.AppendNoAlias(ToString(t));
}

template <typename T>
inline std::enable_if_t<std::is_arithmetic<std::remove_cv_t<T>>::value, void>
AppendToString(TString& dst, const T& t) {
    char buf[512];
    dst.append(buf, ToString<std::remove_cv_t<T>>(t, buf, sizeof(buf)));
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
        for (TIter pos = beg; ++pos != end;) {
            total += delim.length() + ::NPrivate::GetLength(*pos);
        }
        if (total > 0) {
            out.reserve(total);
        }

        AppendToString(out, *beg);
        for (TIter pos = beg; ++pos != end;) {
            AppendJoinNoReserve(out, delim, *pos);
        }
    }

    return out;
}

template <typename TContainer>
TString JoinSeq(const TStringBuf delim, const TContainer& data) {
    using std::begin;
    using std::end;
    return JoinRange(delim, begin(data), end(data));
}

/** \brief Functor for streaming iterative objects from TIterB e to TIterE b, separated with delim.
 *         Difference from JoinSeq, JoinRange, Join is the lack of TString object - all depends on operator<< for the type and
 *         realization of IOutputStream
 */
template<class TIterB, class TIterE>
struct TRangeJoiner{
    friend constexpr IOutputStream& operator<<(IOutputStream& stream, const TRangeJoiner<TIterB, TIterE>& rangeJoiner) {
        if(rangeJoiner.b != rangeJoiner.e) {
            stream << *rangeJoiner.b;

            for(auto it = std::next(rangeJoiner.b); it != rangeJoiner.e; ++it)
                stream << rangeJoiner.delim << *it;
        }
        return stream;
    }

    constexpr TRangeJoiner(TStringBuf delim, TIterB && b, TIterE && e) : delim(delim), b(std::forward<TIterB>(b)), e(std::forward<TIterE>(e)) {}
private:
    const TStringBuf delim;
    const TIterB b;
    const TIterE e;
};

template<class TIterB, class TIterE = TIterB> constexpr auto MakeRangeJoiner(TStringBuf delim, TIterB && b, TIterE && e) {
    return TRangeJoiner<TIterB, TIterE>(delim, std::forward<TIterB>(b), std::forward<TIterE>(e));
}

template<class TContainer> constexpr auto MakeRangeJoiner(TStringBuf delim, const TContainer& data) {
    return MakeRangeJoiner(delim, std::cbegin(data), std::cend(data));
}

template<class TVal> constexpr auto MakeRangeJoiner(TStringBuf delim, const std::initializer_list<TVal>& data) {
    return MakeRangeJoiner(delim, std::cbegin(data), std::cend(data));
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
inline std::enable_if_t<
    !std::is_same<std::decay_t<T>, TString>::value && !std::is_same<std::decay_t<T>, const char*>::value,
    TString>
JoinSeq(const TStringBuf delim, const std::initializer_list<T>& data) {
    return JoinRange(delim, data.begin(), data.end());
}

inline TString JoinSeq(const TStringBuf delim, const std::initializer_list<TStringBuf>& data) {
    return JoinRange(delim, data.begin(), data.end());
}
