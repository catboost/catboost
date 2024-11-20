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

template <typename TCharType, typename T>
inline std::enable_if_t<!std::is_arithmetic<std::remove_cv_t<T>>::value, void>
AppendToString(TBasicString<TCharType>& dst, const T& t) {
    dst.AppendNoAlias(ToString(t));
}

template <typename TCharType, typename T>
inline std::enable_if_t<std::is_arithmetic<std::remove_cv_t<T>>::value, void>
AppendToString(TBasicString<TCharType>& dst, const T& t) {
    char buf[512];
    dst.append(buf, ToString<std::remove_cv_t<T>>(t, buf, sizeof(buf)));
}

template <typename TCharType>
inline void AppendToString(TBasicString<TCharType>& dst, const TCharType* t) {
    dst.append(t);
}

template <typename TCharType>
inline void AppendToString(TBasicString<TCharType>& dst, TBasicStringBuf<TCharType> t) {
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
        return (s ? std::char_traits<char>::length(s) : 0);
    }

    inline size_t GetAppendLength(const TStringBuf /*delim*/) {
        return 0;
    }

    template <typename TFirst, typename... TRest>
    size_t GetAppendLength(const TStringBuf delim, const TFirst& f, const TRest&... r) {
        return delim.length() + ::NPrivate::GetLength(f) + ::NPrivate::GetAppendLength(delim, r...);
    }
} // namespace NPrivate

template <typename TCharType>
inline void AppendJoinNoReserve(TBasicString<TCharType>&, TBasicStringBuf<TCharType>) {
}

template <typename TCharType, typename TFirst, typename... TRest>
inline void AppendJoinNoReserve(TBasicString<TCharType>& dst, TBasicStringBuf<TCharType> delim, const TFirst& f, const TRest&... r) {
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

namespace NPrivate {
    template <typename TCharType, typename TIter>
    inline TBasicString<TCharType> JoinRange(TBasicStringBuf<TCharType> delim, const TIter beg, const TIter end) {
        TBasicString<TCharType> out;
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

} // namespace NPrivate

template <typename TIter>
TString JoinRange(std::string_view delim, const TIter beg, const TIter end) {
    return ::NPrivate::JoinRange<char>(delim, beg, end);
}

template <typename TIter>
TString JoinRange(char delim, const TIter beg, const TIter end) {
    TStringBuf delimBuf(&delim, 1);
    return ::NPrivate::JoinRange<char>(delimBuf, beg, end);
}

template <typename TIter>
TUtf16String JoinRange(std::u16string_view delim, const TIter beg, const TIter end) {
    return ::NPrivate::JoinRange<wchar16>(delim, beg, end);
}

template <typename TIter>
TUtf16String JoinRange(wchar16 delim, const TIter beg, const TIter end) {
    TWtringBuf delimBuf(&delim, 1);
    return ::NPrivate::JoinRange<wchar16>(delimBuf, beg, end);
}

template <typename TIter>
TUtf32String JoinRange(std::u32string_view delim, const TIter beg, const TIter end) {
    return ::NPrivate::JoinRange<wchar32>(delim, beg, end);
}

template <typename TIter>
TUtf32String JoinRange(wchar32 delim, const TIter beg, const TIter end) {
    TUtf32StringBuf delimBuf(&delim, 1);
    return ::NPrivate::JoinRange<wchar32>(delimBuf, beg, end);
}

template <typename TCharType, typename TContainer>
inline TBasicString<TCharType> JoinSeq(std::basic_string_view<TCharType> delim, const TContainer& data) {
    using std::begin;
    using std::end;
    return JoinRange(delim, begin(data), end(data));
}

template <typename TCharType, typename TContainer>
inline TBasicString<TCharType> JoinSeq(const TCharType* delim, const TContainer& data) {
    TBasicStringBuf<TCharType> delimBuf = delim;
    return JoinSeq(delimBuf, data);
}

template <typename TCharType, typename TContainer>
inline TBasicString<TCharType> JoinSeq(const TBasicString<TCharType>& delim, const TContainer& data) {
    TBasicStringBuf<TCharType> delimBuf = delim;
    return JoinSeq(delimBuf, data);
}

template <typename TCharType, typename TContainer>
inline std::enable_if_t<
    std::is_same_v<TCharType, char> ||
        std::is_same_v<TCharType, char16_t> ||
        std::is_same_v<TCharType, char32_t>,
    TBasicString<TCharType>>
JoinSeq(TCharType delim, const TContainer& data) {
    TBasicStringBuf<TCharType> delimBuf(&delim, 1);
    return JoinSeq(delimBuf, data);
}

/** \brief Functor for streaming iterative objects from TIterB e to TIterE b, separated with delim.
 *         Difference from JoinSeq, JoinRange, Join is the lack of TString object - all depends on operator<< for the type and
 *         realization of IOutputStream
 */
template <class TIterB, class TIterE>
struct TRangeJoiner {
    friend constexpr IOutputStream& operator<<(IOutputStream& stream Y_LIFETIME_BOUND, const TRangeJoiner<TIterB, TIterE>& rangeJoiner) {
        if (rangeJoiner.b != rangeJoiner.e) {
            stream << *rangeJoiner.b;

            for (auto it = std::next(rangeJoiner.b); it != rangeJoiner.e; ++it) {
                stream << rangeJoiner.delim << *it;
            }
        }
        return stream;
    }

    constexpr TRangeJoiner(TStringBuf delim, TIterB&& b, TIterE&& e)
        : delim(delim)
        , b(std::forward<TIterB>(b))
        , e(std::forward<TIterE>(e))
    {
    }

private:
    const TStringBuf delim;
    const TIterB b;
    const TIterE e;
};

template <class TIterB, class TIterE = TIterB>
constexpr auto MakeRangeJoiner(TStringBuf delim, TIterB&& b, TIterE&& e) {
    return TRangeJoiner<TIterB, TIterE>(delim, std::forward<TIterB>(b), std::forward<TIterE>(e));
}

template <class TContainer>
constexpr auto MakeRangeJoiner(TStringBuf delim, const TContainer& data) {
    return MakeRangeJoiner(delim, std::cbegin(data), std::cend(data));
}

template <class TVal>
constexpr auto MakeRangeJoiner(TStringBuf delim, const std::initializer_list<TVal>& data) {
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
