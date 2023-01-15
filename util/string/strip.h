#pragma once

#include "ascii.h"

#include <util/generic/string.h>
#include <util/generic/strbuf.h>
#include <utility>

template <class It>
struct TIsAsciiSpaceAdapter {
    bool operator()(const It& it) const noexcept {
        return IsAsciiSpace(*it);
    }
};

template <class It>
TIsAsciiSpaceAdapter<It> IsAsciiSpaceAdapter(It) {
    return {};
}

template <class TChar>
struct TEqualsStripAdapter {
    TEqualsStripAdapter(TChar ch)
        : Ch(ch)
    {
    }

    template <class It>
    bool operator()(const It& it) const noexcept {
        return *it == Ch;
    }

    const TChar Ch;
};

template <class TChar>
TEqualsStripAdapter<TChar> EqualsStripAdapter(TChar ch) {
    return {ch};
}

template <class It, class TStripCriterion>
inline void StripRangeBegin(It& b, const It& e, TStripCriterion&& criterion) noexcept {
    while (b < e && criterion(b)) {
        ++b;
    }
}

template <class It>
inline void StripRangeBegin(It& b, const It& e) noexcept {
    StripRangeBegin(b, e, IsAsciiSpaceAdapter(b));
}

template <class It, class TStripCriterion>
inline void StripRangeEnd(const It& b, It& e, TStripCriterion&& criterion) noexcept {
    while (b < e && criterion(e - 1)) {
        --e;
    }
}

template <class It>
inline void StripRangeEnd(const It& b, It& e) noexcept {
    StripRangeEnd(b, e, IsAsciiSpaceAdapter(b));
}

template <bool stripBeg, bool stripEnd>
struct TStripImpl {
    template <class It, class TStripCriterion>
    static inline bool StripRange(It& b, It& e, TStripCriterion&& criterion) noexcept {
        const size_t oldLen = e - b;

        if (stripBeg) {
            StripRangeBegin(b, e, std::forward<TStripCriterion>(criterion));
        }

        if (stripEnd) {
            StripRangeEnd(b, e, std::forward<TStripCriterion>(criterion));
        }

        const size_t newLen = e - b;
        return newLen != oldLen;
    }

    template <class T, class TStripCriterion>
    static inline bool StripString(const T& from, T& to, TStripCriterion&& criterion) {
        auto b = from.begin();
        auto e = from.end();

        if (StripRange(b, e, std::forward<TStripCriterion>(criterion))) {
            to = T(b, e - b);

            return true;
        }

        to = from;

        return false;
    }

    template <class T, class TStripCriterion>
    static inline T StripString(const T& from, TStripCriterion&& criterion) {
        T ret;
        StripString(from, ret, std::forward<TStripCriterion>(criterion));
        return ret;
    }

    template <class T>
    static inline T StripString(const T& from) {
        return StripString(from, IsAsciiSpaceAdapter(from.begin()));
    }
};

template <class It, class TStripCriterion>
inline bool StripRange(It& b, It& e, TStripCriterion&& criterion) noexcept {
    return TStripImpl<true, true>::StripRange(b, e, std::forward<TStripCriterion>(criterion));
}

template <class It>
inline bool StripRange(It& b, It& e) noexcept {
    return StripRange(b, e, IsAsciiSpaceAdapter(b));
}

template <class It, class TStripCriterion>
inline bool Strip(It& b, size_t& len, TStripCriterion&& criterion) noexcept {
    It e = b + len;

    if (StripRange(b, e, std::forward<TStripCriterion>(criterion))) {
        len = e - b;

        return true;
    }

    return false;
}

template <class It>
inline bool Strip(It& b, size_t& len) noexcept {
    return Strip(b, len, IsAsciiSpaceAdapter(b));
}

template <class T, class TStripCriterion>
static inline bool StripString(const T& from, T& to, TStripCriterion&& criterion) {
    return TStripImpl<true, true>::StripString(from, to, std::forward<TStripCriterion>(criterion));
}

template <class T>
static inline bool StripString(const T& from, T& to) {
    return StripString(from, to, IsAsciiSpaceAdapter(from.begin()));
}

template <class T, class TStripCriterion>
static inline T StripString(const T& from, TStripCriterion&& criterion) {
    return TStripImpl<true, true>::StripString(from, std::forward<TStripCriterion>(criterion));
}

template <class T>
static inline T StripString(const T& from) {
    return TStripImpl<true, true>::StripString(from);
}

template <class T>
static inline T StripStringLeft(const T& from) {
    return TStripImpl<true, false>::StripString(from);
}

template <class T>
static inline T StripStringRight(const T& from) {
    return TStripImpl<false, true>::StripString(from);
}

template <class T, class TStripCriterion>
static inline T StripStringLeft(const T& from, TStripCriterion&& criterion) {
    return TStripImpl<true, false>::StripString(from, std::forward<TStripCriterion>(criterion));
}

template <class T, class TStripCriterion>
static inline T StripStringRight(const T& from, TStripCriterion&& criterion) {
    return TStripImpl<false, true>::StripString(from, std::forward<TStripCriterion>(criterion));
}

/// Copies the given string removing leading and trailing spaces.
static inline bool Strip(const TString& from, TString& to) {
    return StripString(from, to);
}

// deprecated version, use Strip instead
static inline bool strip(const TString& from, TString& to) {
    return StripString(from, to);
}

/// Removes leading and trailing spaces from the string.
inline TString& Strip(TString& s) {
    Strip(s, s);
    return s;
}

// deprecated version, use Strip(TString& s) instead
inline TString& strip(TString& s) {
    Strip(s, s);
    return s;
}

/// Returns a copy of the given string with removed leading and trailing spaces.
inline TString Strip(const TString& s) Y_WARN_UNUSED_RESULT;
inline TString Strip(const TString& s) {
    TString ret = s;
    Strip(ret, ret);
    return ret;
}

// deprecated version, use TString Strip(const TString& s) instead
inline TString strip(const TString& s) Y_WARN_UNUSED_RESULT;
inline TString strip(const TString& s) {
    return Strip(s);
}

template <class TChar, class TWhitespaceFunc>
size_t CollapseImpl(TChar* s, size_t n, const TWhitespaceFunc& isWhitespace) {
    size_t newLen = 0;
    for (size_t i = 0; i < n; ++i, ++newLen) {
        size_t nextNonSpace = i;
        while (nextNonSpace < n && isWhitespace(s[nextNonSpace])) {
            ++nextNonSpace;
        }
        size_t numSpaces = nextNonSpace - i;
        if (numSpaces > 1 || (numSpaces == 1 && s[i] != ' ')) {
            s[newLen] = ' ';
            i = nextNonSpace - 1;
        } else {
            s[newLen] = s[i];
        }
    }
    return newLen;
}

template <class TStringType, class TWhitespaceFunc>
bool CollapseImpl(const TStringType& from, TStringType& to, size_t maxLen, const TWhitespaceFunc& isWhitespace) {
    to = from;
    maxLen = maxLen ? Min(maxLen, to.size()) : to.size();
    for (size_t i = 0; i < maxLen; ++i) {
        if (isWhitespace(to[i]) && (to[i] != ' ' || isWhitespace(to[i + 1]))) {
            size_t tailSize = maxLen - i;
            size_t newTailSize = CollapseImpl(to.begin() + i, tailSize, isWhitespace);
            to.remove(i + newTailSize, tailSize - newTailSize);
            return true;
        }
    }
    return false;
}

bool Collapse(const TString& from, TString& to, size_t maxLen = 0);

// deprecated version, use Collapse instead
inline bool collapse(const TString& from, TString& to, size_t maxLen = 0) {
    return Collapse(from, to, maxLen);
}

/// Replaces several consequtive space symbols with one (processing is limited to maxLen bytes)
inline TString& Collapse(TString& s, size_t maxLen = 0) {
    Collapse(s, s, maxLen);
    return s;
}
// deprecated version, use Collapse instead
inline TString& collapse(TString& s, size_t maxLen = 0) {
    Collapse(s, s, maxLen);
    return s;
}

/// Replaces several consequtive space symbols with one (processing is limited to maxLen bytes)
inline TString Collapse(const TString& s, size_t maxLen = 0) {
    TString ret;
    Collapse(s, ret, maxLen);
    return ret;
}
// deprecated version, use Collapse instead
inline TString collapse(const TString& s, size_t maxLen = 0) {
    return Collapse(s, maxLen);
}

void CollapseText(const TString& from, TString& to, size_t maxLen);

/// The same as Collapse() + truncates the string to maxLen.
/// @details An ellipsis is inserted at the end of the truncated line.
inline void CollapseText(TString& s, size_t maxLen) {
    TString to;
    CollapseText(s, to, maxLen);
    s = to;
}
