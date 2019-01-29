#pragma once

#include "fwd.h"
#include "string.h"
#include "utility.h"
#include "typetraits.h"

template <typename TChar, typename TTraits>
class TStringBufImpl: public TFixedString<TChar, TTraits>, public TStringBase<TStringBufImpl<TChar, TTraits>, TChar, TTraits> {
    using TdSelf = TStringBufImpl;
    using TBaseStr = TFixedString<TChar, TTraits>;
    using TBase = TStringBase<TdSelf, TChar, TTraits>;

public:
    using char_type = TChar;
    using traits_type = TTraits;

    constexpr inline TStringBufImpl(const TChar* data, size_t len) noexcept
        : TBaseStr(data, len)
    {
    }

    inline TStringBufImpl(const TChar* data) noexcept
        : TBaseStr(data, TBase::StrLen(data))
    {
    }

    inline TStringBufImpl(const TChar* beg, const TChar* end) noexcept
        : TBaseStr(beg, end)
    {
        Y_ASSERT(beg <= end);
    }

    template <typename D, typename T>
    inline TStringBufImpl(const TStringBase<D, TChar, T>& str) noexcept
        : TBaseStr(str)
    {
    }

    template <typename T, typename A>
    inline TStringBufImpl(const std::basic_string<TChar, T, A>& str) noexcept
        : TBaseStr(str)
    {
    }

    constexpr TStringBufImpl() noexcept
        : TBaseStr()
    {
    }

    constexpr inline TStringBufImpl(const TBaseStr& src) noexcept
        : TBaseStr(src)
    {
    }

    inline TStringBufImpl(const TBaseStr& src, size_t pos, size_t n) noexcept
        : TBaseStr(src)
    {
        Skip(pos).Trunc(n);
    }

    inline TStringBufImpl(const TBaseStr& src, size_t pos) noexcept
        : TStringBufImpl(src, pos, TBase::npos)
    {
    }

public: // required by TStringBase
    constexpr inline const TChar* data() const noexcept {
        return Start;
    }

    constexpr inline size_t length() const noexcept {
        return Length;
    }

public:
    void Clear() {
        *this = TdSelf();
    }

    constexpr bool IsInited() const noexcept {
        return nullptr != Start;
    }

public:
    /**
     * Tries to split string in two parts using given delimiter character.
     * Searches for the delimiter, scanning string from the beginning.
     * The delimiter is excluded from the result. Both out parameters are
     * left unmodified if there was no delimiter character in string.
     *
     * @param[in] delim                 Delimiter character.
     * @param[out] l                    The first part of split result.
     * @param[out] r                    The second part of split result.
     * @returns                         Whether the split was actually performed.
     */
    inline bool TrySplit(TChar delim, TdSelf& l, TdSelf& r) const noexcept {
        return TrySplitOn(TBase::find(delim), l, r);
    }

    /**
     * Tries to split string in two parts using given delimiter character.
     * Searches for the delimiter, scanning string from the end.
     * The delimiter is excluded from the result. Both out parameters are
     * left unmodified if there was no delimiter character in string.
     *
     * @param[in] delim                 Delimiter character.
     * @param[out] l                    The first part of split result.
     * @param[out] r                    The second part of split result.
     * @returns                         Whether the split was actually performed.
     */
    inline bool TryRSplit(TChar delim, TdSelf& l, TdSelf& r) const noexcept {
        return TrySplitOn(TBase::rfind(delim), l, r);
    }

    /**
     * Tries to split string in two parts using given delimiter sequence.
     * Searches for the delimiter, scanning string from the beginning.
     * The delimiter sequence is excluded from the result. Both out parameters
     * are left unmodified if there was no delimiter character in string.
     *
     * @param[in] delim                 Delimiter sequence.
     * @param[out] l                    The first part of split result.
     * @param[out] r                    The second part of split result.
     * @returns                         Whether the split was actually performed.
     */
    inline bool TrySplit(TdSelf delim, TdSelf& l, TdSelf& r) const noexcept {
        return TrySplitOn(TBase::find(delim), l, r, delim.size());
    }

    /**
     * Tries to split string in two parts using given delimiter sequence.
     * Searches for the delimiter, scanning string from the end.
     * The delimiter sequence is excluded from the result. Both out parameters
     * are left unmodified if there was no delimiter character in string.
     *
     * @param[in] delim                 Delimiter sequence.
     * @param[out] l                    The first part of split result.
     * @param[out] r                    The second part of split result.
     * @returns                         Whether the split was actually performed.
     */
    inline bool TryRSplit(TdSelf delim, TdSelf& l, TdSelf& r) const noexcept {
        return TrySplitOn(TBase::rfind(delim), l, r, delim.size());
    }

    inline void Split(TChar delim, TdSelf& l, TdSelf& r) const noexcept {
        SplitTemplate(delim, l, r);
    }

    inline void RSplit(TChar delim, TdSelf& l, TdSelf& r) const noexcept {
        RSplitTemplate(delim, l, r);
    }

    inline void Split(TdSelf delim, TdSelf& l, TdSelf& r) const noexcept {
        SplitTemplate(delim, l, r);
    }

    inline void RSplit(TdSelf delim, TdSelf& l, TdSelf& r) const noexcept {
        RSplitTemplate(delim, l, r);
    }

private:
    // splits on a delimiter at a given position; delimiter is excluded
    void DoSplitOn(size_t pos, TdSelf& l, TdSelf& r, size_t len) const noexcept {
        Y_ASSERT(pos != TBase::npos);

        // make a copy in case one of l/r is really *this
        const TdSelf tok = SubStr(pos + len);
        l = Head(pos);
        r = tok;
    }

public:
    // In all methods below with @pos parameter, @pos is supposed to be
    // a result of string find()/rfind()/find_first() or other similiar functions,
    // returning either position within string length [0..Length) or npos.
    // For all other @pos values (out of string index range) the behaviour isn't well defined
    // For example, for TStringBuf s("abc"):
    // s.TrySplitOn(s.find('z'), ...) is false, but s.TrySplitOn(100500, ...) is true.

    bool TrySplitOn(size_t pos, TdSelf& l, TdSelf& r, size_t len = 1) const noexcept {
        if (TBase::npos == pos)
            return false;

        DoSplitOn(pos, l, r, len);
        return true;
    }

    void SplitOn(size_t pos, TdSelf& l, TdSelf& r, size_t len = 1) const noexcept {
        if (!TrySplitOn(pos, l, r, len)) {
            l = *this;
            r = TdSelf();
        }
    }

    bool TrySplitAt(size_t pos, TdSelf& l, TdSelf& r) const noexcept {
        return TrySplitOn(pos, l, r, 0);
    }

    void SplitAt(size_t pos, TdSelf& l, TdSelf& r) const noexcept {
        SplitOn(pos, l, r, 0);
    }

    /*
    // Not implemented intentionally, use TrySplitOn() instead
    void RSplitOn(size_t pos, TdSelf& l, TdSelf& r) const noexcept;
    void RSplitAt(size_t pos, TdSelf& l, TdSelf& r) const noexcept;
*/

public:
    Y_PURE_FUNCTION
    inline TdSelf After(TChar c) const noexcept {
        TdSelf l, r;
        return TrySplit(c, l, r) ? r : *this;
    }

    Y_PURE_FUNCTION
    inline TdSelf Before(TChar c) const noexcept {
        TdSelf l, r;
        return TrySplit(c, l, r) ? l : *this;
    }

    Y_PURE_FUNCTION
    inline TdSelf RAfter(TChar c) const noexcept {
        TdSelf l, r;
        return TryRSplit(c, l, r) ? r : *this;
    }

    Y_PURE_FUNCTION
    inline TdSelf RBefore(TChar c) const noexcept {
        TdSelf l, r;
        return TryRSplit(c, l, r) ? l : *this;
    }

public:
    inline bool AfterPrefix(const TdSelf& prefix, TdSelf& result) const noexcept {
        if (this->StartsWith(prefix)) {
            result = Tail(prefix.Length);
            return true;
        }
        return false;
    }

    inline bool BeforeSuffix(const TdSelf& suffix, TdSelf& result) const noexcept {
        if (this->EndsWith(suffix)) {
            result = Head(Length - suffix.Length);
            return true;
        }
        return false;
    }

    inline bool SkipPrefix(const TdSelf& prefix) noexcept {
        return AfterPrefix(prefix, *this);
    }

    inline bool ChopSuffix(const TdSelf& suffix) noexcept {
        return BeforeSuffix(suffix, *this);
    }

public:
    // returns tail, including pos
    TdSelf SplitOffAt(size_t pos) {
        const TdSelf tok = SubStr(pos);
        Trunc(pos);
        return tok;
    }

    // returns head, tail includes pos
    TdSelf NextTokAt(size_t pos) {
        const TdSelf tok = Head(pos);
        Skip(pos);
        return tok;
    }

    TdSelf SplitOffOn(size_t pos) {
        TdSelf tok;
        SplitOn(pos, *this, tok);
        return tok;
    }

    TdSelf NextTokOn(size_t pos) {
        TdSelf tok;
        SplitOn(pos, tok, *this);
        return tok;
    }
    /*
    // See comment on RSplitOn() above
    TdSelf RSplitOffOn(size_t pos);
    TdSelf RNextTokOn(size_t pos);
*/

public:
    TdSelf SplitOff(TChar delim) {
        TdSelf tok;
        Split(delim, *this, tok);
        return tok;
    }

    TdSelf RSplitOff(TChar delim) {
        TdSelf tok;
        RSplit(delim, tok, *this);
        return tok;
    }

    bool NextTok(TChar delim, TdSelf& tok) {
        return NextTokTemplate(delim, tok);
    }

    bool NextTok(TdSelf delim, TdSelf& tok) {
        return NextTokTemplate(delim, tok);
    }

    bool RNextTok(TChar delim, TdSelf& tok) {
        return RNextTokTemplate(delim, tok);
    }

    bool RNextTok(TdSelf delim, TdSelf& tok) {
        return RNextTokTemplate(delim, tok);
    }

    bool ReadLine(TdSelf& tok) {
        if (NextTok('\n', tok)) {
            while (!tok.empty() && tok.back() == '\r') {
                --tok.Length;
            }

            return true;
        }

        return false;
    }

    TdSelf NextTok(TChar delim) {
        return NextTokTemplate(delim);
    }

    TdSelf RNextTok(TChar delim) {
        return RNextTokTemplate(delim);
    }

    TdSelf NextTok(TdSelf delim) {
        return NextTokTemplate(delim);
    }

    TdSelf RNextTok(TdSelf delim) {
        return RNextTokTemplate(delim);
    }

public: // string subsequences
    /// Cut last @c shift characters (or less if length is less than @c shift)
    inline TdSelf& Chop(size_t shift) noexcept {
        ChopImpl(shift);

        return *this;
    }

    /// Cut first @c shift characters (or less if length is less than @c shift)
    inline TdSelf& Skip(size_t shift) noexcept {
        Start += ChopImpl(shift);

        return *this;
    }

    /// Sets the start pointer to a position relative to the end
    inline TdSelf& RSeek(size_t len) noexcept {
        if (Length > len) {
            Start += Length - len;
            Length = len;
        }

        return *this;
    }

    inline TdSelf& Trunc(size_t len) noexcept {
        if (Length > len) {
            Length = len;
        }

        return *this;
    }

    Y_PURE_FUNCTION
    inline TdSelf SubStr(size_t beg) const noexcept {
        return TdSelf(*this).Skip(beg);
    }

    Y_PURE_FUNCTION
    inline TdSelf SubStr(size_t beg, size_t len) const noexcept {
        return SubStr(beg).Trunc(len);
    }

    Y_PURE_FUNCTION
    inline TdSelf Head(size_t pos) const noexcept {
        return TdSelf(*this).Trunc(pos);
    }

    Y_PURE_FUNCTION
    inline TdSelf Tail(size_t pos) const noexcept {
        return SubStr(pos);
    }

    Y_PURE_FUNCTION
    inline TdSelf Last(size_t len) const noexcept {
        return TdSelf(*this).RSeek(len);
    }

    TGenericString<TChar> ToString() const {
        return {Start, Length};
    }

    TGenericString<TChar> Quote() const {
        return ToString().Quote();
    }

private:
    inline size_t ChopImpl(size_t shift) noexcept {
        if (shift > length())
            shift = length();
        Length -= shift;
        return shift;
    }

    template <typename TDelimiterType>
    TdSelf NextTokTemplate(TDelimiterType delim) {
        TdSelf tok;
        Split(delim, tok, *this);
        return tok;
    }

    template <typename TDelimiterType>
    TdSelf RNextTokTemplate(TDelimiterType delim) {
        TdSelf tok;
        RSplit(delim, *this, tok);
        return tok;
    }

    template <typename TDelimiterType>
    bool NextTokTemplate(TDelimiterType delim, TdSelf& tok) {
        if (!this->empty()) {
            tok = NextTokTemplate(delim);
            return true;
        }
        return false;
    }

    template <typename TDelimiterType>
    bool RNextTokTemplate(TDelimiterType delim, TdSelf& tok) {
        if (!this->empty()) {
            tok = RNextTokTemplate(delim);
            return true;
        }
        return false;
    }

    template <typename TDelimiterType>
    inline void SplitTemplate(TDelimiterType delim, TdSelf& l, TdSelf& r) const noexcept {
        if (!TrySplit(delim, l, r)) {
            l = *this;
            r = TdSelf();
        }
    }

    template <typename TDelimiterType>
    inline void RSplitTemplate(TDelimiterType delim, TdSelf& l, TdSelf& r) const noexcept {
        if (!TryRSplit(delim, l, r)) {
            r = *this;
            l = TdSelf();
        }
    }

private:
    using TBaseStr::Length;
    using TBaseStr::Start;
};

std::ostream& operator<< (std::ostream& os, TStringBuf buf);

//string type -> stringbuf type
template <class TStringType>
class TToStringBuf {
public:
    using TType = TGenericStringBuf<std::remove_cv_t<std::remove_reference_t<decltype(*std::declval<TStringType>().begin())>>>;
};

static inline TString ToString(const TStringBuf str) {
    return TString(str);
}

static inline TUtf16String ToWtring(const TWtringBuf wtr) {
    return TUtf16String(wtr);
}

static inline TUtf32String ToUtf32String(const TUtf32String wtr) {
    return TUtf32String(wtr);
}

template <typename TChar, size_t size>
constexpr inline TStringBufImpl<TChar> AsStringBuf(const TChar (&str)[size]) noexcept {
    return TStringBufImpl<TChar>(str, size - 1);
}
