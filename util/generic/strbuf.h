#pragma once

#include "fwd.h"
#include "strbase.h"
#include "utility.h"
#include "typetraits.h"

#include <util/system/compiler.h>

#include <string_view>

using namespace std::string_view_literals;

template <typename TCharType, typename TTraits>
class TBasicStringBuf: public std::basic_string_view<TCharType>,
                       public TStringBase<TBasicStringBuf<TCharType, TTraits>, TCharType, TTraits> {
private:
    using TdSelf = TBasicStringBuf;
    using TBase = TStringBase<TdSelf, TCharType, TTraits>;
    using TStringView = std::basic_string_view<TCharType>;

public:
    using char_type = TCharType; // TODO: DROP
    using traits_type = TTraits;

    // Resolving some ambiguity between TStringBase and std::basic_string_view
    // for typenames
    using typename TStringView::const_iterator;
    using typename TStringView::const_reference;
    using typename TStringView::const_reverse_iterator;
    using typename TStringView::iterator;
    using typename TStringView::reference;
    using typename TStringView::reverse_iterator;
    using typename TStringView::size_type;
    using typename TStringView::value_type;

    // for constants
    using TStringView::npos;

    // for methods and operators
    using TStringView::begin;
    using TStringView::cbegin;
    using TStringView::cend;
    using TStringView::crbegin;
    using TStringView::crend;
    using TStringView::end;
    using TStringView::rbegin;
    using TStringView::rend;

    using TStringView::data;
    using TStringView::empty;
    using TStringView::size;

    using TStringView::operator[];

    /*
     * WARN:
     * TBase::at silently return 0 in case of range error,
     * while std::string_view throws std::out_of_range.
     */
    using TBase::at;
    using TStringView::back;
    using TStringView::front;

    using TStringView::find;
    /*
     * WARN:
     *      TBase::*find* methods take into account TCharTraits,
     *      while TTStringView::*find* would use default std::char_traits.
     */
    using TBase::find_first_not_of;
    using TBase::find_first_of;
    using TBase::find_last_not_of;
    using TBase::find_last_of;
    using TBase::rfind;

    using TStringView::copy;
    /*
     * WARN:
     *  TBase::compare takes into account TCharTraits,
     *  thus making it possible to implement case-insensitive string buffers,
     *  if it is using TStringBase::compare
     */
    using TBase::compare;

    /*
     * WARN:
     *  TBase::substr properly checks boundary cases and clamps them with maximum valid values,
     *  while TStringView::substr throws std::out_of_range error.
     */
    using TBase::substr;

    /*
     * WARN:
     *  Constructing std::string_view(nullptr, non_zero_size) ctor
     *  results in undefined behavior according to the standard.
     *  In libc++ this UB results in runtime assertion, though it is better
     *  to generate compilation error instead.
     */
    constexpr inline TBasicStringBuf(std::nullptr_t begin, size_t size) = delete;

    constexpr inline TBasicStringBuf(const TCharType* data Y_LIFETIME_BOUND, size_t size) noexcept
        : TStringView(data, size)
    {
    }

    constexpr TBasicStringBuf(const TCharType* data Y_LIFETIME_BOUND) noexcept
        /*
         * WARN: TBase::StrLen properly handles nullptr,
         * while std::string_view (using std::char_traits) will abort in such case
         */
        : TStringView(data, TBase::StrLen(data))
    {
    }

    constexpr inline TBasicStringBuf(const TCharType* beg Y_LIFETIME_BOUND, const TCharType* end Y_LIFETIME_BOUND) noexcept
        : TStringView(beg, end - beg)
    {
    }

    template <typename D, typename T>
    inline TBasicStringBuf(const TStringBase<D, TCharType, T>& str) noexcept
        : TStringView(str.data(), str.size())
    {
    }

    template <typename T>
    inline TBasicStringBuf(const TBasicString<TCharType, T>& str Y_STRING_LIFETIME_BOUND) noexcept
        : TStringView(str.data(), str.size())
    {
    }

    template <typename T, typename A>
    inline TBasicStringBuf(const std::basic_string<TCharType, T, A>& str Y_LIFETIME_BOUND) noexcept
        : TStringView(str)
    {
    }

    template <typename TCharTraits>
    constexpr TBasicStringBuf(std::basic_string_view<TCharType, TCharTraits> view Y_LIFETIME_BOUND) noexcept
        : TStringView(view)
    {
    }

    template <typename TCharTraits>
    constexpr TBasicStringBuf(TBasicStringBuf<TCharType, TCharTraits> sb Y_LIFETIME_BOUND) noexcept
        : TStringView(sb)
    {
    }

    constexpr inline TBasicStringBuf() noexcept {
        /*
         * WARN:
         *  This ctor can not be defaulted due to the following feature of default initialization:
         *  If T is a const-qualified type, it must be a class type with a user-provided default constructor.
         *  (see https://en.cppreference.com/w/cpp/language/default_initialization).
         *
         *  This means, that a class with default ctor can not be a constant member of another class with default ctor.
         */
    }

    inline TBasicStringBuf(const TBasicStringBuf src Y_LIFETIME_BOUND, size_t pos, size_t n) noexcept
        : TBasicStringBuf(src)
    {
        Skip(pos).Trunc(n);
    }

    inline TBasicStringBuf(const TBasicStringBuf src Y_LIFETIME_BOUND, size_t pos) noexcept
        : TBasicStringBuf(src, pos, TBase::npos)
    {
    }

    Y_PURE_FUNCTION inline TBasicStringBuf SubString(size_t pos, size_t n) const noexcept {
        pos = Min(pos, size());
        n = Min(n, size() - pos);
        return TBasicStringBuf(data() + pos, n);
    }

public:
    void Clear() {
        *this = TdSelf();
    }

    constexpr bool IsInited() const noexcept {
        return data() != nullptr;
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
    inline bool TrySplit(TCharType delim, TdSelf& l, TdSelf& r) const noexcept {
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
    inline bool TryRSplit(TCharType delim, TdSelf& l, TdSelf& r) const noexcept {
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

    inline void Split(TCharType delim, TdSelf& l, TdSelf& r) const noexcept {
        SplitTemplate(delim, l, r);
    }

    inline void RSplit(TCharType delim, TdSelf& l, TdSelf& r) const noexcept {
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
    // returning either position within string length [0..size()) or npos.
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
    Y_PURE_FUNCTION inline TdSelf After(TCharType c) const noexcept {
        TdSelf l, r;
        return TrySplit(c, l, r) ? r : *this;
    }

    Y_PURE_FUNCTION inline TdSelf Before(TCharType c) const noexcept {
        TdSelf l, r;
        return TrySplit(c, l, r) ? l : *this;
    }

    Y_PURE_FUNCTION inline TdSelf RAfter(TCharType c) const noexcept {
        TdSelf l, r;
        return TryRSplit(c, l, r) ? r : *this;
    }

    Y_PURE_FUNCTION inline TdSelf RBefore(TCharType c) const noexcept {
        TdSelf l, r;
        return TryRSplit(c, l, r) ? l : *this;
    }

public:
    inline bool AfterPrefix(const TdSelf& prefix, TdSelf& result) const noexcept {
        if (this->StartsWith(prefix)) {
            result = Tail(prefix.size());
            return true;
        }
        return false;
    }

    inline bool BeforeSuffix(const TdSelf& suffix, TdSelf& result) const noexcept {
        if (this->EndsWith(suffix)) {
            result = Head(size() - suffix.size());
            return true;
        }
        return false;
    }

    // returns true if string started with `prefix`, false otherwise
    inline bool SkipPrefix(const TdSelf& prefix) noexcept {
        return AfterPrefix(prefix, *this);
    }

    // returns true if string ended with `suffix`, false otherwise
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
    TdSelf SplitOff(TCharType delim) {
        TdSelf tok;
        Split(delim, *this, tok);
        return tok;
    }

    TdSelf RSplitOff(TCharType delim) {
        TdSelf tok;
        RSplit(delim, tok, *this);
        return tok;
    }

    bool NextTok(TCharType delim, TdSelf& tok) {
        return NextTokTemplate(delim, tok);
    }

    bool NextTok(TdSelf delim, TdSelf& tok) {
        return NextTokTemplate(delim, tok);
    }

    bool RNextTok(TCharType delim, TdSelf& tok) {
        return RNextTokTemplate(delim, tok);
    }

    bool RNextTok(TdSelf delim, TdSelf& tok) {
        return RNextTokTemplate(delim, tok);
    }

    bool ReadLine(TdSelf& tok) {
        if (NextTok('\n', tok)) {
            while (!tok.empty() && tok.back() == '\r') {
                tok.remove_suffix(1);
            }

            return true;
        }

        return false;
    }

    TdSelf NextTok(TCharType delim) {
        return NextTokTemplate(delim);
    }

    TdSelf RNextTok(TCharType delim) {
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
        this->remove_suffix(std::min(shift, size()));
        return *this;
    }

    /// Cut first @c shift characters (or less if length is less than @c shift)
    inline TdSelf& Skip(size_t shift) noexcept {
        this->remove_prefix(std::min(shift, size()));
        return *this;
    }

    /// Sets the start pointer to a position relative to the end
    inline TdSelf& RSeek(size_t tailSize) noexcept {
        if (size() > tailSize) {
            // WARN: removing TStringView:: will lead to an infinite recursion
            *this = TStringView::substr(size() - tailSize, tailSize);
        }

        return *this;
    }

    // coverity[exn_spec_violation]
    inline TdSelf& Trunc(size_t targetSize) noexcept {
        // Coverity false positive issue
        // exn_spec_violation: An exception of type "std::out_of_range" is thrown but the exception specification "noexcept" doesn't allow it to be thrown. This will result in a call to terminate().
        // fun_call_w_exception: Called function TStringView::substr throws an exception of type "std::out_of_range".
        // Suppress this issue because we pass argument pos=0 and string_view can't throw std::out_of_range.
        *this = TStringView::substr(0, targetSize); // WARN: removing TStringView:: will lead to an infinite recursion
        return *this;
    }

    Y_PURE_FUNCTION inline TdSelf SubStr(size_t beg) const noexcept {
        return TdSelf(*this).Skip(beg);
    }

    Y_PURE_FUNCTION inline TdSelf SubStr(size_t beg, size_t len) const noexcept {
        return SubStr(beg).Trunc(len);
    }

    Y_PURE_FUNCTION inline TdSelf Head(size_t pos) const noexcept {
        return TdSelf(*this).Trunc(pos);
    }

    Y_PURE_FUNCTION inline TdSelf Tail(size_t pos) const noexcept {
        return SubStr(pos);
    }

    Y_PURE_FUNCTION inline TdSelf Last(size_t len) const noexcept {
        return TdSelf(*this).RSeek(len);
    }

private:
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
        if (!empty()) {
            tok = NextTokTemplate(delim);
            return true;
        }
        return false;
    }

    template <typename TDelimiterType>
    bool RNextTokTemplate(TDelimiterType delim, TdSelf& tok) {
        if (!empty()) {
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
};

std::ostream& operator<<(std::ostream& os, TStringBuf buf);

constexpr TStringBuf operator""_sb(const char* str, size_t len) {
    return TStringBuf{str, len};
}
