#pragma once

#include "recode_result.h"
#include "unidata.h"
#include "utf8.h"
#include "wide_specific.h"

#include <util/generic/algorithm.h>
#include <util/generic/string.h>
#include <util/generic/yexception.h>
#include <util/memory/tempbuf.h>
#include <util/system/compiler.h>
#include <util/system/cpu_id.h>
#include <util/system/yassert.h>

#include <cstring>

#ifdef _sse2_
    #include <emmintrin.h>
#endif

template <class T>
class TTempArray;
using TCharTemp = TTempArray<wchar16>;

namespace NDetail {
    inline TString InStringMsg(const char* s, size_t len) {
        return (len <= 50) ? " in string " + TString(s, len).Quote() : TString();
    }

    template <bool isPointer>
    struct TSelector;

    template <>
    struct TSelector<false> {
        template <class T>
        static inline void WriteSymbol(wchar16 s, T& dest) noexcept {
            dest.push_back(s);
        }
    };

    template <>
    struct TSelector<true> {
        template <class T>
        static inline void WriteSymbol(wchar16 s, T& dest) noexcept {
            *(dest++) = s;
        }
    };

    inline wchar32 ReadSurrogatePair(const wchar16* chars) noexcept {
        const wchar32 SURROGATE_OFFSET = static_cast<wchar32>(0x10000 - (0xD800 << 10) - 0xDC00);
        wchar32 lead = chars[0];
        wchar32 tail = chars[1];

        Y_ASSERT(IsW16SurrogateLead(lead));
        Y_ASSERT(IsW16SurrogateTail(tail));

        return (static_cast<wchar32>(lead) << 10) + tail + SURROGATE_OFFSET;
    }

    template <class T>
    inline void WriteSurrogatePair(wchar32 s, T& dest) noexcept;

}

inline wchar16* SkipSymbol(wchar16* begin, const wchar16* end) noexcept {
    return begin + W16SymbolSize(begin, end);
}
inline const wchar16* SkipSymbol(const wchar16* begin, const wchar16* end) noexcept {
    return begin + W16SymbolSize(begin, end);
}
inline wchar32* SkipSymbol(wchar32* begin, const wchar32* end) noexcept {
    Y_ASSERT(begin < end);
    return begin + 1;
}
inline const wchar32* SkipSymbol(const wchar32* begin, const wchar32* end) noexcept {
    Y_ASSERT(begin < end);
    return begin + 1;
}

inline wchar32 ReadSymbol(const wchar16* begin, const wchar16* end) noexcept {
    Y_ASSERT(begin < end);
    if (IsW16SurrogateLead(*begin)) {
        if (begin + 1 < end && IsW16SurrogateTail(*(begin + 1)))
            return ::NDetail::ReadSurrogatePair(begin);

        return BROKEN_RUNE;
    } else if (IsW16SurrogateTail(*begin)) {
        return BROKEN_RUNE;
    }

    return *begin;
}

inline wchar32 ReadSymbol(const wchar32* begin, const wchar32* end) noexcept {
    Y_ASSERT(begin < end);
    return *begin;
}

//! presuming input data is either big enought of null terminated
inline wchar32 ReadSymbolAndAdvance(const char16_t*& begin) noexcept {
    Y_ASSERT(*begin);
    if (IsW16SurrogateLead(begin[0])) {
        if (IsW16SurrogateTail(begin[1])) {
            Y_ASSERT(begin[1] != 0);
            const wchar32 c = ::NDetail::ReadSurrogatePair(begin);
            begin += 2;
            return c;
        }
        ++begin;
        return BROKEN_RUNE;
    } else if (IsW16SurrogateTail(begin[0])) {
        ++begin;
        return BROKEN_RUNE;
    }
    return *(begin++);
}

//! presuming input data is either big enought of null terminated
inline wchar32 ReadSymbolAndAdvance(const char32_t*& begin) noexcept {
    Y_ASSERT(*begin);
    return *(begin++);
}

inline wchar32 ReadSymbolAndAdvance(const wchar_t*& begin) noexcept {
    // According to
    // https://en.cppreference.com/w/cpp/language/types
    // wchar_t holds UTF-16 on Windows and UTF-32 on Linux / macOS
    //
    // Apply reinterpret cast and dispatch to a proper type

#ifdef _win_
    using TDistinctChar = char16_t;
#else
    using TDistinctChar = char32_t;
#endif
    const TDistinctChar*& distinctBegin = reinterpret_cast<const TDistinctChar*&>(begin);
    wchar32 result = ReadSymbolAndAdvance(distinctBegin);
    begin = reinterpret_cast<const wchar_t*&>(distinctBegin);
    return result;
}

inline wchar32 ReadSymbolAndAdvance(const char16_t*& begin, const char16_t* end) noexcept {
    Y_ASSERT(begin < end);
    if (IsW16SurrogateLead(begin[0])) {
        if (begin + 1 != end && IsW16SurrogateTail(begin[1])) {
            const wchar32 c = ::NDetail::ReadSurrogatePair(begin);
            begin += 2;
            return c;
        }
        ++begin;
        return BROKEN_RUNE;
    } else if (IsW16SurrogateTail(begin[0])) {
        ++begin;
        return BROKEN_RUNE;
    }
    return *(begin++);
}

inline wchar32 ReadSymbolAndAdvance(const wchar32*& begin, const wchar32* end) noexcept {
    Y_ASSERT(begin < end);
    return *(begin++);
}

inline wchar32 ReadSymbolAndAdvance(const wchar_t*& begin, const wchar_t* end) noexcept {
    // According to
    // https://en.cppreference.com/w/cpp/language/types
    // wchar_t holds UTF-16 on Windows and UTF-32 on Linux / macOS
    //
    // Apply reinterpret cast and dispatch to a proper type

#ifdef _win_
    using TDistinctChar = char16_t;
#else
    using TDistinctChar = char32_t;
#endif
    const TDistinctChar* distinctBegin = reinterpret_cast<const TDistinctChar*>(begin);
    const TDistinctChar* distinctEnd = reinterpret_cast<const TDistinctChar*>(end);
    wchar32 result = ::ReadSymbolAndAdvance(distinctBegin, distinctEnd);
    begin = reinterpret_cast<const wchar_t*>(distinctBegin);
    return result;
}

template <class T>
inline size_t WriteSymbol(wchar16 s, T& dest) noexcept {
    ::NDetail::TSelector<std::is_pointer<T>::value>::WriteSymbol(s, dest);
    return 1;
}

template <class T>
inline size_t WriteSymbol(wchar32 s, T& dest) noexcept {
    if (s > 0xFFFF) {
        if (s >= ::NUnicode::UnicodeInstancesLimit()) {
            return WriteSymbol(static_cast<wchar16>(BROKEN_RUNE), dest);
        }

        ::NDetail::WriteSurrogatePair(s, dest);
        return 2;
    }

    return WriteSymbol(static_cast<wchar16>(s), dest);
}

inline bool WriteSymbol(wchar32 s, wchar16*& dest, const wchar16* destEnd) noexcept {
    Y_ASSERT(dest < destEnd);

    if (s > 0xFFFF) {
        if (s >= NUnicode::UnicodeInstancesLimit()) {
            *(dest++) = static_cast<wchar16>(BROKEN_RUNE);
            return true;
        }

        if (dest + 2 > destEnd)
            return false;

        ::NDetail::WriteSurrogatePair(s, dest);
    } else {
        *(dest++) = static_cast<wchar16>(s);
    }

    return true;
}

inline size_t WriteSymbol(wchar32 s, wchar32*& dest) noexcept {
    *(dest++) = s;
    return 1;
}

inline bool WriteSymbol(wchar32 s, wchar32*& dest, const wchar32* destEnd) noexcept {
    Y_ASSERT(dest < destEnd);

    *(dest++) = s;

    return true;
}

template <class T>
inline void ::NDetail::WriteSurrogatePair(wchar32 s, T& dest) noexcept {
    const wchar32 LEAD_OFFSET = 0xD800 - (0x10000 >> 10);
    Y_ASSERT(s > 0xFFFF && s < ::NUnicode::UnicodeInstancesLimit());

    wchar16 lead = LEAD_OFFSET + (static_cast<wchar16>(s >> 10));
    wchar16 tail = 0xDC00 + static_cast<wchar16>(s & 0x3FF);
    Y_ASSERT(IsW16SurrogateLead(lead));
    Y_ASSERT(IsW16SurrogateTail(tail));

    WriteSymbol(lead, dest);
    WriteSymbol(tail, dest);
}

class TCharIterator {
private:
    const wchar16* Begin;
    const wchar16* End;

public:
    inline explicit TCharIterator(const wchar16* end)
        : Begin(end)
        , End(end)
    {
    }

    inline TCharIterator(const wchar16* begin, const wchar16* end)
        : Begin(begin)
        , End(end)
    {
    }

    inline TCharIterator& operator++() {
        Begin = SkipSymbol(Begin, End);

        return *this;
    }

    inline bool operator==(const wchar16* other) const {
        return Begin == other;
    }
    inline bool operator!=(const wchar16* other) const {
        return !(*this == other);
    }

    inline bool operator==(const TCharIterator& other) const {
        return *this == other.Begin;
    }
    inline bool operator!=(const TCharIterator& other) const {
        return *this != other.Begin;
    }

    inline wchar32 operator*() const {
        return ReadSymbol(Begin, End);
    }

    inline const wchar16* Get() const {
        return Begin;
    }
};

namespace NDetail {
    template <bool robust, typename TCharType>
    inline void UTF8ToWideImplScalar(const unsigned char*& cur, const unsigned char* last, TCharType*& dest) noexcept {
        wchar32 rune = BROKEN_RUNE;

        while (cur != last) {
            if (ReadUTF8CharAndAdvance(rune, cur, last) != RECODE_OK) {
                if (robust) {
                    rune = BROKEN_RUNE;
                    ++cur;
                } else {
                    break;
                }
            }

            Y_ASSERT(cur <= last);
            WriteSymbol(rune, dest);
        }
    }

    template <typename TCharType>
    inline void UTF16ToUTF32ImplScalar(const wchar16* cur, const wchar16* last, TCharType*& dest) noexcept {
        wchar32 rune = BROKEN_RUNE;

        while (cur != last) {
            rune = ReadSymbolAndAdvance(cur, last);
            Y_ASSERT(cur <= last);
            WriteSymbol(rune, dest);
        }
    }

    template <class TCharType>
    inline void UTF8ToWideImplSSE41(const unsigned char*& /*cur*/, const unsigned char* /*last*/, TCharType*& /*dest*/) noexcept {
    }

    void UTF8ToWideImplSSE41(const unsigned char*& cur, const unsigned char* last, wchar16*& dest) noexcept;

    void UTF8ToWideImplSSE41(const unsigned char*& cur, const unsigned char* last, wchar32*& dest) noexcept;
}

//! @return len if robust and position where encoding stopped if not
template <bool robust, typename TCharType>
inline size_t UTF8ToWideImpl(const char* text, size_t len, TCharType* dest, size_t& written) noexcept {
    const unsigned char* cur = reinterpret_cast<const unsigned char*>(text);
    const unsigned char* last = cur + len;
    TCharType* p = dest;
#ifdef _sse_ // can't check for sse4, as we build most of arcadia without sse4 support even on platforms that support it
    if (cur + 16 <= last && NX86::CachedHaveSSE41()) {
        ::NDetail::UTF8ToWideImplSSE41(cur, last, p);
    }
#endif

    ::NDetail::UTF8ToWideImplScalar<robust>(cur, last, p);
    written = p - dest;
    return cur - reinterpret_cast<const unsigned char*>(text);
}

template <typename TCharType>
inline size_t UTF8ToWideImpl(const char* text, size_t len, TCharType* dest, size_t& written) {
    return UTF8ToWideImpl<false>(text, len, dest, written);
}

template <bool robust>
inline TUtf16String UTF8ToWide(const char* text, size_t len) {
    TUtf16String w = TUtf16String::Uninitialized(len);
    size_t written;
    size_t pos = UTF8ToWideImpl<robust>(text, len, w.begin(), written);
    if (pos != len)
        ythrow yexception() << "failed to decode UTF-8 string at pos " << pos << ::NDetail::InStringMsg(text, len);
    Y_ASSERT(w.size() >= written);
    w.remove(written);
    return w;
}

template <bool robust, typename TCharType>
inline bool UTF8ToWide(const char* text, size_t len, TCharType* dest, size_t& written) noexcept {
    return UTF8ToWideImpl<robust>(text, len, dest, written) == len;
}

//! converts text from UTF8 to unicode, stops immediately it UTF8 byte sequence is wrong
//! @attention destination buffer must be long enough to fit all characters of the text,
//!            conversion stops if a broken symbol is met
//! @return @c true if all the text converted successfully, @c false - a broken symbol was found
template <typename TCharType>
inline bool UTF8ToWide(const char* text, size_t len, TCharType* dest, size_t& written) noexcept {
    return UTF8ToWide<false>(text, len, dest, written);
}

template <bool robust>
inline TWtringBuf UTF8ToWide(const TStringBuf src, TUtf16String& dst) {
    dst.ReserveAndResize(src.size());
    size_t written = 0;
    UTF8ToWideImpl<robust>(src.data(), src.size(), dst.begin(), written);
    dst.resize(written);
    return dst;
}

//! if not robust will stop at first error position
template <bool robust>
inline TUtf32StringBuf UTF8ToUTF32(const TStringBuf src, TUtf32String& dst) {
    dst.ReserveAndResize(src.size());
    size_t written = 0;
    UTF8ToWideImpl<robust>(src.data(), src.size(), dst.begin(), written);
    dst.resize(written);
    return dst;
}

inline TWtringBuf UTF8ToWide(const TStringBuf src, TUtf16String& dst) {
    return UTF8ToWide<false>(src, dst);
}

inline TUtf16String UTF8ToWide(const char* text, size_t len) {
    return UTF8ToWide<false>(text, len);
}

template <bool robust>
inline TUtf16String UTF8ToWide(const TStringBuf s) {
    return UTF8ToWide<robust>(s.data(), s.size());
}

template <bool robust>
inline TUtf32String UTF8ToUTF32(const TStringBuf s) {
    TUtf32String r;
    UTF8ToUTF32<robust>(s, r);
    return r;
}

inline TUtf16String UTF8ToWide(const TStringBuf s) {
    return UTF8ToWide<false>(s.data(), s.size());
}

//! converts text from unicode to UTF8
//! @attention destination buffer must be long enough to fit all characters of the text,
//!            @c WriteUTF8Char converts @c wchar32 into maximum 4 bytes of UTF8 so
//!            destination buffer must have length equal to <tt> len * 4 </tt>
template <typename TCharType>
inline void WideToUTF8(const TCharType* text, size_t len, char* dest, size_t& written) {
    const TCharType* const last = text + len;
    unsigned char* p = reinterpret_cast<unsigned char*>(dest);
    size_t runeLen;
    for (const TCharType* cur = text; cur != last;) {
        WriteUTF8Char(ReadSymbolAndAdvance(cur, last), runeLen, p);
        Y_ASSERT(runeLen <= 4);
        p += runeLen;
    }
    written = p - reinterpret_cast<unsigned char*>(dest);
}

constexpr size_t WideToUTF8BufferSize(const size_t inputStringSize) noexcept {
    return inputStringSize * 4; // * 4 because the conversion functions can convert unicode character into maximum 4 bytes of UTF8
}

inline TStringBuf WideToUTF8(const TWtringBuf src, TString& dst) {
    dst.ReserveAndResize(WideToUTF8BufferSize(src.size()));
    size_t written = 0;
    WideToUTF8(src.data(), src.size(), dst.begin(), written);
    Y_ASSERT(dst.size() >= written);
    dst.remove(written);
    return dst;
}

inline TString WideToUTF8(const wchar16* text, size_t len) {
    TString s = TString::Uninitialized(WideToUTF8BufferSize(len));
    size_t written = 0;
    WideToUTF8(text, len, s.begin(), written);
    Y_ASSERT(s.size() >= written);
    s.remove(written);
    return s;
}

#if defined(_win_)
inline TString WideToUTF8(const wchar_t* text, size_t len) {
    return WideToUTF8(reinterpret_cast<const wchar16*>(text), len);
}

inline std::string WideToUTF8(std::wstring_view text) {
    return WideToUTF8(text.data(), text.size()).ConstRef();
}
#endif

inline TString WideToUTF8(const wchar32* text, size_t len) {
    TString s = TString::Uninitialized(WideToUTF8BufferSize(len));
    size_t written = 0;
    WideToUTF8(text, len, s.begin(), written);
    Y_ASSERT(s.size() >= written);
    s.remove(written);
    return s;
}

inline TString WideToUTF8(const TWtringBuf w) {
    return WideToUTF8(w.data(), w.size());
}

inline TString WideToUTF8(const TUtf32StringBuf w) {
    return WideToUTF8(w.data(), w.size());
}

inline TUtf16String UTF32ToWide(const wchar32* begin, size_t len) {
    TUtf16String res;
    res.reserve(len);

    const wchar32* end = begin + len;
    for (const wchar32* i = begin; i != end; ++i) {
        WriteSymbol(*i, res);
    }

    return res;
}

// adopted from https://chromium.googlesource.com/chromium/src/+/master/base/strings/string_util.cc
// Assuming that a pointer is the size of a "machine word", then
// uintptr_t is an integer type that is also a machine word.

namespace NDetail {
    using TMachineWord = uintptr_t;
    const uintptr_t kMachineWordAlignmentMask = sizeof(TMachineWord) - 1;

    inline bool IsAlignedToMachineWord(const void* pointer) {
        return !(reinterpret_cast<TMachineWord>(pointer) & kMachineWordAlignmentMask);
    }

    template <typename T>
    inline T* AlignToMachineWord(T* pointer) {
        return reinterpret_cast<T*>(reinterpret_cast<TMachineWord>(pointer) & ~kMachineWordAlignmentMask);
    }

    template <size_t size, typename CharacterType>
    struct NonASCIIMask;

    template <>
    struct
        NonASCIIMask<4, wchar16> {
        static constexpr ui32 Value() {
            return 0xFF80FF80U;
        }
    };

    template <>
    struct
        NonASCIIMask<4, char> {
        static constexpr ui32 Value() {
            return 0x80808080U;
        }
    };

    template <>
    struct
        NonASCIIMask<8, wchar16> {
        static constexpr ui64 Value() {
            return 0xFF80FF80FF80FF80ULL;
        }
    };

    template <>
    struct
        NonASCIIMask<8, char> {
        static constexpr ui64 Value() {
            return 0x8080808080808080ULL;
        }
    };

    template <typename TChar>
    inline bool DoIsStringASCIISlow(const TChar* first, const TChar* last) {
        using TUnsignedChar = std::make_unsigned_t<TChar>;
        Y_ASSERT(first <= last);
        for (; first != last; ++first) {
            if (static_cast<TUnsignedChar>(*first) > 0x7F) {
                return false;
            }
        }
        return true;
    }

    template <typename TChar>
    inline bool DoIsStringASCII(const TChar* first, const TChar* last) {
        if (last - first < 10) {
            return DoIsStringASCIISlow(first, last);
        }
        TMachineWord allCharBits = 0;
        TMachineWord nonAsciiBitMask = NonASCIIMask<sizeof(TMachineWord), TChar>::Value();

        // Prologue: align the input.
        while (!IsAlignedToMachineWord(first) && first != last) {
            allCharBits |= *first;
            ++first;
        }

        // Compare the values of CPU word size.
        const TChar* word_end = AlignToMachineWord(last);
        const size_t loopIncrement = sizeof(TMachineWord) / sizeof(TChar);
        while (first < word_end) {
            allCharBits |= *(reinterpret_cast<const TMachineWord*>(first));
            first += loopIncrement;

            // fast exit
            if (allCharBits & nonAsciiBitMask) {
                return false;
            }
        }

        // Process the remaining bytes.
        while (first != last) {
            allCharBits |= *first;
            ++first;
        }

        return !(allCharBits & nonAsciiBitMask);
    }

#ifdef _sse2_
    inline bool DoIsStringASCIISSE(const unsigned char* first, const unsigned char* last) {
        // scalar version for short strings
        if (first + 8 > last) {
            return ::NDetail::DoIsStringASCIISlow(first, last);
        }

        alignas(16) unsigned char buf[16];

        while (first + 16 <= last) {
            memcpy(buf, first, 16);
            __m128i chunk = _mm_load_si128(reinterpret_cast<__m128i*>(buf));

            int asciiMask = _mm_movemask_epi8(chunk);
            if (asciiMask) {
                return false;
            }
            first += 16;
        }

        if (first + 8 <= last) {
            memcpy(buf, first, 8);
            __m128i chunk = _mm_loadl_epi64(reinterpret_cast<__m128i*>(buf));

            int asciiMask = _mm_movemask_epi8(chunk);
            if (asciiMask) {
                return false;
            }
            first += 8;
        }

        return ::NDetail::DoIsStringASCIISlow(first, last);
    }
#endif // _sse2_

}

//! returns @c true if character sequence has no symbols with value greater than 0x7F
template <typename TChar>
inline bool IsStringASCII(const TChar* first, const TChar* last) {
    return ::NDetail::DoIsStringASCII(first, last);
}

#ifdef _sse2_
template <>
inline bool IsStringASCII<unsigned char>(const unsigned char* first, const unsigned char* last) {
    return ::NDetail::DoIsStringASCIISSE(first, last);
}
template <>
inline bool IsStringASCII<char>(const char* first, const char* last) {
    return ::NDetail::DoIsStringASCIISSE(reinterpret_cast<const unsigned char*>(first), reinterpret_cast<const unsigned char*>(last));
}
#endif

//! copies elements from one character sequence to another using memcpy
//! for compatibility only
template <typename TChar>
inline void Copy(const TChar* first, size_t len, TChar* result) {
    memcpy(result, first, len * sizeof(TChar));
}

template <typename TChar1, typename TChar2>
inline void Copy(const TChar1* first, size_t len, TChar2* result) {
    Copy(first, first + len, result);
}

//! copies symbols from one character sequence to another without any conversion
//! @note this function can be used instead of the template constructor of @c std::basic_string:
//!       template <typename InputIterator>
//!       basic_string(InputIterator begin, InputIterator end, const Allocator& a = Allocator());
//!       and the family of template member functions: append, assign, insert, replace.
template <typename TStringType, typename TChar>
inline TStringType CopyTo(const TChar* first, const TChar* last) {
    Y_ASSERT(first <= last);
    TStringType str = TStringType::Uninitialized(last - first);
    Copy(first, last, str.begin());
    return str;
}

template <typename TStringType, typename TChar>
inline TStringType CopyTo(const TChar* s, size_t n) {
    TStringType str = TStringType::Uninitialized(n);
    Copy(s, n, str.begin());
    return str;
}

inline TString WideToASCII(const TWtringBuf w) {
    Y_ASSERT(IsStringASCII(w.begin(), w.end()));
    return CopyTo<TString>(w.begin(), w.end());
}

inline TUtf16String ASCIIToWide(const TStringBuf s) {
    Y_ASSERT(IsStringASCII(s.begin(), s.end()));
    return CopyTo<TUtf16String>(s.begin(), s.end());
}

inline TUtf32String ASCIIToUTF32(const TStringBuf s) {
    Y_ASSERT(IsStringASCII(s.begin(), s.end()));
    return CopyTo<TUtf32String>(s.begin(), s.end());
}

//! returns @c true if string contains whitespace characters only
inline bool IsSpace(const wchar16* s, size_t n) {
    if (n == 0)
        return false;

    Y_ASSERT(s);

    const wchar16* const e = s + n;
    for (const wchar16* p = s; p != e; ++p) {
        if (!IsWhitespace(*p))
            return false;
    }
    return true;
}

//! returns @c true if string contains whitespace characters only
inline bool IsSpace(const TWtringBuf s) {
    return IsSpace(s.data(), s.length());
}

//! replaces multiple sequential whitespace characters with a single space character
void Collapse(TUtf16String& w);

//! @return new length
size_t Collapse(wchar16* s, size_t n);

//! Removes leading whitespace characters
TWtringBuf StripLeft(const TWtringBuf text) noexcept Y_WARN_UNUSED_RESULT;
void StripLeft(TUtf16String& text);

//! Removes trailing whitespace characters
TWtringBuf StripRight(const TWtringBuf text) noexcept Y_WARN_UNUSED_RESULT;
void StripRight(TUtf16String& text);

//! Removes leading and trailing whitespace characters
TWtringBuf Strip(const TWtringBuf text) noexcept Y_WARN_UNUSED_RESULT;
void Strip(TUtf16String& text);

/* Check if given word is lowercase/uppercase. Will return false if string contains any
 * non-alphabetical symbols. It is expected that `text` is a correct UTF-16 string.
 *
 * For example `IsLowerWord("hello")` will return `true`, when `IsLowerWord("hello there")` will
 * return false because of the space in the middle of the string. Empty string is also considered
 * lowercase.
 */
bool IsLowerWord(const TWtringBuf text) noexcept;
bool IsUpperWord(const TWtringBuf text) noexcept;

/* Will check if given word starts with capital letter and the rest of the word is lowercase. Will
 * return `false` for empty string. See also `IsLowerWord`.
 */
bool IsTitleWord(const TWtringBuf text) noexcept;

/* Check if given string is lowercase/uppercase. Will return `true` if all alphabetic symbols are
 * in proper case, all other symbols are ignored. It is expected that `text` is a correct UTF-16
 * string.
 *
 * For example `IsLowerWord("hello")` will return `true` and `IsLowerWord("hello there")` will
 * also return true because. Empty string is also considered lowercase.
 *
 * NOTE: for any case where `IsLowerWord` returns `true` `IsLower` will also return `true`.
 */
bool IsLower(const TWtringBuf text) noexcept;
bool IsUpper(const TWtringBuf text) noexcept;

/* Lowercase/uppercase given string inplace. Any alphabetic symbol will be converted to a proper
 * case, the rest of the symbols will be kept the same. It is expected that `text` is a correct
 * UTF-16 string.
 *
 * For example `ToLower("heLLo")` will return `"hello"`.
 *
 * @param text      String to modify
 * @param pos       Position of the first character to modify
 * @param count     Length of the substring
 * @returns         `true` if `text` was changed
 *
 * NOTE: `pos` and `count` are measured in `wchar16`, not in codepoints.
 */
bool ToLower(TUtf16String& text, size_t pos = 0, size_t count = TUtf16String::npos);
bool ToUpper(TUtf16String& text, size_t pos = 0, size_t count = TUtf16String::npos);

/* Lowercase/uppercase given string inplace. Any alphabetic symbol will be converted to a proper
 * case, the rest of the symbols will be kept the same. It is expected that `text` is a correct
 * UTF-32 string.
 *
 * For example `ToLower("heLLo")` will return `"hello"`.
 *
 * @param text      String to modify
 * @param pos       Position of the first character to modify
 * @param count     Length of the substring
 * @returns         `true` if `text` was changed
 *
 * NOTE: `pos` and `count` are measured in `wchar16`, not in codepoints.
 */
bool ToLower(TUtf32String& /*text*/, size_t /*pos*/ = 0, size_t /*count*/ = TUtf16String::npos);
bool ToUpper(TUtf32String& /*text*/, size_t /*pos*/ = 0, size_t /*count*/ = TUtf16String::npos);

/* Titlecase first symbol and lowercase the rest, see `ToLower` for more details.
 */
bool ToTitle(TUtf16String& text, size_t pos = 0, size_t count = TUtf16String::npos);

/* Titlecase first symbol and lowercase the rest, see `ToLower` for more details.
 */
bool ToTitle(TUtf32String& /*text*/, size_t /*pos*/ = 0, size_t /*count*/ = TUtf16String::npos);

/* @param text      Pointer to the string to modify
 * @param length    Length of the string to modify
 * @param out       Pointer to the character array to write to
 *
 * NOTE: [text, text+length) and [out, out+length) should not interleave.
 *
 * TODO(yazevnul): replace these functions with `bool(const TWtringBuf, const TArrayRef<wchar16>)`
 * overload.
 */
bool ToLower(const wchar16* text, size_t length, wchar16* out) noexcept;
bool ToUpper(const wchar16* text, size_t length, wchar16* out) noexcept;
bool ToTitle(const wchar16* text, size_t length, wchar16* out) noexcept;

bool ToLower(const wchar32* text, size_t length, wchar32* out) noexcept;
bool ToUpper(const wchar32* text, size_t length, wchar32* out) noexcept;
bool ToTitle(const wchar32* text, size_t length, wchar32* out) noexcept;

/* @param text      Pointer to the string to modify
 * @param length    Length of the string to modify
 *
 * TODO(yazevnul): replace these functions with `bool(const TArrayRef<wchar16>)` overload.
 */
bool ToLower(wchar16* text, size_t length) noexcept;
bool ToUpper(wchar16* text, size_t length) noexcept;
bool ToTitle(wchar16* text, size_t length) noexcept;

bool ToLower(wchar32* text, size_t length) noexcept;
bool ToUpper(wchar32* text, size_t length) noexcept;
bool ToTitle(wchar32* text, size_t length) noexcept;

/* Convenience wrappers for `ToLower`, `ToUpper` and `ToTitle`.
 */
TUtf16String ToLowerRet(TUtf16String text, size_t pos = 0, size_t count = TUtf16String::npos) Y_WARN_UNUSED_RESULT;
TUtf16String ToUpperRet(TUtf16String text, size_t pos = 0, size_t count = TUtf16String::npos) Y_WARN_UNUSED_RESULT;
TUtf16String ToTitleRet(TUtf16String text, size_t pos = 0, size_t count = TUtf16String::npos) Y_WARN_UNUSED_RESULT;

TUtf16String ToLowerRet(const TWtringBuf text, size_t pos = 0, size_t count = TWtringBuf::npos) Y_WARN_UNUSED_RESULT;
TUtf16String ToUpperRet(const TWtringBuf text, size_t pos = 0, size_t count = TWtringBuf::npos) Y_WARN_UNUSED_RESULT;
TUtf16String ToTitleRet(const TWtringBuf text, size_t pos = 0, size_t count = TWtringBuf::npos) Y_WARN_UNUSED_RESULT;

TUtf32String ToLowerRet(const TUtf32StringBuf text, size_t pos = 0, size_t count = TWtringBuf::npos) Y_WARN_UNUSED_RESULT;
TUtf32String ToUpperRet(const TUtf32StringBuf text, size_t pos = 0, size_t count = TWtringBuf::npos) Y_WARN_UNUSED_RESULT;
TUtf32String ToTitleRet(const TUtf32StringBuf text, size_t pos = 0, size_t count = TWtringBuf::npos) Y_WARN_UNUSED_RESULT;

//! replaces the '<', '>' and '&' characters in string with '&lt;', '&gt;' and '&amp;' respectively
// insertBr=true - replace '\r' and '\n' with "<BR>"
template <bool insertBr>
void EscapeHtmlChars(TUtf16String& str);

//! returns number of characters in range. Handle surrogate pairs as one character.
inline size_t CountWideChars(const wchar16* b, const wchar16* e) {
    size_t count = 0;
    Y_ENSURE(b <= e, TStringBuf("invalid iterators"));
    while (b < e) {
        b = SkipSymbol(b, e);
        ++count;
    }
    return count;
}

inline size_t CountWideChars(const TWtringBuf str) {
    return CountWideChars(str.begin(), str.end());
}

//! checks whether the range is valid UTF-16 sequence
inline bool IsValidUTF16(const wchar16* b, const wchar16* e) {
    Y_ENSURE(b <= e, TStringBuf("invalid iterators"));
    while (b < e) {
        wchar32 symbol = ReadSymbolAndAdvance(b, e);
        if (symbol == BROKEN_RUNE)
            return false;
    }
    return true;
}

inline bool IsValidUTF16(const TWtringBuf str) {
    return IsValidUTF16(str.begin(), str.end());
}
