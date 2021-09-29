#include "escape.h"
#include "cast.h"

#include <util/system/defaults.h>
#include <util/charset/utf8.h>
#include <util/charset/wide.h>

/// @todo: escape trigraphs (eg "??/" is "\")

/* REFEREBCES FOR ESCAPE SEQUENCE INTERPRETATION:
 *   C99 p. 6.4.3   Universal character names.
 *   C99 p. 6.4.4.4 Character constants.
 *
 * <simple-escape-sequence> ::= {
 *      \' , \" , \? , \\ ,
 *      \a , \b , \f , \n , \r , \t , \v
 * }
 *
 * <octal-escape-sequence>       ::= \  <octal-digit> {1, 3}
 * <hexadecimal-escape-sequence> ::= \x <hexadecimal-digit> +
 * <universal-character-name>    ::= \u <hexadecimal-digit> {4}
 *                                || \U <hexadecimal-digit> {8}
 *
 * NOTE (6.4.4.4.7):
 * Each octal or hexadecimal escape sequence is the longest sequence of characters that can
 * constitute the escape sequence.
 *
 * THEREFORE:
 *  - Octal escape sequence spans until rightmost non-octal-digit character.
 *  - Octal escape sequence always terminates after three octal digits.
 *  - Hexadecimal escape sequence spans until rightmost non-hexadecimal-digit character.
 *  - Universal character name consists of exactly 4 or 8 hexadecimal digit.
 *
 * by kerzum@
 * It is also required to escape trigraphs that are enabled in compilers by default and
 * are also processed inside string literals
 *      The nine trigraphs and their replacements are
 *
 *      Trigraph:       ??(  ??)  ??<  ??>  ??=  ??/  ??'  ??!  ??-
 *      Replacement:      [    ]    {    }    #    \    ^    |    ~
 *
 */
namespace {
    template <typename TChar>
    static inline char HexDigit(TChar value) {
        Y_ASSERT(value < 16);
        if (value < 10) {
            return '0' + value;
        } else {
            return 'A' + value - 10;
        }
    }

    template <typename TChar>
    static inline char OctDigit(TChar value) {
        Y_ASSERT(value < 8);
        return '0' + value;
    }

    template <typename TChar>
    static inline bool IsPrintable(TChar c) {
        return c >= 32 && c <= 126;
    }

    template <typename TChar>
    static inline bool IsHexDigit(TChar c) {
        return (c >= '0' && c <= '9') || (c >= 'A' && c <= 'F') || (c >= 'a' && c <= 'f');
    }

    template <typename TChar>
    static inline bool IsOctDigit(TChar c) {
        return c >= '0' && c <= '7';
    }

    template <typename TChar>
    struct TEscapeUtil;

    template <>
    struct TEscapeUtil<char> {
        static const size_t ESCAPE_C_BUFFER_SIZE = 4;

        template <typename TNextChar, typename TBufferChar>
        static inline size_t EscapeC(unsigned char c, TNextChar next, TBufferChar r[ESCAPE_C_BUFFER_SIZE]) {
            // (1) Printable characters go as-is, except backslash and double quote.
            // (2) Characters \r, \n, \t and \0 ... \7 replaced by their simple escape characters (if possible).
            // (3) Otherwise, character is encoded using hexadecimal escape sequence (if possible), or octal.
            if (c == '\"') {
                r[0] = '\\';
                r[1] = '\"';
                return 2;
            } else if (c == '\\') {
                r[0] = '\\';
                r[1] = '\\';
                return 2;
            } else if (IsPrintable(c) && (!(c == '?' && next == '?'))) {
                r[0] = c;
                return 1;
            } else if (c == '\r') {
                r[0] = '\\';
                r[1] = 'r';
                return 2;
            } else if (c == '\n') {
                r[0] = '\\';
                r[1] = 'n';
                return 2;
            } else if (c == '\t') {
                r[0] = '\\';
                r[1] = 't';
                return 2;
            } else if (c < 8 && !IsOctDigit(next)) {
                r[0] = '\\';
                r[1] = OctDigit(c);
                return 2;
            } else if (!IsHexDigit(next)) {
                r[0] = '\\';
                r[1] = 'x';
                r[2] = HexDigit((c & 0xF0) >> 4);
                r[3] = HexDigit((c & 0x0F) >> 0);
                return 4;
            } else {
                r[0] = '\\';
                r[1] = OctDigit((c & 0700) >> 6);
                r[2] = OctDigit((c & 0070) >> 3);
                r[3] = OctDigit((c & 0007) >> 0);
                return 4;
            }
        }
    };

    template <>
    struct TEscapeUtil<wchar16> {
        static const size_t ESCAPE_C_BUFFER_SIZE = 6;

        template <typename TNextChar, typename TBufferChar>
        static inline size_t EscapeC(wchar16 c, TNextChar next, TBufferChar r[ESCAPE_C_BUFFER_SIZE]) {
            if (c < 0x100) {
                return TEscapeUtil<char>::EscapeC(char(c), next, r);
            } else {
                r[0] = '\\';
                r[1] = 'u';
                r[2] = HexDigit((c & 0xF000) >> 12);
                r[3] = HexDigit((c & 0x0F00) >> 8);
                r[4] = HexDigit((c & 0x00F0) >> 4);
                r[5] = HexDigit((c & 0x000F) >> 0);
                return 6;
            }
        }
    };
}

template <class TChar>
TBasicString<TChar>& EscapeCImpl(const TChar* str, size_t len, TBasicString<TChar>& r) {
    using TEscapeUtil = ::TEscapeUtil<TChar>;

    TChar buffer[TEscapeUtil::ESCAPE_C_BUFFER_SIZE];

    size_t i, j;
    for (i = 0, j = 0; i < len; ++i) {
        size_t rlen = TEscapeUtil::EscapeC(str[i], (i + 1 < len ? str[i + 1] : 0), buffer);

        if (rlen > 1) {
            r.append(str + j, i - j);
            j = i + 1;
            r.append(buffer, rlen);
        }
    }

    if (j > 0) {
        r.append(str + j, len - j);
    } else {
        r.append(str, len);
    }

    return r;
}

template TString& EscapeCImpl<TString::TChar>(const TString::TChar* str, size_t len, TString& r);
template TUtf16String& EscapeCImpl<TUtf16String::TChar>(const TUtf16String::TChar* str, size_t len, TUtf16String& r);

namespace {
    template <class TStr>
    inline void AppendUnicode(TStr& s, wchar32 v) {
        char buf[10];
        size_t sz = 0;

        WriteUTF8Char(v, sz, (ui8*)buf);
        s.AppendNoAlias(buf, sz);
    }

    inline void AppendUnicode(TUtf16String& s, wchar32 v) {
        WriteSymbol(v, s);
    }

    template <ui32 sz, typename TChar>
    inline size_t CountHex(const TChar* p, const TChar* pe) {
        auto b = p;
        auto e = Min(p + sz, pe);

        while (b < e && IsHexDigit(*b)) {
            ++b;
        }

        return b - p;
    }

    template <size_t sz, typename TChar, typename T>
    inline bool ParseHex(const TChar* p, const TChar* pe, T& t) noexcept {
        return (p + sz <= pe) && TryIntFromString<16>(p, sz, t);
    }

    template <ui32 sz, typename TChar>
    inline size_t CountOct(const TChar* p, const TChar* pe) {
        ui32 maxsz = Min<size_t>(sz, pe - p);

        if (3 == sz && 3 == maxsz && !(*p >= '0' && *p <= '3')) {
            maxsz = 2;
        }

        for (ui32 i = 0; i < maxsz; ++i, ++p) {
            if (!IsOctDigit(*p)) {
                return i;
            }
        }

        return maxsz;
    }
}

template <class TChar, class TStr>
static TStr& DoUnescapeC(const TChar* p, size_t sz, TStr& res) {
    const TChar* pe = p + sz;

    while (p != pe) {
        if ('\\' == *p) {
            ++p;

            if (p == pe) {
                return res;
            }

            switch (*p) {
                default:
                    res.append(*p);
                    break;
                case 'a':
                    res.append('\a');
                    break;
                case 'b':
                    res.append('\b');
                    break;
                case 'f':
                    res.append('\f');
                    break;
                case 'n':
                    res.append('\n');
                    break;
                case 'r':
                    res.append('\r');
                    break;
                case 't':
                    res.append('\t');
                    break;
                case 'v':
                    res.append('\v');
                    break;
                case 'u': {
                    ui16 cp[2];

                    if (ParseHex<4>(p + 1, pe, cp[0])) {
                        if (Y_UNLIKELY(cp[0] >= 0xD800 && cp[0] <= 0xDBFF && ParseHex<4>(p + 7, pe, cp[1]) && p[5] == '\\' && p[6] == 'u')) {
                            const wchar16 wbuf[] = {wchar16(cp[0]), wchar16(cp[1])};
                            AppendUnicode(res, ReadSymbol(wbuf, wbuf + 2));
                            p += 10;
                        } else {
                            AppendUnicode(res, (wchar32)cp[0]);
                            p += 4;
                        }
                    } else {
                        res.append(*p);
                    }

                    break;
                }

                case 'U':
                    if (CountHex<8>(p + 1, pe) != 8) {
                        res.append(*p);
                    } else {
                        AppendUnicode(res, IntFromString<ui32, 16>(p + 1, 8));
                        p += 8;
                    }
                    break;
                case 'x':
                    if (ui32 v = CountHex<2>(p + 1, pe)) {
                        res.append((TChar)IntFromString<ui32, 16>(p + 1, v));
                        p += v;
                    } else {
                        res.append(*p);
                    }

                    break;
                case '0':
                case '1':
                case '2':
                case '3': {
                    ui32 v = CountOct<3>(p, pe); // v is always positive
                    res.append((TChar)IntFromString<ui32, 8>(p, v));
                    p += v - 1;
                } break;
                case '4':
                case '5':
                case '6':
                case '7': {
                    ui32 v = CountOct<2>(p, pe); // v is always positive
                    res.append((TChar)IntFromString<ui32, 8>(p, v));
                    p += v - 1;
                } break;
            }

            ++p;
        } else {
            const auto r = std::basic_string_view<TChar>(p, pe - p).find('\\');
            const auto n = r != std::string::npos ? p + r : pe;

            res.append(p, n);
            p = n;
        }
    }

    return res;
}

template <class TChar>
TBasicString<TChar>& UnescapeCImpl(const TChar* p, size_t sz, TBasicString<TChar>& res) {
    return DoUnescapeC(p, sz, res);
}

template <class TChar>
TChar* UnescapeC(const TChar* str, size_t len, TChar* buf) {
    struct TUnboundedString {
        void append(TChar ch) noexcept {
            *P++ = ch;
        }

        void append(const TChar* b, const TChar* e) noexcept {
            while (b != e) {
                append(*b++);
            }
        }

        void AppendNoAlias(const TChar* s, size_t l) noexcept {
            append(s, s + l);
        }

        TChar* P;
    } bufbuf = {buf};

    return DoUnescapeC(str, len, bufbuf).P;
}

template TString& UnescapeCImpl<TString::TChar>(const TString::TChar* str, size_t len, TString& r);
template TUtf16String& UnescapeCImpl<TUtf16String::TChar>(const TUtf16String::TChar* str, size_t len, TUtf16String& r);

template char* UnescapeC<char>(const char* str, size_t len, char* buf);

template <class TChar>
size_t UnescapeCCharLen(const TChar* begin, const TChar* end) {
    if (begin >= end) {
        return 0;
    }
    if (*begin != '\\') {
        return 1;
    }
    if (++begin == end) {
        return 1;
    }

    switch (*begin) {
        default:
            return 2;
        case 'u':
            return CountHex<4>(begin + 1, end) == 4 ? 6 : 2;
        case 'U':
            return CountHex<8>(begin + 1, end) == 8 ? 10 : 2;
        case 'x':
            return 2 + CountHex<2>(begin + 1, end);
        case '0':
        case '1':
        case '2':
        case '3':
            return 1 + CountOct<3>(begin, end); // >= 2
        case '4':
        case '5':
        case '6':
        case '7':
            return 1 + CountOct<2>(begin, end); // >= 2
    }
}

template size_t UnescapeCCharLen<char>(const char* begin, const char* end);
template size_t UnescapeCCharLen<TUtf16String::TChar>(const TUtf16String::TChar* begin, const TUtf16String::TChar* end);

TString& EscapeC(const TStringBuf str, TString& s) {
    return EscapeC(str.data(), str.size(), s);
}

TUtf16String& EscapeC(const TWtringBuf str, TUtf16String& w) {
    return EscapeC(str.data(), str.size(), w);
}

TString EscapeC(const TString& str) {
    return EscapeC(str.data(), str.size());
}

TUtf16String EscapeC(const TUtf16String& str) {
    return EscapeC(str.data(), str.size());
}

TString& UnescapeC(const TStringBuf str, TString& s) {
    return UnescapeC(str.data(), str.size(), s);
}

TUtf16String& UnescapeC(const TWtringBuf str, TUtf16String& w) {
    return UnescapeC(str.data(), str.size(), w);
}

TString UnescapeC(const TStringBuf str) {
    return UnescapeC(str.data(), str.size());
}

TUtf16String UnescapeC(const TWtringBuf str) {
    return UnescapeC(str.data(), str.size());
}
