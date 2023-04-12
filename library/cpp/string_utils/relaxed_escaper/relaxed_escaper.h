#pragma once

#include <util/stream/output.h>
#include <util/string/escape.h>
#include <util/memory/tempbuf.h>
#include <util/generic/strbuf.h>

namespace NEscJ {
    // almost copypaste from util/string/escape.h
    // todo: move there (note difference in IsPrintable and handling of string)

    inline char HexDigit(char value) {
        if (value < 10)
            return '0' + value;
        else
            return 'A' + value - 10;
    }

    inline char OctDigit(char value) {
        return '0' + value;
    }

    inline bool IsUTF8(ui8 c) {
        return c < 0xf5 && c != 0xC0 && c != 0xC1;
    }

    inline bool IsControl(ui8 c) {
        return c < 0x20 || c == 0x7f;
    }

    inline bool IsPrintable(ui8 c) {
        return IsUTF8(c) && !IsControl(c);
    }

    inline bool IsHexDigit(ui8 c) {
        return (c >= '0' && c <= '9') || (c >= 'A' && c <= 'F') || (c >= 'a' && c <= 'f');
    }

    inline bool IsOctDigit(ui8 c) {
        return c >= '0' && c <= '7';
    }

    struct TEscapeUtil {
        static constexpr size_t ESCAPE_C_BUFFER_SIZE = 6;

        template <bool asunicode, bool hasCustomSafeUnsafe>
        static inline size_t EscapeJ(ui8 c, ui8 next, char r[ESCAPE_C_BUFFER_SIZE], TStringBuf safe, TStringBuf unsafe) {
            // (1) Printable characters go as-is, except backslash and double quote.
            // (2) Characters \r, \n, \t and \0 ... \7 replaced by their simple escape characters (if possible).
            // (3) Otherwise, character is encoded using hexadecimal escape sequence (if possible), or octal.
            if (hasCustomSafeUnsafe && safe.find(c) != TStringBuf::npos) {
                r[0] = c;
                return 1;
            }
            if (c == '\"') {
                r[0] = '\\';
                r[1] = '\"';
                return 2;
            } else if (c == '\\') {
                r[0] = '\\';
                r[1] = '\\';
                return 2;
            } else if (IsPrintable(c) && (!hasCustomSafeUnsafe || unsafe.find(c) == TStringBuf::npos)) {
                r[0] = c;
                return 1;
            } else if (c == '\b') {
                r[0] = '\\';
                r[1] = 'b';
                return 2;
            } else if (c == '\f') {
                r[0] = '\\';
                r[1] = 'f';
                return 2;
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
            } else if (asunicode && IsUTF8(c)) { // utf8 controls escape for json
                r[0] = '\\';
                r[1] = 'u';
                r[2] = '0';
                r[3] = '0';
                r[4] = HexDigit((c & 0xF0) >> 4);
                r[5] = HexDigit((c & 0x0F) >> 0);
                return 6;
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

    inline size_t SuggestBuffer(size_t len) {
        return len * TEscapeUtil::ESCAPE_C_BUFFER_SIZE;
    }

    template <bool tounicode, bool hasCustomSafeUnsafe>
    inline size_t EscapeJImpl(const char* str, size_t len, char* out, TStringBuf safe, TStringBuf unsafe) {
        char* out0 = out;
        char buffer[TEscapeUtil::ESCAPE_C_BUFFER_SIZE];

        size_t i, j;
        for (i = 0, j = 0; i < len; ++i) {
            size_t rlen = TEscapeUtil::EscapeJ<tounicode, hasCustomSafeUnsafe>(str[i], (i + 1 < len ? str[i + 1] : 0), buffer, safe, unsafe);

            if (rlen > 1) {
                memcpy(out, str + j, i - j);
                out += i - j;
                j = i + 1;

                memcpy(out, buffer, rlen);
                out += rlen;
            }
        }

        if (j > 0) {
            memcpy(out, str + j, len - j);
            out += len - j;
        } else {
            memcpy(out, str, len);
            out += len;
        }

        return out - out0;
    }

    template <bool tounicode>
    inline size_t EscapeJ(const char* str, size_t len, char* out, TStringBuf safe = TStringBuf(), TStringBuf unsafe = TStringBuf()) {
        if (Y_LIKELY(safe.empty() && unsafe.empty())) {
            return EscapeJImpl<tounicode, false>(str, len, out, safe, unsafe);
        }
        return EscapeJImpl<tounicode, true>(str, len, out, safe, unsafe);
    }

    template <bool quote, bool tounicode>
    inline void EscapeJ(TStringBuf in, IOutputStream& out, TStringBuf safe = TStringBuf(), TStringBuf unsafe = TStringBuf()) {
        TTempBuf b(SuggestBuffer(in.size()) + 2);

        if (quote)
            b.Append("\"", 1);

        b.Proceed(EscapeJ<tounicode>(in.data(), in.size(), b.Current(), safe, unsafe));

        if (quote)
            b.Append("\"", 1);

        out.Write(b.Data(), b.Filled());
    }

    template <bool quote, bool tounicode>
    inline void EscapeJ(TStringBuf in, TString& out, TStringBuf safe = TStringBuf(), TStringBuf unsafe = TStringBuf()) {
        TTempBuf b(SuggestBuffer(in.size()) + 2);

        if (quote)
            b.Append("\"", 1);

        b.Proceed(EscapeJ<tounicode>(in.data(), in.size(), b.Current(), safe, unsafe));

        if (quote)
            b.Append("\"", 1);

        out.append(b.Data(), b.Filled());
    }

    template <bool quote, bool tounicode>
    inline TString EscapeJ(TStringBuf in, TStringBuf safe = TStringBuf(), TStringBuf unsafe = TStringBuf()) {
        TString s;
        EscapeJ<quote, tounicode>(in, s, safe, unsafe);
        return s;
    }

    // If the template parameter "tounicode" is ommited, then use the default value false
    inline size_t EscapeJ(const char* str, size_t len, char* out, TStringBuf safe = TStringBuf(), TStringBuf unsafe = TStringBuf()) {
        return EscapeJ<false>(str, len, out, safe, unsafe);
    }

    template <bool quote>
    inline void EscapeJ(TStringBuf in, IOutputStream& out, TStringBuf safe = TStringBuf(), TStringBuf unsafe = TStringBuf()) {
        EscapeJ<quote, false>(in, out, safe, unsafe);
    }

    template <bool quote>
    inline void EscapeJ(TStringBuf in, TString& out, TStringBuf safe = TStringBuf(), TStringBuf unsafe = TStringBuf()) {
        EscapeJ<quote, false>(in, out, safe, unsafe);
    }

    template <bool quote>
    inline TString EscapeJ(TStringBuf in, TStringBuf safe = TStringBuf(), TStringBuf unsafe = TStringBuf()) {
        return EscapeJ<quote, false>(in, safe, unsafe);
    }
}
