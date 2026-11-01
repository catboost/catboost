#include "relaxed_escaper.h"

#include <util/charset/utf8.h>

namespace {
    constexpr char HexDigits[] = "0123456789ABCDEF";

    void AppendUnicodeEscape(TString& result, ui16 codeUnit) {
        result += "\\u";
        result += HexDigits[(codeUnit >> 12) & 0x0F];
        result += HexDigits[(codeUnit >> 8) & 0x0F];
        result += HexDigits[(codeUnit >> 4) & 0x0F];
        result += HexDigits[codeUnit & 0x0F];
    }

    void AppendEscapedRune(TString& result, wchar32 rune) {
        switch (rune) {
            case '\n':
                result += "\\n";
                return;
            case '\t':
                result += "\\t";
                return;
            case '\r':
                result += "\\r";
                return;
            case '\b':
                result += "\\b";
                return;
            case '\f':
                result += "\\f";
                return;
            case '\v':
                result += "\\v";
                return;
            case '\a':
                result += "\\a";
                return;
            case '\\':
                result += "\\\\";
                return;
            case '"':
                result += "\\\"";
                return;
            case '/':
                result += "\\/";
                return;
            case '\0':
                result += "\\u0000";
                return;
        }

        if (rune < 32 || rune == 0x7F || (rune >= 0x80 && rune <= 0xFFFF)) {
            AppendUnicodeEscape(result, rune);
        } else if (rune > 0xFFFF) {
            rune -= 0x10000;
            AppendUnicodeEscape(result, 0xD800 + ((rune >> 10) & 0x03FF));
            AppendUnicodeEscape(result, 0xDC00 + (rune & 0x03FF));
        } else {
            result += static_cast<char>(rune);
        }
    }
} // namespace

TString NEscJ::EscapeJsonStringToAscii(TStringBuf input) {
    TString result;
    result.reserve(input.size());

    const auto* current = reinterpret_cast<const unsigned char*>(input.begin());
    const auto* end = reinterpret_cast<const unsigned char*>(input.end());
    while (current != end) {
        wchar32 rune = 0;
        size_t runeLength = 0;
        if (SafeReadUTF8Char(rune, runeLength, current, end) == RECODE_OK) {
            current += runeLength;
        } else {
            rune = *current++;
        }
        AppendEscapedRune(result, rune);
    }
    return result;
}
