#pragma once

#include <util/generic/string.h>
#include <util/generic/yexception.h>
#include <util/system/yassert.h>

inline static char DigitToChar(unsigned char digit) {
    if (digit < 10) {
        return (char)digit + '0';
    }

    return (char)(digit - 10) + 'A';
}

extern const char* const Char2DigitTable;

inline static int Char2Digit(char ch) {
    char result = Char2DigitTable[(unsigned char)ch];
    Y_ENSURE(result != '\xff', "invalid hex character " << (int)ch);
    return result;
}

//! Convert a hex string of exactly 2 chars to int
/*! @example String2Byte("10") => 16 */
inline static int String2Byte(const char* s) {
    return Char2Digit(*s) * 16 + Char2Digit(*(s + 1));
}

char* HexEncode(const void* in, size_t len, char* out);

TString HexEncode(const void* in, size_t len);

inline TString HexEncode(const TStringBuf h) {
    return HexEncode(h.data(), h.size());
}

//! Convert a hex string @c in of @c len chars (case-insensitive) to array of ints stored at @c ptr and return this array.
/*! @note len must be even (len % 2 == 0), otherwise an exception will be thrown.
 *  @return @c ptr, which is an array of chars, where each char holds the numeric value
 *          equal to the corresponding 2 digits of the input stream.
 *  @warning You must ensure that @c ptr has (len/2) allocated bytes, otherwise SIGSEGV will happen.
 *
 * @example HexDecode("beef", 4, ptr) => {190, 239}
 */
void* HexDecode(const void* in, size_t len, void* ptr);

//! Convert a hex string @c in of @c len chars (case-insensitive) to array of ints and return this array.
/*! @note len must be even (len % 2 == 0), otherwise an exception will be thrown.
 *  @return an array of chars, where each char holds the numeric value equal to the corresponding 2 digits
 *          of the input stream.
 *
 * @example HexDecode("beef", 4) => {190, 239}
 */
TString HexDecode(const void* in, size_t len);

//! Convert an ASCII hex-string (case-insensitive) to the binary form. Note that h.Size() must be even (+h % 2 == 0).
inline TString HexDecode(const TStringBuf h) {
    return HexDecode(h.data(), h.size());
}
