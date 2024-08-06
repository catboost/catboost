#pragma once

#include "recode_result.h"

#include <util/generic/strbuf.h>
#include <util/generic/string.h>
#include <util/generic/yexception.h>
#include <util/system/defaults.h>
#include <util/system/yassert.h>

extern const wchar32 BROKEN_RUNE;

inline unsigned char UTF8LeadByteMask(size_t utf8_rune_len) {
    // Y_ASSERT (utf8_rune_len <= 4);
    return "\0\0\037\017\007"[utf8_rune_len];
}

inline size_t UTF8RuneLen(const unsigned char lead_byte) {
    //b0XXXXXXX
    if ((lead_byte & 0x80) == 0x00) {
        return 1;
    }
    //b110XXXXX
    if ((lead_byte & 0xe0) == 0xc0) {
        return 2;
    }
    //b1110XXXX
    if ((lead_byte & 0xf0) == 0xe0) {
        return 3;
    }
    //b11110XXX
    if ((lead_byte & 0xf8) == 0xf0) {
        return 4;
    }
    //b10XXXXXX
    return 0;
}

inline size_t UTF8RuneLenByUCS(wchar32 rune) {
    if (rune < 0x80)
        return 1U;
    else if (rune < 0x800)
        return 2U;
    else if (rune < 0x10000)
        return 3U;
    else if (rune < 0x200000)
        return 4U;
    else if (rune < 0x4000000)
        return 5U;
    else
        return 6U;
}

inline void PutUTF8LeadBits(wchar32& rune, unsigned char c, size_t len) {
    rune = c;
    rune &= UTF8LeadByteMask(len);
}

inline void PutUTF8SixBits(wchar32& rune, unsigned char c) {
    rune <<= 6;
    rune |= c & 0x3F;
}

inline bool IsUTF8ContinuationByte(unsigned char c) {
    return (c & static_cast<unsigned char>(0xC0)) == static_cast<unsigned char>(0x80);
}

//! returns length of the current UTF8 character
//! @param n    length of the current character, it is assigned in case of valid UTF8 byte sequence
//! @param p    pointer to the current character
//! @param e    end of the character sequence
inline RECODE_RESULT GetUTF8CharLen(size_t& n, const unsigned char* p, const unsigned char* e) {
    Y_ASSERT(p < e); // since p < e then we will check RECODE_EOINPUT only for n > 1 (see calls of this functions)
    switch (UTF8RuneLen(*p)) {
        case 0:
            return RECODE_BROKENSYMBOL; //[BROKENSYMBOL] in first byte

        case 1:
            n = 1;
            return RECODE_OK;

        case 2:
            if (p + 2 > e) {
                return RECODE_EOINPUT;
            } else if (!IsUTF8ContinuationByte(p[1])) {
                return RECODE_BROKENSYMBOL;
            } else {
                n = 2;
                return RECODE_OK;
            }
        case 3:
            if (p + 3 > e) {
                return RECODE_EOINPUT;
            } else if (!IsUTF8ContinuationByte(p[1]) || !IsUTF8ContinuationByte(p[2])) {
                return RECODE_BROKENSYMBOL;
            } else {
                n = 3;
                return RECODE_OK;
            }
        default: // actually 4
            if (p + 4 > e) {
                return RECODE_EOINPUT;
            } else if (!IsUTF8ContinuationByte(p[1]) || !IsUTF8ContinuationByte(p[2]) || !IsUTF8ContinuationByte(p[3])) {
                return RECODE_BROKENSYMBOL;
            } else {
                n = 4;
                return RECODE_OK;
            }
    }
}

//! returns number of characters in UTF8 encoded text, stops immediately if UTF8 byte sequence is wrong
//! @param text     UTF8 encoded text
//! @param len      the length of the text in bytes
//! @param number   number of encoded symbols in the text
inline bool GetNumberOfUTF8Chars(const char* text, size_t len, size_t& number) {
    const unsigned char* cur = reinterpret_cast<const unsigned char*>(text);
    const unsigned char* const last = cur + len;
    number = 0;
    size_t runeLen;
    bool res = true;
    while (cur != last) {
        if (GetUTF8CharLen(runeLen, cur, last) != RECODE_OK) { // actually it could be RECODE_BROKENSYMBOL only
            res = false;
            break;
        }
        cur += runeLen;
        Y_ASSERT(cur <= last);
        ++number;
    }
    return res;
}

inline size_t GetNumberOfUTF8Chars(TStringBuf text) {
    size_t number;
    if (!GetNumberOfUTF8Chars(text.data(), text.size(), number)) {
        ythrow yexception() << "GetNumberOfUTF8Chars failed on invalid utf-8 " << TString(text.substr(0, 50)).Quote();
    }
    return number;
}

enum class StrictUTF8 {
    Yes,
    No
};

template <size_t runeLen, StrictUTF8 strictMode>
inline bool IsValidUTF8Rune(wchar32 rune);

template <>
inline bool IsValidUTF8Rune<2, StrictUTF8::Yes>(wchar32 rune) {
    // check for overlong encoding
    return rune >= 0x80;
}

template <>
inline bool IsValidUTF8Rune<2, StrictUTF8::No>(wchar32 rune) {
    return IsValidUTF8Rune<2, StrictUTF8::Yes>(rune);
}

template <>
inline bool IsValidUTF8Rune<3, StrictUTF8::Yes>(wchar32 rune) {
    // surrogates are forbidden by RFC3629 section 3
    return rune >= 0x800 && (rune < 0xD800 || rune > 0xDFFF);
}

template <>
inline bool IsValidUTF8Rune<3, StrictUTF8::No>(wchar32 rune) {
    // check for overlong encoding
    return rune >= 0x800;
}

template <>
inline bool IsValidUTF8Rune<4, StrictUTF8::Yes>(wchar32 rune) {
    // check if this is a valid sumbod without overlong encoding
    return rune <= 0x10FFFF && rune >= 0x10000;
}

template <>
inline bool IsValidUTF8Rune<4, StrictUTF8::No>(wchar32 rune) {
    return IsValidUTF8Rune<4, StrictUTF8::Yes>(rune);
}

//! reads one unicode symbol from a character sequence encoded UTF8 and checks for overlong encoding
//! @param rune      value of the current character
//! @param rune_len  length of the UTF8 bytes sequence that has been read
//! @param s         pointer to the current character
//! @param end       the end of the character sequence
template <StrictUTF8 strictMode = StrictUTF8::No>
inline RECODE_RESULT SafeReadUTF8Char(wchar32& rune, size_t& rune_len, const unsigned char* s, const unsigned char* end) {
    rune = BROKEN_RUNE;
    rune_len = 0;
    wchar32 _rune;

    size_t _len = UTF8RuneLen(*s);
    if (s + _len > end)
        return RECODE_EOINPUT; //[EOINPUT]
    if (_len == 0)
        return RECODE_BROKENSYMBOL; //[BROKENSYMBOL] in first byte
    _rune = *s++;                   //[00000000 0XXXXXXX]

    if (_len > 1) {
        _rune &= UTF8LeadByteMask(_len);
        unsigned char ch = *s++;
        if (!IsUTF8ContinuationByte(ch))
            return RECODE_BROKENSYMBOL; //[BROKENSYMBOL] in second byte
        PutUTF8SixBits(_rune, ch);      //[00000XXX XXYYYYYY]
        if (_len > 2) {
            ch = *s++;
            if (!IsUTF8ContinuationByte(ch))
                return RECODE_BROKENSYMBOL; //[BROKENSYMBOL] in third byte
            PutUTF8SixBits(_rune, ch);      //[XXXXYYYY YYZZZZZZ]
            if (_len > 3) {
                ch = *s;
                if (!IsUTF8ContinuationByte(ch))
                    return RECODE_BROKENSYMBOL; //[BROKENSYMBOL] in fourth byte
                PutUTF8SixBits(_rune, ch);      //[XXXYY YYYYZZZZ ZZQQQQQQ]
                if (!IsValidUTF8Rune<4, strictMode>(_rune))
                    return RECODE_BROKENSYMBOL;
            } else {
                if (!IsValidUTF8Rune<3, strictMode>(_rune))
                    return RECODE_BROKENSYMBOL;
            }
        } else {
            if (!IsValidUTF8Rune<2, strictMode>(_rune))
                return RECODE_BROKENSYMBOL;
        }
    }
    rune_len = _len;
    rune = _rune;
    return RECODE_OK;
}

//! reads one unicode symbol from a character sequence encoded UTF8 and moves pointer to the next character
//! @param c    value of the current character
//! @param p    pointer to the current character, it will be changed in case of valid UTF8 byte sequence
//! @param e    the end of the character sequence
template <StrictUTF8 strictMode = StrictUTF8::No>
Y_FORCE_INLINE RECODE_RESULT ReadUTF8CharAndAdvance(wchar32& rune, const unsigned char*& p, const unsigned char* e) noexcept {
    Y_ASSERT(p < e); // since p < e then we will check RECODE_EOINPUT only for n > 1 (see calls of this functions)
    switch (UTF8RuneLen(*p)) {
        case 0:
            rune = BROKEN_RUNE;
            return RECODE_BROKENSYMBOL; //[BROKENSYMBOL] in first byte

        case 1:
            rune = *p; //[00000000 0XXXXXXX]
            ++p;
            return RECODE_OK;

        case 2:
            if (p + 2 > e) {
                return RECODE_EOINPUT;
            } else if (!IsUTF8ContinuationByte(p[1])) {
                rune = BROKEN_RUNE;
                return RECODE_BROKENSYMBOL;
            } else {
                PutUTF8LeadBits(rune, *p++, 2); //[00000000 000XXXXX]
                PutUTF8SixBits(rune, *p++);     //[00000XXX XXYYYYYY]
                if (!IsValidUTF8Rune<2, strictMode>(rune)) {
                    p -= 2;
                    rune = BROKEN_RUNE;
                    return RECODE_BROKENSYMBOL;
                }
                return RECODE_OK;
            }
        case 3:
            if (p + 3 > e) {
                return RECODE_EOINPUT;
            } else if (!IsUTF8ContinuationByte(p[1]) || !IsUTF8ContinuationByte(p[2])) {
                rune = BROKEN_RUNE;
                return RECODE_BROKENSYMBOL;
            } else {
                PutUTF8LeadBits(rune, *p++, 3); //[00000000 0000XXXX]
                PutUTF8SixBits(rune, *p++);     //[000000XX XXYYYYYY]
                PutUTF8SixBits(rune, *p++);     //[XXXXYYYY YYZZZZZZ]
                // check for overlong encoding and surrogates
                if (!IsValidUTF8Rune<3, strictMode>(rune)) {
                    p -= 3;
                    rune = BROKEN_RUNE;
                    return RECODE_BROKENSYMBOL;
                }
                return RECODE_OK;
            }
        case 4:
            if (p + 4 > e) {
                return RECODE_EOINPUT;
            } else if (!IsUTF8ContinuationByte(p[1]) || !IsUTF8ContinuationByte(p[2]) || !IsUTF8ContinuationByte(p[3])) {
                rune = BROKEN_RUNE;
                return RECODE_BROKENSYMBOL;
            } else {
                PutUTF8LeadBits(rune, *p++, 4); //[00000000 00000000 00000XXX]
                PutUTF8SixBits(rune, *p++);     //[00000000 0000000X XXYYYYYY]
                PutUTF8SixBits(rune, *p++);     //[00000000 0XXXYYYY YYZZZZZZ]
                PutUTF8SixBits(rune, *p++);     //[000XXXYY YYYYZZZZ ZZQQQQQQ]
                if (!IsValidUTF8Rune<4, strictMode>(rune)) {
                    p -= 4;
                    rune = BROKEN_RUNE;
                    return RECODE_BROKENSYMBOL;
                }
                return RECODE_OK;
            }
        default: // >4
            rune = BROKEN_RUNE;
            return RECODE_BROKENSYMBOL;
    }
}

//! writes one unicode symbol into a character sequence encoded UTF8
//! checks for end of the buffer and returns the result of encoding
//! @param rune      value of the current character
//! @param rune_len  length of the UTF8 byte sequence that has been written
//! @param s         pointer to the output buffer
//! @param tail      available size of the buffer
inline RECODE_RESULT SafeWriteUTF8Char(wchar32 rune, size_t& rune_len, unsigned char* s, size_t tail) {
    rune_len = 0;
    if (rune < 0x80) {
        if (tail <= 0)
            return RECODE_EOOUTPUT;
        *s = static_cast<unsigned char>(rune);
        rune_len = 1;
        return RECODE_OK;
    }
    if (rune < 0x800) {
        if (tail <= 1)
            return RECODE_EOOUTPUT;
        *s++ = static_cast<unsigned char>(0xC0 | (rune >> 6));
        *s = static_cast<unsigned char>(0x80 | (rune & 0x3F));
        rune_len = 2;
        return RECODE_OK;
    }
    if (rune < 0x10000) {
        if (tail <= 2)
            return RECODE_EOOUTPUT;
        *s++ = static_cast<unsigned char>(0xE0 | (rune >> 12));
        *s++ = static_cast<unsigned char>(0x80 | ((rune >> 6) & 0x3F));
        *s = static_cast<unsigned char>(0x80 | (rune & 0x3F));
        rune_len = 3;
        return RECODE_OK;
    }
    /*if (rune < 0x200000)*/ {
        if (tail <= 3)
            return RECODE_EOOUTPUT;
        *s++ = static_cast<unsigned char>(0xF0 | ((rune >> 18) & 0x07));
        *s++ = static_cast<unsigned char>(0x80 | ((rune >> 12) & 0x3F));
        *s++ = static_cast<unsigned char>(0x80 | ((rune >> 6) & 0x3F));
        *s = static_cast<unsigned char>(0x80 | (rune & 0x3F));
        rune_len = 4;
        return RECODE_OK;
    }
}

inline RECODE_RESULT SafeWriteUTF8Char(wchar32 rune, size_t& rune_len, unsigned char* s, const unsigned char* end) {
    return SafeWriteUTF8Char(rune, rune_len, s, end - s);
}

//! writes one unicode symbol into a character sequence encoded UTF8
//! @attention       this function works as @c SafeWriteUTF8Char it does not check
//!                  the size of the output buffer, it supposes that buffer is long enough
//! @param rune      value of the current character
//! @param rune_len  length of the UTF8 byte sequence that has been written
//! @param s         pointer to the output buffer
inline void WriteUTF8Char(wchar32 rune, size_t& rune_len, unsigned char* s) {
    if (rune < 0x80) {
        *s = static_cast<unsigned char>(rune);
        rune_len = 1;
        return;
    }
    if (rune < 0x800) {
        *s++ = static_cast<unsigned char>(0xC0 | (rune >> 6));
        *s = static_cast<unsigned char>(0x80 | (rune & 0x3F));
        rune_len = 2;
        return;
    }
    if (rune < 0x10000) {
        *s++ = static_cast<unsigned char>(0xE0 | (rune >> 12));
        *s++ = static_cast<unsigned char>(0x80 | ((rune >> 6) & 0x3F));
        *s = static_cast<unsigned char>(0x80 | (rune & 0x3F));
        rune_len = 3;
        return;
    }
    /*if (rune < 0x200000)*/ {
        *s++ = static_cast<unsigned char>(0xF0 | ((rune >> 18) & 0x07));
        *s++ = static_cast<unsigned char>(0x80 | ((rune >> 12) & 0x3F));
        *s++ = static_cast<unsigned char>(0x80 | ((rune >> 6) & 0x3F));
        *s = static_cast<unsigned char>(0x80 | (rune & 0x3F));
        rune_len = 4;
    }
}

TStringBuf SubstrUTF8(const TStringBuf str Y_LIFETIME_BOUND, size_t pos, size_t len);

enum EUTF8Detect {
    NotUTF8,
    UTF8,
    ASCII
};

EUTF8Detect UTF8Detect(const char* s, size_t len);

inline EUTF8Detect UTF8Detect(const TStringBuf input) {
    return UTF8Detect(input.data(), input.size());
}

inline bool IsUtf(const char* input, size_t len) {
    return UTF8Detect(input, len) != NotUTF8;
}

inline bool IsUtf(const TStringBuf input) {
    return IsUtf(input.data(), input.size());
}

//! returns true, if result is not the same as input, and put it in newString
//! returns false, if result is unmodified
bool ToLowerUTF8Impl(const char* beg, size_t n, TString& newString);

TString ToLowerUTF8(const TString& s);
TString ToLowerUTF8(TStringBuf s);
TString ToLowerUTF8(const char* s);

inline TString ToLowerUTF8(const std::string& s) {
    return ToLowerUTF8(TStringBuf(s));
}

//! returns true, if result is not the same as input, and put it in newString
//! returns false, if result is unmodified
bool ToUpperUTF8Impl(const char* beg, size_t n, TString& newString);

TString ToUpperUTF8(const TString& s);
TString ToUpperUTF8(TStringBuf s);
TString ToUpperUTF8(const char* s);
