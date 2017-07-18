#include "unidata.h"
#include "utf8.h"

extern const wchar32 BROKEN_RUNE = 0xFFFD;

static const char* SkipUTF8Chars(const char* begin, const char* end, size_t numChars) {
    const unsigned char* uEnd = reinterpret_cast<const unsigned char*>(end);
    while (begin != end && numChars > 0) {
        const unsigned char* uBegin = reinterpret_cast<const unsigned char*>(begin);
        size_t runeLen;
        if (GetUTF8CharLen(runeLen, uBegin, uEnd) != RECODE_OK) {
            ythrow yexception() << "invalid UTF-8 char";
        }
        begin += runeLen;
        Y_ASSERT(begin <= end);
        --numChars;
    }
    return begin;
}

TStringBuf SubstrUTF8(const TStringBuf str, size_t pos, size_t len) {
    const char* start = SkipUTF8Chars(str.begin(), str.end(), pos);
    const char* end = SkipUTF8Chars(start, str.end(), len);
    return TStringBuf(start, end - start);
}

EUTF8Detect UTF8Detect(const char* s, size_t len) {
    const unsigned char* s0 = (const unsigned char*)s;
    const unsigned char* send = s0 + len;
    wchar32 rune;
    size_t rune_len;
    EUTF8Detect res = ASCII;

    while (s0 < send) {
        RECODE_RESULT rr = SafeReadUTF8Char(rune, rune_len, s0, send);

        if (rr != RECODE_OK) {
            return NotUTF8;
        }

        if (rune_len > 1) {
            res = UTF8;
        }

        s0 += rune_len;
    }

    return res;
}

bool ToLowerUTF8Impl(const char* beg, size_t n, TString& newString) {
    const unsigned char* p = (const unsigned char*)beg;
    const unsigned char* const end = p + n;

    //first loop searches for the first character, which is changed by ToLower
    //if there is no changed character, we don't need reallocation/copy
    wchar32 cNew = 0;
    size_t cLen = 0;
    while (p < end) {
        wchar32 c;
        if (RECODE_OK != SafeReadUTF8Char(c, cLen, p, end)) {
            ythrow yexception() << "failed to decode UTF-8 string at pos " << ((const char*)p - beg);
        }
        cNew = ToLower(c);

        if (cNew != c)
            break;
        p += cLen;
    }
    if (p == end) {
        return false;
    }

    //some character changed after ToLower. Write new string to newString.
    newString.resize(n);

    size_t written = (char*)p - beg;
    char* writePtr = newString.begin();
    memcpy(writePtr, beg, written);
    writePtr += written;
    size_t destSpace = n - written;

    //before each iteration (including the first one) variable 'cNew' contains unwritten symbol
    while (1) {
        size_t cNewLen;
        Y_ASSERT((writePtr - ~newString) + destSpace == +newString);
        if (RECODE_EOOUTPUT == SafeWriteUTF8Char(cNew, cNewLen, (unsigned char*)writePtr, destSpace)) {
            destSpace += +newString;
            newString.resize(+newString * 2);
            writePtr = newString.begin() + (+newString - destSpace);
            continue;
        }
        destSpace -= cNewLen;
        writePtr += cNewLen;
        p += cLen;
        if (p == end) {
            newString.resize(+newString - destSpace);
            return true;
        }
        wchar32 c = 0;
        if (RECODE_OK != SafeReadUTF8Char(c, cLen, p, end)) {
            ythrow yexception() << "failed to decode UTF-8 string at pos " << ((const char*)p - beg);
        }
        cNew = ToLower(c);
    }
    Y_ASSERT(false);
    return false;
}

TString ToLowerUTF8(const TString& s) {
    TString newString;
    bool changed = ToLowerUTF8Impl(~s, +s, newString);
    return changed ? newString : s;
}

TString ToLowerUTF8(TStringBuf s) {
    TString newString;
    bool changed = ToLowerUTF8Impl(~s, +s, newString);
    return changed ? newString : TString(~s, +s);
}

TString ToLowerUTF8(const char* s) {
    return ToLowerUTF8(TStringBuf(s));
}
