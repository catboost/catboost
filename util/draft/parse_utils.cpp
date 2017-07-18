#include <util/charset/utf8.h>
#include <util/charset/wide.h>

#include <util/string/url.h>
#include <util/system/maxlen.h>

#include "parse_utils.h"

bool IsNonBMPUTF8(TStringBuf s) {
    wchar32 rune;
    size_t runeLen;

    const char* p = s.begin();
    const char* end = s.end();

    for (; p != end; p += runeLen) {
        if (SafeReadUTF8Char(rune, runeLen, (const ui8*)p, (const ui8*)end) != RECODE_OK)
            ythrow yexception() << "Non-valid UTF8";
        if (rune & 0xFFFF0000)
            return true;
    }
    return false;
}
