#include "ci_string.h"
#include "codepage.h"
#include "recyr.hh"

#include <util/system/hi_lo.h>
#include <util/generic/vector.h>

template <typename TxChar>
static inline RECODE_RESULT utf8_read_rune_from_unknown_plane(TxChar& rune, size_t& rune_len, const TxChar* s, const TxChar* end) {
    if ((*s & 0xFF00) != 0xF000) {
        rune_len = 1;
        rune = *s;
        return RECODE_OK;
    }

    rune_len = 0;

    size_t _len = UTF8RuneLen((unsigned char)(*s));
    if (s + _len > end)
        return RECODE_EOINPUT; //[EOINPUT]
    if (_len == 0)
        return RECODE_BROKENSYMBOL; //[BROKENSYMBOL] in first byte

    wchar32 _rune = (ui8)(*s++); //[00000000 0XXXXXXX]
    if (_len > 1) {
        _rune &= UTF8LeadByteMask(_len);
        wchar32 ch = *s++;
        if ((ch & 0xFFC0) != 0xF080)
            return RECODE_BROKENSYMBOL; //[BROKENSYMBOL] in second byte
        _rune <<= 6;
        _rune |= ch & 0x3F; //[00000XXX XXYYYYYY]
        if (_len > 2) {
            ch = *s++;
            if ((ch & 0xFFC0) != 0xF080)
                return RECODE_BROKENSYMBOL; //[BROKENSYMBOL] in third byte
            _rune <<= 6;
            _rune |= ch & 0x3F; //[XXXXYYYY YYZZZZZZ]
            if (_len > 3) {
                ch = *s;
                if ((ch & 0xFFC0) != 0xF080)
                    return RECODE_BROKENSYMBOL; //[BROKENSYMBOL] in fourth byte
                _rune <<= 6;
                _rune |= ch & 0x3F; //[XXXYY YYYYZZZZ ZZQQQQQQ]
            }
        }
    }
    rune_len = _len;
    if (_rune > Max<TxChar>())
        rune = ' '; // maybe put sequence
    else
        rune = TxChar(_rune);
    return RECODE_OK;
}

template <typename TxChar>
void DoDecodeUnknownPlane(TxChar* str, TxChar*& ee, const ECharset enc) {
    TxChar* e = ee;
    if (SingleByteCodepage(enc)) {
        const CodePage* cp = CodePageByCharset(enc);
        for (TxChar* s = str; s < e; s++) {
            if (Hi8(Lo16(*s)) == 0xF0)
                *s = (TxChar)cp->unicode[Lo8(Lo16(*s))]; // NOT mb compliant
        }
    } else if (enc == CODES_UTF8) {
        TxChar* s;
        TxChar* d;

        for (s = d = str; s < e;) {
            size_t l = 0;

            if (utf8_read_rune_from_unknown_plane(*d, l, s, e) == RECODE_OK) {
                d++, s += l;
            } else {
                *d++ = BROKEN_RUNE;
                ++s;
            }
        }
        e = d;
    } else if (enc == CODES_UNKNOWN) {
        for (TxChar* s = str; s < e; s++) {
            if (Hi8(Lo16(*s)) == 0xF0)
                *s = Lo8(Lo16(*s));
        }
    } else {
        Y_ASSERT(!SingleByteCodepage(enc));

        TxChar* s = str;
        TxChar* d = str;

        TVector<char> buf;

        size_t read = 0;
        size_t written = 0;
        for (; s < e; ++s) {
            if (Hi8(Lo16(*s)) == 0xF0) {
                buf.push_back(Lo8(Lo16(*s)));
            } else {
                if (!buf.empty()) {
                    if (RecodeToUnicode(enc, buf.data(), d, buf.size(), e - d, read, written) == RECODE_OK) {
                        Y_ASSERT(read == buf.size());
                        d += written;
                    } else { // just copying broken symbols
                        Y_ASSERT(buf.size() <= static_cast<size_t>(e - d));
                        Copy(buf.data(), buf.size(), d);
                        d += buf.size();
                    }
                    buf.clear();
                }
                *d++ = *s;
            }
        }
    }
    ee = e;
}

void DecodeUnknownPlane(wchar16* str, wchar16*& ee, const ECharset enc) {
    DoDecodeUnknownPlane(str, ee, enc);
}
void DecodeUnknownPlane(wchar32* str, wchar32*& ee, const ECharset enc) {
    DoDecodeUnknownPlane(str, ee, enc);
}
