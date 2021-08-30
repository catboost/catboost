#include "charfilter.h"
#include <library/cpp/unicode/normalization/normalization.h>
#include <util/charset/unidata.h>

namespace NUnicode {
    namespace NPrivate {
        const TDecompositionTable& LemmerDecomposition();
    }
}

static const wchar32* LemmerDecompositionInt(wchar32 ch, bool advancedGermanUmlauts, bool extTable) {
    static const wchar32 ae[] = {'a', 'e', 0};
    static const wchar32 oe[] = {'o', 'e', 0};
    static const wchar32 ue[] = {'u', 'e', 0};

    if (advancedGermanUmlauts) {
        switch (ch) {
            case 0x00E4: // ä
                return ae;
            case 0x00F6: // ö
                return oe;
            case 0x00FC: // ü
                return ue;
        }
    }

    if (extTable)
        return NUnicode::NPrivate::Decomposition(NUnicode::NPrivate::LemmerDecomposition(), ch);

    static const wchar32 I[] = {'I', 0};
    static const wchar32 i[] = {'i', 0};
    static const wchar32 ss[] = {'s', 's', 0};

    switch (ch) {
            //      case 0x040E:    // Ў
            //      case 0x045E:    // ў
        case 0x0419: // Й
        case 0x0439: // й
        case 0x0407: // Ї
        case 0x0457: // ї
            return nullptr;
        case 0x0130: // I with dot
            return I;
        case 0x0131: // dotless i
            return i;
        case 0x00DF: // ß
            return ss;
    }
    return NUnicode::Decomposition<true>(ch);
}

const wchar32* LemmerDecomposition(wchar32 ch, bool advancedGermanUmlauts, bool extTable) {
    const wchar32* dec = LemmerDecompositionInt(ch, advancedGermanUmlauts, extTable);
    if (dec && dec[0] == ch && dec[1] == 0)
        return nullptr;
    return dec;
}

static size_t CharSize(wchar32 c) {
    if (c <= 0xFFFF)
        return 1;
    return 2;
}

static void CheckAddChar(wchar16*& r, size_t& bufLen, wchar32 c) {
    if (IsCombining(c))
        return;
    c = ToLower(c);
    if (CharSize(c) > bufLen) {
        bufLen = 0;
        return;
    }
    size_t sz = WriteSymbol(c, r);
    bufLen -= sz;
}

bool IsDecomp(ui16 c, bool extTable) {
    const wchar32* decomp = LemmerDecompositionInt(c, false, extTable);
    return decomp != nullptr && (decomp[0] != c || decomp[1] != 0);
}

bool IsDecomp(ui16 c) {
    return IsDecomp(c, false) || IsDecomp(c, true);
}

const ui32 UI16_COUNT = 0x10000;

class TLower {
public:
    static const TLower DefaultTLower;

public:
    ui16 Lower[UI16_COUNT];

public:
    TLower() {
        for (ui32 i = 0; i < UI16_COUNT; i++) {
            if (IsW16SurrogateLead(i) || IsW16SurrogateTail(i) || IsDecomp(i) || IsCombining(i)) {
                Lower[i] = 0;
            } else {
                Lower[i] = ::ToLower(i);
            }
        }
    }

    inline ui16 ToLower(ui16 c) const noexcept {
        return Lower[c];
    }
};

const TLower TLower::DefaultTLower;

bool NormalizeUnicodeInt(const wchar16* word, size_t length, wchar16*& res, size_t bufLen, bool advancedGermanUmlauts, bool extTable) {
    const wchar16* end = word + length;
    while (word != end && bufLen > 0) {
        wchar16 lw = TLower::DefaultTLower.ToLower(*word);
        if (lw != 0) {
            *(res++) = lw;
            word++;
            bufLen--;
            continue;
        }
        wchar32 ch = ReadSymbolAndAdvance(word, end);
        const wchar32* decomp = LemmerDecompositionInt(ch, advancedGermanUmlauts, extTable);
        if (decomp != nullptr) {
            for (; *decomp != 0 && bufLen > 0; ++decomp)
                CheckAddChar(res, bufLen, *decomp);
        } else {
            CheckAddChar(res, bufLen, ch);
        }
    }
    return word >= end;
}

size_t NormalizeUnicode(const wchar16* word, size_t length, wchar16* converted, size_t bufLen, bool advancedGermanUmlauts, bool extTable) {
    wchar16* p = converted;
    NormalizeUnicodeInt(word, length, p, bufLen, advancedGermanUmlauts, extTable);
    return p - converted;
}

const ui32 MAX_DECOMPOSED_LEN = 18;

bool NormalizeUnicode(const TWtringBuf& wbuf, bool advancedGermanUmlauts, bool extTable, TUtf16String& ret, ui32 mult) {
    size_t buflen = wbuf.size() * mult + MAX_DECOMPOSED_LEN; // for 1 symbol with longest sequence
    ret.reserve(buflen);
    wchar16* p = ret.begin();
    wchar16* converted = p;
    bool ok = NormalizeUnicodeInt(wbuf.data(), wbuf.size(), p, buflen, advancedGermanUmlauts, extTable);
    if (!ok) {
#ifndef NDEBUG
        fprintf(stderr, "[WARNING]\tOut of buffer %zu %u\n", wbuf.size(), (unsigned int)mult);
#endif
        return false;
    }
    ret.ReserveAndResize(p - converted);
    return true;
}

TUtf16String NormalizeUnicode(const TWtringBuf& wbuf, bool advancedGermanUmlauts, bool extTable) {
    TUtf16String ret;
    if (NormalizeUnicode(wbuf, advancedGermanUmlauts, extTable, ret, 2)) // First try buffer with size twice of original, enough in most cases
        return ret;
    NormalizeUnicode(wbuf, advancedGermanUmlauts, extTable, ret, MAX_DECOMPOSED_LEN); // 18 is enough, because 1 source char can produce no more than 18
    return ret;
}

TUtf16String NormalizeUnicode(const TUtf16String& word, bool advancedGermanUmlauts, bool extTable) {
    return NormalizeUnicode(TWtringBuf(word), advancedGermanUmlauts, extTable);
}
