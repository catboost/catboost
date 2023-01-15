#include "charfilter.h"
#include <library/cpp/unicode/normalization/normalization.h>
#include <util/charset/unidata.h>

namespace {
    struct TRange {
        wchar16 First;
        wchar16 Last;
    };
}

TAccentTable::TAccentTable() {
    // values of yc_80 copied from "library/cpp/tokenizer/charclasses_16.rl"
    TRange ranges[] = {
        {0x0300, 0x0357}, {0x035D, 0x036F}, {0x0483, 0x0486}, {0x0488, 0x0489}, {0x0591, 0x05A1}, {0x05A3, 0x05B9}, {0x05BB, 0x05BD}, {0x05BF, 0x0000}, {0x05C1, 0x05C2}, {0x05C4, 0x0000}, {0x0610, 0x0615}, {0x064B, 0x0658}, {0x0670, 0x0000}, {0x06D6, 0x06DC}, {0x06DE, 0x06E4}, {0x06E7, 0x06E8}, {0x06EA, 0x06ED}, {0x0711, 0x0000}, {0x0730, 0x074A}, {0x07A6, 0x07B0}, {0x0901, 0x0903}, {0x093C, 0x0000}, {0x093E, 0x094D}, {0x0951, 0x0954}, {0x0962, 0x0963}, {0x0981, 0x0983}, {0x09BC, 0x0000}, {0x09BE, 0x09C4}, {0x09C7, 0x09C8}, {0x09CB, 0x09CD}, {0x09D7, 0x0000}, {0x09E2, 0x09E3}, {0x0A01, 0x0A03}, {0x0A3C, 0x0000}, {0x0A3E, 0x0A42}, {0x0A47, 0x0A48}, {0x0A4B, 0x0A4D}, {0x0A70, 0x0A71}, {0x0A81, 0x0A83}, {0x0ABC, 0x0000}, {0x0ABE, 0x0AC5}, {0x0AC7, 0x0AC9}, {0x0ACB, 0x0ACD}, {0x0AE2, 0x0AE3}, {0x0B01, 0x0B03}, {0x0B3C, 0x0000}, {0x0B3E, 0x0B43}, {0x0B47, 0x0B48}, {0x0B4B, 0x0B4D}, {0x0B56, 0x0B57}, {0x0B82, 0x0000}, {0x0BBE, 0x0BC2}, {0x0BC6, 0x0BC8}, {0x0BCA, 0x0BCD}, {0x0BD7, 0x0000}, {0x0C01, 0x0C03}, {0x0C3E, 0x0C44}, {0x0C46, 0x0C48}, {0x0C4A, 0x0C4D}, {0x0C55, 0x0C56}, {0x0C82, 0x0C83}, {0x0CBC, 0x0000}, {0x0CBE, 0x0CC4}, {0x0CC6, 0x0CC8}, {0x0CCA, 0x0CCD}, {0x0CD5, 0x0CD6}, {0x0D02, 0x0D03}, {0x0D3E, 0x0D43}, {0x0D46, 0x0D48}, {0x0D4A, 0x0D4D}, {0x0D57, 0x0000}, {0x0D82, 0x0D83}, {0x0DCA, 0x0000}, {0x0DCF, 0x0DD4}, {0x0DD6, 0x0000}, {0x0DD8, 0x0DDF}, {0x0DF2, 0x0DF3}, {0x0E31, 0x0000}, {0x0E34, 0x0E3A}, {0x0E47, 0x0E4E}, {0x0EB1, 0x0000}, {0x0EB4, 0x0EB9}, {0x0EBB, 0x0EBC}, {0x0EC8, 0x0ECD}, {0x0F18, 0x0F19}, {0x0F35, 0x0000}, {0x0F37, 0x0000}, {0x0F39, 0x0000}, {0x0F3E, 0x0F3F}, {0x0F71, 0x0F84}, {0x0F86, 0x0F87}, {0x0F90, 0x0F97}, {0x0F99, 0x0FBC}, {0x0FC6, 0x0000}, {0x102C, 0x1032}, {0x1036, 0x1039}, {0x1056, 0x1059}, {0x1712, 0x1714}, {0x1732, 0x1734}, {0x1752, 0x1753}, {0x1772, 0x1773}, {0x17B6, 0x17D3}, {0x17DD, 0x0000}, {0x180B, 0x180D}, {0x18A9, 0x0000}, {0x1920, 0x192B}, {0x1930, 0x193B}, {0x20D0, 0x20EA}, {0x302A, 0x302F}, {0x3099, 0x309A}, {0xFB1E, 0x0000}, {0xFE00, 0xFE0F}, {0xFE20, 0xFE23}};

    TRange* const e = ranges + Y_ARRAY_SIZE(ranges);

    // @todo remove this line for static Data
    memset(Data, 0, DATA_SIZE);

    for (TRange* r = ranges; r != e; ++r) {
        if (r->Last) {
            for (wchar16 c = r->First; c <= r->Last; ++c) {
                Y_ASSERT((int)c < DATA_SIZE);
                Data[c] = 1;
            }
        } else {
            Y_ASSERT((int)r->First < DATA_SIZE);
            Data[r->First] = 1;
        }
    }
}

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
        fprintf(stderr, "Out of buffer %zu %u\n", wbuf.size(), (unsigned int)mult);
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
