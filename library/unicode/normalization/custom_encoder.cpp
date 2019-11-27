#include "custom_encoder.h"
#include "normalization.h"

#include <util/string/cast.h>
#include <util/stream/output.h>

void TCustomEncoder::addToTable(wchar32 ucode, unsigned char code, const CodePage* target) {
    unsigned char plane = (unsigned char)(ucode >> 8);
    unsigned char pos = (unsigned char)(ucode & 255);
    if (Table[plane] == DefaultPlane) {
        Table[plane] = new char[256];
        memset(Table[plane], 0, 256 * sizeof(char));
    }

    if (Table[plane][pos] == 0) {
        Table[plane][pos] = code;
    } else {
        Y_ASSERT(target && *target->Names);
        if (static_cast<unsigned char>(Table[plane][pos]) > 127 && code) {
            Cerr << "WARNING: Only lower part of ASCII should have duplicate encodings "
                 << target->Names[0]
                 << " " << IntToString<16>(ucode)
                 << " " << IntToString<16>(code)
                 << " " << IntToString<16>(static_cast<unsigned char>(Table[plane][pos]))
                 << Endl;
        }
    }
}

bool isGoodDecomp(wchar32 rune, wchar32 decomp) {
    if (
        (NUnicode::NPrivate::CharInfo(rune) == NUnicode::NPrivate::CharInfo(decomp)) || (IsAlpha(rune) && IsAlpha(decomp)) || (IsNumeric(rune) && IsNumeric(decomp)) || (IsQuotation(rune) && IsQuotation(decomp)))
    {
        return true;
    }
    return false;
}

void TCustomEncoder::Create(const CodePage* target, bool extended) {
    Y_ASSERT(target);

    DefaultChar = (const char*)target->DefaultChar;

    DefaultPlane = new char[256];

    memset(DefaultPlane, 0, 256 * sizeof(char));
    for (size_t i = 0; i != 256; ++i)
        Table[i] = DefaultPlane;

    for (size_t i = 0; i != 256; ++i) {
        wchar32 ucode = target->unicode[i];
        if (ucode != BROKEN_RUNE) // always UNASSIGNED
            addToTable(ucode, (unsigned char)i, target);
    }

    if (!extended)
        return;

    for (wchar32 w = 1; w < 65535; w++) {
        if (Code(w) == 0) {
            wchar32 dw = w;
            while (IsComposed(dw) && Code(dw) == 0) {
                const wchar32* decomp_p = NUnicode::Decomposition<true>(dw);
                Y_ASSERT(decomp_p != nullptr);

                dw = decomp_p[0];
                if (TCharTraits<wchar32>::GetLength(decomp_p) > 1 && (dw == (wchar32)' ' || dw == (wchar32)'('))
                    dw = decomp_p[1];
            }
            if (Code(dw) != 0 && isGoodDecomp(w, dw))
                addToTable(w, Code(dw), target);
        }
    }
}

TCustomEncoder::~TCustomEncoder() {
    for (size_t i = 0; i != 256; ++i) {
        if (Table[i] != DefaultPlane) {
            delete[] Table[i];
        }
    }
    delete[] DefaultPlane;
}
