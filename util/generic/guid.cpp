#include "guid.h"
#include "ylimits.h"
#include "string.h"

#include <util/string/ascii.h>
#include <util/stream/mem.h>
#include <util/stream/format.h>
#include <util/system/unaligned_mem.h>
#include <util/random/easy.h>

void CreateGuid(TGUID* res) {
    ui64* dw = reinterpret_cast<ui64*>(res->dw);

    WriteUnaligned<ui64>(&dw[0], RandomNumber<ui64>());
    WriteUnaligned<ui64>(&dw[1], RandomNumber<ui64>());
}

TString GetGuidAsString(const TGUID& g) {
    char buf[50];
    TMemoryOutput mo(buf, sizeof(buf));

    mo << Hex(g.dw[0], 0) << '-' << Hex(g.dw[1], 0) << '-' << Hex(g.dw[2], 0) << '-' << Hex(g.dw[3], 0);

    char* e = mo.Buf();

    // TODO - implement LowerCaseHex
    for (char* b = buf; b != e; ++b) {
        *b = AsciiToLower(*b);
    }

    return TString(buf, e);
}

TString CreateGuidAsString() {
    TGUID guid;

    CreateGuid(&guid);

    return GetGuidAsString(guid);
}

static bool GetDigit(const char c, ui32& digit) {
    digit = 0;
    if ('0' <= c && c <= '9') {
        digit = c - '0';
    } else if ('a' <= c && c <= 'f') {
        digit = c - 'a' + 10;
    } else if ('A' <= c && c <= 'F') {
        digit = c - 'A' + 10;
    } else {
        return false; // non-hex character
    }
    return true;
}

bool GetGuid(const TString& s, TGUID& result) {
    size_t partId = 0;
    ui64 partValue = 0;
    bool isEmptyPart = true;

    for (size_t i = 0; i != s.size(); ++i) {
        const char c = s[i];

        if (c == '-') {
            if (isEmptyPart || partId == 3) { // x-y--z, -x-y-z or x-y-z-m-...
                return false;
            }
            result.dw[partId] = static_cast<ui32>(partValue);
            ++partId;
            partValue = 0;
            isEmptyPart = true;
            continue;
        }

        ui32 digit = 0;
        if (!GetDigit(c, digit)) {
            return false;
        }

        partValue = partValue * 16 + digit;
        isEmptyPart = false;

        // overflow check
        if (partValue > Max<ui32>()) {
            return false;
        }
    }

    if (partId != 3 || isEmptyPart) { // x-y or x-y-z-
        return false;
    }
    result.dw[partId] = static_cast<ui32>(partValue);
    return true;
}

// Parses GUID from s and checks that it's valid.
// In case of error returns TGUID().
TGUID GetGuid(const TString& s) {
    TGUID result;

    if (GetGuid(s, result)) {
        return result;
    }

    return TGUID();
}

bool GetUuid(const TString& s, TGUID& result) {
    if (s.size() != 36) {
        return false;
    }

    size_t partId = 0;
    ui64 partValue = 0;
    size_t digitCount = 0;

    for (size_t i = 0; i < s.size(); ++i) {
        const char c = s[i];

        if (c == '-') {
            if (i != 8 && i != 13 && i != 18 && i != 23) {
                return false;
            }
            continue;
        }

        ui32 digit = 0;
        if (!GetDigit(c, digit)) {
            return false;
        }

        partValue = partValue * 16 + digit;

        if (++digitCount == 8) {
            result.dw[partId++] = partValue;
            digitCount = 0;
        }
    }
    return true;
}

// Parses GUID from uuid and checks that it's valid.
// In case of error returns TGUID().
TGUID GetUuid(const TString& s) {
    TGUID result;

    if (GetUuid(s, result)) {
        return result;
    }

    return TGUID();
}
