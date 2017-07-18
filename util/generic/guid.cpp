#include "guid.h"
#include "ylimits.h"
#include "reinterpretcast.h"
#include "string.h"

#include <util/string/ascii.h>
#include <util/stream/mem.h>
#include <util/stream/format.h>
#include <util/system/atomic.h>
#include <util/system/unaligned_mem.h>
#include <util/random/easy.h>

void CreateGuid(TGUID* res) {
    static TAtomic cnt = 0;

    WriteUnaligned(res->dw, RandomNumber<ui64>());

    res->dw[2] = Random();
    res->dw[3] = AtomicAdd(cnt, 1);
}

TString GetGuidAsString(const TGUID& g) {
    char buf[50];
    TMemoryOutput mo(buf, sizeof(buf));

    mo << Hex(g.dw[0], nullptr) << '-' << Hex(g.dw[1], nullptr) << '-' << Hex(g.dw[2], nullptr) << '-' << Hex(g.dw[3], nullptr);

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
        if ('0' <= c && c <= '9') {
            digit = c - '0';
        } else if ('a' <= c && c <= 'f') {
            digit = c - 'a' + 10;
        } else if ('A' <= c && c <= 'F') {
            digit = c - 'A' + 10;
        } else {
            return false; // non-hex character
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
