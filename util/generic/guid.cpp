#include "guid.h"
#include "ylimits.h"
#include "string.h"

#include <util/string/ascii.h>
#include <util/string/builder.h>
#include <util/stream/format.h>
#include <util/system/unaligned_mem.h>
#include <util/random/easy.h>

namespace {
    inline void LowerCaseHex(TString& s) {
        for (auto&& c : s) {
            c = AsciiToLower(c);
        }
    }
} // namespace

TString TGUID::AsGuidString() const {
    TStringBuilder s;
    s.reserve(50);
    s << Hex(dw[0], 0) << '-' << Hex(dw[1], 0) << '-' << Hex(dw[2], 0) << '-' << Hex(dw[3], 0);
    LowerCaseHex(s);
    return std::move(s);
}

TString TGUID::AsUuidString() const {
    TStringBuilder s;
    s.reserve(50);
    s << Hex(dw[0], HF_FULL) << '-';
    s << Hex(static_cast<ui16>(dw[1] >> 16), HF_FULL) << '-' << Hex(static_cast<ui16>(dw[1]), HF_FULL) << '-';
    s << Hex(static_cast<ui16>(dw[2] >> 16), HF_FULL) << '-' << Hex(static_cast<ui16>(dw[2]), HF_FULL);
    s << Hex(dw[3], HF_FULL);
    LowerCaseHex(s);
    return std::move(s);
}

TGUID TGUID::Create() {
    TGUID result;
    CreateGuid(&result);
    return result;
}

void CreateGuid(TGUID* res) {
    ui64* dw = reinterpret_cast<ui64*>(res->dw);

    WriteUnaligned<ui64>(&dw[0], RandomNumber<ui64>());
    WriteUnaligned<ui64>(&dw[1], RandomNumber<ui64>());
}

TGUID TGUID::CreateTimebased() {
    TGUID result;
    // GUID_EPOCH_OFFSET is the number of 100-ns intervals between the
    // UUID epoch 1582-10-15 00:00:00 and the Unix epoch 1970-01-01 00:00:00.
    constexpr ui64 GUID_EPOCH_OFFSET = 0x01b21dd213814000;
    const ui64 timestamp = Now().NanoSeconds() / 100 + GUID_EPOCH_OFFSET;
    result.dw[0] = ui32(timestamp & 0xffffffff); // time low
    const ui32 timeMid = ui32((timestamp >> 32) & 0xffff);
    constexpr ui32 UUID_VERSION = 1;
    const ui32 timeHighAndVersion = ui16((timestamp >> 48) & 0x0fff) | (UUID_VERSION << 12);
    result.dw[1] = (timeMid << 16) | timeHighAndVersion;
    const ui32 clockSeq = RandomNumber<ui32>(0x3fff) | 0x8000;
    result.dw[2] = (clockSeq << 16) | RandomNumber<ui16>();
    result.dw[3] = RandomNumber<ui32>() | (1 << 24);
    return result;
}

TString GetGuidAsString(const TGUID& g) {
    return g.AsGuidString();
}

TString CreateGuidAsString() {
    return TGUID::Create().AsGuidString();
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

bool GetGuid(const TStringBuf s, TGUID& result) {
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
TGUID GetGuid(const TStringBuf s) {
    TGUID result;

    if (GetGuid(s, result)) {
        return result;
    }

    return TGUID();
}

bool GetUuid(const TStringBuf s, TGUID& result) {
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
TGUID GetUuid(const TStringBuf s) {
    TGUID result;

    if (GetUuid(s, result)) {
        return result;
    }

    return TGUID();
}
