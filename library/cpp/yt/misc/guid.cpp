#include "guid.h"

#include <library/cpp/yt/exception/exception.h>

#include <util/random/random.h>

#include <util/string/printf.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

namespace {

const ui8 HexDigits[16] = {
    0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66
};

} // anonymous namespace

////////////////////////////////////////////////////////////////////////////////

TGuid TGuid::Create()
{
    return TGuid(RandomNumber<ui64>(), RandomNumber<ui64>());
}

TGuid TGuid::FromString(TStringBuf str)
{
    TGuid guid;
    if (!FromString(str, &guid)) {
        throw TSimpleException(Sprintf("Error parsing GUID \"%s\"",
            std::string(str).c_str()));
    }
    return guid;
}

bool TGuid::FromString(TStringBuf str, TGuid* result)
{
    size_t partId = 3;
    ui64 partValue = 0;
    bool isEmptyPart = true;

    for (size_t i = 0; i != str.size(); ++i) {
        const char c = str[i];

        if (c == '-') {
            if (isEmptyPart || partId == 0) { // x-y--z, -x-y-z or x-y-z-m-...
                return false;
            }
            result->Parts32[partId] = static_cast<ui32>(partValue);
            --partId;
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

    if (partId != 0 || isEmptyPart) { // x-y or x-y-z-
        return false;
    }
    result->Parts32[partId] = static_cast<ui32>(partValue);
    return true;
}

TGuid TGuid::FromStringHex32(TStringBuf str)
{
    TGuid guid;
    if (!FromStringHex32(str, &guid)) {
        throw TSimpleException(Sprintf("Error parsing Hex32 GUID \"%s\"",
            str.data()));
    }
    return guid;
}

bool TGuid::FromStringHex32(TStringBuf str, TGuid* result)
{
    if (str.size() != 32) {
        return false;
    }

    bool ok = true;
    int i = 0;
    auto parseChar = [&] {
        const char c = str[i++];
        ui32 digit = 0;
        if ('0' <= c && c <= '9') {
            digit = c - '0';
        } else if ('a' <= c && c <= 'f') {
            digit = c - 'a' + 10;
        } else if ('A' <= c && c <= 'F') {
            digit = c - 'A' + 10;
        } else {
            ok = false;
        }
        return digit;
    };

    for (size_t j = 0; j < 16; ++j) {
        result->ReversedParts8[15 - j] = parseChar() * 16 + parseChar();
    }

    return ok;
}

char* WriteGuidToBuffer(char* ptr, TGuid value)
{
    // Each 32-bit component is emitted as lowercase hex with leading zeros
    // stripped (so 1..8 digits). We derive the exact digit count from the
    // position of the highest set bit and fill the digits back-to-front; this
    // avoids the long branch cascade of the naive per-magnitude approach and
    // writes exactly as many bytes as the component requires.
    auto writeComponent = [&] (ui32 x) {
        int digits = x == 0 ? 1 : (35 - __builtin_clz(x)) >> 2;
        char* start = ptr;
        char* cursor = ptr + digits;
        ptr = cursor;
        do {
            *--cursor = HexDigits[x & 0xf];
            x >>= 4;
        } while (cursor != start);
    };

    writeComponent(value.Parts32[3]);
    *ptr++ = '-';
    writeComponent(value.Parts32[2]);
    *ptr++ = '-';
    writeComponent(value.Parts32[1]);
    *ptr++ = '-';
    writeComponent(value.Parts32[0]);

    return ptr;
}

////////////////////////////////////////////////////////////////////////////////

TFormattableGuid::TFormattableGuid(TGuid guid)
    : End_(WriteGuidToBuffer(Buffer_.data(), guid))
{ }

TStringBuf TFormattableGuid::ToStringBuf() const
{
    return {Buffer_.data(), End_};
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
