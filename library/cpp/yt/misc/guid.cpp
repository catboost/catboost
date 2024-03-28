#include "guid.h"

#include <library/cpp/yt/exception/exception.h>

#include <util/random/random.h>

#include <util/string/printf.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

namespace {

const ui8 HexDigits1[16] = {
    0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66
};

const ui16 HexDigits2[256] = {
    0x3030, 0x3130, 0x3230, 0x3330, 0x3430, 0x3530, 0x3630, 0x3730, 0x3830, 0x3930, 0x6130, 0x6230, 0x6330, 0x6430, 0x6530, 0x6630,
    0x3031, 0x3131, 0x3231, 0x3331, 0x3431, 0x3531, 0x3631, 0x3731, 0x3831, 0x3931, 0x6131, 0x6231, 0x6331, 0x6431, 0x6531, 0x6631,
    0x3032, 0x3132, 0x3232, 0x3332, 0x3432, 0x3532, 0x3632, 0x3732, 0x3832, 0x3932, 0x6132, 0x6232, 0x6332, 0x6432, 0x6532, 0x6632,
    0x3033, 0x3133, 0x3233, 0x3333, 0x3433, 0x3533, 0x3633, 0x3733, 0x3833, 0x3933, 0x6133, 0x6233, 0x6333, 0x6433, 0x6533, 0x6633,
    0x3034, 0x3134, 0x3234, 0x3334, 0x3434, 0x3534, 0x3634, 0x3734, 0x3834, 0x3934, 0x6134, 0x6234, 0x6334, 0x6434, 0x6534, 0x6634,
    0x3035, 0x3135, 0x3235, 0x3335, 0x3435, 0x3535, 0x3635, 0x3735, 0x3835, 0x3935, 0x6135, 0x6235, 0x6335, 0x6435, 0x6535, 0x6635,
    0x3036, 0x3136, 0x3236, 0x3336, 0x3436, 0x3536, 0x3636, 0x3736, 0x3836, 0x3936, 0x6136, 0x6236, 0x6336, 0x6436, 0x6536, 0x6636,
    0x3037, 0x3137, 0x3237, 0x3337, 0x3437, 0x3537, 0x3637, 0x3737, 0x3837, 0x3937, 0x6137, 0x6237, 0x6337, 0x6437, 0x6537, 0x6637,
    0x3038, 0x3138, 0x3238, 0x3338, 0x3438, 0x3538, 0x3638, 0x3738, 0x3838, 0x3938, 0x6138, 0x6238, 0x6338, 0x6438, 0x6538, 0x6638,
    0x3039, 0x3139, 0x3239, 0x3339, 0x3439, 0x3539, 0x3639, 0x3739, 0x3839, 0x3939, 0x6139, 0x6239, 0x6339, 0x6439, 0x6539, 0x6639,
    0x3061, 0x3161, 0x3261, 0x3361, 0x3461, 0x3561, 0x3661, 0x3761, 0x3861, 0x3961, 0x6161, 0x6261, 0x6361, 0x6461, 0x6561, 0x6661,
    0x3062, 0x3162, 0x3262, 0x3362, 0x3462, 0x3562, 0x3662, 0x3762, 0x3862, 0x3962, 0x6162, 0x6262, 0x6362, 0x6462, 0x6562, 0x6662,
    0x3063, 0x3163, 0x3263, 0x3363, 0x3463, 0x3563, 0x3663, 0x3763, 0x3863, 0x3963, 0x6163, 0x6263, 0x6363, 0x6463, 0x6563, 0x6663,
    0x3064, 0x3164, 0x3264, 0x3364, 0x3464, 0x3564, 0x3664, 0x3764, 0x3864, 0x3964, 0x6164, 0x6264, 0x6364, 0x6464, 0x6564, 0x6664,
    0x3065, 0x3165, 0x3265, 0x3365, 0x3465, 0x3565, 0x3665, 0x3765, 0x3865, 0x3965, 0x6165, 0x6265, 0x6365, 0x6465, 0x6565, 0x6665,
    0x3066, 0x3166, 0x3266, 0x3366, 0x3466, 0x3566, 0x3666, 0x3766, 0x3866, 0x3966, 0x6166, 0x6266, 0x6366, 0x6466, 0x6566, 0x6666
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
            TString(str).c_str()));
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
            TString(str).c_str()));
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
    auto writeHex1 = [&] (ui8 x) {
        *ptr = HexDigits1[x];
        ptr += 1;
    };

    auto writeHex2 = [&] (ui8 x) {
        ::memcpy(ptr, &HexDigits2[x], 2);
        ptr += 2;
    };

    auto writeComponent = [&] (ui32 x) {
        /*  */ if (x >= 0x10000000) {
            writeHex2((x >> 24) & 0xff);
            writeHex2((x >> 16) & 0xff);
            writeHex2((x >>  8) & 0xff);
            writeHex2( x        & 0xff);
        } else if (x >= 0x1000000) {
            writeHex1( x >> 24);
            writeHex2((x >> 16) & 0xff);
            writeHex2((x >>  8) & 0xff);
            writeHex2( x        & 0xff);
        } else if (x >= 0x100000) {
            writeHex2((x >> 16) & 0xff);
            writeHex2((x >>  8) & 0xff);
            writeHex2( x        & 0xff);
        } else if (x >= 0x10000) {
            writeHex1( x >> 16);
            writeHex2((x >>  8) & 0xff);
            writeHex2( x        & 0xff);
        } else if (x >= 0x1000) {
            writeHex2( x >> 8);
            writeHex2( x        & 0xff);
        } else if (x >= 0x100) {
            writeHex1( x >> 8);
            writeHex2( x        & 0xff);
        } else if (x >= 0x10) {
            writeHex2( x);
        } else {
            writeHex1( x);
        }
    };

    auto writeDash = [&] () {
        *ptr++ = '-';
    };

    writeComponent(value.Parts32[3]);
    writeDash();
    writeComponent(value.Parts32[2]);
    writeDash();
    writeComponent(value.Parts32[1]);
    writeDash();
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
