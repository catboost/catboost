#include "float16.h"
#include <util/system/yassert.h>
#include <util/stream/output.h>


static ui16 ConvertFloat32IntoFloat16Simple(float val) {
    using namespace NFloat16Impl;
    union {
        float fVal;
        ui32 bitView;
    } d;
    d.fVal = val;
    const ui32& bitView = d.bitView;

    ui32 sign = SignFloat32Mask & bitView;
    ui16 resSign = (sign >> (
            Float32ExponentBits + Float32MantisaBits - Float16ExponentBits - Float16MantisaBits
    ));

    if ( (bitView ^ sign) == 0) {//zero or minus-zero
        return 0;
    }

    ui8 exponent = i8((bitView ^ sign) >> Float32MantisaBits);

    ui32 mantisa = (bitView & MantisaFloat32Mask);
    ui16 represantableMantisa = mantisa >> (Float32MantisaBits - Float16MantisaBits);

    if (exponent == 0) {
        if (represantableMantisa == 0) {
            return resSign;
        } else {
            ui8 resExponent = i8(-(Float32MantisaBits - Float16MantisaBits)) + FLoat16ExponentOffset;
            return resSign | (resExponent << Float16MantisaBits) | represantableMantisa;
        }
    } else {
        i8 centeredExponent = exponent - FLoat32ExponentOffset;
        if (centeredExponent > FLoat16ExponentOffset) {
            return ExponentFloat16Mask | resSign;
        }
        if (centeredExponent < -FLoat16ExponentOffset + 1) {
            return resSign;
        }
        return resSign | ( (centeredExponent + FLoat16ExponentOffset) << Float16MantisaBits) | represantableMantisa;
    }
}

ui16 NFloat16Impl::ConvertFloat32IntoFloat16(float val) {
    ui16 res = ConvertFloat32IntoFloat16Simple(val);
    if (res == (1 << FLoat16ExponentOffset)) {
        res = 0;
    }
    return res;
}


ui32 NFloat16Impl::ConvertFloat16IntoFloat32Bitly(ui16 f16) {
    ui16 sign = f16 & SignFloat16Mask;
    ui32 resSign = ui32(sign) << (
        Float32ExponentBits + Float32MantisaBits - Float16ExponentBits - Float16MantisaBits
    );

    if ( (f16 ^ sign) == 0) {
        return resSign;
    }
    ui8 exponent = (f16 ^ sign) >> Float16MantisaBits;
    i8 centeredExponent = exponent - FLoat16ExponentOffset;
    ui32 mantisa = ui32(f16 & MantisaFloat16Mask) << (Float32MantisaBits - Float16MantisaBits);

    if (exponent == 0) {
        return resSign | mantisa;
    }
    if (centeredExponent == FLoat16ExponentOffset + 1) {
        return resSign | ExponentFloat32Mask; // +- infinity
    }

    return resSign | (ui32(centeredExponent + FLoat32ExponentOffset) << Float32MantisaBits) | mantisa;
};


float NFloat16Impl::ConvertFloat16IntoFloat32(ui16 f16) {
    union {
        float fVal;
        ui32 bitView;
    } res;
    res.bitView = ConvertFloat16IntoFloat32Bitly(f16);

    return res.fVal;
}


void NFloat16Ops::UnpackFloat16SequenceAuto(const TFloat16* src, float* dst, size_t len) {
    if (IsIntrisincsAvailableOnHost()) {
        UnpackFloat16SequenceIntrisincs(src, dst, len);
    } else {
        while(len > 0) {
            *dst = src->AsFloat();
            dst += 1;
            src += 1;
            len -= 1;
        }
    }
}

float NFloat16Ops::DotProductOnFloatAuto(const float* f32, const TFloat16* f16, size_t len) {
    Y_ASSERT(size_t(f16) % Float16BufferAlignmentRequirementInBytes == 0);
    Y_ASSERT(size_t(f32) % Float32BufferAlignmentRequirementInBytes == 0);

    if (IsIntrisincsAvailableOnHost()) {
        return DotProductOnFloatIntrisincs(f32, f16, len);
    }
    float res = 0;
    while(len > 0) {
        res += (*f32) * f16->AsFloat();
        f32 += 1;
        f16 += 1;
        len -= 1;
    }
    return res;
}

template <>
void Out<TFloat16>(IOutputStream& out, typename TTypeTraits<TFloat16>::TFuncParam value) {
    out << static_cast<float>(value);
}

void NFloat16Ops::PackFloat16SequenceAuto(const float* src, TFloat16* dst, size_t len) {
    if (IsIntrisincsAvailableOnHost()) {
        PackFloat16SequenceIntrisincs(src, dst, len);
    } else {
        while (len > 0) {
            *dst = *src;
            ++dst;
            ++src;
            --len;
        }
    }
}
