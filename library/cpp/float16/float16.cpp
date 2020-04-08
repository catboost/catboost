#include "float16.h"
#include <util/generic/cast.h>
#include <util/system/yassert.h>
#include <util/stream/output.h>

namespace {
    // Generates a float32 with a given exponent and other bits set to zero
    // We do this a lot in conversion functions, so a short name helps
    constexpr ui32 AsExp32(ui32 e) {
        return e << NFloat16Impl::Float32MantissaBits;
    }
}

ui16 NFloat16Impl::ConvertFloat32IntoFloat16Auto(float val) {
    if (AreConversionIntrinsicsAvailableOnHost()) {
        return ConvertFloat32IntoFloat16Intrinsics(val);
    }

    // This is basically float_to_half_fast3_rtne() from https://gist.github.com/rygorous/2156668
    // modified to match the NaN handling of `vcvtps2ph` on x86 and `fcvt` on ARM

    auto res32 = BitCast<ui32>(val);
    const auto sign = res32 & SignFloat32Mask;
    res32 ^= sign;

    ui16 res = 0;

    if (res32 >= AsExp32(Float32ExponentOffset + Float16ExponentOffset + 1)) {
        // `val` was an Inf or a NaN, set all exponent bits to one
        res = ExponentFloat16Mask;
        if (res32 > ExponentFloat32Mask) {
            // The NaN payload has to be truncated, and the "quiet NaN" bit has to be set
            // See Table 14-9 "Non-Numerical Behavior for VCVTPH2PS, VCVTPS2PH" in
            // "Intel® 64 and IA-32 Architectures Software Developer's Manual"
            res |= (res32 >> MantissaBitsDiff) & MantissaFloat16Mask;
            res |= QuietNanFloat16Mask;
        }
    } else {
        if (res32 < AsExp32(ExponentOffsetDiff + 1)) {
            // Align the bits of our mantissa that are representable in float16 at the very
            // bottom of the 32-bit number using regular floating point addition
            // (see the "Conversion" section in http://www.chrishecker.com/images/f/fb/Gdmfp.pdf)
            constexpr ui32 denormMagic = AsExp32(Float32ExponentOffset - 1);
            const auto afterMagic = BitCast<float>(res32) + BitCast<float>(denormMagic);
            res = static_cast<ui16>(BitCast<ui32>(afterMagic) - denormMagic);
        } else {
            const ui32 mantissaLastBit = (res32 >> MantissaBitsDiff) & 1;
            // Adjust the exponent to use the fp16 offset
            res32 += AsExp32(static_cast<ui32>(-ExponentOffsetDiff));
            // `val` is a normal float32 number which might not be exactly representable as float16,
            // so we need to round it correctly using standard round-to-nearest-even
            // In terms of Table 2.1 in "Handbook of Floating-Point Arithmetic" by Jean-Michel Muller et al.,
            // this line changes the result when round = 1, sticky = 1
            res32 += 0b1111'1111'1111;
            // This line changes the result when round = 0, sticky = 1 and the original mantissa is odd
            res32 += mantissaLastBit;
            res = static_cast<ui16>(res32 >> MantissaBitsDiff);
        }
    }

    return res | static_cast<ui16>(sign >> 16);
}

float NFloat16Impl::ConvertFloat16IntoFloat32Auto(ui16 f16) {
    if (AreConversionIntrinsicsAvailableOnHost()) {
        return ConvertFloat16IntoFloat32Intrinsics(f16);
    }

    // The easy case: all float16 numbers can be losslessly converted to float32
    // This is basically half_to_float() from https://gist.github.com/rygorous/2156668
    // The only major difference is that this version converts signaling NaNs to quiet NaNs
    // to match the behavior of `vcvtph2ps` on x86 and `fcvt` on ARM

    const auto sign = f16 & SignFloat16Mask;
    auto res = static_cast<ui32>(f16 ^ sign) << MantissaBitsDiff;

    ui32 unadjustedExponent = res & (ExponentFloat16Mask << MantissaBitsDiff);
    // For normal numbers, all we have to do is adjust the exponent to use the fp32 offset
    res += AsExp32(ExponentOffsetDiff);

    if (unadjustedExponent == (ExponentFloat16Mask << MantissaBitsDiff)) {
        // This was an Inf or a NaN, so we need another adjustment to set all exponent bits to one
        res += AsExp32(ExponentOffsetDiff);
        if (res > ExponentFloat32Mask) {
            // This was a NaN, so we need to explicitly set the "quiet NaN" bit
            res |= QuietNanFloat32Mask;
        }
    } else if (unadjustedExponent == 0) {
        // `f16` was either denormal or zero (i.e., a number of form 2**-14 * 0.xxxxxxxxxx)
        res += AsExp32(1);
        // `res` now represents 2**-14 * 1.xxxxxxxxxx0...000 as float32
        // Ergo, if we subtract 2**-14 * 1.00000000000...000 from it, the floating point machinery
        // will automatically take care of the final exponent adjustment to make the result normalized
        res = BitCast<ui32>(BitCast<float>(res) - BitCast<float>(AsExp32(ExponentOffsetDiff + 1)));
    }

    return BitCast<float>(res | (static_cast<ui32>(sign) << 16));
}

void NFloat16Ops::UnpackFloat16SequenceAuto(const TFloat16* src, float* dst, size_t len) {
    if (AreIntrinsicsAvailableOnHost()) {
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

    if (AreIntrinsicsAvailableOnHost()) {
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
    if (AreIntrinsicsAvailableOnHost()) {
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
