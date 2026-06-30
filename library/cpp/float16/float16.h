#pragma once

#include <util/system/types.h>
#include <util/system/compiler.h>

#include <limits>

namespace NFloat16Impl {
    constexpr ui16 AllBitsMask16 = ui16(-1);
    constexpr ui32 AllBitsMask32 = ui32(-1);
    constexpr size_t Float16MantissaBits = 10;
    constexpr size_t Float32MantissaBits = 23;
    constexpr size_t MantissaBitsDiff = Float32MantissaBits - Float16MantissaBits;
    constexpr i32 Float32ExponentOffset = 127;
    constexpr i32 Float16ExponentOffset = 15;
    constexpr i32 ExponentOffsetDiff = Float32ExponentOffset - Float16ExponentOffset;

    constexpr ui16 ExponentAndSignFloat16Mask = (AllBitsMask16 >> Float16MantissaBits) << Float16MantissaBits;
    constexpr ui16 ExponentFloat16Mask = ExponentAndSignFloat16Mask & (ExponentAndSignFloat16Mask >> 1);
    constexpr ui32 ExponentAndSignFloat32Mask = (AllBitsMask32 >> Float32MantissaBits) << Float32MantissaBits;

    constexpr ui16 MantissaFloat16Mask = AllBitsMask16 ^ ExponentAndSignFloat16Mask;

    constexpr ui32 SignFloat32Mask = AllBitsMask32 ^ (AllBitsMask32 >> 1);
    constexpr ui16 SignFloat16Mask = AllBitsMask16 ^ (AllBitsMask16 >> 1);
    constexpr ui32 ExponentFloat32Mask = ExponentAndSignFloat32Mask ^ SignFloat32Mask;

    constexpr ui16 QuietNanFloat16Mask = 1 << (Float16MantissaBits - 1);
    constexpr ui32 QuietNanFloat32Mask = 1 << (Float32MantissaBits - 1);

    Y_CONST_FUNCTION
    bool AreConversionIntrinsicsAvailableOnHost();

    Y_CONST_FUNCTION
    ui16 ConvertFloat32IntoFloat16Auto(float val);

    Y_CONST_FUNCTION
    float ConvertFloat16IntoFloat32Auto(ui16 f16);

    Y_CONST_FUNCTION
    ui16 ConvertFloat32IntoFloat16Intrinsics(float val);

    Y_CONST_FUNCTION
    float ConvertFloat16IntoFloat32Intrinsics(ui16 val);
}

struct TFloat16 {
    using TStorageType = ui16;

    TStorageType Data = 0;

    constexpr TFloat16() {}

    TFloat16(float v) {
        Data = NFloat16Impl::ConvertFloat32IntoFloat16Auto(v);
    }

    TFloat16(const TFloat16&) = default;

    TFloat16& operator=(const TFloat16&) = default;
    TFloat16& operator=(float v) {
        Data = NFloat16Impl::ConvertFloat32IntoFloat16Auto(v);
        return *this;
    }

    Y_PURE_FUNCTION
    float AsFloat() const {
        return NFloat16Impl::ConvertFloat16IntoFloat32Auto(Data);
    }

    Y_CONST_FUNCTION
    static constexpr TFloat16 Load(TStorageType d) {
        TFloat16 res;
        res.Data = d;
        return res;
    }

    Y_PURE_FUNCTION
    TStorageType Save() const {
        return Data;
    }

    Y_PURE_FUNCTION
    operator float() const {
        return AsFloat();
    }
};

Y_CONST_FUNCTION inline bool operator<(TFloat16 a, TFloat16 b) {
    return float(a) < float(b);
}

namespace NFloat16Ops {
    constexpr size_t Float32BufferAlignmentRequirementInBytes = 32;
    constexpr size_t Float16BufferAlignmentRequirementInBytes = 16;

    void UnpackFloat16SequenceAuto(const TFloat16* src, float* dst, size_t len);
    void UnpackFloat16SequenceIntrisincs(const TFloat16* src, float* dst, size_t len);

    //NOTE: f32 must be 32 byte aligned, f16 must be 16 byte aligned
    //NOTE: result depends on architecture and do not recomended for canonization
    Y_PURE_FUNCTION
    float DotProductOnFloatAuto(const float* f32, const TFloat16* f16, size_t len);

    Y_PURE_FUNCTION
    float DotProductOnFloatIntrisincs(const float* f32, const TFloat16* f16, size_t len);

    Y_CONST_FUNCTION
    bool AreIntrinsicsAvailableOnHost();

    void PackFloat16SequenceAuto(const float* src, TFloat16* dst, size_t len);
    void PackFloat16SequenceIntrisincs(const float* src, TFloat16* dst, size_t len);
}

namespace std {
    template <>
    class numeric_limits<TFloat16> {
    public:
        static constexpr bool is_specialized = true;
        static constexpr bool is_signed = true;
        static constexpr bool is_integer = false;
        static constexpr bool is_exact = false;
        static constexpr bool has_infinity = true;
        static constexpr bool has_quiet_NaN = true;
        static constexpr bool has_signaling_NaN = true;
        static constexpr std::float_denorm_style has_denorm = std::denorm_present;

        static constexpr TFloat16 min() noexcept {
            return TFloat16::Load(0b0'00001'0000000000);
        }

        static constexpr TFloat16 lowest() noexcept {
            return TFloat16::Load(0b1'11110'1111111111);
        }

        static constexpr TFloat16 denorm_min() noexcept {
            return TFloat16::Load(0b0'00000'0000000001);
        }

        static constexpr TFloat16 max() noexcept {
            return TFloat16::Load(0b0'11110'1111111111);
        }

        static constexpr TFloat16 quiet_NaN() noexcept {
            return TFloat16::Load(0b0'11111'1000000001);
        }

        static constexpr TFloat16 signaling_NaN() noexcept {
            return TFloat16::Load(0b0'11111'0000000001);
        }

        static constexpr TFloat16 infinity() noexcept {
            return TFloat16::Load(0b0'11111'0000000000);
        }
    };
}
