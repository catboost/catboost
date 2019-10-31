#pragma once

#include <util/system/types.h>
#include <util/system/compiler.h>

namespace NFloat16Impl {
    constexpr ui16 AllBitsMask16 = ui16(-1);
    constexpr ui32 AllBitsMask32 = ui32(-1);
    constexpr size_t Float16ExponentBits = 5;
    constexpr size_t Float16MantisaBits = 10;
    constexpr size_t Float32ExponentBits = 8;
    constexpr size_t Float32MantisaBits = 23;
    constexpr i8 FLoat32ExponentOffset = 127;
    constexpr i8 FLoat16ExponentOffset = 15;

    constexpr ui16 ExponentAndSignFloat16Mask = (AllBitsMask16 >> Float16MantisaBits) << Float16MantisaBits;
    constexpr ui16 ExponentFloat16Mask = ExponentAndSignFloat16Mask & (ExponentAndSignFloat16Mask >> 1);
    constexpr ui32 ExponentAndSignFloat32Mask = (AllBitsMask32 >> Float32MantisaBits) << Float32MantisaBits;

    constexpr ui32 MantisaFloat32Mask = AllBitsMask32 ^ ExponentAndSignFloat32Mask;
    constexpr ui16 MantisaFloat16Mask = AllBitsMask16 ^ ExponentAndSignFloat16Mask;

    constexpr ui32 SignFloat32Mask = AllBitsMask32 ^ (AllBitsMask32 >> 1);
    constexpr ui16 SignFloat16Mask = AllBitsMask16 ^ (AllBitsMask16 >> 1);
    constexpr ui32 ExponentFloat32Mask = ExponentAndSignFloat32Mask ^ SignFloat32Mask;

    Y_CONST_FUNCTION
    ui16 ConvertFloat32IntoFloat16(float val);

    Y_CONST_FUNCTION
    ui32 ConvertFloat16IntoFloat32Bitly(ui16 f16);

    Y_CONST_FUNCTION
    float ConvertFloat16IntoFloat32(ui16 f16);
}

struct TFloat16 {
    using TStorageType = ui16;

    TStorageType Data = 0;

    constexpr TFloat16() {}

    TFloat16(float v) {
        Data = NFloat16Impl::ConvertFloat32IntoFloat16(v);
    }

    TFloat16(const TFloat16&) = default;

    TFloat16& operator=(const TFloat16&) = default;
    TFloat16& operator=(float v) {
        Data = NFloat16Impl::ConvertFloat32IntoFloat16(v);
        return *this;
    }

    Y_PURE_FUNCTION
    float AsFloat() const {
        return NFloat16Impl::ConvertFloat16IntoFloat32(Data);
    }

    Y_CONST_FUNCTION
    static TFloat16 Load(TStorageType d) {
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

    //NOTE: src must be 16 byte aligned, dst must be 32 byte alighned
    void UnpackFloat16SequenceAuto(const TFloat16* src, float* dst, size_t len);
    void UnpackFloat16SequenceIntrisincs(const TFloat16* src, float* dst, size_t len);

    //NOTE: f32 must be 32 byte aligned, f16 must be 16 byte alighned
    //NOTE: result depends on architecture and do not recomended for canonization
    Y_PURE_FUNCTION
    float DotProductOnFloatAuto(const float* f32, const TFloat16* f16, size_t len);

    Y_PURE_FUNCTION
    float DotProductOnFloatIntrisincs(const float* f32, const TFloat16* f16, size_t len);

    Y_CONST_FUNCTION
    bool IsIntrisincsAvailableOnHost();

    void PackFloat16SequenceAuto(const float* src, TFloat16* dst, size_t len);
    void PackFloat16SequenceIntrisincs(const float* src, TFloat16* dst, size_t len);
}
