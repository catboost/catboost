#include "float16.h"

#include <library/cpp/testing/unittest/registar.h>
#include <util/generic/bitops.h>
#include <util/generic/cast.h>
#include <util/stream/format.h>
#include <util/string/builder.h>
#include <util/generic/ylimits.h>

static ui32 ReintrepretFloat(float v) {
    return BitCast<ui32>(v);
}

using namespace NFloat16Impl;

Y_UNIT_TEST_SUITE(Conversations) {
    Y_UNIT_TEST(32To16) {
        UNIT_ASSERT_NO_DIFF(
            TStringBuilder{} << Bin(ConvertFloat32IntoFloat16Auto(0)),
            TStringBuilder{} << Bin(ui16(0))
        );
        UNIT_ASSERT_NO_DIFF(
            TStringBuilder{} << Bin(ConvertFloat32IntoFloat16Auto(1)),
            TStringBuilder{} << Bin(ui16(0b0011110000000000))
        );
        UNIT_ASSERT_NO_DIFF(
            TStringBuilder{} << Bin(ConvertFloat32IntoFloat16Auto(-2)),
            TStringBuilder{} <<  Bin(ui16(0b1100000000000000))
        );
        UNIT_ASSERT_NO_DIFF(
            TStringBuilder{} << Bin(ConvertFloat32IntoFloat16Auto(-2)),
            TStringBuilder{} <<  Bin(ui16(0b1100000000000000))
        );
        UNIT_ASSERT_NO_DIFF(
            TStringBuilder{} << Bin(ConvertFloat32IntoFloat16Auto(65504)),
            TStringBuilder{} <<  Bin(ui16(0b0111101111111111))
        );
        UNIT_ASSERT_NO_DIFF(
            TStringBuilder{} << Bin(ConvertFloat32IntoFloat16Auto(1e5)),
            TStringBuilder{} <<  Bin(ui16(0b0111110000000000))
        );
        UNIT_ASSERT_NO_DIFF(
            TStringBuilder{} << Bin(ConvertFloat32IntoFloat16Auto(-1e5)),
            TStringBuilder{} <<  Bin(ui16(0b1111110000000000))
        );
        UNIT_ASSERT_NO_DIFF(
            TStringBuilder{} << Bin(ConvertFloat32IntoFloat16Auto(0.5)),
            TStringBuilder{} <<  Bin(ui16(0b0011100000000000))
        );
        UNIT_ASSERT_NO_DIFF(
            TStringBuilder{} << Bin(ConvertFloat32IntoFloat16Auto(1e-100)),
            TStringBuilder{} <<  Bin(ui16(0))
        );
        UNIT_ASSERT_NO_DIFF(
            TStringBuilder{} << Bin(ConvertFloat32IntoFloat16Auto(-1e-100)),
            TStringBuilder{} <<  Bin(ui16(0b1000000000000000))
        );
    }

    Y_UNIT_TEST(16To32) {
        UNIT_ASSERT_NO_DIFF(
            TStringBuilder{} << Bin(ReintrepretFloat(0.f)),
            TStringBuilder{} << Bin(ReintrepretFloat(ConvertFloat16IntoFloat32Auto(0)))
        );
        UNIT_ASSERT_NO_DIFF(
            TStringBuilder{} << Bin(ReintrepretFloat(1.f)),
            TStringBuilder{} << Bin(ReintrepretFloat(ConvertFloat16IntoFloat32Auto(0b0011110000000000)))
        );
        UNIT_ASSERT_NO_DIFF(
            TStringBuilder{} << Bin(ReintrepretFloat(-2.f)),
            TStringBuilder{} << Bin(ReintrepretFloat(ConvertFloat16IntoFloat32Auto(0b1100000000000000)))
        );
        UNIT_ASSERT_NO_DIFF(
            TStringBuilder{} << Bin(ReintrepretFloat(65504)),
            TStringBuilder{} << Bin(ReintrepretFloat(ConvertFloat16IntoFloat32Auto(0b0111101111111111)))
        );
        UNIT_ASSERT_NO_DIFF(
            TStringBuilder{} << Bin(ReintrepretFloat(std::numeric_limits<float>::infinity())),
            TStringBuilder{} << Bin(ReintrepretFloat(ConvertFloat16IntoFloat32Auto(0b0111110000000000)))
        );
        UNIT_ASSERT_NO_DIFF(
            TStringBuilder{} << Bin(ReintrepretFloat(-std::numeric_limits<float>::infinity())),
            TStringBuilder{} << Bin(ReintrepretFloat(ConvertFloat16IntoFloat32Auto(0b1111110000000000)))
        );
        UNIT_ASSERT_NO_DIFF(
            TStringBuilder{} << Bin(ReintrepretFloat(0.5)),
            TStringBuilder{} << Bin(ReintrepretFloat(ConvertFloat16IntoFloat32Auto(0b0011100000000000)))
        );
    }

    Y_UNIT_TEST(AccuracyCheck) {
        const i64 size = 1e5;
        for(i64 i = -size; i <= size; ++i) {
            float val = i / float(size);
            TFloat16 f16(val);
            UNIT_ASSERT_DOUBLES_EQUAL(float(f16), val, 1e-3);
        }
    }

    Y_UNIT_TEST(Compare) {
        TFloat16 a(0.1);
        TFloat16 b(0.2);

        UNIT_ASSERT(a < b);
    }


    Y_UNIT_TEST(NegativeZero) {
        TFloat16 a(-0.f);
        TFloat16 b(-1.e-10f);

        UNIT_ASSERT_VALUES_EQUAL(a.Data, b.Data);
    }
}


Y_UNIT_TEST_SUITE(Intrisincs) {
    Y_UNIT_TEST(ConvertSequenceToFloat) {
        constexpr size_t len = 10;
        alignas(16) TFloat16 src[len];
        alignas(32) float dst[len];

        for(size_t i = 0; i < len; ++i) {
            src[i] = float(i);
            UNIT_ASSERT_VALUES_EQUAL(src[i].AsFloat(), float(i));
        }
        NFloat16Ops::UnpackFloat16SequenceAuto(src, dst, len);
        for(size_t i = 0; i < len; ++i) {
            UNIT_ASSERT_VALUES_EQUAL(dst[i], float(i));
        }
    }

    Y_UNIT_TEST(DotProduct16) {
        constexpr size_t len = 16;
        alignas(16) TFloat16 f16[len];
        alignas(32) float f32[len];

        float expected_f32 = 0;
        float expected_f32f16 = 0;

        for(size_t i = 1; i < len + 1; ++i) {
            size_t ind = i - 1;
            f16[ind] = float(1. / i);
            f32[ind] = float(1. / i);
            expected_f32 += f32[ind] * f32[ind];
            expected_f32f16 += f32[ind] * float(f16[ind]);
        }
        UNIT_ASSERT_DOUBLES_EQUAL(expected_f32f16, expected_f32, 1e-2);
        float res = NFloat16Ops::DotProductOnFloatAuto(f32, f16, len);
        UNIT_ASSERT_DOUBLES_EQUAL(expected_f32f16, res, 1e-2);
    }

    Y_UNIT_TEST(DotProduct7) {
        constexpr size_t len = 7;
        alignas(16) TFloat16 f16[len];
        alignas(32) float f32[len];

        float expected_f32 = 0;
        float expected_f32f16 = 0;

        for(size_t i = 1; i < len + 1; ++i) {
            size_t ind = i - 1;
            f16[ind] = float(1. / i);
            f32[ind] = float(1. / i);
            expected_f32 += f32[ind] * f32[ind];
            expected_f32f16 += f32[ind] * float(f16[ind]);
        }
        UNIT_ASSERT_DOUBLES_EQUAL(expected_f32f16, expected_f32, 1e-2);
        float res = NFloat16Ops::DotProductOnFloatAuto(f32, f16, len);
        UNIT_ASSERT_DOUBLES_EQUAL(expected_f32f16, res, 1e-2);
    }

    Y_UNIT_TEST(DotProduct24) {
        constexpr size_t len = 24;
        alignas(16) TFloat16 f16[len];
        alignas(32) float f32[len];

        float expected_f32 = 0;
        float expected_f32f16 = 0;

        for(size_t i = 1; i < len + 1; ++i) {
            size_t ind = i - 1;
            f16[ind] = float(1. / i);
            f32[ind] = float(1. / i);
            expected_f32 += f32[ind] * f32[ind];
            expected_f32f16 += f32[ind] * float(f16[ind]);
        }
        UNIT_ASSERT_DOUBLES_EQUAL(expected_f32f16, expected_f32, 1e-2);
        float res = NFloat16Ops::DotProductOnFloatAuto(f32, f16, len);
        UNIT_ASSERT_DOUBLES_EQUAL(expected_f32f16, res, 1e-2);
    }

    Y_UNIT_TEST(DotProduct32) {
        constexpr size_t len = 32;
        alignas(16) TFloat16 f16[len];
        alignas(32) float f32[len];

        float expected_f32 = 0;
        float expected_f32f16 = 0;

        for(size_t i = 1; i < len + 1; ++i) {
            size_t ind = i - 1;
            f16[ind] = float(1. / i);
            f32[ind] = float(1. / i);
            expected_f32 += f32[ind] * f32[ind];
            expected_f32f16 += f32[ind] * float(f16[ind]);
        }
        UNIT_ASSERT_DOUBLES_EQUAL(expected_f32f16, expected_f32, 1e-2);
        float res = NFloat16Ops::DotProductOnFloatAuto(f32, f16, len);
        UNIT_ASSERT_DOUBLES_EQUAL(expected_f32f16, res, 1e-2);
    }

    Y_UNIT_TEST(ConvertSequenceFromFloat) {
        constexpr size_t len = 10;
        alignas(32) float src[len];
        alignas(16) TFloat16 dst[len];

        for(size_t i = 0; i < len; ++i) {
            src[i] = i;
        }
        NFloat16Ops::PackFloat16SequenceAuto(src, dst, len);
        for(size_t i = 0; i < len; ++i) {
            UNIT_ASSERT_VALUES_EQUAL(dst[i], TFloat16(float(i)));
        }
    }

    Y_UNIT_TEST(MaxValues) {
        UNIT_ASSERT_VALUES_EQUAL(65504.0f, Max<TFloat16>());
        UNIT_ASSERT_VALUES_EQUAL(-65504.0f, -Max<TFloat16>());
    }

    Y_UNIT_TEST(StdNumericLimits) {
        UNIT_ASSERT_VALUES_EQUAL(0.00006103515625f, std::numeric_limits<TFloat16>::min());
        UNIT_ASSERT_VALUES_EQUAL(65504.0f, std::numeric_limits<TFloat16>::max());
        UNIT_ASSERT_VALUES_EQUAL(-65504.0f, std::numeric_limits<TFloat16>::lowest());
    }
}

