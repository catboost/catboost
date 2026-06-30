/*
  Unittests for all SSE instrinsics translated to NEON instrinsics or
  software implementation.
  Should be tested both on Intel and ARM64.
 */
/* Author: Vitaliy Manushkin <agri@yandex-team.ru */

#include <library/cpp/testing/unittest/registar.h>

#include <util/generic/typetraits.h>
#include <util/string/hex.h>
#include <util/random/fast.h>
#include <util/stream/output.h>

#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>

template <typename TResult, typename TFunc, TFunc* func>
struct T_mm_CallWrapper {
    TResult Value;

    template <typename... TParams>
    T_mm_CallWrapper(TParams&&... params) {
        Value = func(std::forward<TParams>(params)...);
    }

    operator TResult&() {
        return Value;
    }

    operator const TResult&() const {
        return Value;
    }
};

#if defined(_arm64_)
#include "library/cpp/sse/sse2neon.h"
#elif defined(_i386_) || defined(_x86_64_)
#include <xmmintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>
#elif defined(_ppc64_)
#include "library/cpp/sse/powerpc.h"
#else
#error "Unsupported platform"
#endif

#if defined(_arm64_)
#define Wrap(T_mm_func) T_mm_func
#define WrapF(T_mm_func) T_mm_func
#define WrapD(T_mm_func) T_mm_func
#elif defined(_ppc64_) || defined(_i386_) || defined(_x86_64_)
#define Wrap(_mm_func) \
    T_mm_CallWrapper<__m128i, decltype(_mm_func), _mm_func>
#define WrapF(_mm_func) \
    T_mm_CallWrapper<__m128, decltype(_mm_func), _mm_func>
#define WrapD(_mm_func) \
    T_mm_CallWrapper<__m128d, decltype(_mm_func), _mm_func>
using int8x16_t = std::array<i8, 16>;
using int16x8_t = std::array<i16, 8>;
using int32x4_t = std::array<i32, 4>;
using int64x2_t = std::array<i64, 2>;
using uint8x16_t = std::array<ui8, 16>;
using uint16x8_t = std::array<ui16, 8>;
using uint32x4_t = std::array<ui32, 4>;
using uint64x2_t = std::array<ui64, 2>;
using float32x4_t = std::array<float, 4>;
using float64x2_t = std::array<double, 2>;

template <typename TVectorType>
struct TQType {
    static TVectorType As(__m128i param) {
        TVectorType value;
        _mm_storeu_si128((__m128i*)&value, param);
        return value;
    }
    static TVectorType As(__m128 param) {
        TVectorType value;
        _mm_storeu_ps((float*)&value, param);
        return value;
    }
    static TVectorType As(__m128d param) {
        TVectorType value;
        _mm_storeu_pd((double*)&value, param);
        return value;
    }
};
#endif

template <typename TVectorType>
struct TFuncLoad;
template <typename TVectorType>
struct TFuncStore;

template <>
struct TFuncLoad<__m128i> {
    __m128i Value;

    template <typename TPointer>
    TFuncLoad(TPointer* ptr) {
        Value = _mm_loadu_si128((__m128i*)ptr);
    }

    operator __m128i&() {
        return Value;
    }

    operator const __m128i&() const {
        return Value;
    }
};

template <>
struct TFuncLoad<__m128> {
    __m128 Value;

    template <typename TPointer>
    TFuncLoad(TPointer* ptr) {
        Value = _mm_loadu_ps((float*)ptr);
    }

    operator __m128&() {
        return Value;
    }

    operator const __m128&() const {
        return Value;
    }
};

template <>
struct TFuncLoad<__m128d> {
    __m128d Value;

    template <typename TPointer>
    TFuncLoad(TPointer* ptr) {
        Value = _mm_loadu_pd((double*)ptr);
    }

    operator __m128d&() {
        return Value;
    }

    operator const __m128d&() const {
        return Value;
    }
};

template <>
struct TFuncStore<__m128i> {
    template <typename TPointer>
    TFuncStore(TPointer* ptr, __m128i Value) {
        _mm_storeu_si128((__m128i*)ptr, Value);
    }
};

template <>
struct TFuncStore<__m128> {
    template <typename TPointer>
    TFuncStore(TPointer* ptr, __m128 Value) {
        _mm_storeu_ps((float*)ptr, Value);
    }
};

class TSSEEmulTest: public TTestBase {
private:
    UNIT_TEST_SUITE(TSSEEmulTest);
    UNIT_TEST(Test_mm_load_si128);
    UNIT_TEST(Test_mm_loadu_si128);
    UNIT_TEST(Test_mm_storeu_si128);
    UNIT_TEST(Test_mm_loadu_si128_2);
    UNIT_TEST(Test_mm_loadu_ps);
    UNIT_TEST(Test_mm_storeu_ps);

    UNIT_TEST(Test_mm_slli_epi16);
    UNIT_TEST(Test_mm_slli_epi32);
    UNIT_TEST(Test_mm_slli_epi64);
    UNIT_TEST(Test_mm_slli_si128);

    UNIT_TEST(Test_mm_srli_epi16);
    UNIT_TEST(Test_mm_srli_epi32);
    UNIT_TEST(Test_mm_srli_epi64);
    UNIT_TEST(Test_mm_srli_si128);

    UNIT_TEST(Test_mm_srai_epi16);
    UNIT_TEST(Test_mm_srai_epi32);

    UNIT_TEST(Test_mm_sll_epi16);
    UNIT_TEST(Test_mm_sll_epi32);
    UNIT_TEST(Test_mm_sll_epi64);

    UNIT_TEST(Test_mm_srl_epi16);
    UNIT_TEST(Test_mm_srl_epi32);
    UNIT_TEST(Test_mm_srl_epi64);

    UNIT_TEST(Test_mm_add_epi16);
    UNIT_TEST(Test_mm_add_epi32);
    UNIT_TEST(Test_mm_add_epi64);
    UNIT_TEST(Test_mm_add_ps);
    UNIT_TEST(Test_mm_add_pd);

    UNIT_TEST(Test_mm_madd_epi16);

    UNIT_TEST(Test_mm_sub_epi16);
    UNIT_TEST(Test_mm_sub_epi32);
    UNIT_TEST(Test_mm_sub_epi64);
    UNIT_TEST(Test_mm_sub_ps);
    UNIT_TEST(Test_mm_sub_pd);

    UNIT_TEST(Test_mm_mul_ps);
    UNIT_TEST(Test_mm_mul_pd);
    UNIT_TEST(Test_mm_div_ps);
    UNIT_TEST(Test_mm_div_pd);
    UNIT_TEST(Test_mm_max_ps);
    UNIT_TEST(Test_mm_min_ps);
    UNIT_TEST(Test_mm_and_ps);

    UNIT_TEST(Test_mm_unpacklo_epi8);
    UNIT_TEST(Test_mm_unpackhi_epi8);
    UNIT_TEST(Test_mm_unpacklo_epi16);
    UNIT_TEST(Test_mm_unpackhi_epi16);
    UNIT_TEST(Test_mm_unpacklo_epi32);
    UNIT_TEST(Test_mm_unpackhi_epi32);
    UNIT_TEST(Test_mm_unpacklo_epi64);
    UNIT_TEST(Test_mm_unpackhi_epi64);

    UNIT_TEST(Test_mm_or_si128);
    UNIT_TEST(Test_mm_and_si128);
    UNIT_TEST(Test_mm_andnot_si128);

    UNIT_TEST(Test_mm_cmpeq_epi8);
    UNIT_TEST(Test_mm_cmpeq_epi16);
    UNIT_TEST(Test_mm_cmpeq_epi32);
    UNIT_TEST(Test_mm_cmpeq_ps);

    UNIT_TEST(Test_mm_cmpgt_epi8);
    UNIT_TEST(Test_mm_cmpgt_epi16);
    UNIT_TEST(Test_mm_cmpgt_epi32);
    UNIT_TEST(Test_mm_cmpgt_ps);

    UNIT_TEST(Test_mm_cmplt_epi8);
    UNIT_TEST(Test_mm_cmplt_epi16);
    UNIT_TEST(Test_mm_cmplt_epi32);

    UNIT_TEST(Test_mm_set1_epi8);
    UNIT_TEST(Test_mm_set1_epi16);
    UNIT_TEST(Test_mm_set1_epi32);
    UNIT_TEST(Test_mm_set1_ps);
    UNIT_TEST(Test_mm_set_ps1);

    UNIT_TEST(Test_mm_setzero_si128);
    UNIT_TEST(Test_mm_setzero_ps);
    UNIT_TEST(Test_mm_setzero_pd);

    UNIT_TEST(Test_mm_storel_epi64);
    UNIT_TEST(Test_mm_loadl_epi64);

    UNIT_TEST(Test_mm_loadl_pd);
    UNIT_TEST(Test_mm_loadh_pd);
    UNIT_TEST(Test_mm_cvtsd_f64);

    UNIT_TEST(Test_mm_shuffle_epi32);
    UNIT_TEST(Test_mm_movemask_epi8);
    UNIT_TEST(Test_mm_cvtsi128_si32);
    UNIT_TEST(Test_mm_cvtsi128_si64);

    UNIT_TEST(Test_mm_set_epi16);
    UNIT_TEST(Test_mm_set_epi32);
    UNIT_TEST(Test_mm_set_ps);
    UNIT_TEST(Test_mm_set_pd);

    UNIT_TEST(Test_mm_cvtsi32_si128);
    UNIT_TEST(Test_mm_cvtsi64_si128);

    UNIT_TEST(Test_mm_packs_epi16);
    UNIT_TEST(Test_mm_packs_epi32);
    UNIT_TEST(Test_mm_packus_epi16);

    UNIT_TEST(Test_mm_extract_epi16);
    UNIT_TEST(Test_mm_extract_epi8);
    UNIT_TEST(Test_mm_extract_epi32);
    UNIT_TEST(Test_mm_extract_epi64);

    UNIT_TEST(Test_MM_TRANSPOSE4_PS);
    UNIT_TEST(Test_mm_movemask_ps);
    UNIT_TEST(Test_mm_movemask_ps_2);

    UNIT_TEST(Test_mm_cvtepi32_ps);
    UNIT_TEST(Test_mm_cvtps_epi32);
    UNIT_TEST(Test_mm_cvttps_epi32);

    UNIT_TEST(Test_mm_castsi128_ps);
    UNIT_TEST(Test_mm_castps_si128);

    UNIT_TEST(Test_mm_mul_epu32);

    UNIT_TEST(Test_mm_cmpunord_ps);
    UNIT_TEST(Test_mm_andnot_ps);
    UNIT_TEST(Test_mm_shuffle_ps);
    UNIT_TEST(Test_mm_shuffle_pd);
    UNIT_TEST(Test_mm_or_ps);
    UNIT_TEST(Test_mm_store_ss);
    UNIT_TEST(Test_mm_store_ps);
    UNIT_TEST(Test_mm_storeu_pd);
    UNIT_TEST(Test_mm_loadu_pd);
    UNIT_TEST(Test_mm_rsqrt_ps);
    UNIT_TEST(Test_matrixnet_powerpc);

    UNIT_TEST_SUITE_END();

public:
    void Test_mm_load_si128();
    void Test_mm_loadu_si128();
    void Test_mm_storeu_si128();
    void Test_mm_loadu_si128_2();
    void Test_mm_loadu_ps();
    void Test_mm_storeu_ps();

    template <typename TElem, int bits, int elemCount,
              typename TFunc, typename TShifter, typename TOp, typename TElemFunc>
    void Test_mm_shifter_epiXX();

    enum class EDirection {
        Left,
        Right
    };

    struct TShiftRes {
        __m128i Value[17];
    };

    void Test_mm_byte_shifter(EDirection direction, std::function<TShiftRes (__m128i)> foo);

    void Test_mm_slli_epi16();
    void Test_mm_slli_epi32();
    void Test_mm_slli_epi64();
    void Test_mm_slli_si128();

    void Test_mm_srli_epi16();
    void Test_mm_srli_epi32();
    void Test_mm_srli_epi64();
    void Test_mm_srli_si128();

    void Test_mm_srai_epi16();
    void Test_mm_srai_epi32();

    void Test_mm_sll_epi16();
    void Test_mm_sll_epi32();
    void Test_mm_sll_epi64();

    void Test_mm_srl_epi16();
    void Test_mm_srl_epi32();
    void Test_mm_srl_epi64();

    void Test_mm_add_epi8();
    void Test_mm_add_epi16();
    void Test_mm_add_epi32();
    void Test_mm_add_epi64();
    void Test_mm_add_ps();
    void Test_mm_add_pd();

    void Test_mm_madd_epi16();

    void Test_mm_sub_epi8();
    void Test_mm_sub_epi16();
    void Test_mm_sub_epi32();
    void Test_mm_sub_epi64();
    void Test_mm_sub_ps();
    void Test_mm_sub_pd();

    void Test_mm_mul_ps();
    void Test_mm_mul_pd();
    void Test_mm_div_ps();
    void Test_mm_div_pd();
    void Test_mm_max_ps();
    void Test_mm_min_ps();
    void Test_mm_and_ps();

    template <typename TElem, int bits, int elemCount, int shift,
              typename TFunc, typename TOp>
    void Test_mm_unpack_epiXX();
    void Test_mm_unpacklo_epi8();
    void Test_mm_unpackhi_epi8();
    void Test_mm_unpacklo_epi16();
    void Test_mm_unpackhi_epi16();
    void Test_mm_unpacklo_epi32();
    void Test_mm_unpackhi_epi32();
    void Test_mm_unpacklo_epi64();
    void Test_mm_unpackhi_epi64();

    template <typename TElem, unsigned elemCount,
              typename TFunc, typename TElemFunc,
              typename TOp, typename TVectorType = __m128i>
    void Test_mm_dualop();

    template <typename TElem, unsigned elemCount,
              typename TFunc, typename TElemFunc,
              typename TOp, typename TVectorType = __m128i>
    void Test_mm_dualcmp();

    void Test_mm_or_si128();
    void Test_mm_and_si128();
    void Test_mm_andnot_si128();

    void Test_mm_cmpeq_epi8();
    void Test_mm_cmpeq_epi16();
    void Test_mm_cmpeq_epi32();
    void Test_mm_cmpeq_ps();

    void Test_mm_cmpgt_epi8();
    void Test_mm_cmpgt_epi16();
    void Test_mm_cmpgt_epi32();
    void Test_mm_cmpgt_ps();

    void Test_mm_cmplt_epi8();
    void Test_mm_cmplt_epi16();
    void Test_mm_cmplt_epi32();

    template <typename TElem, int elemCount,
              typename TFunc, typename TOp, typename TVectorType>
    void Test_mm_setter_epiXX();
    void Test_mm_set1_epi8();
    void Test_mm_set1_epi16();
    void Test_mm_set1_epi32();
    void Test_mm_set1_ps();
    void Test_mm_set_ps1();

    void Test_mm_setzero_si128();
    void Test_mm_setzero_ps();
    void Test_mm_setzero_pd();

    void Test_mm_loadl_epi64();
    void Test_mm_storel_epi64();

    void Test_mm_loadl_pd();
    void Test_mm_loadh_pd();
    void Test_mm_cvtsd_f64();

    void Test_mm_shuffle_epi32();
    void Test_mm_movemask_epi8();
    void Test_mm_cvtsi128_si32();
    void Test_mm_cvtsi128_si64();

    void Test_mm_set_epi16();
    void Test_mm_set_epi32();
    void Test_mm_set_ps();
    void Test_mm_set_pd();

    void Test_mm_cvtsi32_si128();
    void Test_mm_cvtsi64_si128();

    template <typename TElem, typename TNarrow, unsigned elemCount,
              typename TFunc>
    void Test_mm_packs_epiXX();
    void Test_mm_packs_epi16();
    void Test_mm_packs_epi32();
    void Test_mm_packus_epi16();

    void Test_mm_extract_epi16();
    void Test_mm_extract_epi8();
    void Test_mm_extract_epi32();
    void Test_mm_extract_epi64();

    void Test_MM_TRANSPOSE4_PS();
    void Test_mm_movemask_ps();
    void Test_mm_movemask_ps_2();

    template <typename TFrom, typename TTo, unsigned elemCount,
              typename TLoadVector, typename TResultVector,
              typename TElemFunc, typename TFunc, typename TOp>
    void Test_mm_convertop();
    void Test_mm_cvtepi32_ps();
    void Test_mm_cvtps_epi32();
    void Test_mm_cvttps_epi32();

    template <typename TLoadVector, typename TCastVector,
              typename TFunc, TFunc* func>
    void Test_mm_castXX();
    void Test_mm_castsi128_ps();
    void Test_mm_castps_si128();

    void Test_mm_mul_epu32();

    void Test_mm_cmpunord_ps();
    void Test_mm_store_ss();
    void Test_mm_store_ps();
    void Test_mm_storeu_pd();
    void Test_mm_andnot_ps();
    void Test_mm_shuffle_ps();
    void Test_mm_shuffle_pd();
    void Test_mm_or_ps();
    void Test_mm_loadu_pd();
    void Test_mm_rsqrt_ps();
    void Test_mm_rsqrt_ss();
    void Test_matrixnet_powerpc();
};

UNIT_TEST_SUITE_REGISTRATION(TSSEEmulTest);

void TSSEEmulTest::Test_mm_load_si128() {
    alignas(16) char data[16] = {
        '\xAA', '\x00', '\xFF', '\xCC', '\x11', '\x22', '\xBB', '\xAA',
        '\x33', '\x99', '\x44', '\x88', '\x55', '\x77', '\x66', '\x1C'};
    __m128i value = _mm_load_si128((__m128i*)&data);
    UNIT_ASSERT_EQUAL(TQType<uint64x2_t>::As(value)[0], 0xAABB2211CCFF00AAUL);
    UNIT_ASSERT_EQUAL(TQType<uint64x2_t>::As(value)[1], 0x1C66775588449933UL);
}

void TSSEEmulTest::Test_mm_loadu_si128() {
    alignas(16) char data[17] = {
        '\x66',
        '\xAA', '\x00', '\xFF', '\xCC', '\x11', '\x22', '\xBB', '\xAA',
        '\x33', '\x99', '\x44', '\x88', '\x55', '\x77', '\x66', '\x1C'};
    UNIT_ASSERT((ui64(&data[1]) & 0x1) == 0x1);
    __m128i value = _mm_loadu_si128((__m128i*)&data[1]);
    UNIT_ASSERT(TQType<uint64x2_t>::As(value)[0] == 0xAABB2211CCFF00AAUL);
    UNIT_ASSERT(TQType<uint64x2_t>::As(value)[1] == 0x1C66775588449933UL);
}

void TSSEEmulTest::Test_mm_storeu_si128() {
    alignas(16) unsigned char stub[32] = {
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07
    };

    alignas(16) unsigned char value[16] = {
        0xc0, 0xc1, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7,
        0xc8, 0xc9, 0xca, 0xcb, 0xcc, 0xcd, 0xce, 0xcf
    };

    const __m128i val = _mm_loadu_si128((__m128i*)&value[0]);

    for (size_t shift = 0; shift != 17; ++shift) {
        alignas(16) unsigned char res[sizeof(stub)];
        memcpy(res, stub, sizeof(res));

        _mm_storeu_si128((__m128i*)&res[shift], val);


        alignas(16) unsigned char etalon[sizeof(stub)];
        memcpy(etalon, stub, sizeof(etalon));
        for (size_t i = 0; i != sizeof(value); ++i) {
            etalon[shift + i] = value[i];
        }

        for (size_t i = 0; i != sizeof(etalon) / sizeof(etalon[0]); ++i) {
            UNIT_ASSERT_EQUAL_C(res[i], etalon[i], "res: " << HexEncode(res, 32) << " vs etalon: " << HexEncode(etalon, 32));
        }
    }

}


void TSSEEmulTest::Test_mm_loadu_si128_2() {
    alignas(16) unsigned char stub[32] = {
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
        0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
        0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
        0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f
    };

    for (size_t shift = 0; shift != 17; ++shift) {
        const __m128i val = _mm_loadu_si128((const __m128i*)&stub[shift]);
        alignas(16) unsigned char res[16];
        _mm_store_si128((__m128i*)res, val);

        for (size_t i = 0; i != 16; ++i) {
            UNIT_ASSERT_EQUAL_C(res[i], stub[i + shift], "res: " << HexEncode(res, 16) << " vs etalon: " << HexEncode(&stub[shift], 16));
        }
    }
}


void TSSEEmulTest::Test_mm_loadu_ps() {
    alignas(16) float stub[8] = {
        0.f, 1.f, 2.f, 3.f,
        4.f, 5.f, 6.f, 7.f
    };

    for (size_t shift = 0; shift != 5; ++shift) {
        const __m128 val = _mm_loadu_ps(&stub[shift]);
        alignas(16) float res[4];
        _mm_store_ps(res, val);

        for (size_t i = 0; i != 4; ++i) {
            UNIT_ASSERT_EQUAL_C(res[i], stub[shift + i], "res: " << HexEncode(res, 16) << " vs etalon: " << HexEncode(&stub[shift], 16));
        }
    }
}


void TSSEEmulTest::Test_mm_storeu_ps() {
    alignas(16) float stub[8] = {
        0.f, 1.f, 2.f, 3.f,
        4.f, 5.f, 6.f, 7.f
    };

    alignas(16) float value[4] = {
        100.f, 101.f, 102.f, 103.f
    };
    const __m128 val = _mm_load_ps(value);

    for (size_t shift = 0; shift != 5; ++shift) {
        alignas(16) float res[sizeof(stub) / sizeof(stub[0])];
        memcpy(res, stub, sizeof(stub));

        _mm_storeu_ps(&res[shift], val);

        float etalon[sizeof(stub) / sizeof(stub[0])];
        memcpy(etalon, stub, sizeof(stub));
        for (size_t i = 0; i != 4; ++i) {
            etalon[i + shift] = value[i];
        }

        for (size_t i = 0; i != sizeof(stub) / sizeof(stub[0]); ++i) {
            UNIT_ASSERT_EQUAL_C(res[i], etalon[i], "res: " << HexEncode(res, sizeof(res)) << " vs etalon: " << HexEncode(etalon, sizeof(etalon)));
        }
    }
}

template<typename C>
C MakeNumber(unsigned number);

template<>
__m128i MakeNumber<__m128i>(unsigned number) {
    char data[16] = {0};
    memcpy(data, &number, sizeof(number));

    return _mm_loadu_si128((__m128i*)data);
}

template<>
unsigned MakeNumber<unsigned>(unsigned number) {
    return number;
}

template <typename TElem, int bits, int elemCount,
          typename TFunc, typename TShifter, typename TOp, typename TElemFunc>
void TSSEEmulTest::Test_mm_shifter_epiXX() {
    char data[16] = {
        '\xAA', '\x00', '\xFF', '\xCC', '\x11', '\x22', '\xBB', '\xAA',
        '\x33', '\x99', '\x44', '\x88', '\x55', '\x77', '\x66', '\x1C'};
    TElem* dataw = reinterpret_cast<TElem*>(&data);

    __m128i value = _mm_loadu_si128((__m128i*)&data);

    for (unsigned shifter = 0; shifter <= bits; ++shifter) {
        TElem shiftedData[elemCount];
        for (unsigned i = 0; i < elemCount; ++i) {
            shiftedData[i] = TElemFunc::Call(dataw[i], shifter);
        }

        const TShifter adhoc_shifter = MakeNumber<TShifter>(shifter);

        __m128i result = TFunc(value, adhoc_shifter);

        for (unsigned i = 0; i < elemCount; ++i) {
            UNIT_ASSERT_EQUAL(shiftedData[i], TQType<TOp>::As(result)[i]);
        }
    }
}


void TSSEEmulTest::Test_mm_byte_shifter(EDirection direction, std::function<TShiftRes (__m128i)> foo) {
    const char data[48] = {
        '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00',
        '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00',
        '\xAA', '\x00', '\xFF', '\xCC', '\x11', '\x22', '\xBB', '\xAA',
        '\x33', '\x99', '\x44', '\x88', '\x55', '\x77', '\x66', '\x1C',
        '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00',
        '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00'
    };

    const __m128i a = _mm_loadu_si128((__m128i*)(data + 16));
    const TShiftRes res = foo(a);
    for (int shift = 0; shift <= 16; ++shift) {
        const int etalon_offset = 16 + (direction == EDirection::Left ? -shift : shift); //!< specific to little endian byte order.
        const char* etalon = data + etalon_offset;
        const char* res_bytes = (const char*)&res.Value[shift];

        for (size_t byte = 0; byte != 16; ++byte) {
            UNIT_ASSERT_EQUAL(etalon[byte], res_bytes[byte]);
        }
    }
}

template <typename TElem>
struct THelperASHR {
    static TElem Call(const TElem op, const int shift) {
        constexpr int nBitsInOp = sizeof(op) * CHAR_BIT;
        if (op < 0) {
            // Arithmetic shift propagates sign bit to the right
            // while operator>> is implementation defined for negative values,
            // so we can't use it as a reference implementation
            // and we need to write some standard consistent code.
            typedef TFixedWidthUnsignedInt<TElem> TUnsignedElem;
            TUnsignedElem uOp(op);
            const TUnsignedElem signBit = TUnsignedElem(1) << (nBitsInOp - 1);
            Y_ENSURE(shift >= 0);
            for (int i = 0; i != shift; ++i) {
                uOp = signBit | (uOp >> 1);
            }
            // unsigned -> signed conversion is also implementation defined, so we need to use some other method.
            return reinterpret_cast<TElem&>(uOp);
        }
        return shift < nBitsInOp ? op >> shift : 0;
    }
};

template <typename TElem>
struct THelperSHR {
    static TElem Call(const TElem op, const int shift) {
        constexpr int nBitsInOp = sizeof(op) * CHAR_BIT;
        return shift < nBitsInOp ? op >> shift : 0;
    }
};

void TSSEEmulTest::Test_mm_srli_epi16() {
    Test_mm_shifter_epiXX<ui16, 16, 8, Wrap(_mm_srli_epi16), unsigned, uint16x8_t,
                          THelperSHR<ui16>>();
}

void TSSEEmulTest::Test_mm_srli_epi32() {
    Test_mm_shifter_epiXX<ui32, 32, 4, Wrap(_mm_srli_epi32), unsigned, uint32x4_t,
                          THelperSHR<ui32>>();
}

void TSSEEmulTest::Test_mm_srli_epi64() {
    Test_mm_shifter_epiXX<ui64, 64, 2, Wrap(_mm_srli_epi64), unsigned, uint64x2_t,
                          THelperSHR<ui64>>();
}

template <typename TElem>
struct THelperSHL {
    static TElem Call(const TElem op, const int shift) {
        constexpr int nBitsInOp = sizeof(op) * CHAR_BIT;
        return shift < nBitsInOp ? op << shift : 0;
    }
};

void TSSEEmulTest::Test_mm_slli_epi16() {
    Test_mm_shifter_epiXX<ui16, 16, 8, Wrap(_mm_slli_epi16), unsigned, uint16x8_t,
                          THelperSHL<ui16>>();
}

void TSSEEmulTest::Test_mm_slli_epi32() {
    Test_mm_shifter_epiXX<ui32, 32, 4, Wrap(_mm_slli_epi32), unsigned, uint32x4_t,
                          THelperSHL<ui32>>();
}

void TSSEEmulTest::Test_mm_slli_epi64() {
    Test_mm_shifter_epiXX<ui64, 64, 2, Wrap(_mm_slli_epi64), unsigned, uint64x2_t,
                          THelperSHL<ui64>>();
}

void TSSEEmulTest::Test_mm_slli_si128() {
    Test_mm_byte_shifter(EDirection::Left, [] (__m128i a) -> TShiftRes {
        TShiftRes res;
        res.Value[0] = _mm_slli_si128(a, 0);
        res.Value[1] = _mm_slli_si128(a, 1);
        res.Value[2] = _mm_slli_si128(a, 2);
        res.Value[3] = _mm_slli_si128(a, 3);
        res.Value[4] = _mm_slli_si128(a, 4);
        res.Value[5] = _mm_slli_si128(a, 5);
        res.Value[6] = _mm_slli_si128(a, 6);
        res.Value[7] = _mm_slli_si128(a, 7);
        res.Value[8] = _mm_slli_si128(a, 8);
        res.Value[9] = _mm_slli_si128(a, 9);
        res.Value[10] = _mm_slli_si128(a, 10);
        res.Value[11] = _mm_slli_si128(a, 11);
        res.Value[12] = _mm_slli_si128(a, 12);
        res.Value[13] = _mm_slli_si128(a, 13);
        res.Value[14] = _mm_slli_si128(a, 14);
        res.Value[15] = _mm_slli_si128(a, 15);
        res.Value[16] = _mm_slli_si128(a, 16);

        return res;
    });
}

void TSSEEmulTest::Test_mm_srl_epi16() {
    Test_mm_shifter_epiXX<ui16, 16, 8, T_mm_CallWrapper<__m128i, decltype(_mm_srl_epi16), _mm_srl_epi16>, __m128i, uint16x8_t,
                          THelperSHR<ui16>>();
}

void TSSEEmulTest::Test_mm_srl_epi32() {
    Test_mm_shifter_epiXX<ui32, 32, 4, T_mm_CallWrapper<__m128i, decltype(_mm_srl_epi32), _mm_srl_epi32>, __m128i, uint32x4_t,
                          THelperSHR<ui32>>();
}

void TSSEEmulTest::Test_mm_srl_epi64() {
    Test_mm_shifter_epiXX<ui64, 64, 2, T_mm_CallWrapper<__m128i, decltype(_mm_srl_epi64), _mm_srl_epi64>, __m128i, uint64x2_t,
                          THelperSHR<ui64>>();
}

void TSSEEmulTest::Test_mm_srai_epi16() {
    Test_mm_shifter_epiXX<i16, 16, 8, T_mm_CallWrapper<__m128i, decltype(_mm_srai_epi16), _mm_srai_epi16>, unsigned, int16x8_t,
                          THelperASHR<i16>>();
}

void TSSEEmulTest::Test_mm_srai_epi32() {
    Test_mm_shifter_epiXX<i32, 32, 4, T_mm_CallWrapper<__m128i, decltype(_mm_srai_epi32), _mm_srai_epi32>, unsigned, int32x4_t,
                          THelperASHR<i32>>();
}

void TSSEEmulTest::Test_mm_srli_si128() {
    Test_mm_byte_shifter(EDirection::Right, [](__m128i a) -> TShiftRes {
        TShiftRes res;
        res.Value[0] = _mm_srli_si128(a, 0);
        res.Value[1] = _mm_srli_si128(a, 1);
        res.Value[2] = _mm_srli_si128(a, 2);
        res.Value[3] = _mm_srli_si128(a, 3);
        res.Value[4] = _mm_srli_si128(a, 4);
        res.Value[5] = _mm_srli_si128(a, 5);
        res.Value[6] = _mm_srli_si128(a, 6);
        res.Value[7] = _mm_srli_si128(a, 7);
        res.Value[8] = _mm_srli_si128(a, 8);
        res.Value[9] = _mm_srli_si128(a, 9);
        res.Value[10] = _mm_srli_si128(a, 10);
        res.Value[11] = _mm_srli_si128(a, 11);
        res.Value[12] = _mm_srli_si128(a, 12);
        res.Value[13] = _mm_srli_si128(a, 13);
        res.Value[14] = _mm_srli_si128(a, 14);
        res.Value[15] = _mm_srli_si128(a, 15);
        res.Value[16] = _mm_srli_si128(a, 16);

        return res;
    });
}

void TSSEEmulTest::Test_mm_sll_epi16() {
    Test_mm_shifter_epiXX<ui16, 16, 8, T_mm_CallWrapper<__m128i, decltype(_mm_sll_epi16), _mm_sll_epi16>, __m128i, uint16x8_t,
                          THelperSHL<ui16>>();
}

void TSSEEmulTest::Test_mm_sll_epi32() {
    Test_mm_shifter_epiXX<ui32, 32, 4, T_mm_CallWrapper<__m128i, decltype(_mm_sll_epi32), _mm_sll_epi32>, __m128i, uint32x4_t,
                          THelperSHL<ui32>>();
}

void TSSEEmulTest::Test_mm_sll_epi64() {
    Test_mm_shifter_epiXX<ui64, 64, 2, T_mm_CallWrapper<__m128i, decltype(_mm_sll_epi64), _mm_sll_epi64>, __m128i, uint64x2_t,
                          THelperSHL<ui64>>();
}

template <typename TElem>
struct THelperAdd {
    static TElem Call(const TElem op1, const TElem op2) {
        return op1 + op2;
    }
};

void TSSEEmulTest::Test_mm_add_epi16() {
    Test_mm_dualop<ui16, 8, Wrap(_mm_add_epi16), THelperAdd<ui16>, uint16x8_t>();
}

void TSSEEmulTest::Test_mm_add_epi32() {
    Test_mm_dualop<ui32, 4, Wrap(_mm_add_epi32), THelperAdd<ui32>, uint32x4_t>();
}

void TSSEEmulTest::Test_mm_add_epi64() {
    Test_mm_dualop<ui64, 2, Wrap(_mm_add_epi64), THelperAdd<ui64>, uint64x2_t>();
}

void TSSEEmulTest::Test_mm_add_ps() {
    Test_mm_dualop<float, 2, WrapF(_mm_add_ps),
                   THelperAdd<float>, float32x4_t, __m128>();
}

void TSSEEmulTest::Test_mm_add_pd() {
    Test_mm_dualop<double, 2, WrapD(_mm_add_pd),
                   THelperAdd<double>, float64x2_t, __m128d>();
}

void TSSEEmulTest::Test_mm_madd_epi16() {
    alignas(16) const char data1[16] = {
        '\xAA', '\x00', '\xFF', '\xCC', '\x11', '\x22', '\xBB', '\xAA',
        '\x33', '\x99', '\x44', '\x88', '\x55', '\x77', '\x66', '\x1C'
    };
    alignas(16) const char data2[16] = {
        '\x99', '\x33', '\x1C', '\x55', '\x88', '\x66', '\x77', '\x44',
        '\x00', '\xAA', '\xAA', '\x11', '\xCC', '\xBB', '\x22', '\xFF'
    };

    const __m128i value1 = TFuncLoad<__m128i>(&data1);
    const __m128i value2 = TFuncLoad<__m128i>(&data2);
    const __m128i res = _mm_madd_epi16(value1, value2);

    const i16* dataw1 = reinterpret_cast<const i16*>(&data1);
    const i16* dataw2 = reinterpret_cast<const i16*>(&data2);

    for (size_t i = 0; i != 4; ++i) {
        const size_t dataIdx = i * 2;
        const i32 etalonResult = (i32) dataw1[dataIdx] * (i32) dataw2[dataIdx] + (i32) dataw1[dataIdx + 1] * (i32) dataw2[dataIdx + 1];
        const i32 value = TQType<int32x4_t>::As(res)[i];
        UNIT_ASSERT_EQUAL(value, etalonResult);
    }
}


template <typename TElem>
struct THelperSub {
    static TElem Call(const TElem op1, const TElem op2) {
        return op1 - op2;
    }
};

void TSSEEmulTest::Test_mm_sub_epi16() {
    Test_mm_dualop<ui16, 8, Wrap(_mm_sub_epi16), THelperSub<ui16>, uint16x8_t>();
}

void TSSEEmulTest::Test_mm_sub_epi32() {
    Test_mm_dualop<ui32, 4, Wrap(_mm_sub_epi32), THelperSub<ui32>, uint32x4_t>();
}

void TSSEEmulTest::Test_mm_sub_epi64() {
    Test_mm_dualop<ui64, 2, Wrap(_mm_sub_epi64), THelperSub<ui64>, uint64x2_t>();
}

void TSSEEmulTest::Test_mm_sub_ps() {
    Test_mm_dualop<float, 4, WrapF(_mm_sub_ps), THelperSub<float>,
                   float32x4_t, __m128>();
}

void TSSEEmulTest::Test_mm_sub_pd() {
    Test_mm_dualop<double, 2, WrapD(_mm_sub_pd), THelperSub<double>,
                   float64x2_t, __m128d>();
}

void TSSEEmulTest::Test_mm_mul_ps() {
    struct THelper {
        static float Call(const float op1, const float op2) {
            return op1 * op2;
        }
    };
    Test_mm_dualop<float, 4, WrapF(_mm_mul_ps), THelper, float32x4_t, __m128>();
}

void TSSEEmulTest::Test_mm_mul_pd() {
    struct THelper {
        static double Call(const double op1, const double op2) {
            return op1 * op2;
        }
    };
    Test_mm_dualop<double, 2, WrapD(_mm_mul_pd), THelper, float64x2_t, __m128d>();
}

void TSSEEmulTest::Test_mm_div_ps() {
    struct THelper {
        static float Call(const float op1, const float op2) {
            return op1 / op2;
        }
    };
    Test_mm_dualop<float, 4, WrapF(_mm_div_ps), THelper, float32x4_t, __m128>();
}

void TSSEEmulTest::Test_mm_div_pd() {
    struct THelper {
        static double Call(const double op1, const double op2) {
            return op1 / op2;
        }
    };
    Test_mm_dualop<double, 2, WrapD(_mm_div_pd), THelper, float64x2_t, __m128d>();
}

void TSSEEmulTest::Test_mm_max_ps() {
    struct THelper {
        static float Call(const float op1, const float op2) {
            return std::max(op1, op2);
        }
    };
    Test_mm_dualop<float, 4, WrapF(_mm_max_ps), THelper, float32x4_t, __m128>();
}

void TSSEEmulTest::Test_mm_min_ps() {
    struct THelper {
        static float Call(const float op1, const float op2) {
            return std::min(op1, op2);
        }
    };
    Test_mm_dualop<float, 4, WrapF(_mm_min_ps), THelper, float32x4_t, __m128>();
}

void TSSEEmulTest::Test_mm_and_ps() {
    struct THelper {
        static float Call(const float op1, const float op2) {
            union Cast {
                unsigned int AsUInt;
                float AsFloat;
            };
            Cast v1, v2, result;
            v1.AsFloat = op1;
            v2.AsFloat = op2;
            result.AsUInt = v1.AsUInt & v2.AsUInt;
            return result.AsFloat;
        }
    };
    Test_mm_dualcmp<float, 4, WrapF(_mm_and_ps),
                    THelper, float32x4_t, __m128>();
}

template <typename TElem, int bits, int elemCount, int shift,
          typename TFunc, typename TOp>
void TSSEEmulTest::Test_mm_unpack_epiXX() {
    char data1[16] = {
        '\xAA', '\x00', '\xFF', '\xCC', '\x11', '\x22', '\xBB', '\xAA',
        '\x33', '\x99', '\x44', '\x88', '\x55', '\x77', '\x66', '\x1C'};
    char data2[16] = {
        '\x99', '\x33', '\x1C', '\x55', '\x88', '\x66', '\x77', '\x44',
        '\x00', '\xAA', '\xAA', '\x11', '\xCC', '\xBB', '\x22', '\xFF'};
    TElem* dataw1 = reinterpret_cast<TElem*>(&data1);
    TElem* dataw2 = reinterpret_cast<TElem*>(&data2);

    __m128i value1 = _mm_loadu_si128((__m128i*)&data1);
    __m128i value2 = _mm_loadu_si128((__m128i*)&data2);

    TElem zippedData[elemCount];
    for (unsigned i = 0; i < elemCount / 2; ++i) {
        zippedData[i * 2] = dataw1[i + shift];
        zippedData[i * 2 + 1] = dataw2[i + shift];
    }
    __m128i result = TFunc(value1, value2);

    for (unsigned i = 0; i < elemCount / 2; ++i) {
        UNIT_ASSERT_EQUAL(zippedData[i * 2], TQType<TOp>::As(result)[i * 2]);
        UNIT_ASSERT_EQUAL(zippedData[i * 2 + 1],
                          TQType<TOp>::As(result)[i * 2 + 1]);
    }
}

void TSSEEmulTest::Test_mm_unpacklo_epi8() {
    Test_mm_unpack_epiXX<ui8, 8, 16, 0, Wrap(_mm_unpacklo_epi8), uint8x16_t>();
}

void TSSEEmulTest::Test_mm_unpackhi_epi8() {
    Test_mm_unpack_epiXX<ui8, 8, 16, 8, Wrap(_mm_unpackhi_epi8), uint8x16_t>();
}

void TSSEEmulTest::Test_mm_unpacklo_epi16() {
    Test_mm_unpack_epiXX<ui16, 16, 8, 0, Wrap(_mm_unpacklo_epi16), uint16x8_t>();
}

void TSSEEmulTest::Test_mm_unpackhi_epi16() {
    Test_mm_unpack_epiXX<ui16, 16, 8, 4, Wrap(_mm_unpackhi_epi16), uint16x8_t>();
}

void TSSEEmulTest::Test_mm_unpacklo_epi32() {
    Test_mm_unpack_epiXX<ui32, 32, 4, 0, Wrap(_mm_unpacklo_epi32), uint32x4_t>();
}

void TSSEEmulTest::Test_mm_unpackhi_epi32() {
    Test_mm_unpack_epiXX<ui32, 32, 4, 2, Wrap(_mm_unpackhi_epi32), uint32x4_t>();
}

void TSSEEmulTest::Test_mm_unpacklo_epi64() {
    Test_mm_unpack_epiXX<ui64, 64, 2, 0, Wrap(_mm_unpacklo_epi64), uint64x2_t>();
}

void TSSEEmulTest::Test_mm_unpackhi_epi64() {
    Test_mm_unpack_epiXX<ui64, 64, 2, 1, Wrap(_mm_unpackhi_epi64), uint64x2_t>();
}

template <typename TElem, unsigned elemCount,
          typename TFunc, typename TElemFunc,
          typename TOp, typename TVectorType>
void TSSEEmulTest::Test_mm_dualop() {
    char data1[16] = {
        '\xAA', '\x00', '\xFF', '\xCC', '\x11', '\x22', '\xBB', '\xAA',
        '\x33', '\x99', '\x44', '\x88', '\x55', '\x77', '\x66', '\x1C'};
    char data2[16] = {
        '\x99', '\x33', '\x1C', '\x55', '\x88', '\x66', '\x77', '\x44',
        '\x00', '\xAA', '\xAA', '\x11', '\xCC', '\xBB', '\x22', '\xFF'};
    TElem* dataw1 = reinterpret_cast<TElem*>(&data1);
    TElem* dataw2 = reinterpret_cast<TElem*>(&data2);

    TVectorType value1 = TFuncLoad<TVectorType>(&data1);
    TVectorType value2 = TFuncLoad<TVectorType>(&data2);

    TElem procData[elemCount];
    for (unsigned i = 0; i < elemCount; ++i) {
        procData[i] = TElemFunc::Call(dataw1[i], dataw2[i]);
    }
    TVectorType result = TFunc(value1, value2);

    for (unsigned i = 0; i < elemCount; ++i) {
        UNIT_ASSERT_EQUAL(procData[i], TQType<TOp>::As(result)[i]);
    }
}

/* This is almost the same as Test_mm_dualop,
   but different data1 and data2 */
template <typename TElem, unsigned elemCount,
          typename TFunc, typename TElemFunc,
          typename TOp, typename TVectorType>
void TSSEEmulTest::Test_mm_dualcmp() {
    char data1[16] = {
        '\xAA', '\x00', '\xFF', '\xCC', '\x11', '\x66', '\x77', '\xAA',
        '\x33', '\x99', '\x44', '\x88', '\xCC', '\xBB', '\x66', '\x1C'};
    char data2[16] = {
        '\x99', '\x33', '\xFF', '\xCC', '\x88', '\x66', '\x77', '\x44',
        '\x33', '\x99', '\x44', '\x88', '\xCC', '\xBB', '\x22', '\xFF'};
    TElem* dataw1 = reinterpret_cast<TElem*>(&data1);
    TElem* dataw2 = reinterpret_cast<TElem*>(&data2);

    TVectorType value1 = TFuncLoad<TVectorType>(&data1);
    TVectorType value2 = TFuncLoad<TVectorType>(&data2);

    TElem procData[elemCount];
    for (unsigned i = 0; i < elemCount; ++i) {
        procData[i] = TElemFunc::Call(dataw1[i], dataw2[i]);
    }
    TVectorType result = TFunc(value1, value2);

    for (unsigned i = 0; i < elemCount; ++i) {
        /* memcmp is for compare to invalid floats in results */
        const TElem value = TQType<TOp>::As(result)[i];
        UNIT_ASSERT(memcmp(&(procData[i]), &value, sizeof(TElem)) == 0);
    }
}

void TSSEEmulTest::Test_mm_or_si128() {
    struct THelper {
        static ui64 Call(const ui64 op1, const ui64 op2) {
            return op1 | op2;
        }
    };

    Test_mm_dualop<ui64, 2, Wrap(_mm_or_si128), THelper, uint64x2_t>();
}

void TSSEEmulTest::Test_mm_and_si128() {
    struct THelper {
        static ui64 Call(const ui64 op1, const ui64 op2) {
            return op1 & op2;
        }
    };

    Test_mm_dualop<ui64, 2, Wrap(_mm_and_si128), THelper, uint64x2_t>();
}

void TSSEEmulTest::Test_mm_andnot_si128() {
    struct THelper {
        static ui64 Call(const ui64 op1, const ui64 op2) {
            return (~op1) & op2;
        }
    };

    Test_mm_dualop<ui64, 2, Wrap(_mm_andnot_si128), THelper, uint64x2_t>();
}

template <typename TElem>
struct THelperCMPEQ {
    static TElem Call(const TElem op1, const TElem op2) {
        return op1 == op2 ? ~TElem(0) : TElem(0);
    }
};

void TSSEEmulTest::Test_mm_cmpeq_epi8() {
    Test_mm_dualcmp<ui8, 16, Wrap(_mm_cmpeq_epi8),
                    THelperCMPEQ<ui8>, uint8x16_t>();
}

void TSSEEmulTest::Test_mm_cmpeq_epi16() {
    Test_mm_dualcmp<ui16, 8, Wrap(_mm_cmpeq_epi16),
                    THelperCMPEQ<ui16>, uint16x8_t>();
}

void TSSEEmulTest::Test_mm_cmpeq_epi32() {
    Test_mm_dualcmp<ui32, 4, Wrap(_mm_cmpeq_epi32),
                    THelperCMPEQ<ui32>, uint32x4_t>();
}

void TSSEEmulTest::Test_mm_cmpeq_ps() {
    struct THelperFloat {
        static float Call(const float op1, const float op2) {
            union Cast {
                unsigned int AsUInt;
                float AsFloat;
            };
            Cast value;
            value.AsUInt = op1 == op2 ? 0xFFFFFFFF : 0;
            return value.AsFloat;
        }
    };

    Test_mm_dualcmp<float, 4, WrapF(_mm_cmpeq_ps),
                    THelperFloat, float32x4_t, __m128>();
}

template <typename TElem>
struct THelperCMPGT {
    static TElem Call(const TElem op1, const TElem op2) {
        return op1 > op2 ? ~TElem(0) : TElem(0);
    }
};

void TSSEEmulTest::Test_mm_cmpgt_epi8() {
    Test_mm_dualcmp<i8, 16, Wrap(_mm_cmpgt_epi8),
                    THelperCMPGT<i8>, int8x16_t>();
}

void TSSEEmulTest::Test_mm_cmpgt_epi16() {
    Test_mm_dualcmp<i16, 8, Wrap(_mm_cmpgt_epi16),
                    THelperCMPGT<i16>, int16x8_t>();
}

void TSSEEmulTest::Test_mm_cmpgt_epi32() {
    Test_mm_dualcmp<i32, 4, Wrap(_mm_cmpgt_epi32),
                    THelperCMPGT<i32>, int32x4_t>();
}

void TSSEEmulTest::Test_mm_cmpgt_ps() {
    struct THelperFloat {
        static float Call(const float op1, const float op2) {
            union Cast {
                unsigned int AsUInt;
                float AsFloat;
            };
            Cast value;
            value.AsUInt = op1 > op2 ? 0xFFFFFFFF : 0;
            return value.AsFloat;
        }
    };

    Test_mm_dualcmp<float, 4, WrapF(_mm_cmpgt_ps),
                    THelperFloat, float32x4_t, __m128>();
}

template <typename TElem>
struct THelperCMPLT {
    static TElem Call(const TElem op1, const TElem op2) {
        return op1 < op2 ? ~TElem(0) : TElem(0);
    }
};

void TSSEEmulTest::Test_mm_cmplt_epi8() {
    Test_mm_dualcmp<i8, 16, Wrap(_mm_cmplt_epi8),
                    THelperCMPLT<i8>, int8x16_t>();
}

void TSSEEmulTest::Test_mm_cmplt_epi16() {
    Test_mm_dualcmp<i16, 8, Wrap(_mm_cmplt_epi16),
                    THelperCMPLT<i16>, int16x8_t>();
}

void TSSEEmulTest::Test_mm_cmplt_epi32() {
    Test_mm_dualcmp<i32, 4, Wrap(_mm_cmplt_epi32),
                    THelperCMPLT<i32>, int32x4_t>();
}

template <typename TElem, int elemCount,
          typename TFunc, typename TOp, typename TVectorType>
void TSSEEmulTest::Test_mm_setter_epiXX() {
    char data[64] = {
        '\xAA', '\x00', '\xFF', '\xCC', '\x11', '\x22', '\xBB', '\xAA',
        '\x33', '\x99', '\x44', '\x00', '\x55', '\x77', '\x66', '\x1C',
        '\x99', '\x33', '\x1C', '\x55', '\x88', '\x66', '\x77', '\x44',
        '\x00', '\xAA', '\xAA', '\x11', '\xCC', '\xBB', '\x22', '\xFF',
        '\xAA', '\x00', '\xFF', '\xCC', '\x11', '\x00', '\x00', '\x00',
        '\x33', '\x99', '\x44', '\x88', '\xCC', '\xBB', '\x66', '\x1C',
        '\x99', '\x33', '\xFF', '\xCC', '\x88', '\x66', '\x77', '\x44',
        '\x33', '\x99', '\x44', '\x88', '\xCC', '\xBB', '\x22', '\xFF'};
    TElem* dataw = reinterpret_cast<TElem*>(&data);

    for (unsigned dataItem = 0; dataItem < elemCount * 4; ++dataItem) {
        TVectorType value = TFunc(dataw[dataItem]);

        for (unsigned i = 0; i < elemCount; ++i)
            UNIT_ASSERT_EQUAL(dataw[dataItem], TQType<TOp>::As(value)[i]);
    }
}

void TSSEEmulTest::Test_mm_set1_epi8() {
    Test_mm_setter_epiXX<i8, 16, Wrap(_mm_set1_epi8), int8x16_t, __m128i>();
}
void TSSEEmulTest::Test_mm_set1_epi16() {
    Test_mm_setter_epiXX<i16, 8, Wrap(_mm_set1_epi16), int16x8_t, __m128i>();
}
void TSSEEmulTest::Test_mm_set1_epi32() {
    Test_mm_setter_epiXX<i32, 4, Wrap(_mm_set1_epi32), int32x4_t, __m128i>();
}
void TSSEEmulTest::Test_mm_set1_ps() {
    Test_mm_setter_epiXX<float, 4, WrapF(_mm_set1_ps), float32x4_t, __m128>();
}

void TSSEEmulTest::Test_mm_set_ps1() {
    Test_mm_setter_epiXX<float, 4, WrapF(_mm_set_ps1), float32x4_t, __m128>();
}

void TSSEEmulTest::Test_mm_setzero_si128() {
    __m128i value = _mm_setzero_si128();
    for (unsigned i = 0; i < 4; ++i)
        UNIT_ASSERT_EQUAL(0, TQType<uint32x4_t>::As(value)[i]);
}

void TSSEEmulTest::Test_mm_setzero_ps() {
    __m128 value = _mm_setzero_ps();
    for (unsigned i = 0; i < 4; ++i)
        UNIT_ASSERT_EQUAL(0.0, TQType<float32x4_t>::As(value)[i]);
}

void TSSEEmulTest::Test_mm_setzero_pd() {
    __m128d value = _mm_setzero_pd();
    for (unsigned i = 0; i < 2; ++i)
        UNIT_ASSERT_EQUAL(0.0, TQType<float64x2_t>::As(value)[i]);
}

void TSSEEmulTest::Test_mm_loadl_epi64() {
    char data[64] = {
        '\xAA', '\x00', '\xFF', '\xCC', '\x11', '\x22', '\xBB', '\xAA',
        '\x33', '\x99', '\x44', '\x00', '\x55', '\x77', '\x66', '\x1C',
        '\x99', '\x33', '\x1C', '\x55', '\x88', '\x66', '\x77', '\x44',
        '\x00', '\xAA', '\xAA', '\x11', '\xCC', '\xBB', '\x22', '\xFF',
        '\xAA', '\x00', '\xFF', '\xCC', '\x11', '\x00', '\x00', '\x00',
        '\x33', '\x99', '\x44', '\x88', '\xCC', '\xBB', '\x66', '\x1C',
        '\x99', '\x33', '\xFF', '\xCC', '\x88', '\x66', '\x77', '\x44',
        '\x33', '\x99', '\x44', '\x88', '\xCC', '\xBB', '\x22', '\xFF'};
    ui64* dataw = reinterpret_cast<ui64*>(&data);

    for (unsigned dataItem = 0; dataItem < 8; ++dataItem) {
        __m128i value = _mm_loadl_epi64((__m128i const*)&dataw[dataItem]);

        UNIT_ASSERT_EQUAL(dataw[dataItem], TQType<uint64x2_t>::As(value)[0]);
        UNIT_ASSERT_EQUAL(0, TQType<uint64x2_t>::As(value)[1]);
    }
}

void TSSEEmulTest::Test_mm_storel_epi64() {
    char data[64] = {
        '\xAA', '\x00', '\xFF', '\xCC', '\x11', '\x22', '\xBB', '\xAA',
        '\x33', '\x99', '\x44', '\x00', '\x55', '\x77', '\x66', '\x1C',
        '\x99', '\x33', '\x1C', '\x55', '\x88', '\x66', '\x77', '\x44',
        '\x00', '\xAA', '\xAA', '\x11', '\xCC', '\xBB', '\x22', '\xFF',
        '\xAA', '\x00', '\xFF', '\xCC', '\x11', '\x00', '\x00', '\x00',
        '\x33', '\x99', '\x44', '\x88', '\xCC', '\xBB', '\x66', '\x1C',
        '\x99', '\x33', '\xFF', '\xCC', '\x88', '\x66', '\x77', '\x44',
        '\x33', '\x99', '\x44', '\x88', '\xCC', '\xBB', '\x22', '\xFF'};
    ui64* dataw = reinterpret_cast<ui64*>(&data);

    for (unsigned dataItem = 0; dataItem < 4; ++dataItem) {
        __m128i value = _mm_loadu_si128((__m128i*)&dataw[dataItem * 2]);

        ui64 buf[2] = {55, 81};
        _mm_storel_epi64((__m128i*)&buf, value);

        UNIT_ASSERT_EQUAL(dataw[dataItem * 2], buf[0]);
        UNIT_ASSERT_EQUAL(81, buf[1]);
    }
}

void TSSEEmulTest::Test_mm_shuffle_epi32() {
    char data[16] = {
        '\xAA', '\x00', '\xFF', '\xCC', '\x11', '\x22', '\xBB', '\xAA',
        '\x33', '\x99', '\x44', '\x88', '\x55', '\x77', '\x66', '\x1C'};
    ui32* dataw = reinterpret_cast<ui32*>(&data);
    __m128i value = _mm_loadu_si128((__m128i*)&data);

    int coding[4] = {1, 3, 0, 2};
    __m128i result = _mm_shuffle_epi32(value, _MM_SHUFFLE(2, 0, 3, 1));

    for (unsigned i = 0; i < 4; ++i)
        UNIT_ASSERT_EQUAL(dataw[coding[i]],
                          TQType<uint32x4_t>::As(result)[i]);
}

static int GetHighBitAt(char data, int at) {
    ui8 udata = data & 0x80;
    return int(udata >> 7) << at;
}

void TSSEEmulTest::Test_mm_movemask_epi8() {
    char data[16] = {
        '\xAA', '\x00', '\xFF', '\xCC', '\x11', '\x22', '\xBB', '\xAA',
        '\x33', '\x99', '\x44', '\x88', '\x55', '\x77', '\x66', '\x1C'};
    __m128i value = _mm_loadu_si128((__m128i*)&data);

    int result = _mm_movemask_epi8(value);
    int verify = 0;
    for (unsigned i = 0; i < 16; ++i) {
        verify |= GetHighBitAt(data[i], i);
    }

    UNIT_ASSERT_EQUAL(result, verify);
}

void TSSEEmulTest::Test_mm_movemask_ps() {
    char data[16] = {
        '\xAA', '\x00', '\xFF', '\xCC', '\x11', '\x22', '\xBB', '\xAA',
        '\x33', '\x99', '\x44', '\x88', '\x55', '\x77', '\x66', '\x1C'};
    __m128 value = _mm_loadu_ps((float*)&data);

    int result = _mm_movemask_ps(value);
    int verify = 0;
    for (unsigned i = 0; i < 4; ++i) {
        verify |= GetHighBitAt(data[i * 4 + 3], i);
    }

    UNIT_ASSERT_EQUAL(result, verify);
}

void TSSEEmulTest::Test_mm_movemask_ps_2() {
    char data[16] = {
        '\xFF', '\xFF', '\xFF', '\xFF', '\xFF', '\xFF', '\xFF', '\xFF',
        '\xFF', '\xFF', '\xFF', '\xFF', '\xFF', '\xFF', '\xFF', '\xFF'};
    __m128 value = _mm_loadu_ps((float*)&data);

    int result = _mm_movemask_ps(value);
    UNIT_ASSERT_EQUAL(result, 0xf);
}

void TSSEEmulTest::Test_mm_cvtsi128_si32() {
    char data[16] = {
        '\xAA', '\x00', '\xFF', '\xCC', '\x11', '\x22', '\xBB', '\xAA',
        '\x33', '\x99', '\x44', '\x88', '\x55', '\x77', '\x66', '\x1C'};
    __m128i value = _mm_loadu_si128((__m128i*)&data);

    int result = _mm_cvtsi128_si32(value);
    i32* datap = reinterpret_cast<i32*>(&data);
    int verify = datap[0];

    UNIT_ASSERT_EQUAL(result, verify);
}

void TSSEEmulTest::Test_mm_cvtsi128_si64() {
    char data[16] = {
        '\xAA', '\x00', '\xFF', '\xCC', '\x11', '\x22', '\xBB', '\xAA',
        '\x33', '\x99', '\x44', '\x88', '\x55', '\x77', '\x66', '\x1C'};
    __m128i value = _mm_loadu_si128((__m128i*)&data);

    i64 result = _mm_cvtsi128_si64(value);
    i64* datap = reinterpret_cast<i64*>(&data);
    i64 verify = datap[0];

    UNIT_ASSERT_EQUAL(result, verify);
}

void TSSEEmulTest::Test_mm_set_epi16() {
    char data[16] = {
        '\xAA', '\x00', '\xFF', '\xCC', '\x11', '\x22', '\xBB', '\xAA',
        '\x33', '\x99', '\x44', '\x88', '\x55', '\x77', '\x66', '\x1C'};
    i16* dataw = reinterpret_cast<i16*>(&data);
    ui64* dataq = reinterpret_cast<ui64*>(&data);

    __m128i result = _mm_set_epi16(dataw[7], dataw[6], dataw[5], dataw[4],
                                   dataw[3], dataw[2], dataw[1], dataw[0]);
    ui64 buf[2] = {53, 81};
    _mm_storeu_si128((__m128i*)&buf, result);

    UNIT_ASSERT_EQUAL(buf[0], dataq[0]);
    UNIT_ASSERT_EQUAL(buf[1], dataq[1]);
}

void TSSEEmulTest::Test_mm_set_epi32() {
    char data[16] = {
        '\xAA', '\x00', '\xFF', '\xCC', '\x11', '\x22', '\xBB', '\xAA',
        '\x33', '\x99', '\x44', '\x88', '\x55', '\x77', '\x66', '\x1C'};
    i32* dataw = reinterpret_cast<i32*>(&data);
    ui64* dataq = reinterpret_cast<ui64*>(&data);

    __m128i result = _mm_set_epi32(dataw[3], dataw[2], dataw[1], dataw[0]);
    ui64 buf[2] = {53, 81};
    _mm_storeu_si128((__m128i*)&buf, result);

    UNIT_ASSERT_EQUAL(buf[0], dataq[0]);
    UNIT_ASSERT_EQUAL(buf[1], dataq[1]);
}

void TSSEEmulTest::Test_mm_set_ps() {
    char data[16] = {
        '\xAA', '\x00', '\xFF', '\xCC', '\x11', '\x22', '\xBB', '\xAA',
        '\x33', '\x99', '\x44', '\x88', '\x55', '\x77', '\x66', '\x1C'};
    float* dataw = reinterpret_cast<float*>(&data);
    ui64* dataq = reinterpret_cast<ui64*>(&data);

    __m128 result = _mm_set_ps(dataw[3], dataw[2], dataw[1], dataw[0]);
    ui64 buf[2] = {53, 81};
    _mm_storeu_ps((float*)&buf, result);

    UNIT_ASSERT_EQUAL(buf[0], dataq[0]);
    UNIT_ASSERT_EQUAL(buf[1], dataq[1]);
}

void TSSEEmulTest::Test_mm_set_pd() {
    char data[16] = {
        '\xAA', '\x00', '\xFF', '\xCC', '\x11', '\x22', '\xBB', '\xAA',
        '\x33', '\x99', '\x44', '\x88', '\x55', '\x77', '\x66', '\x1C'};
    double* dataw = reinterpret_cast<double*>(&data);
    ui64* dataq = reinterpret_cast<ui64*>(&data);

    __m128d result = _mm_set_pd(dataw[1], dataw[0]);
    ui64 buf[2] = {53, 81};
    _mm_storeu_pd((double*)&buf, result);

    UNIT_ASSERT_EQUAL(buf[0], dataq[0]);
    UNIT_ASSERT_EQUAL(buf[1], dataq[1]);
}

void TSSEEmulTest::Test_mm_cvtsi32_si128() {
    char data[16] = {
        '\xAA', '\x00', '\xFF', '\xCC', '\x11', '\x22', '\xBB', '\xAA',
        '\x33', '\x99', '\x44', '\x88', '\x55', '\x77', '\x66', '\x1C'};
    i32* dataw = reinterpret_cast<i32*>(&data);

    __m128i result = _mm_cvtsi32_si128(dataw[0]);
    i32 buf[4] = {53, 81, -43, 2132};
    _mm_storeu_si128((__m128i*)&buf, result);

    UNIT_ASSERT_EQUAL(buf[0], dataw[0]);
    UNIT_ASSERT_EQUAL(buf[1], 0);
    UNIT_ASSERT_EQUAL(buf[2], 0);
    UNIT_ASSERT_EQUAL(buf[3], 0);
}

void TSSEEmulTest::Test_mm_cvtsi64_si128() {
    char data[16] = {
        '\xAA', '\x00', '\xFF', '\xCC', '\x11', '\x22', '\xBB', '\xAA',
        '\x33', '\x99', '\x44', '\x88', '\x55', '\x77', '\x66', '\x1C'};
    i64* dataw = reinterpret_cast<i64*>(&data);

    __m128i result = _mm_cvtsi64_si128(dataw[0]);
    i64 buf[2] = {7, 8};
    _mm_storeu_si128((__m128i*)&buf, result);

    UNIT_ASSERT_EQUAL(buf[0], dataw[0]);
    UNIT_ASSERT_EQUAL(buf[1], 0);
}

template <typename TElem, typename TNarrow, unsigned elemCount, typename TFunc>
void TSSEEmulTest::Test_mm_packs_epiXX() {
    char data[32] = {
        '\xAA', '\x00', '\xFF', '\xCC', '\x11', '\x22', '\xBB', '\xAA',
        '\x33', '\x99', '\x44', '\x88', '\x55', '\x00', '\x66', '\x1C',
        '\x99', '\x33', '\x1C', '\x55', '\x00', '\x00', '\x00', '\x00',
        '\x00', '\xAA', '\x00', '\x00', '\xCC', '\xBB', '\x22', '\xFF'};
    __m128i value0 = _mm_loadu_si128((__m128i*)&data);
    __m128i value1 = _mm_loadu_si128(((__m128i*)&data) + 1);
    TElem* dataw = reinterpret_cast<TElem*>(&data);

    __m128i result = TFunc(value0, value1);

    TNarrow verify[elemCount];
    for (unsigned i = 0; i < elemCount; ++i) {
        TElem sum = dataw[i];
        if (sum > std::numeric_limits<TNarrow>::max())
            sum = std::numeric_limits<TNarrow>::max();
        if (sum < std::numeric_limits<TNarrow>::min())
            sum = std::numeric_limits<TNarrow>::min();
        verify[i] = TNarrow(sum);
    }

    ui64* verifyp = (ui64*)&verify;
    UNIT_ASSERT_EQUAL(verifyp[0], TQType<uint64x2_t>::As(result)[0]);
    UNIT_ASSERT_EQUAL(verifyp[1], TQType<uint64x2_t>::As(result)[1]);
}

void TSSEEmulTest::Test_mm_packs_epi16() {
    Test_mm_packs_epiXX<i16, i8, 16, Wrap(_mm_packs_epi16)>();
}
void TSSEEmulTest::Test_mm_packs_epi32() {
    Test_mm_packs_epiXX<i32, i16, 8, Wrap(_mm_packs_epi32)>();
}
void TSSEEmulTest::Test_mm_packus_epi16() {
    Test_mm_packs_epiXX<i16, ui8, 16, Wrap(_mm_packus_epi16)>();
}

void TSSEEmulTest::Test_mm_extract_epi8() {
    alignas(16) char data[16] = {
        '\xAA', '\x00', '\xFF', '\xCC', '\x11', '\x22', '\xBB', '\xAA',
        '\x33', '\x99', '\x44', '\x88', '\x55', '\x77', '\x66', '\x1C'};
    const ui8* dataw = reinterpret_cast<const ui8*>(&data);
    const __m128i value = _mm_loadu_si128((__m128i*)&data);

    UNIT_ASSERT_EQUAL((_mm_extract_epi16(value, 0)), int(dataw[0]));
    UNIT_ASSERT_EQUAL((_mm_extract_epi8(value, 1)), int(dataw[1]));
    UNIT_ASSERT_EQUAL((_mm_extract_epi8(value, 2)), int(dataw[2]));
    UNIT_ASSERT_EQUAL((_mm_extract_epi8(value, 3)), int(dataw[3]));
    UNIT_ASSERT_EQUAL((_mm_extract_epi8(value, 4)), int(dataw[4]));
    UNIT_ASSERT_EQUAL((_mm_extract_epi8(value, 5)), int(dataw[5]));
    UNIT_ASSERT_EQUAL((_mm_extract_epi8(value, 6)), int(dataw[6]));
    UNIT_ASSERT_EQUAL((_mm_extract_epi8(value, 7)), int(dataw[7]));
    UNIT_ASSERT_EQUAL((_mm_extract_epi8(value, 8)), int(dataw[8]));
    UNIT_ASSERT_EQUAL((_mm_extract_epi8(value, 9)), int(dataw[9]));
    UNIT_ASSERT_EQUAL((_mm_extract_epi8(value, 10)), int(dataw[10]));
    UNIT_ASSERT_EQUAL((_mm_extract_epi8(value, 11)), int(dataw[11]));
    UNIT_ASSERT_EQUAL((_mm_extract_epi8(value, 12)), int(dataw[12]));
    UNIT_ASSERT_EQUAL((_mm_extract_epi8(value, 13)), int(dataw[13]));
    UNIT_ASSERT_EQUAL((_mm_extract_epi8(value, 14)), int(dataw[14]));
    UNIT_ASSERT_EQUAL((_mm_extract_epi8(value, 15)), int(dataw[15]));
}

void TSSEEmulTest::Test_mm_extract_epi16() {
    alignas(16) char data[16] = {
        '\xAA', '\x00', '\xFF', '\xCC', '\x11', '\x22', '\xBB', '\xAA',
        '\x33', '\x99', '\x44', '\x88', '\x55', '\x77', '\x66', '\x1C'};
    const ui16* dataw = reinterpret_cast<const ui16*>(&data);
    const __m128i value = _mm_loadu_si128((__m128i*)&data);

    UNIT_ASSERT_EQUAL((_mm_extract_epi16(value, 0)), int(dataw[0]));
    UNIT_ASSERT_EQUAL((_mm_extract_epi16(value, 1)), int(dataw[1]));
    UNIT_ASSERT_EQUAL((_mm_extract_epi16(value, 2)), int(dataw[2]));
    UNIT_ASSERT_EQUAL((_mm_extract_epi16(value, 3)), int(dataw[3]));
    UNIT_ASSERT_EQUAL((_mm_extract_epi16(value, 4)), int(dataw[4]));
    UNIT_ASSERT_EQUAL((_mm_extract_epi16(value, 5)), int(dataw[5]));
    UNIT_ASSERT_EQUAL((_mm_extract_epi16(value, 6)), int(dataw[6]));
    UNIT_ASSERT_EQUAL((_mm_extract_epi16(value, 7)), int(dataw[7]));
}

void TSSEEmulTest::Test_mm_extract_epi64() {
    alignas(16) char data[16] = {
        '\xAA', '\x00', '\xFF', '\xCC', '\x11', '\x22', '\xBB', '\xAA',
        '\x33', '\x99', '\x44', '\x88', '\x55', '\x77', '\x66', '\x1C'};
    const ui64* dataw = reinterpret_cast<const ui64*>(&data);
    const __m128i value = _mm_loadu_si128((__m128i*)&data);

    UNIT_ASSERT_EQUAL((_mm_extract_epi64(value, 0)), (long long)(dataw[0]));
    UNIT_ASSERT_EQUAL((_mm_extract_epi64(value, 1)), (long long)(dataw[1]));
}

void TSSEEmulTest::Test_mm_extract_epi32() {
    alignas(16) char data[16] = {
        '\xAA', '\x00', '\xFF', '\xCC', '\x11', '\x22', '\xBB', '\xAA',
        '\x33', '\x99', '\x44', '\x88', '\x55', '\x77', '\x66', '\x1C'};
    const ui32* dataw = reinterpret_cast<const ui32*>(&data);
    const __m128i value = _mm_loadu_si128((__m128i*)&data);

    UNIT_ASSERT_EQUAL((_mm_extract_epi32(value, 0)), int(dataw[0]));
    UNIT_ASSERT_EQUAL((_mm_extract_epi32(value, 1)), int(dataw[1]));
    UNIT_ASSERT_EQUAL((_mm_extract_epi32(value, 2)), int(dataw[2]));
    UNIT_ASSERT_EQUAL((_mm_extract_epi32(value, 3)), int(dataw[3]));
}

void TSSEEmulTest::Test_MM_TRANSPOSE4_PS() {
    char data0[16] = {
        '\xAA', '\x00', '\xFF', '\xCC', '\x11', '\x22', '\xBB', '\xAA',
        '\x33', '\x99', '\x44', '\x88', '\x55', '\x77', '\x66', '\x1C'};
    char data1[16] = {
        '\x99', '\x33', '\x1C', '\x55', '\x88', '\x66', '\x77', '\x44',
        '\x00', '\xAA', '\xAA', '\x11', '\xCC', '\xBB', '\x22', '\xFF'};
    char data2[16] = {
        '\xAA', '\x00', '\xFF', '\xCC', '\x11', '\x22', '\xBB', '\xAA',
        '\x33', '\x99', '\x44', '\x88', '\x55', '\x77', '\x66', '\x1C'};
    char data3[16] = {
        '\x99', '\x33', '\x1C', '\x55', '\x88', '\x66', '\x77', '\x44',
        '\x00', '\xAA', '\xAA', '\x11', '\xCC', '\xBB', '\x22', '\xFF'};

    __m128 value0 = _mm_loadu_ps((float*)&data0);
    __m128 value1 = _mm_loadu_ps((float*)&data1);
    __m128 value2 = _mm_loadu_ps((float*)&data2);
    __m128 value3 = _mm_loadu_ps((float*)&data3);

    _MM_TRANSPOSE4_PS(value0, value1, value2, value3);

    ui64 tbuf0[2] = {0, 0};
    ui64 tbuf1[2] = {0, 0};
    ui64 tbuf2[2] = {0, 0};
    ui64 tbuf3[2] = {0, 0};

    _mm_storeu_ps((float*)&tbuf0, value0);
    _mm_storeu_ps((float*)&tbuf1, value1);
    _mm_storeu_ps((float*)&tbuf2, value2);
    _mm_storeu_ps((float*)&tbuf3, value3);

    char tdata0[16] = {
        '\xAA', '\x00', '\xFF', '\xCC', '\x99', '\x33', '\x1C', '\x55',
        '\xAA', '\x00', '\xFF', '\xCC', '\x99', '\x33', '\x1C', '\x55'};
    char tdata1[16] = {
        '\x11', '\x22', '\xBB', '\xAA', '\x88', '\x66', '\x77', '\x44',
        '\x11', '\x22', '\xBB', '\xAA', '\x88', '\x66', '\x77', '\x44'};
    char tdata2[16] = {
        '\x33', '\x99', '\x44', '\x88', '\x00', '\xAA', '\xAA', '\x11',
        '\x33', '\x99', '\x44', '\x88', '\x00', '\xAA', '\xAA', '\x11'};
    char tdata3[16] = {
        '\x55', '\x77', '\x66', '\x1C', '\xCC', '\xBB', '\x22', '\xFF',
        '\x55', '\x77', '\x66', '\x1C', '\xCC', '\xBB', '\x22', '\xFF'};

    UNIT_ASSERT(memcmp(tbuf0, tdata0, 16) == 0);
    UNIT_ASSERT(memcmp(tbuf1, tdata1, 16) == 0);
    UNIT_ASSERT(memcmp(tbuf2, tdata2, 16) == 0);
    UNIT_ASSERT(memcmp(tbuf3, tdata3, 16) == 0);
}

template <typename TFrom, typename TTo, unsigned elemCount,
          typename TLoadVector, typename TResultVector,
          typename TElemFunc, typename TFunc, typename TOp>
void TSSEEmulTest::Test_mm_convertop() {
    char data[16] = {
        '\xAA', '\x00', '\xFF', '\xCC', '\x11', '\x22', '\xBB', '\xAA',
        '\x33', '\x99', '\x44', '\x88', '\x55', '\x77', '\x66', '\x1C'};
    TFrom* datap = reinterpret_cast<TFrom*>(&data);

    TLoadVector value = TFuncLoad<TLoadVector>(&data);

    TTo procData[elemCount];
    for (unsigned i = 0; i < elemCount; ++i) {
        procData[i] = TElemFunc::Call(datap[i]);
    }

    TResultVector result = TFunc(value);

    for (unsigned i = 0; i < elemCount; ++i) {
        UNIT_ASSERT_EQUAL(procData[i], TQType<TOp>::As(result)[i]);
    }
}

void TSSEEmulTest::Test_mm_cvtepi32_ps() {
    struct THelper {
        static float Call(const i32 op) {
            return float(op);
        }
    };
    Test_mm_convertop<i32, float, 4, __m128i, __m128,
                      THelper, WrapF(_mm_cvtepi32_ps), float32x4_t>();
}

void TSSEEmulTest::Test_mm_cvtps_epi32() {
    struct THelper {
        static i32 Call(const float op) {
            return i32(op);
        }
    };
    Test_mm_convertop<float, i32, 4, __m128, __m128i,
                      THelper, T_mm_CallWrapper<__m128i, decltype(_mm_cvtps_epi32), _mm_cvtps_epi32>, int32x4_t>();
}

void TSSEEmulTest::Test_mm_cvttps_epi32() {
    struct THelper {
        static i32 Call(const float op) {
            return i32(op);
        }
    };
    Test_mm_convertop<float, i32, 4, __m128, __m128i,
                      THelper, Wrap(_mm_cvttps_epi32), int32x4_t>();
}

template <typename TLoadVector, typename TCastVector,
          typename TFunc, TFunc* func>
void TSSEEmulTest::Test_mm_castXX() {
    char data[16] = {
        '\xAA', '\x00', '\xFF', '\xCC', '\x11', '\x22', '\xBB', '\xAA',
        '\x33', '\x99', '\x44', '\x88', '\x55', '\x77', '\x66', '\x1C'};

    TLoadVector value = TFuncLoad<TLoadVector>(&data);
    const TLoadVector constvalue = TFuncLoad<TLoadVector>(&data);
    TCastVector casted = func(value);
    const TCastVector constcasted = func(constvalue);
    char verify[16];
    char constverify[16];
    TFuncStore<TCastVector>(&verify, casted);
    TFuncStore<TCastVector>(&constverify, constcasted);

    UNIT_ASSERT(memcmp(&data, &verify, 16) == 0);
    UNIT_ASSERT(memcmp(&data, &constverify, 16) == 0);
}

void TSSEEmulTest::Test_mm_castsi128_ps() {
    Test_mm_castXX<__m128i, __m128,
                   decltype(_mm_castsi128_ps), _mm_castsi128_ps>();
}

void TSSEEmulTest::Test_mm_castps_si128() {
    Test_mm_castXX<__m128, __m128i,
                   decltype(_mm_castps_si128), _mm_castps_si128>();
}

void TSSEEmulTest::Test_mm_mul_epu32() {
    char data0[16] = {
        '\xAA', '\x00', '\xFF', '\xCC', '\x11', '\x22', '\xBB', '\xAA',
        '\x33', '\x99', '\x44', '\x88', '\x55', '\x77', '\x66', '\x1C'};
    char data1[16] = {
        '\x99', '\x33', '\x1C', '\x55', '\x88', '\x66', '\x77', '\x44',
        '\x00', '\xAA', '\xAA', '\x11', '\xCC', '\xBB', '\x22', '\xFF'};
    ui32* dataw0 = reinterpret_cast<ui32*>(&data0);
    ui32* dataw1 = reinterpret_cast<ui32*>(&data1);

    __m128i value0 = _mm_loadu_si128((__m128i*)&data0);
    __m128i value1 = _mm_loadu_si128((__m128i*)&data1);

    ui64 mul0 = (ui64) dataw0[0] * (ui64) dataw1[0];
    ui64 mul1 = (ui64) dataw0[2] * (ui64) dataw1[2];

    __m128i result = _mm_mul_epu32(value0, value1);

    UNIT_ASSERT_EQUAL(mul0, TQType<uint64x2_t>::As(result)[0]);
    UNIT_ASSERT_EQUAL(mul1, TQType<uint64x2_t>::As(result)[1]);
}

void TSSEEmulTest::Test_mm_cmpunord_ps() {
    alignas(16) float valuesBits[4] = {1.f, 2.f, 3.f, 4.f};
    alignas(16) float values2Bits[4] = {5.f, 6.f, 7.f, 8.f};

    alignas(16) char allfs[16] = {
        '\xff', '\xff', '\xff', '\xff', '\xff', '\xff', '\xff', '\xff',
        '\xff', '\xff', '\xff', '\xff', '\xff', '\xff', '\xff', '\xff'
    };

    alignas(16) char allzeroes[16] = {
        '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00',
        '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00'
    };

    const __m128 qnan = _mm_set_ps1(std::numeric_limits<float>::quiet_NaN());
    const __m128 snan = _mm_set_ps1(std::numeric_limits<float>::signaling_NaN());
    const __m128 values = _mm_loadu_ps((const float*) valuesBits);
    const __m128 values2 = _mm_loadu_ps((const float*) values2Bits);

    const __m128 mask1 = _mm_cmpunord_ps(qnan, qnan);
    UNIT_ASSERT_EQUAL(::memcmp(&mask1, &allfs, sizeof(allfs)), 0);

    const __m128 mask2 = _mm_cmpunord_ps(values, values);
    UNIT_ASSERT_EQUAL(::memcmp(&mask2, &allzeroes, sizeof(allzeroes)), 0);

    const __m128 mask3 = _mm_cmpunord_ps(snan, snan);
    UNIT_ASSERT_EQUAL(::memcmp(&mask3, &allfs, sizeof(allfs)), 0);

    const __m128 mask4 = _mm_cmpunord_ps(qnan, values);
    UNIT_ASSERT_EQUAL(::memcmp(&mask4, &allfs, sizeof(allfs)), 0);

    const __m128 mask5 = _mm_cmpunord_ps(snan, values);
    UNIT_ASSERT_EQUAL(::memcmp(&mask5, &allfs, sizeof(allfs)), 0);

    const __m128 mask6 = _mm_cmpunord_ps(qnan, snan);
    UNIT_ASSERT_EQUAL(::memcmp(&mask6, &allfs, sizeof(allfs)), 0);

    const __m128 mask7 = _mm_cmpunord_ps(values, values2);
    UNIT_ASSERT_EQUAL(::memcmp(&mask7, &allzeroes, sizeof(allzeroes)), 0);
}

void TSSEEmulTest::Test_mm_store_ss() {
    alignas(16) const float valueBits[4] = {1.f, 2.f, 3.f, 4.f};
    const __m128 value = _mm_loadu_ps(valueBits);
    float res = std::numeric_limits<float>::signaling_NaN();
    _mm_store_ss(&res, value);
    UNIT_ASSERT_EQUAL(res, 1.f);
}

void TSSEEmulTest::Test_mm_store_ps() {
    alignas(16) const float valueBits[4] = {1.f, 2.f, 3.f, 4.f};
    const __m128 value = _mm_loadu_ps(valueBits);
    float res[4] = {0.f};
    _mm_storeu_ps(res, value);
    UNIT_ASSERT_EQUAL(res[0], 1.f);
    UNIT_ASSERT_EQUAL(res[1], 2.f);
    UNIT_ASSERT_EQUAL(res[2], 3.f);
    UNIT_ASSERT_EQUAL(res[3], 4.f);
}

void TSSEEmulTest::Test_mm_storeu_pd() {
    alignas(16) const double valueBits[4] = {1., 2., 3., 4.};
    for (size_t i = 0; i != 3; ++i) {
        const __m128d value = _mm_loadu_pd(&valueBits[i]);
        alignas(16) double res[4];
        for (size_t shift = 0; shift != 3; ++shift) {
            _mm_storeu_pd(&res[shift], value);
            for (size_t j = 0; j != 2; ++j) {
                UNIT_ASSERT_EQUAL_C(res[j + shift], valueBits[i + j], "res: " << HexEncode(&res[shift], 16) << " vs etalon: " << HexEncode(&valueBits[i], 16));
            }
        }
    }
}

void TSSEEmulTest::Test_mm_andnot_ps() {
    alignas(16) const char firstBits[16] = {
        '\x00', '\x00', '\xff', '\xff', '\x00', '\x00', '\xff', '\xff',
        '\x00', '\x00', '\xff', '\xff', '\x00', '\x00', '\xff', '\xff'
    };

    alignas(16) const char secondBits[16] = {
        '\x00', '\xff', '\x00', '\xff', '\x00', '\xff', '\x00', '\xff',
        '\x00', '\xff', '\x00', '\xff', '\x00', '\xff', '\x00', '\xff'
    };

    alignas(16) const char resBits[16] = {
        '\x00', '\xff', '\x00', '\x00', '\x00', '\xff', '\x00', '\x00',
        '\x00', '\xff', '\x00', '\x00', '\x00', '\xff', '\x00', '\x00'
    };

    const __m128 value1 = _mm_loadu_ps((const float*) firstBits);
    const __m128 value2 = _mm_loadu_ps((const float*) secondBits);
    const __m128 res = _mm_andnot_ps(value1, value2);

    UNIT_ASSERT_EQUAL(::memcmp(&res, resBits, sizeof(resBits)), 0);
}

void TSSEEmulTest::Test_mm_shuffle_ps() {
    alignas(16) const float first[4] = {1.f, 2.f, 3.f, 4.f};
    alignas(16) const float second[4] = {5.f, 6.f, 7.f, 8.f};
    alignas(16) const float etalon[4] = {3.f, 4.f, 5.f, 6.f};

    const __m128 value1 = _mm_loadu_ps(first);
    const __m128 value2 = _mm_loadu_ps(second);
    const __m128 res = _mm_shuffle_ps(value1, value2, _MM_SHUFFLE(1, 0, 3, 2));

    UNIT_ASSERT_EQUAL(::memcmp(&res, etalon, sizeof(etalon)), 0);
}

void TSSEEmulTest::Test_mm_shuffle_pd() {
    const double first[2] = {1.3, 2.3};
    const double second[2] = {5.3, 6.3};
    const double etalon0[2] = {1.3, 5.3};
    const double etalon1[2] = {2.3, 5.3};
    const double etalon2[2] = {1.3, 6.3};
    const double etalon3[2] = {2.3, 6.3};

    const __m128d value1 = _mm_loadu_pd(first);
    const __m128d value2 = _mm_loadu_pd(second);

    __m128d res = _mm_shuffle_pd(value1, value2, 0);
    UNIT_ASSERT_EQUAL(::memcmp(&res, etalon0, sizeof(etalon0)), 0);

    res = _mm_shuffle_pd(value1, value2, 1);
    UNIT_ASSERT_EQUAL(::memcmp(&res, etalon1, sizeof(etalon1)), 0);

    res = _mm_shuffle_pd(value1, value2, 2);
    UNIT_ASSERT_EQUAL(::memcmp(&res, etalon2, sizeof(etalon2)), 0);

    res = _mm_shuffle_pd(value1, value2, 3);
    UNIT_ASSERT_EQUAL(::memcmp(&res, etalon3, sizeof(etalon3)), 0);
}

void TSSEEmulTest::Test_mm_cvtsd_f64() {
    const double first[2] = {1.3, 2.3};
    const double second[2] = {5.3, 6.3};

    const __m128d value1 = _mm_loadu_pd(first);
    const __m128d value2 = _mm_loadu_pd(second);

    UNIT_ASSERT_EQUAL(_mm_cvtsd_f64(value1), 1.3);
    UNIT_ASSERT_EQUAL(_mm_cvtsd_f64(value2), 5.3);
}

void TSSEEmulTest::Test_mm_loadl_pd() {
    const double first[2] = {1.3, 2.3};
    const double second[2] = {5.3, 6.3};
    const double firstEtalon[2] = {10.13, 2.3};
    const double secondEtalon[2] = {11.13, 6.3};

    double newFirst = 10.13;
    double newSecond = 11.13;

    __m128d value1 = _mm_loadu_pd(first);
    __m128d value2 = _mm_loadu_pd(second);
    value1 = _mm_loadl_pd(value1, &newFirst);
    value2 = _mm_loadl_pd(value2, &newSecond);
    UNIT_ASSERT_EQUAL(::memcmp(&value1, firstEtalon, sizeof(firstEtalon)), 0);
    UNIT_ASSERT_EQUAL(::memcmp(&value2, secondEtalon, sizeof(secondEtalon)), 0);
}

void TSSEEmulTest::Test_mm_loadh_pd() {
    const double first[2] = {1.3, 2.3};
    const double second[2] = {5.3, 6.3};
    const double firstEtalon[2] = {1.3, 10.13};
    const double secondEtalon[2] = {5.3, 11.13};

    double newFirst = 10.13;
    double newSecond = 11.13;

    __m128d value1 = _mm_loadu_pd(first);
    __m128d value2 = _mm_loadu_pd(second);
    value1 = _mm_loadh_pd(value1, &newFirst);
    value2 = _mm_loadh_pd(value2, &newSecond);
    UNIT_ASSERT_EQUAL(::memcmp(&value1, firstEtalon, sizeof(firstEtalon)), 0);
    UNIT_ASSERT_EQUAL(::memcmp(&value2, secondEtalon, sizeof(secondEtalon)), 0);
}

void TSSEEmulTest::Test_mm_or_ps() {
    alignas(16) const char bytes1[16] = {
        '\x00', '\x00', '\xff', '\xff', '\x00', '\x00', '\xff', '\xff',
        '\x00', '\x00', '\xff', '\xff', '\x00', '\x00', '\xff', '\xff'
    };

    alignas(16) const char bytes2[16] = {
        '\x00', '\xff', '\x00', '\xff', '\x00', '\xff', '\x00', '\xff',
        '\x00', '\xff', '\x00', '\xff', '\x00', '\xff', '\x00', '\xff'
    };

    alignas(16) const char etalon[16] = {
        '\x00', '\xff', '\xff', '\xff', '\x00', '\xff', '\xff', '\xff',
        '\x00', '\xff', '\xff', '\xff', '\x00', '\xff', '\xff', '\xff'
    };

    const __m128 value1 = _mm_loadu_ps((const float*) bytes1);
    const __m128 value2 = _mm_loadu_ps((const float*) bytes2);
    const __m128 res = _mm_or_ps(value1, value2);

    UNIT_ASSERT_EQUAL(::memcmp(&res, etalon, sizeof(etalon)), 0);
}

void TSSEEmulTest::Test_mm_loadu_pd() {
    alignas(16) double stub[4] = {
        0.f, 1.f,
        2.f, 3.f
    };

    for (size_t shift = 0; shift != 3; ++shift) {
        const __m128d val = _mm_loadu_pd(&stub[shift]);
        alignas(16) double res[2];
        _mm_store_pd(res, val);

        for (size_t i = 0; i != 2; ++i) {
            UNIT_ASSERT_EQUAL_C(res[i], stub[shift + i], "res: " << HexEncode(res, 16) << " vs etalon: " << HexEncode(&stub[shift], 16));
        }
    }
}

void TSSEEmulTest::Test_mm_rsqrt_ps() {
    alignas(16) const char bytes[16] = {
        '\x00', '\x00', '\x28', '\x42', // 42.f
        '\x00', '\x98', '\x84', '\x45', // 4243.f
        '\x60', '\x26', '\xcf', '\x48', // 424243.f
        '\xed', '\xd5', '\x21', '\x4c'  // 42424243.f
    };
    const __m128 value = _mm_loadu_ps((const float*)bytes);
    const __m128 result = _mm_rsqrt_ps(value);
    alignas(16) float res[4];
    _mm_store_ps(res, result);
    float fResult = 0.f;
    for (size_t i = 0; i < 4; ++i) {
        memcpy(&fResult, &bytes[i * 4], 4);
        fResult = 1.f / std::sqrt(fResult);
        UNIT_ASSERT_DOUBLES_EQUAL_C(res[i], fResult, 1e-3, "res: " << fResult << " vs etalon " << res[i]);
    }
}

namespace NHelpers {

    static __m128i Y_FORCE_INLINE GetCmp16(const __m128 &c0, const __m128 &c1, const __m128 &c2, const __m128 &c3, const __m128 test) {
        const __m128i r0 = _mm_castps_si128(_mm_cmpgt_ps(c0, test));
        const __m128i r1 = _mm_castps_si128(_mm_cmpgt_ps(c1, test));
        const __m128i r2 = _mm_castps_si128(_mm_cmpgt_ps(c2, test));
        const __m128i r3 = _mm_castps_si128(_mm_cmpgt_ps(c3, test));
        const __m128i packed = _mm_packs_epi16(_mm_packs_epi32(r0, r1), _mm_packs_epi32(r2, r3));
        return _mm_and_si128(_mm_set1_epi8(0x01), packed);
    }

    static __m128i Y_FORCE_INLINE GetCmp16(const float *factors, const __m128 test) {
        const __m128 *ptr = (__m128 *)factors;
        return GetCmp16(ptr[0], ptr[1], ptr[2], ptr[3], test);
    }

    template<size_t Num>
    void DoLane(size_t length, const float *factors, ui32 *& dst, const float *&values) {
        for (size_t i = 0; i < length; ++i) {
            __m128 value = _mm_set1_ps(values[i]);
            __m128i agg = GetCmp16(factors, value);
            if (Num > 1) {
                agg = _mm_add_epi16(agg, _mm_slli_epi16(GetCmp16(&factors[64], value), 1));
            }
            _mm_store_si128((__m128i *)&dst[4 * i], agg);
        }
    }
}

void TSSEEmulTest::Test_matrixnet_powerpc() {
    static constexpr size_t length = 10;
    alignas(16) float factors[1024];
    alignas(16) ui32 valP[4 * length] = { 0 };
    float values[length];
    TReallyFastRng32 rng(42);
    for (size_t i = 0; i < 1024; ++i) {
        factors[i] = rng.GenRandReal2();
    }
    for (size_t i = 0; i < length; ++i) {
        values[i] = rng.GenRandReal2();
    }
    ui32* val = reinterpret_cast<ui32*>(valP);
    const float* vals = reinterpret_cast<const float*>(values);
    NHelpers::DoLane<2>(length, factors, val, vals);
    static const ui32 etalon[4 * length] = {
        2, 33554432, 258, 33554433, 50529027,
        50529027, 50529027, 50529027, 50528770,
        33685763, 33555203, 50462723, 50528770,
        33685763, 33555203, 50462723, 50529026,
        33751299, 50529027, 50463491, 2, 33554432,
        258, 33554433, 50397698, 33685761, 259,
        50462721, 50332162, 33554689, 259, 50462721,
        50528770, 33685761, 33555203, 50462723,
        50529026, 33685763, 50463491, 50463235
    };
    for (size_t i = 0; i < 4 * length; ++i) {
        UNIT_ASSERT_EQUAL(valP[i], etalon[i]);
    }
}
