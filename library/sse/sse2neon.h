#pragma once

/*
  The header contains inlining code
  which translates SSE intrinsics to NEON intrinsics or software emulation.
  You are encouraged for commitments.
  Add missing intrinsics, add unittests, purify the implementation,
  merge and simplify templates.
  Warning: The code is made in deep nights, so it surely contains bugs,
  imperfections, flaws and all other kinds of errors and mistakes.
*/
/* Author: Vitaliy Manushkin <agri@yandex-team.ru> */

#include <util/system/platform.h>
#include <util/system/compiler.h>
#include <util/system/types.h>

#if !defined(_arm64_)
#error "This header is for ARM64 (aarch64) platform only. " \
    "Include sse.h instead of including this header directly."
#endif

#include <arm_neon.h>

union __m128i {
    uint64x2_t AsUi64x2;
    int64x2_t AsSi64x2;

    uint32x4_t AsUi32x4;
    int32x4_t AsSi32x4;

    uint16x8_t AsUi16x8;
    int16x8_t AsSi16x8;

    uint8x16_t AsUi8x16;
    int8x16_t AsSi8x16;

    float32x4_t AsFloat32x4;
    float64x2_t AsFloat64x2;
};

union __m128 {
    float32x4_t AsFloat32x4;
    float64x2_t AsFloat64x2;

    uint32x4_t AsUi32x4;
    int32x4_t AsSi32x4;

    uint64x2_t AsUi64x2;
    int64x2_t AsSi64x2;

    uint8x16_t AsUi8x16;
    int8x16_t AsSi8x16;

    __m128i As128i;
};

typedef float64x2_t __m128d;

enum _mm_hint
{
  /* _MM_HINT_ET is _MM_HINT_T with set 3rd bit.  */
  _MM_HINT_ET0 = 7,
  _MM_HINT_ET1 = 6,
  _MM_HINT_T0 = 3,
  _MM_HINT_T1 = 2,
  _MM_HINT_T2 = 1,
  _MM_HINT_NTA = 0
};

Y_FORCE_INLINE void _mm_prefetch(const void *p, enum _mm_hint) {
    __builtin_prefetch(p);
}

template <typename TType>
struct TQType;

template <>
struct TQType<uint8x16_t> {
    static inline uint8x16_t& As(__m128i& value) {
        return value.AsUi8x16;
    }
    static inline const uint8x16_t& As(const __m128i& value) {
        return value.AsUi8x16;
    }
};

template <>
struct TQType<int8x16_t> {
    static inline int8x16_t& As(__m128i& value) {
        return value.AsSi8x16;
    }
    static inline const int8x16_t& As(const __m128i& value) {
        return value.AsSi8x16;
    }
};

template <>
struct TQType<uint16x8_t> {
    static inline uint16x8_t& As(__m128i& value) {
        return value.AsUi16x8;
    }
    static inline const uint16x8_t& As(const __m128i& value) {
        return value.AsUi16x8;
    }
};

template <>
struct TQType<int16x8_t> {
    static inline int16x8_t& As(__m128i& value) {
        return value.AsSi16x8;
    }
    static inline const int16x8_t& As(const __m128i& value) {
        return value.AsSi16x8;
    }
};

template <>
struct TQType<uint32x4_t> {
    static inline uint32x4_t& As(__m128i& value) {
        return value.AsUi32x4;
    }
    static inline const uint32x4_t& As(const __m128i& value) {
        return value.AsUi32x4;
    }
};

template <>
struct TQType<int32x4_t> {
    static inline int32x4_t& As(__m128i& value) {
        return value.AsSi32x4;
    }
    static inline const int32x4_t& As(const __m128i& value) {
        return value.AsSi32x4;
    }
};

template <>
struct TQType<uint64x2_t> {
    static inline uint64x2_t& As(__m128i& value) {
        return value.AsUi64x2;
    }
    static inline const uint64x2_t& As(const __m128i& value) {
        return value.AsUi64x2;
    }
    static inline uint64x2_t& As(__m128& value) {
        return value.AsUi64x2;
    }
    static inline const uint64x2_t& As(const __m128& value) {
        return value.AsUi64x2;
    }
};

template <>
struct TQType<int64x2_t> {
    static inline int64x2_t& As(__m128i& value) {
        return value.AsSi64x2;
    }
    static inline const int64x2_t& As(const __m128i& value) {
        return value.AsSi64x2;
    }
};

template <typename TValue>
struct TBaseWrapper {
    TValue Value;

    Y_FORCE_INLINE
    operator TValue&() {
        return Value;
    }

    Y_FORCE_INLINE
    operator const TValue&() const {
        return Value;
    }
};

template <typename TOp, typename TFunc, TFunc* func,
          typename TDup, TDup* dupfunc>
struct TWrapperSingleDup: public TBaseWrapper<__m128i> {
    Y_FORCE_INLINE
    TWrapperSingleDup(const __m128i& op, const int shift) {
        TQType<TOp>::As(Value) = func(TQType<TOp>::As(op), dupfunc(shift));
    }
};

template <typename TOp, typename TFunc, TFunc* func,
          typename TDup, TDup* dupfunc>
struct TWrapperSingleNegDup: public TBaseWrapper<__m128i> {
    Y_FORCE_INLINE
    TWrapperSingleNegDup(const __m128i& op, const int shift) {
        TQType<TOp>::As(Value) = func(TQType<TOp>::As(op), dupfunc(-shift));
    }
};

inline __m128i _mm_srl_epi16(__m128i a, __m128i count) {
    __m128i res;
    res.AsUi16x8 = vshlq_u16(a.AsUi16x8, vdupq_n_s16(-count.AsUi16x8[0]));
    return res;
}


inline __m128i _mm_srl_epi32(__m128i a, __m128i count) {
    __m128i res;
    res.AsUi32x4 = vshlq_u32(a.AsUi32x4, vdupq_n_s32(-count.AsUi32x4[0]));
    return res;
}

inline __m128i _mm_srl_epi64(__m128i a, __m128i count) {
    __m128i res;
    res.AsUi64x2 = vshlq_u64(a.AsUi64x2, vdupq_n_s64(-count.AsUi64x2[0]));
    return res;
}

inline __m128i _mm_srai_epi16(__m128i a, int count) {
    __m128i res;
    res.AsSi16x8 = vqshlq_s16(a.AsSi16x8, vdupq_n_s16(-count));
    return res;
}

inline __m128i _mm_srai_epi32(__m128i a, int count) {
    __m128i res;
    res.AsSi32x4 = vqshlq_s32(a.AsSi32x4, vdupq_n_s32(-count));
    return res;
}

using _mm_srli_epi16 =
    TWrapperSingleNegDup<uint16x8_t, decltype(vshlq_u16), vshlq_u16,
                         decltype(vdupq_n_s16), vdupq_n_s16>;
using _mm_srli_epi32 =
    TWrapperSingleNegDup<uint32x4_t, decltype(vshlq_u32), vshlq_u32,
                         decltype(vdupq_n_s32), vdupq_n_s32>;
using _mm_srli_epi64 =
    TWrapperSingleNegDup<uint64x2_t, decltype(vshlq_u64), vshlq_u64,
                         decltype(vdupq_n_s64), vdupq_n_s64>;


inline __m128i _mm_sll_epi16(__m128i a, __m128i count) {
    __m128i res;
    res.AsUi16x8 = vshlq_u16(a.AsUi16x8, vdupq_n_s16(count.AsUi16x8[0]));
    return res;
}


inline __m128i _mm_sll_epi32(__m128i a, __m128i count) {
    __m128i res;
    res.AsUi32x4 = vshlq_u32(a.AsUi32x4, vdupq_n_s32(count.AsUi32x4[0]));
    return res;
}

inline __m128i _mm_sll_epi64(__m128i a, __m128i count) {
    __m128i res;
    res.AsUi64x2 = vshlq_u64(a.AsUi64x2, vdupq_n_s64(count.AsUi64x2[0]));
    return res;
}

using _mm_slli_epi16 =
    TWrapperSingleDup<uint16x8_t, decltype(vshlq_u16), vshlq_u16,
                      decltype(vdupq_n_s16), vdupq_n_s16>;
using _mm_slli_epi32 =
    TWrapperSingleDup<uint32x4_t, decltype(vshlq_u32), vshlq_u32,
                      decltype(vdupq_n_s32), vdupq_n_s32>;
using _mm_slli_epi64 =
    TWrapperSingleDup<uint64x2_t, decltype(vshlq_u64), vshlq_u64,
                      decltype(vdupq_n_s64), vdupq_n_s64>;

template <typename TOp, typename TFunc, TFunc* func, typename... TParams>
struct TWrapperDual : TBaseWrapper<__m128i> {
    Y_FORCE_INLINE
    TWrapperDual(const __m128i& op1, const __m128i& op2, TParams... params) {
        TQType<TOp>::As(Value) = (TOp)
            func(TQType<TOp>::As(op1),
                 TQType<TOp>::As(op2),
                 params...);
    }
};

template <typename TOp, typename TFunc, TFunc* func, typename... TParams>
struct TWrapperDualSwap : TBaseWrapper<__m128i> {
    Y_FORCE_INLINE
    TWrapperDualSwap(const __m128i& op1, const __m128i& op2, TParams... params) {
        TQType<TOp>::As(Value) =
            func(TQType<TOp>::As(op2),
                 TQType<TOp>::As(op1),
                 params...);
    }
};

template <typename TOp, typename TFunc, TFunc* func, typename TArgument = __m128>
struct TWrapperDualF : TBaseWrapper<TArgument> {
    Y_FORCE_INLINE
    TWrapperDualF(const TArgument& op1, const TArgument& op2) {
        TQType<TOp>::As(TBaseWrapper<TArgument>::Value) = (TOp) func(TQType<TOp>::As(op1), TQType<TOp>::As(op2));
    }
};

using _mm_or_si128 = TWrapperDual<uint64x2_t, decltype(vorrq_u64), vorrq_u64>;
using _mm_and_si128 = TWrapperDual<uint64x2_t, decltype(vandq_u64), vandq_u64>;
using _mm_andnot_si128 =
    TWrapperDualSwap<uint64x2_t, decltype(vbicq_u64), vbicq_u64>;
using _mm_xor_si128 = TWrapperDual<uint64x2_t, decltype(veorq_u64), veorq_u64>;

using _mm_add_epi8 = TWrapperDual<uint8x16_t, decltype(vaddq_u8), vaddq_u8>;
using _mm_add_epi16 = TWrapperDual<uint16x8_t, decltype(vaddq_u16), vaddq_u16>;
using _mm_add_epi32 = TWrapperDual<uint32x4_t, decltype(vaddq_u32), vaddq_u32>;
using _mm_add_epi64 = TWrapperDual<uint64x2_t, decltype(vaddq_u64), vaddq_u64>;

inline __m128i _mm_madd_epi16(__m128i a, __m128i b) {
    int32x4_t aLow;
    int32x4_t aHigh;
    int32x4_t bLow;
    int32x4_t bHigh;
    #ifdef __LITTLE_ENDIAN__
        aLow[0] = a.AsSi16x8[0]; //!< I couldn't find vector instructions to do that. Feel free to fix this code.
        aLow[1] = a.AsSi16x8[2];
        aLow[2] = a.AsSi16x8[4];
        aLow[3] = a.AsSi16x8[6];

        aHigh[0] = a.AsSi16x8[1];
        aHigh[1] = a.AsSi16x8[3];
        aHigh[2] = a.AsSi16x8[5];
        aHigh[3] = a.AsSi16x8[7];

        bLow[0] = b.AsSi16x8[0];
        bLow[1] = b.AsSi16x8[2];
        bLow[2] = b.AsSi16x8[4];
        bLow[3] = b.AsSi16x8[6];

        bHigh[0] = b.AsSi16x8[1];
        bHigh[1] = b.AsSi16x8[3];
        bHigh[2] = b.AsSi16x8[5];
        bHigh[3] = b.AsSi16x8[7];
    #else
        #error Not implemented yet. Do it yourself.
    #endif

    const int32x4_t lowMul = vmulq_u32(aLow, bLow);
    const int32x4_t highMul = vmulq_u32(aHigh, bHigh);
    __m128i res;
    res.AsSi32x4 = vaddq_u32(lowMul, highMul);
    return res;
}

using _mm_sub_epi8 = TWrapperDual<uint8x16_t, decltype(vsubq_u8), vsubq_u8>;
using _mm_sub_epi16 = TWrapperDual<uint16x8_t, decltype(vsubq_u16), vsubq_u16>;
using _mm_sub_epi32 = TWrapperDual<uint32x4_t, decltype(vsubq_u32), vsubq_u32>;
using _mm_sub_epi64 = TWrapperDual<uint64x2_t, decltype(vsubq_u64), vsubq_u64>;

using _mm_unpacklo_epi8 =
    TWrapperDual<uint8x16_t, decltype(vzip1q_u8), vzip1q_u8>;
using _mm_unpackhi_epi8 =
    TWrapperDual<uint8x16_t, decltype(vzip2q_u8), vzip2q_u8>;
using _mm_unpacklo_epi16 =
    TWrapperDual<uint16x8_t, decltype(vzip1q_u16), vzip1q_u16>;
using _mm_unpackhi_epi16 =
    TWrapperDual<uint16x8_t, decltype(vzip2q_u16), vzip2q_u16>;
using _mm_unpacklo_epi32 =
    TWrapperDual<uint32x4_t, decltype(vzip1q_u32), vzip1q_u32>;
using _mm_unpackhi_epi32 =
    TWrapperDual<uint32x4_t, decltype(vzip2q_u32), vzip2q_u32>;
using _mm_unpacklo_epi64 =
    TWrapperDual<uint64x2_t, decltype(vzip1q_u64), vzip1q_u64>;
using _mm_unpackhi_epi64 =
    TWrapperDual<uint64x2_t, decltype(vzip2q_u64), vzip2q_u64>;

using _mm_cmpeq_epi8 =
    TWrapperDual<uint8x16_t, decltype(vceqq_u8), vceqq_u8>;
using _mm_cmpeq_epi16 =
    TWrapperDual<uint16x8_t, decltype(vceqq_u16), vceqq_u16>;
using _mm_cmpeq_epi32 =
    TWrapperDual<uint32x4_t, decltype(vceqq_u32), vceqq_u32>;

using _mm_cmpgt_epi8 =
    TWrapperDual<int8x16_t, decltype(vcgtq_s8), vcgtq_s8>;
using _mm_cmpgt_epi16 =
    TWrapperDual<int16x8_t, decltype(vcgtq_s16), vcgtq_s16>;
using _mm_cmpgt_epi32 =
    TWrapperDual<int32x4_t, decltype(vcgtq_s32), vcgtq_s32>;

using _mm_cmplt_epi8 =
    TWrapperDual<int8x16_t, decltype(vcltq_s8), vcltq_s8>;
using _mm_cmplt_epi16 =
    TWrapperDual<int16x8_t, decltype(vcltq_s16), vcltq_s16>;
using _mm_cmplt_epi32 =
    TWrapperDual<int32x4_t, decltype(vcltq_s32), vcltq_s32>;

Y_FORCE_INLINE __m128i _mm_load_si128(const __m128i* ptr) {
    __m128i result;
    result.AsUi64x2 = vld1q_u64((const uint64_t*)ptr);
    return result;
}

Y_FORCE_INLINE __m128i _mm_loadu_si128(const __m128i* ptr) {
    __m128i result;
    result.AsUi64x2 = vld1q_u64((const uint64_t*)ptr);
    return result;
}

Y_FORCE_INLINE __m128i _mm_lddqu_si128(const __m128i* ptr) {
    return _mm_loadu_si128(ptr);
}

Y_FORCE_INLINE void _mm_storeu_si128(__m128i* ptr, const __m128i& op) {
    vst1q_u64((uint64_t*)ptr, op.AsUi64x2);
}

Y_FORCE_INLINE void
_mm_store_si128(__m128i* ptr, const __m128i& op) {
    vst1q_u64((uint64_t*)ptr, op.AsUi64x2);
}

template <typename TOp, typename TFunc, TFunc* func, typename... TParams>
struct TWrapperSimple : TBaseWrapper<__m128i> {
    Y_FORCE_INLINE
    TWrapperSimple(TParams... params) {
        TQType<TOp>::As(Value) = func(params...);
    }
};

template <typename TOp, typename TFunc, TFunc* func, typename... TParams>
struct TWrapperSimpleF : TBaseWrapper<__m128> {
    Y_FORCE_INLINE
    TWrapperSimpleF(TParams... params) {
        TQType<TOp>::As(Value) = func(params...);
    }
};

using _mm_set1_epi8 =
    TWrapperSimple<int8x16_t, decltype(vdupq_n_s8), vdupq_n_s8, const char>;
using _mm_set1_epi16 =
    TWrapperSimple<int16x8_t, decltype(vdupq_n_s16), vdupq_n_s16, const ui16>;
using _mm_set1_epi32 =
    TWrapperSimple<int32x4_t, decltype(vdupq_n_s32), vdupq_n_s32, const ui32>;

struct _mm_setzero_si128 : TBaseWrapper<__m128i> {
    Y_FORCE_INLINE
    _mm_setzero_si128() {
        TQType<uint64x2_t>::As(Value) = vdupq_n_u64(0);
    }
};

struct _mm_loadl_epi64 : TBaseWrapper<__m128i> {
    Y_FORCE_INLINE
    _mm_loadl_epi64(const __m128i* p) {
        uint64x1_t im = vld1_u64((const uint64_t*)p);
        TQType<uint64x2_t>::As(Value) = vcombine_u64(im, vdup_n_u64(0));
    }
};

struct _mm_storel_epi64 : TBaseWrapper<__m128i> {
    Y_FORCE_INLINE
    _mm_storel_epi64(__m128i* a, __m128i op) {
        vst1_u64((uint64_t*)a, vget_low_u64(op.AsUi64x2));
    }
};

struct ShuffleStruct4 {
    ui8 x[4];
};

Y_FORCE_INLINE ShuffleStruct4
_MM_SHUFFLE(ui8 x4, ui8 x3, ui8 x2, ui8 x1) {
    ShuffleStruct4 result;
    result.x[0] = x1;
    result.x[1] = x2;
    result.x[2] = x3;
    result.x[3] = x4;
    return result;
}

Y_FORCE_INLINE __m128i
_mm_shuffle_epi32(const __m128i& op1, const ShuffleStruct4& op2) {
    __m128i result;
    const ui8 xi[4] = {
        ui8(op2.x[0] * 4), ui8(op2.x[1] * 4),
        ui8(op2.x[2] * 4), ui8(op2.x[3] * 4)
    };
    const uint8x16_t transform = {
        ui8(xi[0]), ui8(xi[0] + 1), ui8(xi[0] + 2), ui8(xi[0] + 3),
        ui8(xi[1]), ui8(xi[1] + 1), ui8(xi[1] + 2), ui8(xi[1] + 3),
        ui8(xi[2]), ui8(xi[2] + 1), ui8(xi[2] + 2), ui8(xi[2] + 3),
        ui8(xi[3]), ui8(xi[3] + 1), ui8(xi[3] + 2), ui8(xi[3] + 3)
    };
    result.AsUi8x16 = vqtbl1q_u8(op1.AsUi8x16, transform);
    return result;
}

Y_FORCE_INLINE int
_mm_movemask_epi8(const __m128i& op) {
    uint8x16_t mask = {0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
                       0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80};
    uint8x16_t opmasked = vandq_u8(op.AsUi8x16, mask);
    int8x16_t byteshifter = {
        0, -7, 0, -7, 0, -7, 0, -7, 0, -7, 0, -7, 0, -7, 0, -7};
    uint8x16_t opshifted = vshlq_u8(opmasked, byteshifter);
    int16x8_t wordshifter = {-7, -5, -3, -1, 1, 3, 5, 7};
    uint16x8_t wordshifted =
        vshlq_u16(vreinterpretq_u16_u8(opshifted), wordshifter);
    return vaddvq_u16(wordshifted);
}

template <int imm>
struct THelper_mm_srli_si128 : TBaseWrapper<__m128i> {
    Y_FORCE_INLINE
    THelper_mm_srli_si128(const __m128i a) {
        const auto zero = vdupq_n_u8(0);
        TQType<uint8x16_t>::As(Value) = vextq_u8(a.AsUi8x16, zero, imm);
    }
};

template <>
struct THelper_mm_srli_si128<16> : TBaseWrapper<__m128i> {
    Y_FORCE_INLINE
    THelper_mm_srli_si128(const __m128i /* a */) {
        const auto zero = vdupq_n_u8(0);
        TQType<uint8x16_t>::As(Value) = zero;
    }
};

#define _mm_srli_si128(a, imm) THelper_mm_srli_si128<imm>(a)

template<int imm>
inline uint8x16_t vextq_u8_function(uint8x16_t a, uint8x16_t b) {
    return vextq_u8(a, b, imm);
}

template<>
inline uint8x16_t vextq_u8_function<16>(uint8x16_t /* a */, uint8x16_t b) {
    return b;
}


template <int imm>
struct THelper_mm_slli_si128 : TBaseWrapper<__m128i> {
    Y_FORCE_INLINE
    THelper_mm_slli_si128(const __m128i a) {
        auto zero = vdupq_n_u8(0);
        TQType<uint8x16_t>::As(Value) = vextq_u8_function<16 - imm>(zero, a.AsUi8x16);
    }
};

#define _mm_slli_si128(a, imm) THelper_mm_slli_si128<imm>(a)

Y_FORCE_INLINE int _mm_cvtsi128_si32(const __m128i& op) {
    return vgetq_lane_s32(op.AsSi32x4, 0);
}

struct _mm_set_epi16 : TBaseWrapper<__m128i> {
    Y_FORCE_INLINE
    _mm_set_epi16(const short w7, const short w6,
                  const short w5, const short w4,
                  const short w3, const short w2,
                  const short w1, const short w0) {
        int16x4_t d0 = {w0, w1, w2, w3};
        int16x4_t d1 = {w4, w5, w6, w7};
        TQType<int16x8_t>::As(Value) = vcombine_s16(d0, d1);
    }
};

struct _mm_setr_epi16 : TBaseWrapper<__m128i> {
    Y_FORCE_INLINE
    _mm_setr_epi16(const short w7, const short w6,
                  const short w5, const short w4,
                  const short w3, const short w2,
                  const short w1, const short w0) {
        int16x4_t d0 = {w7, w6, w5, w4};
        int16x4_t d1 = {w3, w2, w1, w0};
        TQType<int16x8_t>::As(Value) = vcombine_s16(d0, d1);
    }
};

struct _mm_set_epi32 : TBaseWrapper<__m128i> {
    Y_FORCE_INLINE
    _mm_set_epi32(const int x3, const int x2,
                  const int x1, const int x0) {
        int32x2_t d0 = {x0, x1};
        int32x2_t d1 = {x2, x3};
        TQType<int32x4_t>::As(Value) = vcombine_s32(d0, d1);
    }
};

struct _mm_setr_epi32 : TBaseWrapper<__m128i> {
    Y_FORCE_INLINE
    _mm_setr_epi32(const int x3, const int x2,
                  const int x1, const int x0) {
        int32x2_t d0 = {x3, x2};
        int32x2_t d1 = {x1, x0};
        TQType<int32x4_t>::As(Value) = vcombine_s32(d0, d1);
    }
};

struct _mm_cvtsi32_si128 : TBaseWrapper<__m128i> {
    Y_FORCE_INLINE
    _mm_cvtsi32_si128(int op) {
        auto zero = vdupq_n_s32(0);
        TQType<int32x4_t>::As(Value) = vsetq_lane_s32(op, zero, 0);
    }
};

struct _mm_cvtsi64_si128 : TBaseWrapper<__m128i> {
    Y_FORCE_INLINE
    _mm_cvtsi64_si128(i64 op) {
        auto zero = vdupq_n_s64(0);
        TQType<int64x2_t>::As(Value) = vsetq_lane_s64(op, zero, 0);
    }
};

template <typename TOpOut, typename TOpIn,
          typename TFunc, TFunc* func,
          typename TCombine, TCombine* combine>
struct TCombineWrapper : TBaseWrapper<__m128i> {
    Y_FORCE_INLINE
    TCombineWrapper(const __m128i op1, const __m128i op2) {
        TQType<TOpOut>::As(Value) =
            combine(func(TQType<TOpIn>::As(op1)),
                    func(TQType<TOpIn>::As(op2)));
    }
};

using _mm_packs_epi16 =
    TCombineWrapper<int8x16_t, int16x8_t,
                    decltype(vqmovn_s16), vqmovn_s16,
                    decltype(vcombine_s8), vcombine_s8>;
using _mm_packs_epi32 =
    TCombineWrapper<int16x8_t, int32x4_t,
                    decltype(vqmovn_s32), vqmovn_s32,
                    decltype(vcombine_s16), vcombine_s16>;
using _mm_packus_epi16 =
    TCombineWrapper<uint8x16_t, int16x8_t,
                    decltype(vqmovun_s16), vqmovun_s16,
                    decltype(vcombine_u8), vcombine_u8>;

template <typename TOpOut, typename TOpIn,
          typename TFunc, TFunc* func, typename... TParams>
struct TScalarOutWrapper : TBaseWrapper<TOpOut> {
    Y_FORCE_INLINE
    TScalarOutWrapper(const __m128i op, TParams... params) {
        TBaseWrapper<TOpOut>::Value =
            func(TQType<TOpIn>::As(op), params...);
    }
};

template<int imm>
int extract_epi8_arm(__m128i arg) {
    return vgetq_lane_u8(arg.AsUi8x16, imm);
}

template<int imm>
int extract_epi16_arm(__m128i arg) {
    return vgetq_lane_u16(arg.AsUi16x8, imm);
}

template<int imm>
int extract_epi32_arm(__m128i arg) {
    return vgetq_lane_s32(arg.AsSi32x4, imm);
}

template<int imm>
long long extract_epi64_arm(__m128i arg) {
    return vgetq_lane_s64(arg.AsSi64x2, imm);
}

#define _mm_extract_epi8(op, imm) extract_epi8_arm<imm>(op)
#define _mm_extract_epi16(op, imm) extract_epi16_arm<imm>(op)
#define _mm_extract_epi32(op, imm) extract_epi32_arm<imm>(op)
#define _mm_extract_epi64(op, imm) extract_epi64_arm<imm>(op)
#define _mm_extract_ps(op, imm) _mm_extract_epi32(op, imm)

static Y_FORCE_INLINE
__m128i _mm_mul_epu32(__m128i op1, __m128i op2) {
    __m128i result;
    uint32x4_t r1 = vuzp1q_u32(op1.AsUi32x4, op2.AsUi32x4);
    uint32x4_t r2 = vuzp1q_u32(op2.AsUi32x4, op1.AsUi32x4);
    result.AsUi64x2 = vmull_u32(vget_low_u32(r1), vget_low_u32(r2));
    return result;
}

template <>
struct TQType<float32x4_t> {
    static inline float32x4_t& As(__m128& value) {
        return value.AsFloat32x4;
    }

    static inline const float32x4_t& As(const __m128& value) {
        return value.AsFloat32x4;
    }

    static inline float32x4_t& As(__m128i& value) {
        return value.AsFloat32x4;
    }

    static inline const float32x4_t& As(const __m128i& value) {
        return value.AsFloat32x4;
    }
};

template <>
struct TQType<float64x2_t> {
    static inline float64x2_t& As(__m128& value) {
        return value.AsFloat64x2;
    }

    static inline const float64x2_t& As(const __m128& value) {
        return value.AsFloat64x2;
    }

    static inline float64x2_t& As(__m128i& value) {
        return value.AsFloat64x2;
    }

    static inline const float64x2_t& As(const __m128i& value) {
        return value.AsFloat64x2;
    }

    static inline float64x2_t& As(__m128d& value) {
        return value;
    }

    static inline const float64x2_t& As(const __m128d& value) {
        return value;
    }
};

using _mm_set1_ps = TWrapperSimpleF<float32x4_t,
                                    decltype(vdupq_n_f32), vdupq_n_f32, const float>;
using _mm_set_ps1 = TWrapperSimpleF<float32x4_t,
                                    decltype(vdupq_n_f32), vdupq_n_f32, const float>;

struct _mm_setzero_ps : TBaseWrapper<__m128> {
    Y_FORCE_INLINE
    _mm_setzero_ps() {
        TQType<float32x4_t>::As(Value) = vdupq_n_f32(0.);
    }
};

Y_FORCE_INLINE __m128d _mm_setzero_pd() {
    return vdupq_n_f64(0.);
}

Y_FORCE_INLINE __m128 _mm_loadu_ps(const float* ptr) {
    __m128 result;
    result.AsFloat32x4 = vld1q_f32(ptr);
    return result;
}

Y_FORCE_INLINE __m128 _mm_load_ps(const float* ptr) {
    __m128 result;
    result.AsFloat32x4 = vld1q_f32(ptr);
    return result;
}

Y_FORCE_INLINE void _mm_storeu_ps(float* ptr, const __m128& op) {
    vst1q_f32(ptr, op.AsFloat32x4);
}

Y_FORCE_INLINE void _mm_store_ps(float* ptr, const __m128& op) {
    vst1q_f32(ptr, op.AsFloat32x4);
}

struct _mm_set_ps : TBaseWrapper<__m128> {
    Y_FORCE_INLINE
    _mm_set_ps(const float x3, const float x2,
               const float x1, const float x0) {
        float32x2_t d0 = {x0, x1};
        float32x2_t d1 = {x2, x3};
        TQType<float32x4_t>::As(Value) = vcombine_f32(d0, d1);
    }
};

Y_FORCE_INLINE __m128d _mm_set_pd(double d1, double d0) {
    const float64x1_t p0 = {d0};
    const float64x1_t p1 = {d1};
    return vcombine_f64(p0, p1);
}

Y_FORCE_INLINE __m128d _mm_loadu_pd(const double* d) {
    __m128d res;
    res = vld1q_f64(d);
    return res;
}

Y_FORCE_INLINE void _mm_storeu_pd(double* res, __m128d a) {
    vst1q_f64(res, a);
}

Y_FORCE_INLINE void _mm_store_pd(double* res, __m128d a) {
    vst1q_f64(res, a);
}

using _mm_add_ps = TWrapperDualF<float32x4_t, decltype(vaddq_f32), vaddq_f32>;
using _mm_sub_ps = TWrapperDualF<float32x4_t, decltype(vsubq_f32), vsubq_f32>;
using _mm_mul_ps = TWrapperDualF<float32x4_t, decltype(vmulq_f32), vmulq_f32>;
using _mm_div_ps = TWrapperDualF<float32x4_t, decltype(vdivq_f32), vdivq_f32>;
using _mm_cmpeq_ps = TWrapperDualF<float32x4_t, decltype(vceqq_f32), vceqq_f32>;
using _mm_cmpgt_ps = TWrapperDualF<float32x4_t, decltype(vcgtq_f32), vcgtq_f32>;
using _mm_max_ps = TWrapperDualF<float32x4_t, decltype(vmaxq_f32), vmaxq_f32>;
using _mm_min_ps = TWrapperDualF<float32x4_t, decltype(vminq_f32), vminq_f32>;

using _mm_add_pd = TWrapperDualF<float64x2_t, decltype(vaddq_f64), vaddq_f64, __m128d>;
using _mm_sub_pd = TWrapperDualF<float64x2_t, decltype(vsubq_f64), vsubq_f64, __m128d>;
using _mm_mul_pd = TWrapperDualF<float64x2_t, decltype(vmulq_f64), vmulq_f64, __m128d>;
using _mm_div_pd = TWrapperDualF<float64x2_t, decltype(vdivq_f64), vdivq_f64, __m128d>;

struct _mm_and_ps : TBaseWrapper<__m128> {
    Y_FORCE_INLINE
    _mm_and_ps(const __m128& op1, const __m128& op2) {
        TQType<uint64x2_t>::As(Value) =
            vandq_u64(TQType<uint64x2_t>::As(op1),
                      TQType<uint64x2_t>::As(op2));
    }
};

Y_FORCE_INLINE __m128d _mm_and_pd(__m128d a, __m128d b) {
    return vandq_u64(a, b);
}

Y_FORCE_INLINE void _MM_TRANSPOSE4_PS(__m128& op0, __m128& op1, __m128& op2, __m128& op3) {
    float64x2_t im0 =
        (float64x2_t)vtrn1q_f32(op0.AsFloat32x4, op1.AsFloat32x4);
    float64x2_t im1 =
        (float64x2_t)vtrn2q_f32(op0.AsFloat32x4, op1.AsFloat32x4);
    float64x2_t im2 =
        (float64x2_t)vtrn1q_f32(op2.AsFloat32x4, op3.AsFloat32x4);
    float64x2_t im3 =
        (float64x2_t)vtrn2q_f32(op2.AsFloat32x4, op3.AsFloat32x4);

    TQType<float64x2_t>::As(op0) = vtrn1q_f64(im0, im2);
    TQType<float64x2_t>::As(op1) = vtrn1q_f64(im1, im3);
    TQType<float64x2_t>::As(op2) = vtrn2q_f64(im0, im2);
    TQType<float64x2_t>::As(op3) = vtrn2q_f64(im1, im3);
};

Y_FORCE_INLINE __m128 _mm_castsi128_ps(__m128i op) {
    return reinterpret_cast<__m128&>(op);
}

Y_FORCE_INLINE __m128i _mm_castps_si128(__m128 op) {
    return reinterpret_cast<__m128i&>(op);
}

template <typename TOpOut, typename TOpIn,
          typename TFunc, TFunc* func, typename... TParams>
struct TCvtS2FWrapperSingle : TBaseWrapper<__m128> {
    Y_FORCE_INLINE
    TCvtS2FWrapperSingle(const __m128i& op, TParams... params) {
        TQType<TOpOut>::As(Value) =
            func(TQType<TOpIn>::As(op), params...);
    }
};

using _mm_cvtepi32_ps =
    TCvtS2FWrapperSingle<float32x4_t, int32x4_t,
                         decltype(vcvtq_f32_s32), vcvtq_f32_s32>;

template <typename TOpOut, typename TOpIn,
          typename TFunc, TFunc* func, typename... TParams>
struct TCvtF2SWrapperSingle : TBaseWrapper<__m128i> {
    Y_FORCE_INLINE
    TCvtF2SWrapperSingle(const __m128& op, TParams... params) {
        TQType<TOpOut>::As(Value) =
            func(TQType<TOpIn>::As(op), params...);
    }
};

inline __m128i _mm_cvtps_epi32(__m128 a) {
    /// vcvtq_s32_f32 rounds to zero, but we need to round to the nearest.
    static const float32x4_t half = vdupq_n_f32(0.5f);
    static const float32x4_t negHalf = vdupq_n_f32(-0.5f);
    static const float32x4_t zero = vdupq_n_f32(0.0f);
    const float32x4_t corrections = vbslq_f32(vcgeq_f32(a.AsFloat32x4, zero), half, negHalf);
    __m128i res;
    res.AsSi32x4 = vcvtq_s32_f32(vaddq_f32(a.AsFloat32x4, corrections));
    return res;
}

using _mm_cvttps_epi32 =
    TCvtF2SWrapperSingle<int32x4_t, float32x4_t,
                         decltype(vcvtq_s32_f32), vcvtq_s32_f32>;

Y_FORCE_INLINE int
_mm_movemask_ps(const __m128& op) {
    uint32x4_t mask = {0x80000000, 0x80000000, 0x80000000, 0x80000000};
    uint32x4_t bits = vandq_u32(op.AsUi32x4, mask);
    int32x4_t shifts = {-31, -30, -29, -28};
    bits = vshlq_u32(bits, shifts);
    return vaddvq_u32(bits);
}

Y_FORCE_INLINE i64 _mm_cvtsi128_si64(__m128i a) {
    return vgetq_lane_s64(a.AsSi64x2, 0);
}

static inline void _mm_pause() {
    __asm__ ("YIELD");
}

static inline __m128 _mm_rsqrt_ps(__m128 a) {
    __m128 res;
    res.AsFloat32x4 = vrsqrteq_f32(a.AsFloat32x4);
    return res;
}

inline float _mm_cvtss_f32(__m128 a) {
    return a.AsFloat32x4[0];
}

inline __m128 _mm_cmpunord_ps(__m128 a, __m128 b) {
    __m128 res;
    res.AsUi32x4 = vorrq_u32(
        vmvnq_u32(vceqq_f32(a.AsFloat32x4, a.AsFloat32x4)), //!< 0xffffffff for all nans in a.
        vmvnq_u32(vceqq_f32(b.AsFloat32x4, b.AsFloat32x4)) //!< 0xffffffff all nans in b.
    );
    return res;
}

inline __m128 _mm_andnot_ps(__m128 a, __m128 b) {
    __m128 res;
    res.AsFloat32x4 = vandq_u32(vmvnq_u32(a.AsUi32x4), b.AsUi32x4);
    return res;
}

inline void _mm_store_ss(float* p, __m128 a) {
    *p = vgetq_lane_f32(a.AsFloat32x4, 0);
}

inline float vgetg_lane_f32_switch(float32x4_t a, ui8 b) {
    switch (b & 0x3) {
        case 0:
            return vgetq_lane_f32(a, 0);
        case 1:
            return vgetq_lane_f32(a, 1);
        case 2:
            return vgetq_lane_f32(a, 2);
        case 3:
            return vgetq_lane_f32(a, 3);
    }
    return 0;
}

inline __m128 _mm_shuffle_ps(__m128 a, __m128 b, const ShuffleStruct4& shuf) {
    __m128 ret;
    ret.AsFloat32x4 = vmovq_n_f32(vgetg_lane_f32_switch(a.AsFloat32x4, shuf.x[0]));
    ret.AsFloat32x4 = vsetq_lane_f32(vgetg_lane_f32_switch(a.AsFloat32x4, shuf.x[1]), ret.AsFloat32x4, 1);
    ret.AsFloat32x4 = vsetq_lane_f32(vgetg_lane_f32_switch(b.AsFloat32x4, shuf.x[2]), ret.AsFloat32x4, 2);
    ret.AsFloat32x4 = vsetq_lane_f32(vgetg_lane_f32_switch(b.AsFloat32x4, shuf.x[3]), ret.AsFloat32x4, 3);
    return ret;
}

inline __m128 _mm_or_ps(__m128 a, __m128 b) {
    __m128 res;
    res.AsUi32x4 = vorrq_u32(a.AsUi32x4, b.AsUi32x4);
    return res;
}

inline __m128i _mm_sad_epu8(__m128i a, __m128i b) {
    uint16x8_t t = vpaddlq_u8(vabdq_u8(a.AsUi8x16, b.AsUi8x16));
    uint16_t r0 = t[0] + t[1] + t[2] + t[3];
    uint16_t r4 = t[4] + t[5] + t[6] + t[7];
    uint16x8_t r = vsetq_lane_u16(r0, vdupq_n_u16(0), 0);
    __m128i ans;
    ans.AsUi16x8 = vsetq_lane_u16(r4, r, 4);
    return ans;
}

Y_FORCE_INLINE __m128i _mm_subs_epi8(__m128i a, __m128i b) {
    __m128i ans;
    ans.AsSi8x16 = vqsubq_s8(a.AsSi8x16, b.AsSi8x16);
    return ans;
}

Y_FORCE_INLINE __m128i _mm_subs_epi16(__m128i a, __m128i b) {
    __m128i ans;
    ans.AsSi16x8 = vqsubq_s16(a.AsSi16x8, b.AsSi16x8);
    return ans;
}

Y_FORCE_INLINE __m128i _mm_subs_epu8(__m128i a, __m128i b) {
    __m128i ans;
    ans.AsUi8x16 = vqsubq_u8(a.AsUi8x16, b.AsUi8x16);
    return ans;
}

Y_FORCE_INLINE __m128i _mm_subs_epu16(__m128i a, __m128i b) {
    __m128i ans;
    ans.AsUi16x8 = vqsubq_u16(a.AsUi16x8, b.AsUi16x8);
    return ans;
}

Y_FORCE_INLINE __m128d _mm_castsi128_pd(__m128i __A) {
    return reinterpret_cast<__m128d&>(__A);
}

Y_FORCE_INLINE __m128i _mm_set_epi8(ui8 i15, ui8 i14, ui8 i13, ui8 i12, ui8 i11, ui8 i10, ui8 i9, ui8 i8,
        ui8 i7, ui8 i6, ui8 i5, ui8 i4, ui8 i3, ui8 i2, ui8 i1, ui8 i0)
{
    int a0 =  i0 |  (i1<<8) |  (i2<<16) |  (i3<<24);
    int a1 =  i4 |  (i5<<8) |  (i6<<16) |  (i7<<24);
    int a2 =  i8 |  (i9<<8) | (i10<<16) | (i11<<24);
    int a3 = i12 | (i13<<8) | (i14<<16) | (i15<<24);
    return _mm_set_epi32(a3, a2, a1, a0);
}

Y_FORCE_INLINE __m128i _mm_max_epu8(__m128i a, __m128i b) {
    __m128i ans;
    ans.AsUi8x16 = vmaxq_u8(a.AsUi8x16, b.AsUi8x16);
    return ans;
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuninitialized"
Y_FORCE_INLINE __m128d _mm_undefined_pd(void) {
    __m128d ans = ans;
    return ans;
}
#pragma GCC diagnostic pop

Y_FORCE_INLINE __m128d _mm_loadh_pd(__m128d a, const double* b) {
    a[1] = *b;
    return a;
}

Y_FORCE_INLINE __m128d _mm_loadl_pd(__m128d a, const double* b) {
    a[0] = *b;
    return a;
}

Y_FORCE_INLINE double _mm_cvtsd_f64(__m128d a) {
    return a[0];
}

Y_FORCE_INLINE __m128d _mm_shuffle_pd(__m128d a, __m128d b, int mask) {
    __m128d result;
    const int litmsk = mask & 0x3;

    if (litmsk == 0)
        result = vzip1q_f64(a, b);
    else if (litmsk == 1)
        result = __builtin_shufflevector(a, b, 1, 2);
    else if (litmsk == 2)
        result = __builtin_shufflevector(a, b, 0, 3);
    else
        result = vzip2q_f64(a, b);
    return result;
}
