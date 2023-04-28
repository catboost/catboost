#pragma once

/*
    The header contains code which translates SSE intrinsics
    to PowerPC AltiVec or software emulation.

    See also: https://www.ibm.com/developerworks/community/wikis/home?lang=en#!/wiki/W51a7ffcf4dfd_4b40_9d82_446ebc23c550/page/Intel%20SSE%20to%20PowerPC%20AltiVec%20migration
*/
/* Author: Vadim Rumyantsev <rumvadim@yandex-team.ru> */

#if !defined(_ppc64_)
#error "This header is for PowerPC (ppc64) platform only." \
    "Include sse.h instead of including this header directly."
#endif

#include <util/system/types.h>
#include <util/system/compiler.h>

#include <altivec.h>

typedef __attribute__((__aligned__(8))) unsigned long long __m64;
typedef __attribute__((__aligned__(16), __may_alias__)) vector float __m128;
typedef __attribute__((__aligned__(16), __may_alias__)) vector unsigned char __m128i;
typedef __attribute__((__aligned__(16), __may_alias__)) vector double __m128d;

using __v2df = __vector double;
using __v2di = __vector long long;
using __v2du = __vector unsigned long long;
using __v4si = __vector int;
using __v4su = __vector unsigned int;
using __v8hi = __vector short;
using __v8hu = __vector unsigned short;
using __v16qi = __vector signed char;
using __v16qu = __vector unsigned char;
using __v4sf = __vector float;

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

#define _MM_SHUFFLE(a, b, c, d) ((signed char)(a * 64 + b * 16 + c * 4 + d))

/// Functions that work with floats.

Y_FORCE_INLINE __m128 _mm_setzero_ps() {
    return (__m128){0.0f, 0.0f, 0.0f, 0.0f};
};

Y_FORCE_INLINE __m128d _mm_setzero_pd() {
    return (__m128d)vec_splats((double)0);
}

// bug in clang compiler until 7.0.0 inclusive, Y_NO_INLINE is vital/essential
static Y_NO_INLINE __m128 _mm_set1_ps(float f) {
    return (vector float)f;
}

Y_FORCE_INLINE __m128 _mm_set_ps1(float f) {
    return _mm_set1_ps(f);
}

Y_FORCE_INLINE __m128 _mm_set_ps(float v3, float v2, float v1, float v0) {
    return (__m128)(__v4sf){v0, v1, v2, v3};
}

Y_FORCE_INLINE __m128d _mm_set_pd(double d1, double d0) {
    return (__m128d){d0, d1};
}

Y_FORCE_INLINE __m128 _mm_loadu_ps(const float* p) {
    return vec_vsx_ld(0, p);
}

Y_FORCE_INLINE __m128 _mm_load_ps(const float* p) {
    return (__m128)vec_ld(0, (vector float*)p);
}

Y_FORCE_INLINE __m128d _mm_loadu_pd(const double* d) {
    return vec_vsx_ld(0, d);
}

Y_FORCE_INLINE void _mm_storeu_ps(float* p, __m128 a) {
    *(__m128*)p = a;
}

Y_FORCE_INLINE __m128 _mm_xor_ps(__m128 a, __m128 b) {
    return (__m128)vec_xor((__v4sf)a, (__v4sf)b);
}

Y_FORCE_INLINE __m128 _mm_xor_pd(__m128d a, __m128d b) {
    return (__m128)vec_xor((__v2df)a, (__v2df)b);
}

Y_FORCE_INLINE __m128 _mm_add_ps(__m128 a, __m128 b) {
    return (__m128)((__v4sf)a + (__v4sf)b);
}

Y_FORCE_INLINE __m128d _mm_add_pd(__m128d a, __m128d b) {
    return (__m128d)((__v2df)a + (__v2df)b);
}

Y_FORCE_INLINE __m128 _mm_sub_ps(__m128 a, __m128 b) {
    return (__m128)((__v4sf)a - (__v4sf)b);
}

Y_FORCE_INLINE __m128d _mm_sub_pd(__m128d a, __m128d b) {
    return (__m128d)((__v2df)a - (__v2df)b);
}

Y_FORCE_INLINE __m128 _mm_mul_ps(__m128 a, __m128 b) {
    return (__m128)((__v4sf)a * (__v4sf)b);
}

Y_FORCE_INLINE __m128d _mm_mul_pd(__m128d a, __m128d b) {
    return (__m128d)((__v2df)a * (__v2df)b);
}

Y_FORCE_INLINE __m128 _mm_div_ps(__m128 a, __m128 b) {
    return (__m128)((__v4sf)a / (__v4sf)b);
}

Y_FORCE_INLINE __m128d _mm_div_pd(__m128d a, __m128d b) {
    return (__m128d)((__v2df)a / (__v2df)b);
}

Y_FORCE_INLINE __m128 _mm_cmpeq_ps(__m128 a, __m128 b) {
    return ((__m128)vec_cmpeq((__v4sf)a, (__v4sf)b));
    ;
}

Y_FORCE_INLINE __m128 _mm_cmpgt_ps(__m128 a, __m128 b) {
    return ((__m128)vec_cmpgt((__v4sf)a, (__v4sf)b));
}

Y_FORCE_INLINE __m128 _mm_max_ps(__m128 a, __m128 b) {
    return (__m128)vec_max((vector float)a, (vector float)b);
}

Y_FORCE_INLINE __m128i _mm_max_epu8(__m128i a, __m128i b) {
    return (__m128i)vec_max((__v16qu)a, (__v16qu)b);
}

Y_FORCE_INLINE __m128 _mm_min_ps(__m128 a, __m128 b) {
    return (__m128)vec_min((vector float)a, (vector float)b);
}

Y_FORCE_INLINE __m128 _mm_and_ps(__m128 a, __m128 b) {
    return ((__m128)vec_and((__v4sf)a, (__v4sf)b));
}

Y_FORCE_INLINE __m128d _mm_and_pd(__m128d a, __m128d b) {
    return vec_and((__v2df)a, (__v2df)b);
}

Y_FORCE_INLINE __m128 _mm_rsqrt_ps(__m128 a) {
    return vec_rsqrte(a);
}

Y_FORCE_INLINE __m128 _mm_rsqrt_ss(__m128 a) {
    __m128 a1, c;
    const vector unsigned int mask = {0xffffffff, 0, 0, 0};
    a1 = vec_splat(a, 0);
    c = vec_rsqrte(a1);
    return (vec_sel((vector float)a, c, mask));
}

Y_FORCE_INLINE int _mm_movemask_ps(__m128 a) {
    __vector unsigned long long result;
    const __vector unsigned int perm_mask =
        {
#ifdef __LITTLE_ENDIAN__
            0x00204060, 0x80808080, 0x80808080, 0x80808080
#elif __BIG_ENDIAN__
            0x80808080, 0x80808080, 0x80808080, 0x00204060
#endif
        };

    result = (__vector unsigned long long)vec_vbpermq((__vector unsigned char)a,
                                                      (__vector unsigned char)perm_mask);

#ifdef __LITTLE_ENDIAN__
    return result[1];
#elif __BIG_ENDIAN__
    return result[0];
#endif
}

Y_FORCE_INLINE __m128 _mm_cvtepi32_ps(__m128i a) {
    return ((__m128)vec_ctf((__v4si)a, 0));
}

Y_FORCE_INLINE float _mm_cvtss_f32(__m128 a) {
    return ((__v4sf)a)[0];
}

Y_FORCE_INLINE __m128 _mm_cmpunord_ps(__m128 A, __m128 B) {
    __vector unsigned int a, b;
    __vector unsigned int c, d;
    const __vector unsigned int float_exp_mask =
        {0x7f800000, 0x7f800000, 0x7f800000, 0x7f800000};

    a = (__vector unsigned int)vec_abs((__v4sf)A);
    b = (__vector unsigned int)vec_abs((__v4sf)B);
    c = (__vector unsigned int)vec_cmpgt(a, float_exp_mask);
    d = (__vector unsigned int)vec_cmpgt(b, float_exp_mask);
    return ((__m128)vec_or(c, d));
}

Y_FORCE_INLINE __m128 _mm_andnot_ps(__m128 a, __m128 b) {
    return ((__m128)vec_andc((__v4sf)b, (__v4sf)a));
}

Y_FORCE_INLINE __m128 _mm_or_ps(__m128 a, __m128 b) {
    return ((__m128)vec_or((__v4sf)a, (__v4sf)b));
}

Y_FORCE_INLINE void _mm_store_ss(float* p, __m128 a) {
    *p = ((__v4sf)a)[0];
}

Y_FORCE_INLINE void _mm_store_ps(float* p, __m128 a) {
    vec_st(a, 0, p);
}

Y_FORCE_INLINE void _mm_storeu_pd(double* p, __m128d a) {
    *(__m128d*)p = a;
}

Y_FORCE_INLINE void _mm_store_pd(double* p, __m128d a) {
    vec_st((vector unsigned char)a, 0, (vector unsigned char*)p);
}

Y_FORCE_INLINE __m128 _mm_shuffle_ps(__m128 a, __m128 b, long shuff) {
    unsigned long element_selector_10 = shuff & 0x03;
    unsigned long element_selector_32 = (shuff >> 2) & 0x03;
    unsigned long element_selector_54 = (shuff >> 4) & 0x03;
    unsigned long element_selector_76 = (shuff >> 6) & 0x03;
    const unsigned int permute_selectors[4] =
        {
#ifdef __LITTLE_ENDIAN__
            0x03020100, 0x07060504, 0x0B0A0908, 0x0F0E0D0C
#elif __BIG_ENDIAN__
            0x0C0D0E0F, 0x08090A0B, 0x04050607, 0x00010203
#endif
        };
    __vector unsigned int t;

#ifdef __LITTLE_ENDIAN__
    t[0] = permute_selectors[element_selector_10];
    t[1] = permute_selectors[element_selector_32];
    t[2] = permute_selectors[element_selector_54] + 0x10101010;
    t[3] = permute_selectors[element_selector_76] + 0x10101010;
#elif __BIG_ENDIAN__
    t[3] = permute_selectors[element_selector_10] + 0x10101010;
    t[2] = permute_selectors[element_selector_32] + 0x10101010;
    t[1] = permute_selectors[element_selector_54];
    t[0] = permute_selectors[element_selector_76];
#endif
    return vec_perm((__v4sf)a, (__v4sf)b, (__vector unsigned char)t);
}

Y_FORCE_INLINE __m128d _mm_shuffle_pd(__m128d a, __m128d b, const int mask) {
    __vector double result;
    const int litmsk = mask & 0x3;

    if (litmsk == 0)
        result = vec_mergeh(a, b);
    else if (litmsk == 1)
        result = vec_xxpermdi(a, b, 2);
    else if (litmsk == 2)
        result = vec_xxpermdi(a, b, 1);
    else
        result = vec_mergel(a, b);
    return result;
}

Y_FORCE_INLINE __m128i _mm_cvtps_epi32(__m128 a) {
    vector float rounded;
    __v4si result;

    rounded = vec_rint((vector float)a);
    result = vec_cts(rounded, 0);
    return (__m128i)result;
}

/// Functions that work with integers.

Y_FORCE_INLINE int _mm_movemask_epi8(__m128i a) {
    __vector unsigned long long result;
    const __vector unsigned char perm_mask =
        {
#ifdef __LITTLE_ENDIAN__
            0x78, 0x70, 0x68, 0x60, 0x58, 0x50, 0x48, 0x40,
            0x38, 0x30, 0x28, 0x20, 0x18, 0x10, 0x08, 0x00
#elif __BIG_ENDIAN__
            0x00, 0x08, 0x10, 0x18, 0x20, 0x28, 0x30, 0x38,
            0x40, 0x48, 0x50, 0x58, 0x60, 0x68, 0x70, 0x78
#endif
        };

    result = (__vector unsigned long long)vec_vbpermq((__vector unsigned char)a,
                                                      (__vector unsigned char)perm_mask);

#ifdef __LITTLE_ENDIAN__
    return result[1];
#elif __BIG_ENDIAN__
    return result[0];
#endif
}

Y_FORCE_INLINE __m128i _mm_cvttps_epi32(__m128 a) {
    __v4si result;

    result = vec_cts((__v4sf)a, 0);
    return (__m128i)result;
}

#define _MM_TRANSPOSE4_PS(row0, row1, row2, row3)                                       \
    do {                                                                                \
        __v4sf __r0 = (row0), __r1 = (row1), __r2 = (row2), __r3 = (row3);              \
        __v4sf __t0 = vec_vmrghw((vector unsigned int)__r0, (vector unsigned int)__r1); \
        __v4sf __t1 = vec_vmrghw((vector unsigned int)__r2, (vector unsigned int)__r3); \
        __v4sf __t2 = vec_vmrglw((vector unsigned int)__r0, (vector unsigned int)__r1); \
        __v4sf __t3 = vec_vmrglw((vector unsigned int)__r2, (vector unsigned int)__r3); \
        (row0) = (__v4sf)vec_mergeh((vector long long)__t0,                             \
                                    (vector long long)__t1);                            \
        (row1) = (__v4sf)vec_mergel((vector long long)__t0,                             \
                                    (vector long long)__t1);                            \
        (row2) = (__v4sf)vec_mergeh((vector long long)__t2,                             \
                                    (vector long long)__t3);                            \
        (row3) = (__v4sf)vec_mergel((vector long long)__t2,                             \
                                    (vector long long)__t3);                            \
    } while (0)

Y_FORCE_INLINE __m128i _mm_or_si128(__m128i a, __m128i b) {
    return (__m128i)vec_or((__v2di)a, (__v2di)b);
}

Y_FORCE_INLINE __m128i _mm_and_si128(__m128i a, __m128i b) {
    return (__m128i)vec_and((__v2di)a, (__v2di)b);
}

Y_FORCE_INLINE __m128i _mm_andnot_si128(__m128i a, __m128i b) {
    return (__m128i)vec_andc((__v2di)b, (__v2di)a);
}

Y_FORCE_INLINE __m128i _mm_xor_si128(__m128i a, __m128i b) {
    return (__m128i)vec_xor((__v2di)a, (__v2di)b);
}

Y_FORCE_INLINE __m128i _mm_setzero_si128() {
    return (__m128i)(__v4si){0, 0, 0, 0};
}

Y_FORCE_INLINE __m128i _mm_shuffle_epi32(__m128i op1, long op2) {
    unsigned long element_selector_10 = op2 & 0x03;
    unsigned long element_selector_32 = (op2 >> 2) & 0x03;
    unsigned long element_selector_54 = (op2 >> 4) & 0x03;
    unsigned long element_selector_76 = (op2 >> 6) & 0x03;
    const unsigned int permute_selectors[4] =
        {
#ifdef __LITTLE_ENDIAN__
            0x03020100, 0x07060504, 0x0B0A0908, 0x0F0E0D0C
#elif __BIG_ENDIAN__
            0x0C0D0E0F, 0x08090A0B, 0x04050607, 0x00010203
#endif
        };
    __v4su t;

#ifdef __LITTLE_ENDIAN__
    t[0] = permute_selectors[element_selector_10];
    t[1] = permute_selectors[element_selector_32];
    t[2] = permute_selectors[element_selector_54] + 0x10101010;
    t[3] = permute_selectors[element_selector_76] + 0x10101010;
#elif __BIG_ENDIAN__
    t[3] = permute_selectors[element_selector_10] + 0x10101010;
    t[2] = permute_selectors[element_selector_32] + 0x10101010;
    t[1] = permute_selectors[element_selector_54];
    t[0] = permute_selectors[element_selector_76];
#endif
    return (__m128i)vec_perm((__v4si)op1, (__v4si)op1, (__vector unsigned char)t);
}

Y_FORCE_INLINE int _mm_extract_epi16(__m128i a, int imm) {
    return (unsigned short)((__v8hi)a)[imm & 7];
}

Y_FORCE_INLINE int _mm_extract_epi8(__m128i a, int imm) {
    return (unsigned char)((__v16qi)a)[imm & 15];
}

Y_FORCE_INLINE int _mm_extract_epi32(__m128i a, int imm) {
    return ((__v4si)a)[imm & 3];
}

Y_FORCE_INLINE long long _mm_extract_epi64(__m128i a, int imm) {
    return ((__v2di)a)[imm & 1];
}

Y_FORCE_INLINE int _mm_extract_ps(__m128 a, int imm) {
    return ((__v4si)a)[imm & 3];
}

Y_FORCE_INLINE __m128i _mm_slli_epi16(__m128i a, int count) {
    __v8hu lshift;
    __v8hi result = {0, 0, 0, 0, 0, 0, 0, 0};

    if (count >= 0 && count < 16) {
        if (__builtin_constant_p(count)) {
            lshift = (__v8hu)vec_splat_s16(count);
        } else {
            lshift = vec_splats((unsigned short)count);
        }

        result = vec_vslh((__v8hi)a, lshift);
    }

    return (__m128i)result;
}

Y_FORCE_INLINE __m128i _mm_slli_epi32(__m128i a, int count) {
    __v4su lshift;
    __v4si result = {0, 0, 0, 0};

    if (count >= 0 && count < 32) {
        if (__builtin_constant_p(count) && count < 16) {
            lshift = (__v4su)vec_splat_s32(count);
        } else {
            lshift = vec_splats((unsigned int)count);
        }

        result = vec_vslw((__v4si)a, lshift);
    }

    return (__m128i)result;
}

Y_FORCE_INLINE __m128i _mm_slli_epi64(__m128i a, int count) {
    __v2du lshift;
    __v2di result = {0, 0};

    if (count >= 0 && count < 64) {
        if (__builtin_constant_p(count) && count < 16) {
            lshift = (__v2du)vec_splat_s32(count);
        } else {
            lshift = (__v2du)vec_splats((unsigned int)count);
        }

        result = vec_sl((__v2di)a, lshift);
    }

    return (__m128i)result;
}

Y_FORCE_INLINE __m128i _mm_slli_si128(__m128i a, int imm) {
    __v16qu result;
    const __v16qu zeros = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    if (imm < 16)
#ifdef __LITTLE_ENDIAN__
        result = vec_sld((__v16qu)a, zeros, imm);
#elif __BIG_ENDIAN__
        result = vec_sld(zeros, (__v16qu)a, (16 - imm));
#endif
    else
        result = zeros;

    return (__m128i)result;
}

Y_FORCE_INLINE __m128i _mm_srli_epi16(__m128i a, int count) {
    if ((unsigned long)count >= 16) {
        /* SSE2 shifts >= element_size or < 0 produce 0; Altivec/MMX shifts by count%element_size. */
        return (__m128i)vec_splats(0);
    } else if (count == 0) {
        return a;
    } else {
        /* The PowerPC Architecture says all shift count fields must contain the same shift count. */
        __v8hi replicated_count;
        replicated_count = vec_splats((short)count);
        return (__m128i)vec_sr((vector signed short)a, (vector unsigned short)replicated_count);
    }
}

Y_FORCE_INLINE __m128i _mm_srli_epi32(__m128i a, int count) {
    if ((unsigned long)count >= 32) {
        /* SSE2 shifts >= element_size or < 0 produce 0; Altivec/MMX shifts by count%element_size. */
        return (__m128i)vec_splats(0);
    } else if (count == 0) {
        return a;
    } else {
        /* The PowerPC Architecture says all shift count fields must contain the same shift count. */
        __v4si replicated_count;
        replicated_count = vec_splats(count);
        return (__m128i)vec_sr((vector signed int)a, (vector unsigned int)replicated_count);
    }
}

Y_FORCE_INLINE __m128i _mm_srli_epi64(__m128i a, int count) {
    if ((unsigned long)count >= 64) {
        /* SSE2 shifts >= element_size or < 0 produce 0; Altivec/MMX shifts by count%element_size. */
        return (__m128i)vec_splats(0);
    } else if (count == 0) {
        return a;
    } else {
        /* The PowerPC Architecture says all shift count fields must contain the same shift count. */
        /* On Power7 vec_slo (vslo) does use just the documented bits 121:124. */
        /* On Power7 vec_sll (vsll) uses the lower 3 bits of each byte instead (legal). */
        __v16qu replicated_count;
        replicated_count = vec_splats((unsigned char)count);
        long long m = 0xFFFFFFFFFFFFFFFFull >> count;
        __v2di mask;
        mask[0] = m;
        mask[1] = m;
        return vec_and(vec_srl(vec_sro(a, (__m128i)replicated_count), (__m128i)replicated_count), (__v16qu)mask);
    }
}

Y_FORCE_INLINE __m128i _mm_bsrli_si128(__m128i a, const int __N) {
    __v16qu result;
    const __v16qu zeros = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    if (__N < 16)
        if (__builtin_constant_p(__N))
            /* Would like to use Vector Shift Left Double by Octet
     Immediate here to use the immediate form and avoid
     load of __N * 8 value into a separate VR.  */
            result = vec_sld(zeros, (__v16qu)a, (16 - __N));
        else {
            __v16qu shift = vec_splats((unsigned char)(__N * 8));
            result = vec_sro((__v16qu)a, shift);
        }
    else
        result = zeros;

    return (__m128i)result;
}

Y_FORCE_INLINE __m128i _mm_srli_si128(__m128i a, int imm) {
    return _mm_bsrli_si128(a, imm);
}

Y_FORCE_INLINE __m128i _mm_srai_epi16(__m128i a, int count) {
    __v8hu rshift = {15, 15, 15, 15, 15, 15, 15, 15};
    __v8hi result;
    if (count < 16) {
        if (__builtin_constant_p(count)) {
            rshift = (__v8hu)vec_splat_s16(count);
        } else {
            rshift = vec_splats((unsigned short)count);
        }
    }
    result = vec_vsrah((__v8hi)a, rshift);
    return (__m128i)result;
}

Y_FORCE_INLINE __m128i _mm_srai_epi32(__m128i a, int count) {
    // return vec_shiftrightarithmetic4wimmediate(a, count); //!< Failes to work with count >= 32.
    __v4su rshift = {31, 31, 31, 31};
    __v4si result;

    if (count < 32) {
        if (__builtin_constant_p(count)) {
            if (count < 16) {
                rshift = (__v4su)vec_splat_s32(count);
            } else {
                rshift = (__v4su)vec_splats((unsigned int)count);
            }
        } else {
            rshift = vec_splats((unsigned int)count);
        }
    }
    result = vec_vsraw((__v4si)a, rshift);
    return (__m128i)result;
}

Y_FORCE_INLINE __m128i _mm_sll_epi16(__m128i a, __m128i count) {
    __v8hu lshift, shmask;
    const __v8hu shmax = {15, 15, 15, 15, 15, 15, 15, 15};
    __v8hu result;

#ifdef __LITTLE_ENDIAN__
    lshift = vec_splat((__v8hu)count, 0);
#elif __BIG_ENDIAN__
    lshift = vec_splat((__v8hu)count, 3);
#endif
    shmask = vec_cmple(lshift, shmax);
    result = vec_vslh((__v8hu)a, lshift);
    result = vec_sel(shmask, result, shmask);
    return (__m128i)result;
}

Y_FORCE_INLINE __m128i _mm_sll_epi32(__m128i a, __m128i count) {
    __v4su lshift, shmask;
    const __v4su shmax = {32, 32, 32, 32};
    __v4su result;
#ifdef __LITTLE_ENDIAN__
    lshift = vec_splat((__v4su)count, 0);
#elif __BIG_ENDIAN__
    lshift = vec_splat((__v4su)count, 1);
#endif
    shmask = vec_cmplt(lshift, shmax);
    result = vec_vslw((__v4su)a, lshift);
    result = vec_sel(shmask, result, shmask);

    return (__m128i)result;
}

Y_FORCE_INLINE __m128i _mm_sll_epi64(__m128i a, __m128i count) {
    __v2du lshift, shmask;
    const __v2du shmax = {64, 64};
    __v2du result;

    lshift = (__v2du)vec_splat((__v2du)count, 0);
    shmask = vec_cmplt(lshift, shmax);
    result = vec_sl((__v2du)a, lshift);
    result = result & shmask;

    return (__m128i)result;
}

Y_FORCE_INLINE __m128i _mm_srl_epi16(__m128i a, __m128i count) {
    __v8hu rshift, shmask;
    const __v8hu shmax = {15, 15, 15, 15, 15, 15, 15, 15};
    __v8hu result;

#ifdef __LITTLE_ENDIAN__
    rshift = vec_splat((__v8hu)count, 0);
#elif __BIG_ENDIAN__
    rshift = vec_splat((__v8hu)count, 3);
#endif
    shmask = vec_cmple(rshift, shmax);
    result = vec_vsrh((__v8hu)a, rshift);
    result = vec_sel(shmask, result, shmask);

    return (__m128i)result;
}

Y_FORCE_INLINE __m128i _mm_srl_epi32(__m128i a, __m128i count) {
    __v4su rshift, shmask;
    const __v4su shmax = {32, 32, 32, 32};
    __v4su result;

#ifdef __LITTLE_ENDIAN__
    rshift = vec_splat((__v4su)count, 0);
#elif __BIG_ENDIAN__
    rshift = vec_splat((__v4su)count, 1);
#endif
    shmask = vec_cmplt(rshift, shmax);
    result = vec_vsrw((__v4su)a, rshift);
    result = vec_sel(shmask, result, shmask);

    return (__m128i)result;
}

Y_FORCE_INLINE __m128i _mm_srl_epi64(__m128i a, __m128i count) {
    __v2du rshift, shmask;
    const __v2du shmax = {64, 64};
    __v2du result;

    rshift = (__v2du)vec_splat((__v2du)count, 0);
    shmask = vec_cmplt(rshift, shmax);
    result = vec_sr((__v2du)a, rshift);
    result = (__v2du)vec_sel((__v2du)shmask, (__v2du)result, (__v2du)shmask);

    return (__m128i)result;
}

Y_FORCE_INLINE void _mm_storeu_si128(__m128i* p, __m128i a) {
    vec_vsx_st(a, 0, p);
}

Y_FORCE_INLINE void _mm_store_si128(__m128i* p, __m128i a) {
    vec_st((__v16qu)a, 0, (__v16qu*)p);
}

Y_FORCE_INLINE __m128i _mm_unpackhi_epi8(__m128i a, __m128i b) {
    return (__m128i)vec_mergel((__v16qu)a, (__v16qu)b);
}

Y_FORCE_INLINE __m128i _mm_unpackhi_epi16(__m128i a, __m128i b) {
    return (__m128i)vec_mergel((__v8hu)a, (__v8hu)b);
}

Y_FORCE_INLINE __m128i _mm_unpackhi_epi32(__m128i a, __m128i b) {
    return (__m128i)vec_mergel((__v4su)a, (__v4su)b);
}

Y_FORCE_INLINE __m128i _mm_unpackhi_epi64(__m128i a, __m128i b) {
    return (__m128i)vec_mergel((vector long long)a, (vector long long)b);
}

Y_FORCE_INLINE __m128i _mm_unpacklo_epi8(__m128i a, __m128i b) {
    return (__m128i)vec_mergeh((__v16qu)a, (__v16qu)b);
}

Y_FORCE_INLINE __m128i _mm_unpacklo_epi16(__m128i a, __m128i b) {
    return (__m128i)vec_mergeh((__v8hi)a, (__v8hi)b);
}

Y_FORCE_INLINE __m128i _mm_unpacklo_epi32(__m128i a, __m128i b) {
    return (__m128i)vec_mergeh((__v4si)a, (__v4si)b);
}

Y_FORCE_INLINE __m128i _mm_unpacklo_epi64(__m128i a, __m128i b) {
    return (__m128i)vec_mergeh((vector long long)a, (vector long long)b);
}

Y_FORCE_INLINE __m128i _mm_add_epi8(__m128i a, __m128i b) {
    return (__m128i)((__v16qu)a + (__v16qu)b);
}

Y_FORCE_INLINE __m128i _mm_add_epi16(__m128i a, __m128i b) {
    return (__m128i)((__v8hu)a + (__v8hu)b);
}

Y_FORCE_INLINE __m128i _mm_add_epi32(__m128i a, __m128i b) {
    return (__m128i)((__v4su)a + (__v4su)b);
}

Y_FORCE_INLINE __m128i _mm_add_epi64(__m128i a, __m128i b) {
    return (__m128i)((__v2du)a + (__v2du)b);
}

Y_FORCE_INLINE __m128i _mm_madd_epi16(__m128i a, __m128i b) {
    const vector signed int zero = {0, 0, 0, 0};
    return (__m128i)vec_vmsumshm((__v8hi)a, (__v8hi)b, zero);
}

Y_FORCE_INLINE __m128i _mm_sub_epi8(__m128i a, __m128i b) {
    return (__m128i)((__v16qu)a - (__v16qu)b);
}

Y_FORCE_INLINE __m128i _mm_sub_epi16(__m128i a, __m128i b) {
    return (__m128i)((__v8hu)a - (__v8hu)b);
}

Y_FORCE_INLINE __m128i _mm_sub_epi32(__m128i a, __m128i b) {
    return (__m128i)((__v4su)a - (__v4su)b);
}

Y_FORCE_INLINE __m128i _mm_sub_epi64(__m128i a, __m128i b) {
    return (__m128i)((__v2du)a - (__v2du)b);
}

Y_FORCE_INLINE __m128i _mm_mul_epu32(__m128i a, __m128i b) {
#ifdef __LITTLE_ENDIAN__
    return (__m128i)vec_mule((__v4su)a, (__v4su)b);
#elif __BIG_ENDIAN__
    return (__m128i)vec_mulo((__v4su)a, (__v4su)b);
#endif
}

Y_FORCE_INLINE __m128i _mm_set_epi8(char q15, char q14, char q13, char q12, char q11, char q10, char q09, char q08, char q07, char q06, char q05, char q04, char q03, char q02, char q01, char q00) {
    return (__m128i)(__v16qi){q00, q01, q02, q03, q04, q05, q06, q07, q08, q09, q10, q11, q12, q13, q14, q15};
};

Y_FORCE_INLINE __m128i _mm_setr_epi8(char q15, char q14, char q13, char q12, char q11, char q10, char q09, char q08, char q07, char q06, char q05, char q04, char q03, char q02, char q01, char q00) {
    return (__m128i)(__v16qi){q15, q14, q13, q12, q11, q10, q09, q08, q07, q06, q05, q04, q03, q02, q01, q00};
};

Y_FORCE_INLINE __m128i _mm_set_epi16(short q7, short q6, short q5, short q4, short q3, short q2, short q1, short q0) {
    return (__m128i)(__v8hi){q0, q1, q2, q3, q4, q5, q6, q7};
}

Y_FORCE_INLINE __m128i _mm_setr_epi16(short q7, short q6, short q5, short q4, short q3, short q2, short q1, short q0) {
    return (__m128i)(__v8hi){q7, q6, q5, q4, q3, q2, q1, q0};
}

Y_FORCE_INLINE __m128i _mm_set_epi32(int q3, int q2, int q1, int q0) {
    return (__m128i)(__v4si){q0, q1, q2, q3};
}

Y_FORCE_INLINE __m128i _mm_setr_epi32(int q3, int q2, int q1, int q0) {
    return (__m128i)(__v4si){q3, q2, q1, q0};
}

Y_FORCE_INLINE __m128i _mm_set1_epi8(char a) {
    return _mm_set_epi8(a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a);
}

Y_FORCE_INLINE __m128i _mm_set1_epi16(short a) {
    return _mm_set_epi16(a, a, a, a, a, a, a, a);
}

Y_FORCE_INLINE __m128i _mm_set1_epi32(int a) {
    return _mm_set_epi32(a, a, a, a);
}

Y_FORCE_INLINE __m128i _mm_cmpeq_epi8(__m128i a, __m128i b) {
    return (__m128i)vec_cmpeq((__v16qi)a, (__v16qi)b);
}

Y_FORCE_INLINE __m128i _mm_cmpeq_epi16(__m128i a, __m128i b) {
    return (__m128i)vec_cmpeq((__v8hi)a, (__v8hi)b);
}

Y_FORCE_INLINE __m128i _mm_cmpeq_epi32(__m128i a, __m128i b) {
    return (__m128i)vec_cmpeq((__v4si)a, (__v4si)b);
}

Y_FORCE_INLINE __m128i _mm_packs_epi16(__m128i a, __m128i b) {
    return (__m128i)vec_packs((__v8hi)a, (__v8hi)b);
}

Y_FORCE_INLINE __m128i _mm_packs_epi32(__m128i a, __m128i b) {
    return (__m128i)vec_packs((__v4si)a, (__v4si)b);
}

Y_FORCE_INLINE __m128i _mm_packus_epi16(__m128i a, __m128i b) {
    return (__m128i)vec_packsu((vector signed short)a, (vector signed short)b);
}

Y_FORCE_INLINE __m128i _mm_cvtsi64_si128(i64 a) {
    return (__m128i)(__v2di){a, 0LL};
}

Y_FORCE_INLINE __m128i _mm_cvtsi32_si128(int a) {
    return _mm_set_epi32(0, 0, 0, a);
}

Y_FORCE_INLINE int _mm_cvtsi128_si32(__m128i a) {
    return ((__v4si)a)[0];
}

Y_FORCE_INLINE i64 _mm_cvtsi128_si64(__m128i a) {
    return ((__v2di)a)[0];
}

Y_FORCE_INLINE __m128i _mm_load_si128(const __m128i* p) {
    return *p;
}

Y_FORCE_INLINE __m128i _mm_loadu_si128(const __m128i* p) {
    return (__m128i)(vec_vsx_ld(0, (signed int const*)p));
}

Y_FORCE_INLINE __m128i _mm_lddqu_si128(const __m128i* p) {
    return _mm_loadu_si128(p);
}

Y_FORCE_INLINE __m128i _mm_loadl_epi64(const __m128i* a) {
#ifdef __LITTLE_ENDIAN__
    const vector bool long long mask = {
        0xFFFFFFFFFFFFFFFFull, 0x0000000000000000ull};
#elif __BIG_ENDIAN__
    const vector bool long long mask = {
        0x0000000000000000ull, 0xFFFFFFFFFFFFFFFFull};
#endif
    return (__m128i)vec_and(_mm_loadu_si128(a), (vector unsigned char)mask);
}

Y_FORCE_INLINE void _mm_storel_epi64(__m128i* a, __m128i b) {
    *(long long*)a = ((__v2di)b)[0];
}

Y_FORCE_INLINE double _mm_cvtsd_f64(__m128d a) {
    return ((__v2df)a)[0];
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuninitialized"
Y_FORCE_INLINE __m128d _mm_undefined_pd(void) {
    __m128d ans = ans;
    return ans;
}
#pragma GCC diagnostic pop

Y_FORCE_INLINE __m128d _mm_loadh_pd(__m128d a, const double* b) {
    __v2df result = (__v2df)a;
    result[1] = *b;
    return (__m128d)result;
}

Y_FORCE_INLINE __m128d _mm_loadl_pd(__m128d a, const double* b) {
    __v2df result = (__v2df)a;
    result[0] = *b;
    return (__m128d)result;
}

Y_FORCE_INLINE __m128 _mm_castsi128_ps(__m128i a) {
    return (__m128)a;
}

Y_FORCE_INLINE __m128i _mm_castps_si128(__m128 a) {
    return (__m128i)a;
}

Y_FORCE_INLINE __m128i _mm_cmpgt_epi8(__m128i a, __m128i b) {
    return (__m128i)vec_cmpgt((__v16qi)a, (__v16qi)b);
}

Y_FORCE_INLINE __m128i _mm_cmpgt_epi16(__m128i a, __m128i b) {
    return (__m128i)vec_cmpgt((__v8hi)a, (__v8hi)b);
}

Y_FORCE_INLINE __m128i _mm_cmpgt_epi32(__m128i a, __m128i b) {
    return (__m128i)vec_cmpgt((__v4si)a, (__v4si)b);
}

Y_FORCE_INLINE __m128i _mm_cmpgt_epi64(__m128i a, __m128i b) {
    return (__m128i)vec_cmpgt((vector signed long long)a, (vector signed long long)b);
}

Y_FORCE_INLINE __m128i _mm_cmplt_epi8(__m128i a, __m128i b) {
    return (__m128i)vec_cmplt((__v16qi)a, (__v16qi)b);
}

Y_FORCE_INLINE __m128i _mm_cmplt_epi16(__m128i a, __m128i b) {
    return (__m128i)vec_cmplt((__v8hi)a, (__v8hi)b);
}

Y_FORCE_INLINE __m128i _mm_cmplt_epi32(__m128i a, __m128i b) {
    return (__m128i)vec_cmplt((__v4si)a, (__v4si)b);
}

Y_FORCE_INLINE __m128i _mm_cmplt_epi64(__m128i a, __m128i b) {
    return (__m128i)vec_cmplt((vector signed long long)a, (vector signed long long)b);
}

Y_FORCE_INLINE __m128i _mm_sad_epu8(__m128i A, __m128i B) {
    __v16qu a, b;
    __v16qu vmin, vmax, vabsdiff;
    __v4si vsum;
    const __v4su zero = {0, 0, 0, 0};
    __v4si result;

    a = (__v16qu)A;
    b = (__v16qu)B;
    vmin = vec_min(a, b);
    vmax = vec_max(a, b);
    vabsdiff = vec_sub(vmax, vmin);
    /* Sum four groups of bytes into integers.  */
    vsum = (__vector signed int)vec_sum4s(vabsdiff, zero);
    /* Sum across four integers with two integer results.  */
    result = vec_sum2s(vsum, (__vector signed int)zero);
    /* Rotate the sums into the correct position.  */
#ifdef __LITTLE_ENDIAN__
    result = vec_sld(result, result, 4);
#elif __BIG_ENDIAN__
    result = vec_sld(result, result, 6);
#endif
    /* Rotate the sums into the correct position.  */
    return (__m128i)result;
}

Y_FORCE_INLINE __m128i _mm_subs_epi8(__m128i a, __m128i b) {
    return (__m128i)vec_subs((__v16qi)a, (__v16qi)b);
}

Y_FORCE_INLINE __m128i _mm_subs_epi16(__m128i a, __m128i b) {
    return (__m128i)vec_subs((__v8hi)a, (__v8hi)b);
}

Y_FORCE_INLINE __m128i _mm_subs_epu8(__m128i a, __m128i b) {
    return (__m128i)vec_subs((__v16qu)a, (__v16qu)b);
}

Y_FORCE_INLINE __m128i _mm_subs_epu16(__m128i a, __m128i b) {
    return (__m128i)vec_subs((__v8hu)a, (__v8hu)b);
}

Y_FORCE_INLINE __m128i _mm_adds_epi8(__m128i a, __m128i b) {
    return (__m128i)vec_adds((__v16qi)a, (__v16qi)b);
}

Y_FORCE_INLINE __m128i _mm_adds_epi16(__m128i a, __m128i b) {
    return (__m128i)vec_adds((__v8hi)a, (__v8hi)b);
}

Y_FORCE_INLINE __m128i _mm_adds_epu8(__m128i a, __m128i b) {
    return (__m128i)vec_adds((__v16qu)a, (__v16qu)b);
}

Y_FORCE_INLINE __m128i _mm_adds_epu16(__m128i a, __m128i b) {
    return (__m128i)vec_adds((__v8hu)a, (__v8hu)b);
}

Y_FORCE_INLINE __m128d _mm_castsi128_pd(__m128i a) {
    return (__m128d)a;
}

Y_FORCE_INLINE void _mm_prefetch(const void *p, enum _mm_hint) {
    __builtin_prefetch(p);
}

Y_FORCE_INLINE __m128i _mm_hadd_epi16(__m128i a, __m128i b) {
    const __v16qu p = {  0,  1,  4,  5,  8,  9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29 };
    const __v16qu q = {  2,  3,  6,  7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31 };
    __v8hi c = vec_perm((__v8hi)a, (__v8hi)b, p);
    __v8hi d = vec_perm((__v8hi)a, (__v8hi)b, q);
    return (__m128i)vec_add(c, d);
}
