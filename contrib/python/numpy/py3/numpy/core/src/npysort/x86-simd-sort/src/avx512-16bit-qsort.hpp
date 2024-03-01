/*******************************************************************
 * Copyright (C) 2022 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 * Authors: Raghuveer Devulapalli <raghuveer.devulapalli@intel.com>
 * ****************************************************************/

#ifndef AVX512_QSORT_16BIT
#define AVX512_QSORT_16BIT

#include "avx512-16bit-common.h"

struct float16 {
    uint16_t val;
};

template <>
struct zmm_vector<float16> {
    using type_t = uint16_t;
    using zmm_t = __m512i;
    using ymm_t = __m256i;
    using opmask_t = __mmask32;
    static const uint8_t numlanes = 32;

    static zmm_t get_network(int index)
    {
        return _mm512_loadu_si512(&network[index - 1][0]);
    }
    static type_t type_max()
    {
        return X86_SIMD_SORT_INFINITYH;
    }
    static type_t type_min()
    {
        return X86_SIMD_SORT_NEGINFINITYH;
    }
    static zmm_t zmm_max()
    {
        return _mm512_set1_epi16(type_max());
    }
    static opmask_t knot_opmask(opmask_t x)
    {
        return _knot_mask32(x);
    }

    static opmask_t ge(zmm_t x, zmm_t y)
    {
        zmm_t sign_x = _mm512_and_si512(x, _mm512_set1_epi16(0x8000));
        zmm_t sign_y = _mm512_and_si512(y, _mm512_set1_epi16(0x8000));
        zmm_t exp_x = _mm512_and_si512(x, _mm512_set1_epi16(0x7c00));
        zmm_t exp_y = _mm512_and_si512(y, _mm512_set1_epi16(0x7c00));
        zmm_t mant_x = _mm512_and_si512(x, _mm512_set1_epi16(0x3ff));
        zmm_t mant_y = _mm512_and_si512(y, _mm512_set1_epi16(0x3ff));

        __mmask32 mask_ge = _mm512_cmp_epu16_mask(
                sign_x, sign_y, _MM_CMPINT_LT); // only greater than
        __mmask32 sign_eq = _mm512_cmpeq_epu16_mask(sign_x, sign_y);
        __mmask32 neg = _mm512_mask_cmpeq_epu16_mask(
                sign_eq,
                sign_x,
                _mm512_set1_epi16(0x8000)); // both numbers are -ve

        // compare exponents only if signs are equal:
        mask_ge = mask_ge
                | _mm512_mask_cmp_epu16_mask(
                          sign_eq, exp_x, exp_y, _MM_CMPINT_NLE);
        // get mask for elements for which both sign and exponents are equal:
        __mmask32 exp_eq = _mm512_mask_cmpeq_epu16_mask(sign_eq, exp_x, exp_y);

        // compare mantissa for elements for which both sign and expponent are equal:
        mask_ge = mask_ge
                | _mm512_mask_cmp_epu16_mask(
                          exp_eq, mant_x, mant_y, _MM_CMPINT_NLT);
        return _kxor_mask32(mask_ge, neg);
    }
    static zmm_t loadu(void const *mem)
    {
        return _mm512_loadu_si512(mem);
    }
    static zmm_t max(zmm_t x, zmm_t y)
    {
        return _mm512_mask_mov_epi16(y, ge(x, y), x);
    }
    static void mask_compressstoreu(void *mem, opmask_t mask, zmm_t x)
    {
        // AVX512_VBMI2
        return _mm512_mask_compressstoreu_epi16(mem, mask, x);
    }
    static zmm_t mask_loadu(zmm_t x, opmask_t mask, void const *mem)
    {
        // AVX512BW
        return _mm512_mask_loadu_epi16(x, mask, mem);
    }
    static zmm_t mask_mov(zmm_t x, opmask_t mask, zmm_t y)
    {
        return _mm512_mask_mov_epi16(x, mask, y);
    }
    static void mask_storeu(void *mem, opmask_t mask, zmm_t x)
    {
        return _mm512_mask_storeu_epi16(mem, mask, x);
    }
    static zmm_t min(zmm_t x, zmm_t y)
    {
        return _mm512_mask_mov_epi16(x, ge(x, y), y);
    }
    static zmm_t permutexvar(__m512i idx, zmm_t zmm)
    {
        return _mm512_permutexvar_epi16(idx, zmm);
    }
    // Apparently this is a terrible for perf, npy_half_to_float seems to work
    // better
    //static float uint16_to_float(uint16_t val)
    //{
    //    // Ideally use _mm_loadu_si16, but its only gcc > 11.x
    //    // TODO: use inline ASM? https://godbolt.org/z/aGYvh7fMM
    //    __m128i xmm = _mm_maskz_loadu_epi16(0x01, &val);
    //    __m128 xmm2 = _mm_cvtph_ps(xmm);
    //    return _mm_cvtss_f32(xmm2);
    //}
    static type_t float_to_uint16(float val)
    {
        __m128 xmm = _mm_load_ss(&val);
        __m128i xmm2 = _mm_cvtps_ph(xmm, _MM_FROUND_NO_EXC);
        return _mm_extract_epi16(xmm2, 0);
    }
    static type_t reducemax(zmm_t v)
    {
        __m512 lo = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(v, 0));
        __m512 hi = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(v, 1));
        float lo_max = _mm512_reduce_max_ps(lo);
        float hi_max = _mm512_reduce_max_ps(hi);
        return float_to_uint16(std::max(lo_max, hi_max));
    }
    static type_t reducemin(zmm_t v)
    {
        __m512 lo = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(v, 0));
        __m512 hi = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(v, 1));
        float lo_max = _mm512_reduce_min_ps(lo);
        float hi_max = _mm512_reduce_min_ps(hi);
        return float_to_uint16(std::min(lo_max, hi_max));
    }
    static zmm_t set1(type_t v)
    {
        return _mm512_set1_epi16(v);
    }
    template <uint8_t mask>
    static zmm_t shuffle(zmm_t zmm)
    {
        zmm = _mm512_shufflehi_epi16(zmm, (_MM_PERM_ENUM)mask);
        return _mm512_shufflelo_epi16(zmm, (_MM_PERM_ENUM)mask);
    }
    static void storeu(void *mem, zmm_t x)
    {
        return _mm512_storeu_si512(mem, x);
    }
};

template <>
struct zmm_vector<int16_t> {
    using type_t = int16_t;
    using zmm_t = __m512i;
    using ymm_t = __m256i;
    using opmask_t = __mmask32;
    static const uint8_t numlanes = 32;

    static zmm_t get_network(int index)
    {
        return _mm512_loadu_si512(&network[index - 1][0]);
    }
    static type_t type_max()
    {
        return X86_SIMD_SORT_MAX_INT16;
    }
    static type_t type_min()
    {
        return X86_SIMD_SORT_MIN_INT16;
    }
    static zmm_t zmm_max()
    {
        return _mm512_set1_epi16(type_max());
    }
    static opmask_t knot_opmask(opmask_t x)
    {
        return _knot_mask32(x);
    }

    static opmask_t ge(zmm_t x, zmm_t y)
    {
        return _mm512_cmp_epi16_mask(x, y, _MM_CMPINT_NLT);
    }
    static zmm_t loadu(void const *mem)
    {
        return _mm512_loadu_si512(mem);
    }
    static zmm_t max(zmm_t x, zmm_t y)
    {
        return _mm512_max_epi16(x, y);
    }
    static void mask_compressstoreu(void *mem, opmask_t mask, zmm_t x)
    {
        // AVX512_VBMI2
        return _mm512_mask_compressstoreu_epi16(mem, mask, x);
    }
    static zmm_t mask_loadu(zmm_t x, opmask_t mask, void const *mem)
    {
        // AVX512BW
        return _mm512_mask_loadu_epi16(x, mask, mem);
    }
    static zmm_t mask_mov(zmm_t x, opmask_t mask, zmm_t y)
    {
        return _mm512_mask_mov_epi16(x, mask, y);
    }
    static void mask_storeu(void *mem, opmask_t mask, zmm_t x)
    {
        return _mm512_mask_storeu_epi16(mem, mask, x);
    }
    static zmm_t min(zmm_t x, zmm_t y)
    {
        return _mm512_min_epi16(x, y);
    }
    static zmm_t permutexvar(__m512i idx, zmm_t zmm)
    {
        return _mm512_permutexvar_epi16(idx, zmm);
    }
    static type_t reducemax(zmm_t v)
    {
        zmm_t lo = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(v, 0));
        zmm_t hi = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(v, 1));
        type_t lo_max = (type_t)_mm512_reduce_max_epi32(lo);
        type_t hi_max = (type_t)_mm512_reduce_max_epi32(hi);
        return std::max(lo_max, hi_max);
    }
    static type_t reducemin(zmm_t v)
    {
        zmm_t lo = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(v, 0));
        zmm_t hi = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(v, 1));
        type_t lo_min = (type_t)_mm512_reduce_min_epi32(lo);
        type_t hi_min = (type_t)_mm512_reduce_min_epi32(hi);
        return std::min(lo_min, hi_min);
    }
    static zmm_t set1(type_t v)
    {
        return _mm512_set1_epi16(v);
    }
    template <uint8_t mask>
    static zmm_t shuffle(zmm_t zmm)
    {
        zmm = _mm512_shufflehi_epi16(zmm, (_MM_PERM_ENUM)mask);
        return _mm512_shufflelo_epi16(zmm, (_MM_PERM_ENUM)mask);
    }
    static void storeu(void *mem, zmm_t x)
    {
        return _mm512_storeu_si512(mem, x);
    }
};
template <>
struct zmm_vector<uint16_t> {
    using type_t = uint16_t;
    using zmm_t = __m512i;
    using ymm_t = __m256i;
    using opmask_t = __mmask32;
    static const uint8_t numlanes = 32;

    static zmm_t get_network(int index)
    {
        return _mm512_loadu_si512(&network[index - 1][0]);
    }
    static type_t type_max()
    {
        return X86_SIMD_SORT_MAX_UINT16;
    }
    static type_t type_min()
    {
        return 0;
    }
    static zmm_t zmm_max()
    {
        return _mm512_set1_epi16(type_max());
    }

    static opmask_t knot_opmask(opmask_t x)
    {
        return _knot_mask32(x);
    }
    static opmask_t ge(zmm_t x, zmm_t y)
    {
        return _mm512_cmp_epu16_mask(x, y, _MM_CMPINT_NLT);
    }
    static zmm_t loadu(void const *mem)
    {
        return _mm512_loadu_si512(mem);
    }
    static zmm_t max(zmm_t x, zmm_t y)
    {
        return _mm512_max_epu16(x, y);
    }
    static void mask_compressstoreu(void *mem, opmask_t mask, zmm_t x)
    {
        return _mm512_mask_compressstoreu_epi16(mem, mask, x);
    }
    static zmm_t mask_loadu(zmm_t x, opmask_t mask, void const *mem)
    {
        return _mm512_mask_loadu_epi16(x, mask, mem);
    }
    static zmm_t mask_mov(zmm_t x, opmask_t mask, zmm_t y)
    {
        return _mm512_mask_mov_epi16(x, mask, y);
    }
    static void mask_storeu(void *mem, opmask_t mask, zmm_t x)
    {
        return _mm512_mask_storeu_epi16(mem, mask, x);
    }
    static zmm_t min(zmm_t x, zmm_t y)
    {
        return _mm512_min_epu16(x, y);
    }
    static zmm_t permutexvar(__m512i idx, zmm_t zmm)
    {
        return _mm512_permutexvar_epi16(idx, zmm);
    }
    static type_t reducemax(zmm_t v)
    {
        zmm_t lo = _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(v, 0));
        zmm_t hi = _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(v, 1));
        type_t lo_max = (type_t)_mm512_reduce_max_epi32(lo);
        type_t hi_max = (type_t)_mm512_reduce_max_epi32(hi);
        return std::max(lo_max, hi_max);
    }
    static type_t reducemin(zmm_t v)
    {
        zmm_t lo = _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(v, 0));
        zmm_t hi = _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(v, 1));
        type_t lo_min = (type_t)_mm512_reduce_min_epi32(lo);
        type_t hi_min = (type_t)_mm512_reduce_min_epi32(hi);
        return std::min(lo_min, hi_min);
    }
    static zmm_t set1(type_t v)
    {
        return _mm512_set1_epi16(v);
    }
    template <uint8_t mask>
    static zmm_t shuffle(zmm_t zmm)
    {
        zmm = _mm512_shufflehi_epi16(zmm, (_MM_PERM_ENUM)mask);
        return _mm512_shufflelo_epi16(zmm, (_MM_PERM_ENUM)mask);
    }
    static void storeu(void *mem, zmm_t x)
    {
        return _mm512_storeu_si512(mem, x);
    }
};

template <>
inline bool comparison_func<zmm_vector<float16>>(const uint16_t &a, const uint16_t &b)
{
    uint16_t signa = a & 0x8000, signb = b & 0x8000;
    uint16_t expa = a & 0x7c00, expb = b & 0x7c00;
    uint16_t manta = a & 0x3ff, mantb = b & 0x3ff;
    if (signa != signb) {
        // opposite signs
        return a > b;
    }
    else if (signa > 0) {
        // both -ve
        if (expa != expb) { return expa > expb; }
        else {
            return manta > mantb;
        }
    }
    else {
        // both +ve
        if (expa != expb) { return expa < expb; }
        else {
            return manta < mantb;
        }
    }

    //return npy_half_to_float(a) < npy_half_to_float(b);
}

X86_SIMD_SORT_INLINE int64_t replace_nan_with_inf(uint16_t *arr,
                                                  int64_t arrsize)
{
    int64_t nan_count = 0;
    __mmask16 loadmask = 0xFFFF;
    while (arrsize > 0) {
        if (arrsize < 16) { loadmask = (0x0001 << arrsize) - 0x0001; }
        __m256i in_zmm = _mm256_maskz_loadu_epi16(loadmask, arr);
        __m512 in_zmm_asfloat = _mm512_cvtph_ps(in_zmm);
        __mmask16 nanmask = _mm512_cmp_ps_mask(
                in_zmm_asfloat, in_zmm_asfloat, _CMP_NEQ_UQ);
        nan_count += _mm_popcnt_u32((int32_t)nanmask);
        _mm256_mask_storeu_epi16(arr, nanmask, YMM_MAX_HALF);
        arr += 16;
        arrsize -= 16;
    }
    return nan_count;
}

X86_SIMD_SORT_INLINE void
replace_inf_with_nan(uint16_t *arr, int64_t arrsize, int64_t nan_count)
{
    for (int64_t ii = arrsize - 1; nan_count > 0; --ii) {
        arr[ii] = 0xFFFF;
        nan_count -= 1;
    }
}

template <>
inline void avx512_qselect(int16_t *arr, int64_t k, int64_t arrsize)
{
    if (arrsize > 1) {
        qselect_16bit_<zmm_vector<int16_t>, int16_t>(
                arr, k, 0, arrsize - 1, 2 * (int64_t)log2(arrsize));
    }
}

template <>
inline void avx512_qselect(uint16_t *arr, int64_t k, int64_t arrsize)
{
    if (arrsize > 1) {
        qselect_16bit_<zmm_vector<uint16_t>, uint16_t>(
                arr, k, 0, arrsize - 1, 2 * (int64_t)log2(arrsize));
    }
}

inline void avx512_qselect_fp16(uint16_t *arr, int64_t k, int64_t arrsize)
{
    if (arrsize > 1) {
        int64_t nan_count = replace_nan_with_inf(arr, arrsize);
        qselect_16bit_<zmm_vector<float16>, uint16_t>(
                arr, k, 0, arrsize - 1, 2 * (int64_t)log2(arrsize));
        replace_inf_with_nan(arr, arrsize, nan_count);
    }
}

template <>
inline void avx512_qsort(int16_t *arr, int64_t arrsize)
{
    if (arrsize > 1) {
        qsort_16bit_<zmm_vector<int16_t>, int16_t>(
                arr, 0, arrsize - 1, 2 * (int64_t)log2(arrsize));
    }
}

template <>
inline void avx512_qsort(uint16_t *arr, int64_t arrsize)
{
    if (arrsize > 1) {
        qsort_16bit_<zmm_vector<uint16_t>, uint16_t>(
                arr, 0, arrsize - 1, 2 * (int64_t)log2(arrsize));
    }
}

inline void avx512_qsort_fp16(uint16_t *arr, int64_t arrsize)
{
    if (arrsize > 1) {
        int64_t nan_count = replace_nan_with_inf(arr, arrsize);
        qsort_16bit_<zmm_vector<float16>, uint16_t>(
                arr, 0, arrsize - 1, 2 * (int64_t)log2(arrsize));
        replace_inf_with_nan(arr, arrsize, nan_count);
    }
}

#endif // AVX512_QSORT_16BIT
