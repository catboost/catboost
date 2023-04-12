#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_AVX512_REORDER_H
#define _NPY_SIMD_AVX512_REORDER_H

// combine lower part of two vectors
#define npyv_combinel_u8(A, B) _mm512_inserti64x4(A, _mm512_castsi512_si256(B), 1)
#define npyv_combinel_s8  npyv_combinel_u8
#define npyv_combinel_u16 npyv_combinel_u8
#define npyv_combinel_s16 npyv_combinel_u8
#define npyv_combinel_u32 npyv_combinel_u8
#define npyv_combinel_s32 npyv_combinel_u8
#define npyv_combinel_u64 npyv_combinel_u8
#define npyv_combinel_s64 npyv_combinel_u8
#define npyv_combinel_f64(A, B) _mm512_insertf64x4(A, _mm512_castpd512_pd256(B), 1)
#ifdef NPY_HAVE_AVX512DQ
    #define npyv_combinel_f32(A, B) \
        _mm512_insertf32x8(A, _mm512_castps512_ps256(B), 1)
#else
    #define npyv_combinel_f32(A, B) \
        _mm512_castsi512_ps(npyv_combinel_u8(_mm512_castps_si512(A), _mm512_castps_si512(B)))
#endif

// combine higher part of two vectors
#define npyv_combineh_u8(A, B) _mm512_inserti64x4(B, _mm512_extracti64x4_epi64(A, 1), 0)
#define npyv_combineh_s8  npyv_combineh_u8
#define npyv_combineh_u16 npyv_combineh_u8
#define npyv_combineh_s16 npyv_combineh_u8
#define npyv_combineh_u32 npyv_combineh_u8
#define npyv_combineh_s32 npyv_combineh_u8
#define npyv_combineh_u64 npyv_combineh_u8
#define npyv_combineh_s64 npyv_combineh_u8
#define npyv_combineh_f64(A, B) _mm512_insertf64x4(B, _mm512_extractf64x4_pd(A, 1), 0)
#ifdef NPY_HAVE_AVX512DQ
    #define npyv_combineh_f32(A, B) \
        _mm512_insertf32x8(B, _mm512_extractf32x8_ps(A, 1), 0)
#else
    #define npyv_combineh_f32(A, B) \
        _mm512_castsi512_ps(npyv_combineh_u8(_mm512_castps_si512(A), _mm512_castps_si512(B)))
#endif

// combine two vectors from lower and higher parts of two other vectors
NPY_FINLINE npyv_m512ix2 npyv__combine(__m512i a, __m512i b)
{
    npyv_m512ix2 r;
    r.val[0] = npyv_combinel_u8(a, b);
    r.val[1] = npyv_combineh_u8(a, b);
    return r;
}
NPY_FINLINE npyv_f32x2 npyv_combine_f32(__m512 a, __m512 b)
{
    npyv_f32x2 r;
    r.val[0] = npyv_combinel_f32(a, b);
    r.val[1] = npyv_combineh_f32(a, b);
    return r;
}
NPY_FINLINE npyv_f64x2 npyv_combine_f64(__m512d a, __m512d b)
{
    npyv_f64x2 r;
    r.val[0] = npyv_combinel_f64(a, b);
    r.val[1] = npyv_combineh_f64(a, b);
    return r;
}
#define npyv_combine_u8  npyv__combine
#define npyv_combine_s8  npyv__combine
#define npyv_combine_u16 npyv__combine
#define npyv_combine_s16 npyv__combine
#define npyv_combine_u32 npyv__combine
#define npyv_combine_s32 npyv__combine
#define npyv_combine_u64 npyv__combine
#define npyv_combine_s64 npyv__combine

// interleave two vectors
#ifndef NPY_HAVE_AVX512BW
    NPYV_IMPL_AVX512_FROM_AVX2_2ARG(npyv__unpacklo_epi8,  _mm256_unpacklo_epi8)
    NPYV_IMPL_AVX512_FROM_AVX2_2ARG(npyv__unpackhi_epi8,  _mm256_unpackhi_epi8)
    NPYV_IMPL_AVX512_FROM_AVX2_2ARG(npyv__unpacklo_epi16, _mm256_unpacklo_epi16)
    NPYV_IMPL_AVX512_FROM_AVX2_2ARG(npyv__unpackhi_epi16, _mm256_unpackhi_epi16)
#endif

NPY_FINLINE npyv_u64x2 npyv_zip_u64(__m512i a, __m512i b)
{
    npyv_u64x2 r;
    r.val[0] = _mm512_permutex2var_epi64(a, npyv_set_u64(0, 8, 1, 9, 2, 10, 3, 11), b);
    r.val[1] = _mm512_permutex2var_epi64(a, npyv_set_u64(4, 12, 5, 13, 6, 14, 7, 15), b);
    return r;
}
#define npyv_zip_s64 npyv_zip_u64

NPY_FINLINE npyv_u8x2 npyv_zip_u8(__m512i a, __m512i b)
{
    npyv_u8x2 r;
#ifdef NPY_HAVE_AVX512VBMI
    r.val[0] = _mm512_permutex2var_epi8(a,
        npyv_set_u8(0,  64, 1,  65, 2,  66, 3,  67, 4,  68, 5,  69, 6,  70, 7,  71,
                    8,  72, 9,  73, 10, 74, 11, 75, 12, 76, 13, 77, 14, 78, 15, 79,
                    16, 80, 17, 81, 18, 82, 19, 83, 20, 84, 21, 85, 22, 86, 23, 87,
                    24, 88, 25, 89, 26, 90, 27, 91, 28, 92, 29, 93, 30, 94, 31, 95), b);
    r.val[1] = _mm512_permutex2var_epi8(a,
        npyv_set_u8(32, 96,  33, 97,  34, 98,  35, 99,  36, 100, 37, 101, 38, 102, 39, 103,
                    40, 104, 41, 105, 42, 106, 43, 107, 44, 108, 45, 109, 46, 110, 47, 111,
                    48, 112, 49, 113, 50, 114, 51, 115, 52, 116, 53, 117, 54, 118, 55, 119,
                    56, 120, 57, 121, 58, 122, 59, 123, 60, 124, 61, 125, 62, 126, 63, 127), b);
#else
    #ifdef NPY_HAVE_AVX512BW
    __m512i ab0 = _mm512_unpacklo_epi8(a, b);
    __m512i ab1 = _mm512_unpackhi_epi8(a, b);
    #else
    __m512i ab0 = npyv__unpacklo_epi8(a, b);
    __m512i ab1 = npyv__unpackhi_epi8(a, b);
    #endif
    r.val[0] = _mm512_permutex2var_epi64(ab0, npyv_set_u64(0, 1, 8, 9, 2, 3, 10, 11), ab1);
    r.val[1] = _mm512_permutex2var_epi64(ab0, npyv_set_u64(4, 5, 12, 13, 6, 7, 14, 15), ab1);
#endif
    return r;
}
#define npyv_zip_s8 npyv_zip_u8

NPY_FINLINE npyv_u16x2 npyv_zip_u16(__m512i a, __m512i b)
{
    npyv_u16x2 r;
#ifdef NPY_HAVE_AVX512BW
    r.val[0] = _mm512_permutex2var_epi16(a,
        npyv_set_u16(0, 32, 1, 33, 2, 34, 3, 35, 4, 36, 5, 37, 6, 38, 7, 39,
                     8, 40, 9, 41, 10, 42, 11, 43, 12, 44, 13, 45, 14, 46, 15, 47), b);
    r.val[1] = _mm512_permutex2var_epi16(a,
        npyv_set_u16(16, 48, 17, 49, 18, 50, 19, 51, 20, 52, 21, 53, 22, 54, 23, 55,
                     24, 56, 25, 57, 26, 58, 27, 59, 28, 60, 29, 61, 30, 62, 31, 63), b);
#else
    __m512i ab0 = npyv__unpacklo_epi16(a, b);
    __m512i ab1 = npyv__unpackhi_epi16(a, b);
    r.val[0] = _mm512_permutex2var_epi64(ab0, npyv_set_u64(0, 1, 8, 9, 2, 3, 10, 11), ab1);
    r.val[1] = _mm512_permutex2var_epi64(ab0, npyv_set_u64(4, 5, 12, 13, 6, 7, 14, 15), ab1);
#endif
    return r;
}
#define npyv_zip_s16 npyv_zip_u16

NPY_FINLINE npyv_u32x2 npyv_zip_u32(__m512i a, __m512i b)
{
    npyv_u32x2 r;
    r.val[0] = _mm512_permutex2var_epi32(a,
        npyv_set_u32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23), b);
    r.val[1] = _mm512_permutex2var_epi32(a,
        npyv_set_u32(8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31), b);
    return r;
}
#define npyv_zip_s32 npyv_zip_u32

NPY_FINLINE npyv_f32x2 npyv_zip_f32(__m512 a, __m512 b)
{
    npyv_f32x2 r;
    r.val[0] = _mm512_permutex2var_ps(a,
        npyv_set_u32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23), b);
    r.val[1] = _mm512_permutex2var_ps(a,
        npyv_set_u32(8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31), b);
    return r;
}

NPY_FINLINE npyv_f64x2 npyv_zip_f64(__m512d a, __m512d b)
{
    npyv_f64x2 r;
    r.val[0] = _mm512_permutex2var_pd(a, npyv_set_u64(0, 8, 1, 9, 2, 10, 3, 11), b);
    r.val[1] = _mm512_permutex2var_pd(a, npyv_set_u64(4, 12, 5, 13, 6, 14, 7, 15), b);
    return r;
}

// Reverse elements of each 64-bit lane
NPY_FINLINE npyv_u8 npyv_rev64_u8(npyv_u8 a)
{
#ifdef NPY_HAVE_AVX512BW
    const __m512i idx = npyv_set_u8(
        7, 6, 5, 4, 3, 2, 1, 0,/*64*/15, 14, 13, 12, 11, 10, 9, 8,
        7, 6, 5, 4, 3, 2, 1, 0,/*64*/15, 14, 13, 12, 11, 10, 9, 8,
        7, 6, 5, 4, 3, 2, 1, 0,/*64*/15, 14, 13, 12, 11, 10, 9, 8,
        7, 6, 5, 4, 3, 2, 1, 0,/*64*/15, 14, 13, 12, 11, 10, 9, 8
    );
    return _mm512_shuffle_epi8(a, idx);
#else
    const __m256i idx = _mm256_setr_epi8(
        7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8,
        7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8
    );
    __m256i lo = _mm256_shuffle_epi8(npyv512_lower_si256(a),  idx);
    __m256i hi = _mm256_shuffle_epi8(npyv512_higher_si256(a), idx);
    return npyv512_combine_si256(lo, hi);
#endif
}
#define npyv_rev64_s8 npyv_rev64_u8

NPY_FINLINE npyv_u16 npyv_rev64_u16(npyv_u16 a)
{
#ifdef NPY_HAVE_AVX512BW
    const __m512i idx = npyv_set_u8(
        6, 7, 4, 5, 2, 3, 0, 1,/*64*/14, 15, 12, 13, 10, 11, 8, 9,
        6, 7, 4, 5, 2, 3, 0, 1,/*64*/14, 15, 12, 13, 10, 11, 8, 9,
        6, 7, 4, 5, 2, 3, 0, 1,/*64*/14, 15, 12, 13, 10, 11, 8, 9,
        6, 7, 4, 5, 2, 3, 0, 1,/*64*/14, 15, 12, 13, 10, 11, 8, 9
    );
    return _mm512_shuffle_epi8(a, idx);
#else
    const __m256i idx = _mm256_setr_epi8(
        6, 7, 4, 5, 2, 3, 0, 1,/*64*/14, 15, 12, 13, 10, 11, 8, 9,
        6, 7, 4, 5, 2, 3, 0, 1,/*64*/14, 15, 12, 13, 10, 11, 8, 9
    );
    __m256i lo = _mm256_shuffle_epi8(npyv512_lower_si256(a),  idx);
    __m256i hi = _mm256_shuffle_epi8(npyv512_higher_si256(a), idx);
    return npyv512_combine_si256(lo, hi);
#endif
}
#define npyv_rev64_s16 npyv_rev64_u16

NPY_FINLINE npyv_u32 npyv_rev64_u32(npyv_u32 a)
{
    return _mm512_shuffle_epi32(a, (_MM_PERM_ENUM)_MM_SHUFFLE(2, 3, 0, 1));
}
#define npyv_rev64_s32 npyv_rev64_u32

NPY_FINLINE npyv_f32 npyv_rev64_f32(npyv_f32 a)
{
    return _mm512_shuffle_ps(a, a, (_MM_PERM_ENUM)_MM_SHUFFLE(2, 3, 0, 1));
}

#endif // _NPY_SIMD_AVX512_REORDER_H
