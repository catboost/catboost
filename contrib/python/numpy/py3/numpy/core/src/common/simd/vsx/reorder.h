#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_VSX_REORDER_H
#define _NPY_SIMD_VSX_REORDER_H

// combine lower part of two vectors
#define npyv__combinel(A, B) vec_mergeh((npyv_u64)(A), (npyv_u64)(B))
#define npyv_combinel_u8(A, B)  ((npyv_u8) npyv__combinel(A, B))
#define npyv_combinel_s8(A, B)  ((npyv_s8) npyv__combinel(A, B))
#define npyv_combinel_u16(A, B) ((npyv_u16)npyv__combinel(A, B))
#define npyv_combinel_s16(A, B) ((npyv_s16)npyv__combinel(A, B))
#define npyv_combinel_u32(A, B) ((npyv_u32)npyv__combinel(A, B))
#define npyv_combinel_s32(A, B) ((npyv_s32)npyv__combinel(A, B))
#define npyv_combinel_u64       vec_mergeh
#define npyv_combinel_s64       vec_mergeh
#define npyv_combinel_f32(A, B) ((npyv_f32)npyv__combinel(A, B))
#define npyv_combinel_f64       vec_mergeh

// combine higher part of two vectors
#define npyv__combineh(A, B) vec_mergel((npyv_u64)(A), (npyv_u64)(B))
#define npyv_combineh_u8(A, B)  ((npyv_u8) npyv__combineh(A, B))
#define npyv_combineh_s8(A, B)  ((npyv_s8) npyv__combineh(A, B))
#define npyv_combineh_u16(A, B) ((npyv_u16)npyv__combineh(A, B))
#define npyv_combineh_s16(A, B) ((npyv_s16)npyv__combineh(A, B))
#define npyv_combineh_u32(A, B) ((npyv_u32)npyv__combineh(A, B))
#define npyv_combineh_s32(A, B) ((npyv_s32)npyv__combineh(A, B))
#define npyv_combineh_u64       vec_mergel
#define npyv_combineh_s64       vec_mergel
#define npyv_combineh_f32(A, B) ((npyv_f32)npyv__combineh(A, B))
#define npyv_combineh_f64       vec_mergel

/*
 * combine: combine two vectors from lower and higher parts of two other vectors
 * zip: interleave two vectors
*/
#define NPYV_IMPL_VSX_COMBINE_ZIP(T_VEC, SFX)                  \
    NPY_FINLINE T_VEC##x2 npyv_combine_##SFX(T_VEC a, T_VEC b) \
    {                                                          \
        T_VEC##x2 r;                                           \
        r.val[0] = NPY_CAT(npyv_combinel_, SFX)(a, b);         \
        r.val[1] = NPY_CAT(npyv_combineh_, SFX)(a, b);         \
        return r;                                              \
    }                                                          \
    NPY_FINLINE T_VEC##x2 npyv_zip_##SFX(T_VEC a, T_VEC b)     \
    {                                                          \
        T_VEC##x2 r;                                           \
        r.val[0] = vec_mergeh(a, b);                           \
        r.val[1] = vec_mergel(a, b);                           \
        return r;                                              \
    }

NPYV_IMPL_VSX_COMBINE_ZIP(npyv_u8,  u8)
NPYV_IMPL_VSX_COMBINE_ZIP(npyv_s8,  s8)
NPYV_IMPL_VSX_COMBINE_ZIP(npyv_u16, u16)
NPYV_IMPL_VSX_COMBINE_ZIP(npyv_s16, s16)
NPYV_IMPL_VSX_COMBINE_ZIP(npyv_u32, u32)
NPYV_IMPL_VSX_COMBINE_ZIP(npyv_s32, s32)
NPYV_IMPL_VSX_COMBINE_ZIP(npyv_u64, u64)
NPYV_IMPL_VSX_COMBINE_ZIP(npyv_s64, s64)
NPYV_IMPL_VSX_COMBINE_ZIP(npyv_f32, f32)
NPYV_IMPL_VSX_COMBINE_ZIP(npyv_f64, f64)

// Reverse elements of each 64-bit lane
NPY_FINLINE npyv_u8 npyv_rev64_u8(npyv_u8 a)
{
#if defined(NPY_HAVE_VSX3) && ((defined(__GNUC__) && __GNUC__ > 7) || defined(__IBMC__))
    return (npyv_u8)vec_revb((npyv_u64)a);
#elif defined(NPY_HAVE_VSX3) && defined(NPY_HAVE_VSX_ASM)
    npyv_u8 ret;
    __asm__ ("xxbrd %x0,%x1" : "=wa" (ret) : "wa" (a));
    return ret;
#else
    const npyv_u8 idx = npyv_set_u8(
        7, 6, 5, 4, 3, 2, 1, 0,/*64*/15, 14, 13, 12, 11, 10, 9, 8
    );
    return vec_perm(a, a, idx);
#endif
}
NPY_FINLINE npyv_s8 npyv_rev64_s8(npyv_s8 a)
{ return (npyv_s8)npyv_rev64_u8((npyv_u8)a); }

NPY_FINLINE npyv_u16 npyv_rev64_u16(npyv_u16 a)
{
    const npyv_u8 idx = npyv_set_u8(
        6, 7, 4, 5, 2, 3, 0, 1,/*64*/14, 15, 12, 13, 10, 11, 8, 9
    );
    return vec_perm(a, a, idx);
}
NPY_FINLINE npyv_s16 npyv_rev64_s16(npyv_s16 a)
{ return (npyv_s16)npyv_rev64_u16((npyv_u16)a); }

NPY_FINLINE npyv_u32 npyv_rev64_u32(npyv_u32 a)
{
    const npyv_u8 idx = npyv_set_u8(
        4, 5, 6, 7, 0, 1, 2, 3,/*64*/12, 13, 14, 15, 8, 9, 10, 11
    );
    return vec_perm(a, a, idx);
}
NPY_FINLINE npyv_s32 npyv_rev64_s32(npyv_s32 a)
{ return (npyv_s32)npyv_rev64_u32((npyv_u32)a); }
NPY_FINLINE npyv_f32 npyv_rev64_f32(npyv_f32 a)
{ return (npyv_f32)npyv_rev64_u32((npyv_u32)a); }

#endif // _NPY_SIMD_VSX_REORDER_H
