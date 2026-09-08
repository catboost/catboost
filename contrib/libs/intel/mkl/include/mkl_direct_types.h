/*******************************************************************************
* Copyright 2016-2022 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/*
!  Content:
!      Intel(R) oneAPI Math Kernel Library (oneMKL) types definition for functions
!      than can be inlined
!******************************************************************************/
#include "mkl_types.h"
#include <math.h>

#undef mkl_dc_type
#undef mkl_dc_real_type
#undef mkl_dc_native_type
#undef MKL_REAL_DATA_TYPE
#undef mkl_dc_veclen
#undef MKL_DC_PREC_LETTER
#undef MKL_DC_NR
#undef mkl_dc_tsqrt
#undef MKL_DC_IS_ONE
#undef MKL_DC_IS_ZERO
#undef MKL_DC_IS_ZERO2
#undef MKL_DC_ABS1
#undef MKL_DC_SET_ONE
#undef MKL_DC_SET_ZERO
#undef MKL_DC_ZERO_IMAG
#undef MKL_DC_NON_POS
#undef MKL_DC_ADD
#undef MKL_DC_SUB
#undef MKL_DC_CONJ
#undef MKL_DC_DIV
#undef MKL_DC_SUB_MUL
#undef MKL_DC_SUB_MUL_CONJ
#undef MKL_DC_MUL
#undef MKL_DC_MUL_ADD
#undef MKL_DC_ZERO_C
#undef MKL_DC_MUL_C
#undef MKL_DC_DIV_B
#undef MKL_DC_SWAP
#undef MKL_DC_NEG
#undef MKL_DC_INV
#undef MKL_DC_SQRT
#undef MKL_DC_CONVERT_INT
#undef MKL_DC_NRM2
#undef MKL_DC_NRM2_VEC
#undef MKL_DC_R_SQRT
#undef MKL_DC_XLAMCH
#undef MKL_DC_PRAGMA_VECTOR

#if defined(MKL_DOUBLE)
#define mkl_dc_type double
#define mkl_dc_real_type double
#define mkl_dc_native_type double
#define MKL_REAL_DATA_TYPE
#define mkl_dc_veclen 4
#define MKL_DC_PREC_LETTER d
#define MKL_DC_NR 1
#define mkl_dc_tsqrt sqrt

#elif defined(MKL_SINGLE)
#define mkl_dc_type float
#define mkl_dc_real_type float
#define mkl_dc_native_type float
#define MKL_REAL_DATA_TYPE
#define mkl_dc_veclen 8
#define MKL_DC_PREC_LETTER s
#define MKL_DC_NR 1
#define mkl_dc_tsqrt sqrtf

#elif defined(MKL_COMPLEX)
#define mkl_dc_type MKL_Complex8
#define mkl_dc_real_type float
#define mkl_dc_native_type float _Complex
#define mkl_dc_veclen 4
#define MKL_DC_PREC_LETTER c
#define MKL_DC_NR 2
#define mkl_dc_tsqrt sqrtf

#elif defined(MKL_COMPLEX16)
#define mkl_dc_type MKL_Complex16
#define mkl_dc_real_type double
#define mkl_dc_native_type double _Complex
#define mkl_dc_veclen 2
#define MKL_DC_PREC_LETTER z
#define MKL_DC_NR 2
#define mkl_dc_tsqrt sqrt

#endif

#define MKL_DC_ACCESS1D(a, i) (a[i])
#define MKL_DC_R_SUB(r, x, y) ((r) = (x) - (y))
#define MKL_DC_R_MUL(z, x, y) ((z) = (x) * (y))
#define MKL_DC_R_DIV(r, x, y) ((r) = (x) / (y))
#define MKL_DC_R_INV(x, y) do { \
    mkl_dc_real_type one; \
    MKL_DC_SET_R_ONE(one); \
    MKL_DC_R_DIV(x, one, y); \
} while (0)
#define MKL_DC_R_NEG(r, x) ((r) = -(x))

#if defined(MKL_COMPLEX) || defined(MKL_COMPLEX16)
#define MKL_DC_ABS1(x) (MKL_DC_ABS((x).real) + MKL_DC_ABS((x).imag))
#define MKL_DC_SET_ONE(x) ((x).real = 1.0, (x).imag = 0.0)
#define MKL_DC_IS_ONE(x) ((x).real == 1.0 && (x).imag == 0.0)
#define MKL_DC_IS_ZERO(x) ((x).real == 0.0 && (x).imag == 0.0)
#define MKL_DC_IS_ZERO2(r, x) do { \
    (r) = ((x).real == 0.0 && (x).imag == 0.0); \
} while(0);
#define MKL_DC_SET_ZERO(x) ((x).real = 0.0, (x).imag = 0.0)
#define MKL_DC_ZERO_IMAG(x) ((x).real = (x).real, (x).imag = 0.0)
#define MKL_DC_NON_POS(x) (!((x).real > 0.0))
#define MKL_DC_ADD(x, y, z) do { \
	(x).real = (y).real + (z).real; \
	(x).imag = (y).imag + (z).imag; \
} while (0)
#define MKL_DC_SUB(x, y, z) do { \
	(x).real = (y).real - (z).real; \
	(x).imag = (y).imag - (z).imag; \
} while (0)
#define MKL_DC_CONJ(x, y) ((x).real = (y).real, (x).imag = - (y).imag)
#define MKL_DC_MUL(z, x, y) do { \
	mkl_dc_type t; \
	t.real = (x).real * (y).real - (x).imag * (y).imag; \
	t.imag = (x).real * (y).imag + (x).imag * (y).real; \
	(z) = t; \
} while (0)
#define MKL_DC_MUL_ADD(r, x, y, z) do { \
	mkl_dc_type t; \
	t.real = (x).real * (y).real - (x).imag * (y).imag; \
	t.imag = (x).real * (y).imag + (x).imag * (y).real; \
	(r).real = t.real + (z).real; \
	(r).imag = t.imag + (z).imag; \
} while (0)
#define MKL_DC_DIV(z, x, y) do { \
	mkl_dc_type t; \
	mkl_dc_real_type r; \
	if (MKL_DC_ABS((y).imag) <= MKL_DC_ABS((y).real)) { \
		r = (y).imag / (y).real; \
		t.real = ((x).real + (x).imag * r) / ((y).real + (y).imag * r); \
		t.imag = ((x).imag - (x).real * r) / ((y).real + (y).imag * r); \
	} else { \
		r = (y).real / (y).imag; \
		t.real = ((x).imag + (x).real * r) / ((y).imag + (y).real * r); \
		t.imag = (-(x).real + (x).imag * r) / ((y).imag + (y).real * r); \
	} \
	(z) = t; \
} while (0)
#define MKL_DC_SUB_MUL(r, x, y, z) do { \
	mkl_dc_type t; \
	t.real = (y).real * (z).real - (y).imag * (z).imag; \
	t.imag = (y).real * (z).imag + (y).imag * (z).real; \
	(r).real = (x).real - t.real; \
	(r).imag = (x).imag - t.imag; \
} while (0)
#define MKL_DC_SUB_MUL_CONJ(r, x, y, z) do { \
	mkl_dc_type t; \
	t.real = (y).real * (z).real + (y).imag * (z).imag; \
	t.imag = (y).imag * (z).real - (y).real * (z).imag; \
	(r).real = (x).real - t.real; \
	(r).imag = (x).imag - t.imag; \
} while (0)
#define MKL_DC_SWAP(x, y) do { \
    mkl_dc_type t; \
    t.real = x.real; \
    t.imag = x.imag; \
    x.real = y.real; \
    x.imag = y.imag; \
    y.real = t.real; \
    y.imag = t.imag; \
} while (0)
#define MKL_DC_NEG(x, y) ((x).real = -(y).real, (x).imag = -(y).imag)
#define MKL_DC_INV(x, y) do { \
    mkl_dc_type one; \
    MKL_DC_SET_ONE(one); \
    MKL_DC_DIV(x, one, y); \
} while (0)
#define MKL_DC_SQRT(x, y) ((x).real = mkl_dc_tsqrt((y).real), (x).imag = (y).imag)
#define MKL_DC_CONVERT_INT(x, y) do { \
    x.real = (mkl_dc_real_type)(y); \
    x.imag = (mkl_dc_real_type)(0); \
} while (0)
#define MKL_DC_NRM2(x) ((x).real * (x).real + (x).imag * (x).imag)
#define MKL_DC_CMPLX(r, x, y) do { \
    (r).real = (x); \
    (r).imag = (y); \
} while (0)
#define MKL_DC_REIM(re, im, x) do { \
    (re) = (x).real; \
    (im) = (x).imag; \
} while (0)

#else
#define MKL_DC_ABS1(x) MKL_DC_ABS(x)
#define MKL_DC_SET_ONE(x) ((x) = 1.0)
#define MKL_DC_IS_ONE(x) ((x) == 1.0)
#define MKL_DC_IS_ZERO(x) ((x) == 0.0)
#define MKL_DC_IS_ZERO2(r, x) do { \
    (r) = ((x) == 0.0); \
} while(0);
#define MKL_DC_SET_ZERO(x) ((x) = 0.0)
#define MKL_DC_ZERO_IMAG(x) ((x) = (x))
#define MKL_DC_NON_POS(x) (!((x) > 0.0))
#define MKL_DC_ADD(x, y, z) ((x) = (y) + (z))
#define MKL_DC_SUB(x, y, z) MKL_DC_R_SUB(x, y, z)
#define MKL_DC_CONJ(x, y) ((x) = (y))
#define MKL_DC_MUL(z, x, y) MKL_DC_R_MUL(z, x, y)
#define MKL_DC_MUL_ADD(r, x, y, z) ((r) = (x) * (y) + (z))
#define MKL_DC_DIV(r, x, y) MKL_DC_R_DIV(r, x, y)
#define MKL_DC_INV(x, y) MKL_DC_R_INV(x, y)
#define MKL_DC_SUB_MUL(r, x, y, z) ((r) = (x) - (y) * (z))
#define MKL_DC_SUB_MUL_CONJ(r, x, y, z) ((r) = (x) - (y) * (z))
#define MKL_DC_SWAP(x, y) do { \
    mkl_dc_type t; \
    t = x; \
    x = y; \
    y = t; \
} while (0)
#define MKL_DC_NEG(x, y) MKL_DC_R_NEG(x, y)
#define MKL_DC_SQRT(x, y) ((x) = mkl_dc_tsqrt(y))
#define MKL_DC_CONVERT_INT(x, y) ((x) = (mkl_dc_type)(y))
#define MKL_DC_NRM2(x) ((x) * (x))
#endif

#define MKL_DC_MisN(t) ((t) == 'N' || (t) == 'n')
#define MKL_DC_MisT(c) ((c) == 'T' || (c) == 't')
#define MKL_DC_MisU(c) ((c) == 'U' || (c) == 'u')
#define MKL_DC_MisL(c) ((c) == 'L' || (c) == 'l')
#define MKL_DC_MN(M, LDM, r, c) (M[(r) + (c) * (LDM)])
#define MKL_DC_MT(M, LDM, r, c) (M[(c) + (r) * (LDM)])
#define MKL_DC_MOV(x, y) x = y
#define MKL_DC_NOOP(x, y)
#define MKL_DC_ZERO_C(c, beta) MKL_DC_SET_ZERO(c)
#define MKL_DC_MUL_C(c, beta) MKL_DC_MUL(c, beta, c)
#define MKL_DC_DIV_B(b, a) MKL_DC_DIV((b), (b), (a))
#define MKL_DC_MIN(a, b) ((a) < (b) ? (a) : (b))
#define MKL_DC_MAX(a, b) ((a) > (b) ? (a) : (b))
#define MKL_DC_MAX3(x, y, z) MKL_DC_MAX(MKL_DC_MAX(x, y), z)
#define MKL_DC_ABS(x) ((x) > 0 ? (x) : -(x))
#define MKL_DC_XCONCAT3(x, y, z) x ## y ## z
#define MKL_DC_CONCAT3(x, y, z) MKL_DC_XCONCAT3(x, y, z)
#define MKL_DC_R_SIGN(r, x, y) do { \
    if ((y) >= 0) { \
        (r) = MKL_DC_ABS(x); \
    } else { \
        (r) = -MKL_DC_ABS(x); \
    } \
} while (0)

#define MKL_DC_SET_R_ONE(x) ((x) = 1.0)
#define MKL_DC_SET_R_ZERO(x) ((x) = 0.0)
#define MKL_DC_IS_R_ZERO(x) ((x) == 0.0)

#if defined(MKL_DOUBLE) || defined(MKL_COMPLEX16)
    #define MKL_DC_R_SQRT(r, x) ((r) = sqrt(x))
    #define MKL_DC_XLAMCH(cmach) dlamch(cmach)
#endif

#if defined(MKL_SINGLE) || defined(MKL_COMPLEX)
    #define MKL_DC_R_SQRT(r, x) ((r) = sqrtf(x))
    #define MKL_DC_XLAMCH(cmach) slamch(cmach)
#endif

#ifdef __GNUC__
#define MKL_DC_PRAGMA_VECTOR 
#else
#define MKL_DC_PRAGMA_VECTOR __pragma(vector always)
#endif

/* Vector operations */
#if MKL_DC_UNSAFE
#define MKL_DC_NRM2_VEC(xnorm, n, x, ldx, a_access) \
do { \
    MKL_INT _i; \
    (xnorm) = 0; \
    for (_i = 0; _i < (n); _i++) { \
        (xnorm) += MKL_DC_NRM2(a_access(x, ldx, _i, 0)); \
    } \
    MKL_DC_R_SQRT(xnorm, xnorm); \
} while (0)

#define MKL_DC_R_PY2(r, x, y) MKL_DC_R_SQRT(r, (x) * (x)  + (y) * (y))
#define MKL_DC_R_PY3(r, x, y, z) MKL_DC_R_SQRT(r, (x) * (x)  + (y) * (y) + (z) * (z))
#else
#if defined(MKL_SINGLE) || defined(MKL_DOUBLE)
#define MKL_DC_NRM2_VEC(xnorm, n, x, ldx, a_access) \
do { \
    MKL_INT _i; \
    mkl_dc_real_type scale; \
    mkl_dc_real_type ssq; \
    MKL_DC_SET_R_ZERO(scale); \
    MKL_DC_SET_R_ONE(ssq); \
    if ((n) < 1) { \
        MKL_DC_SET_R_ZERO(xnorm); \
        break; \
    } else if ((n) == 1) { \
        (xnorm) = MKL_DC_ABS(a_access(x, ldx, 0, 0)); \
        break; \
    } \
    for (_i = 0; _i < (n); _i++) { \
        mkl_dc_real_type xi = a_access(x, ldx, _i, 0); \
        if (!MKL_DC_IS_R_ZERO(xi)) { \
            mkl_dc_real_type _temp = MKL_DC_ABS(xi); \
            if (scale < _temp) { \
                ssq = 1 + ssq * (scale / _temp) * (scale / _temp); \
                scale = _temp; \
            } else { \
                ssq += (_temp / scale) * (_temp / scale); \
            } \
        } \
    } \
    MKL_DC_R_SQRT(xnorm, ssq); \
    (xnorm) *= scale; \
} while (0)
#endif

#if defined(MKL_COMPLEX) || defined(MKL_COMPLEX16)
#define MKL_DC_NRM2_VEC(xnorm, n, x, ldx, a_access) \
do { \
    MKL_INT _i; \
    mkl_dc_real_type scale; \
    mkl_dc_real_type ssq; \
    MKL_DC_SET_R_ZERO(scale); \
    MKL_DC_SET_R_ONE(ssq); \
    if ((n) < 1) { \
        MKL_DC_SET_R_ZERO(xnorm); \
        break; \
    } \
    for (_i = 0; _i < (n); _i++) { \
        mkl_dc_real_type xi_re = a_access(x, ldx, _i, 0).real; \
        mkl_dc_real_type xi_im = a_access(x, ldx, _i, 0).imag; \
        if (!MKL_DC_IS_R_ZERO(xi_re)) { \
            mkl_dc_real_type _temp = MKL_DC_ABS(xi_re); \
            if (scale < _temp) { \
                ssq = 1 + ssq * (scale / _temp) * (scale / _temp); \
                scale = _temp; \
            } else { \
                ssq += (_temp / scale) * (_temp / scale); \
            } \
        } \
        if (!MKL_DC_IS_R_ZERO(xi_im)) { \
            mkl_dc_real_type _temp = MKL_DC_ABS(xi_im); \
            if (scale < _temp) { \
                ssq = 1 + ssq * (scale / _temp) * (scale / _temp); \
                scale = _temp; \
            } else { \
                ssq += (_temp / scale) * (_temp / scale); \
            } \
        } \
    } \
    MKL_DC_R_SQRT(xnorm, ssq); \
    (xnorm) *= scale; \
} while (0)
#endif

#define MKL_DC_R_PY2(r, x, y) \
do { \
    mkl_dc_real_type xabs = MKL_DC_ABS(x); \
    mkl_dc_real_type yabs = MKL_DC_ABS(y); \
    mkl_dc_real_type w = MKL_DC_MAX(xabs, yabs); \
    mkl_dc_real_type z = MKL_DC_MIN(xabs, yabs); \
    if (MKL_DC_IS_R_ZERO(z)) { \
        (r) = w; \
    } else { \
        MKL_DC_R_SQRT(r, 1 + (z / w) * (z / w)); \
        (r) *= w; \
    } \
} while (0)

#define MKL_DC_R_PY3(r, x, y, z) \
do { \
    mkl_dc_real_type xabs = MKL_DC_ABS(x); \
    mkl_dc_real_type yabs = MKL_DC_ABS(y); \
    mkl_dc_real_type zabs = MKL_DC_ABS(z); \
    mkl_dc_real_type w = MKL_DC_MAX3(xabs, yabs, zabs); \
    if (MKL_DC_IS_R_ZERO(w)) { \
        (r) = xabs + yabs + zabs; \
    } else { \
        MKL_DC_R_SQRT(r, (xabs / w) * (xabs / w) + (yabs / w) * (yabs / w) + (zabs / w) * (zabs / w)); \
        (r) *= w; \
    } \
} while (0)

#endif /* MKL_DC_UNSAFE */
