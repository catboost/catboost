#ifndef NUMPY_CORE_SRC_COMMON_NPY_CONFIG_H_
#define NUMPY_CORE_SRC_COMMON_NPY_CONFIG_H_

#include "config.h"
#include "npy_cpu_features.h"
#include "npy_cpu_dispatch.h"
#include "numpy/numpyconfig.h"
#include "numpy/npy_cpu.h"
#include "numpy/npy_os.h"

/* blocklist */

/* Disable broken Sun Workshop Pro math functions */
#ifdef __SUNPRO_C

#undef HAVE_ATAN2
#undef HAVE_ATAN2F
#undef HAVE_ATAN2L

#endif

/* Disable broken functions on z/OS */
#if defined (__MVS__)

#undef HAVE_POWF
#undef HAVE_EXPF
#undef HAVE___THREAD

#endif

/* Disable broken MS math functions */
#if (defined(_MSC_VER) && (_MSC_VER < 1900)) || defined(__MINGW32_VERSION)

#undef HAVE_ATAN2
#undef HAVE_ATAN2F
#undef HAVE_ATAN2L

#undef HAVE_HYPOT
#undef HAVE_HYPOTF
#undef HAVE_HYPOTL

#endif

#if defined(_MSC_VER) && (_MSC_VER >= 1900)

#undef HAVE_CASIN
#undef HAVE_CASINF
#undef HAVE_CASINL
#undef HAVE_CASINH
#undef HAVE_CASINHF
#undef HAVE_CASINHL
#undef HAVE_CATAN
#undef HAVE_CATANF
#undef HAVE_CATANL
#undef HAVE_CATANH
#undef HAVE_CATANHF
#undef HAVE_CATANHL
#undef HAVE_CSQRT
#undef HAVE_CSQRTF
#undef HAVE_CSQRTL
#undef HAVE_CLOG
#undef HAVE_CLOGF
#undef HAVE_CLOGL
#undef HAVE_CACOS
#undef HAVE_CACOSF
#undef HAVE_CACOSL
#undef HAVE_CACOSH
#undef HAVE_CACOSHF
#undef HAVE_CACOSHL

#endif

/* MSVC _hypot messes with fp precision mode on 32-bit, see gh-9567 */
#if defined(_MSC_VER) && (_MSC_VER >= 1900) && !defined(_WIN64)

#undef HAVE_CABS
#undef HAVE_CABSF
#undef HAVE_CABSL

#undef HAVE_HYPOT
#undef HAVE_HYPOTF
#undef HAVE_HYPOTL

#endif


/* Intel C for Windows uses POW for 64 bits longdouble*/
#if defined(_MSC_VER) && defined(__INTEL_COMPILER)
#if defined(HAVE_POWL) && (NPY_SIZEOF_LONGDOUBLE == 8)
#undef HAVE_POWL
#endif
#endif /* defined(_MSC_VER) && defined(__INTEL_COMPILER) */

/* powl gives zero division warning on OS X, see gh-8307 */
#if defined(HAVE_POWL) && defined(NPY_OS_DARWIN)
#undef HAVE_POWL
#endif

#ifdef __CYGWIN__
/* Loss of precision */
#undef HAVE_CASINHL
#undef HAVE_CASINH
#undef HAVE_CASINHF

/* Loss of precision */
#undef HAVE_CATANHL
#undef HAVE_CATANH
#undef HAVE_CATANHF

/* Loss of precision and branch cuts */
#undef HAVE_CATANL
#undef HAVE_CATAN
#undef HAVE_CATANF

/* Branch cuts */
#undef HAVE_CACOSHF
#undef HAVE_CACOSH

/* Branch cuts */
#undef HAVE_CSQRTF
#undef HAVE_CSQRT

/* Branch cuts and loss of precision */
#undef HAVE_CASINF
#undef HAVE_CASIN
#undef HAVE_CASINL

/* Branch cuts */
#undef HAVE_CACOSF
#undef HAVE_CACOS

/* log2(exp2(i)) off by a few eps */
#undef HAVE_LOG2

/* np.power(..., dtype=np.complex256) doesn't report overflow */
#undef HAVE_CPOWL
#undef HAVE_CEXPL

/* Builtin abs reports overflow */
#undef HAVE_CABSL
#undef HAVE_HYPOTL
#endif

/* Disable broken gnu trig functions */
#if defined(HAVE_FEATURES_H)
#include <features.h>

#if defined(__GLIBC__)
#if !__GLIBC_PREREQ(2, 18)

#undef HAVE_CASIN
#undef HAVE_CASINF
#undef HAVE_CASINL
#undef HAVE_CASINH
#undef HAVE_CASINHF
#undef HAVE_CASINHL
#undef HAVE_CATAN
#undef HAVE_CATANF
#undef HAVE_CATANL
#undef HAVE_CATANH
#undef HAVE_CATANHF
#undef HAVE_CATANHL
#undef HAVE_CACOS
#undef HAVE_CACOSF
#undef HAVE_CACOSL
#undef HAVE_CACOSH
#undef HAVE_CACOSHF
#undef HAVE_CACOSHL

#endif  /* __GLIBC_PREREQ(2, 18) */
#endif  /* defined(__GLIBC_PREREQ) */

#endif  /* defined(HAVE_FEATURES_H) */

#endif  /* NUMPY_CORE_SRC_COMMON_NPY_CONFIG_H_ */
