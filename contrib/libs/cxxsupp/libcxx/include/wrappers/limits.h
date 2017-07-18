#if defined(__aarch64__) && !defined(__STDC_VERSION__)
#define __STDC_VERSION__ 199901L
#define __FORCED_STDC_VERSION
#endif

#if defined(_WIN32) && defined(_MSC_VER) && !defined(__clang__)
#include <../../VC/include/limits.h>
#else
#include_next <limits.h>
#endif

#if defined(__FORCED_STDC_VERSION) && defined(__STDC_VERSION__)
#undef __FORCED_STDC_VERSION

#if defined(__GNUC__)
#pragma GCC system_header
#endif

#undef __STDC_VERSION__

#endif
