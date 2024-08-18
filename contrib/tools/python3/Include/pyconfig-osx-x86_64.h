#include "pyconfig-osx-arm64.h"

#undef ALIGNOF_MAX_ALIGN_T
#define ALIGNOF_MAX_ALIGN_T 16

#undef SIZEOF_LONG_DOUBLE
#define SIZEOF_LONG_DOUBLE 16

#define HAVE_GCC_ASM_FOR_X87 1
#define HAVE_GCC_ASM_FOR_X64 1
