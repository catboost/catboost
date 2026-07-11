#include "jemalloc-linux.h"

/* sizeof(void *) == 2^LG_SIZEOF_PTR. ARM is 32-bit. */
#undef LG_SIZEOF_PTR
#define LG_SIZEOF_PTR 2

/* MALLOCX_ALIGN depends on LG_SIZEOF_PTR at preprocessor time inside jemalloc-linux.h,
 * so the 64-bit variant was selected. Override with the correct 32-bit form. */
#undef MALLOCX_ALIGN
#define MALLOCX_ALIGN(a) ((int)(ffs((int)(a))-1))
