// The lack of #pragma once is intentional:
// its presence breaks compilation of contrib/tools/unbound somehow.

#if defined(__GNUC__) || defined(__clang__)
    #include_next <stdlib.h>
#else
    #ifdef Y_UCRT_INCLUDE_NEXT
        #include Y_UCRT_INCLUDE_NEXT(stdlib.h)
    #else
        #define Y_UCRT_INCLUDE_NEXT(x) <Y_UCRT_INCLUDE/x>
        #include Y_UCRT_INCLUDE_NEXT(stdlib.h)
        #undef Y_UCRT_INCLUDE_NEXT
    #endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

void* reallocarray(void*, size_t, size_t);

#ifdef __cplusplus
} // extern "C"
#endif
