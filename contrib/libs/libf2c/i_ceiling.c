#include "f2c.h"

#ifdef KR_headers
integer i_sceiling(x) real *x;
#else
#ifdef __cplusplus
extern "C" {
#endif
integer i_sceiling(real *x)
#endif
{
#define CEIL(x) ((int)(x) + ((x) > 0 && (x) != (int)(x)))

    return (integer) CEIL(*x);
}
#ifdef __cplusplus
}
#endif


#ifdef KR_headers
integer i_dceiling(x) doublereal *x;
#else
#ifdef __cplusplus
extern "C" {
#endif
integer i_dceiling(doublereal *x)
#endif
{
#define CEIL(x) ((int)(x) + ((x) > 0 && (x) != (int)(x)))

    return (integer) CEIL(*x);
}
#ifdef __cplusplus
}
#endif
