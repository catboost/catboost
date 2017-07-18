#include "f2c.h"

#define log10e 0.43429448190325182765

#ifdef KR_headers
double log();
double d_lg10(x) doublereal *x;
#else
#undef abs
#include "math.h"
#ifdef __cplusplus
extern "C" {
#endif
double d_lg10(doublereal *x)
#endif
{
return( log10e * log(*x) );
}
#ifdef __cplusplus
}
#endif
