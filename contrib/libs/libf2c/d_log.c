#include "f2c.h"

#ifdef KR_headers
double log();
double d_log(x) doublereal *x;
#else
#undef abs
#include "math.h"
#ifdef __cplusplus
extern "C" {
#endif
double d_log(doublereal *x)
#endif
{
return( log(*x) );
}
#ifdef __cplusplus
}
#endif
