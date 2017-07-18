#include "f2c.h"

#ifdef KR_headers
double cosh();
double d_cosh(x) doublereal *x;
#else
#undef abs
#include "math.h"
#ifdef __cplusplus
extern "C" {
#endif
double d_cosh(doublereal *x)
#endif
{
return( cosh(*x) );
}
#ifdef __cplusplus
}
#endif
