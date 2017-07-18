#include "f2c.h"

#ifdef KR_headers
double cosh();
double r_cosh(x) real *x;
#else
#undef abs
#include "math.h"
#ifdef __cplusplus
extern "C" {
#endif
double r_cosh(real *x)
#endif
{
return( cosh(*x) );
}
#ifdef __cplusplus
}
#endif
