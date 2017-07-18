#include "f2c.h"

#ifdef KR_headers
double asin();
double r_asin(x) real *x;
#else
#undef abs
#include "math.h"
#ifdef __cplusplus
extern "C" {
#endif
double r_asin(real *x)
#endif
{
return( asin(*x) );
}
#ifdef __cplusplus
}
#endif
