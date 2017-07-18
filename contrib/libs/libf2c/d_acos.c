#include "f2c.h"

#ifdef KR_headers
double acos();
double d_acos(x) doublereal *x;
#else
#undef abs
#include "math.h"
#ifdef __cplusplus
extern "C" {
#endif
double d_acos(doublereal *x)
#endif
{
return( acos(*x) );
}
#ifdef __cplusplus
}
#endif
