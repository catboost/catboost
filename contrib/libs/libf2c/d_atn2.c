#include "f2c.h"

#ifdef KR_headers
double atan2();
double d_atn2(x,y) doublereal *x, *y;
#else
#undef abs
#include "math.h"
#ifdef __cplusplus
extern "C" {
#endif
double d_atn2(doublereal *x, doublereal *y)
#endif
{
return( atan2(*x,*y) );
}
#ifdef __cplusplus
}
#endif
