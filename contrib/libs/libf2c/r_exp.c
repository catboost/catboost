#include "f2c.h"

#ifdef KR_headers
double exp();
double r_exp(x) real *x;
#else
#undef abs
#include "math.h"
#ifdef __cplusplus
extern "C" {
#endif
double r_exp(real *x)
#endif
{
return( exp(*x) );
}
#ifdef __cplusplus
}
#endif
