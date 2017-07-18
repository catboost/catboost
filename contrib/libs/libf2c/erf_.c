#include "f2c.h"
#ifdef __cplusplus
extern "C" {
#endif

#ifndef REAL
#define REAL double
#endif

#ifdef KR_headers
double erf();
REAL erf_(x) real *x;
#else
extern double erf(double);
REAL erf_(real *x)
#endif
{
return( erf((double)*x) );
}
#ifdef __cplusplus
}
#endif
