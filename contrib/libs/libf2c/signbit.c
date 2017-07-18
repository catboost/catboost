#include "arith.h"

#ifndef Long
#define Long long
#endif

 int
#ifdef KR_headers
signbit_f2c(x) double *x;
#else
signbit_f2c(double *x)
#endif
{
#ifdef IEEE_MC68k
	if (*(Long*)x & 0x80000000)
		return 1;
#else
#ifdef IEEE_8087
	if (((Long*)x)[1] & 0x80000000)
		return 1;
#endif /*IEEE_8087*/
#endif /*IEEE_MC68k*/
	return 0;
	}
